"""Smoke tests for audio models: run --mode smoke on tiny synthetic LMDBs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from smoke_helpers import create_audio_lmdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMEOUT = 180


def _has_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


AUDIO_MODELS = [
    pytest.param(
        "audio_models/baseline/logistic_regression.py",
        id="logreg",
    ),
    pytest.param(
        "audio_models/baseline/svm.py",
        id="svm",
    ),
    pytest.param(
        "audio_models/baseline/random_forest.py",
        id="random-forest",
    ),
    pytest.param(
        "audio_models/CNN/CNN.py",
        id="cnn",
        marks=pytest.mark.skipif(
            not _has_module("torch"),
            reason="torch not installed",
        ),
    ),
    pytest.param(
        "audio_models/CNN/CNN_BiLSTM.py",
        id="cnn-bilstm",
        marks=pytest.mark.skipif(
            not _has_module("torch"),
            reason="torch not installed",
        ),
    ),
]


@pytest.fixture(scope="session")
def audio_lmdb_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("smoke_audio")
    train = create_audio_lmdb(tmp / "train", num_samples=80)
    test = create_audio_lmdb(tmp / "test", num_samples=40)
    return train, test


@pytest.mark.parametrize("script_rel", AUDIO_MODELS)
def test_audio_model_smoke(script_rel: str, audio_lmdb_pair: tuple[Path, Path]) -> None:
    train_path, test_path = audio_lmdb_pair
    script = PROJECT_ROOT / "dusha" / "my_experiments" / script_rel
    assert script.exists(), f"Script not found: {script}"

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--mode", "smoke",
            "--train-data-path", str(train_path),
            "--test-data-path", str(test_path),
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    combined = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, (
        f"{script_rel} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "accuracy" in combined.lower(), (
        f"{script_rel}: no 'accuracy' in output"
    )


def test_random_forest_tune_smoke(audio_lmdb_pair: tuple[Path, Path]) -> None:
    """Random Forest --mode tune: RandomizedSearchCV на мини-LMDB."""
    train_path, test_path = audio_lmdb_pair
    script = PROJECT_ROOT / "dusha" / "my_experiments" / "audio_models" / "baseline" / "random_forest.py"
    assert script.exists(), f"Script not found: {script}"

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--mode", "tune",
            "--no-save",
            "--search-iterations", "3",
            "--cv-folds", "2",
            "--max-samples", "40",
            "--train-data-path", str(train_path),
            "--test-data-path", str(test_path),
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    combined = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, (
        f"random_forest --mode tune exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "Лучшие параметры" in combined, (
        "random_forest --mode tune: no best params in output"
    )
    assert "accuracy" in combined.lower(), (
        "random_forest --mode tune: no 'accuracy' in output"
    )


def test_cnn_bilstm_tune_smoke(audio_lmdb_pair: tuple[Path, Path]) -> None:
    """CNN-BiLSTM --mode tune: Optuna-поиск на мини-LMDB (skip без optuna)."""
    pytest.importorskip("optuna", reason="optuna not installed")
    train_path, test_path = audio_lmdb_pair
    script = PROJECT_ROOT / "dusha" / "my_experiments" / "audio_models" / "CNN" / "CNN_BiLSTM.py"
    assert script.exists(), f"Script not found: {script}"

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--mode", "tune",
            "--no-save",
            "--n-trials", "2",
            "--tune-epochs", "2",
            "--retrain-epochs", "1",
            "--max-samples", "40",
            "--device", "cpu",
            "--train-data-path", str(train_path),
            "--test-data-path", str(test_path),
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    combined = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, (
        f"cnn_bilstm --mode tune exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "Лучшие параметры" in combined, (
        "cnn_bilstm --mode tune: no best params in output"
    )
    assert "accuracy" in combined.lower(), (
        "cnn_bilstm --mode tune: no 'accuracy' in output"
    )


def test_stratified_subsample() -> None:
    """_stratified_subsample возвращает ровно max_samples стратифицированных примеров."""
    import numpy as np

    from dusha.my_experiments.audio_models.baseline.random_forest import _stratified_subsample

    X = np.arange(200).reshape(-1, 1).astype(float)
    y = np.array([i % 4 for i in range(200)])

    Xs, ys = _stratified_subsample(X, y, max_samples=40)
    assert len(ys) == 40, f"Expected 40 samples, got {len(ys)}"
    counts = np.bincount(ys)
    assert counts.min() >= 8, f"Stratification lost: {counts}"

    Xs, ys = _stratified_subsample(X, y, max_samples=500)
    assert len(ys) == 200, "max_samples >= len(y): subsample must be ignored"
