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
