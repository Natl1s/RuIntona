"""Smoke tests for text models: run --mode smoke on tiny synthetic LMDBs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from smoke_helpers import create_text_lmdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMEOUT = 600


def _has_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


_FASTTEXT_PATH = PROJECT_ROOT / "dusha" / "my_experiments" / "checkpoints" / "pretrained" / "fasttext" / "cc.ru.300.bin"
_HAS_FASTTEXT = _has_module("gensim") and _FASTTEXT_PATH.exists()

TEXT_MODELS = [
    pytest.param(
        "text_models/baseline/TF-IDF_LogReg.py",
        id="tfidf-logreg",
    ),
    pytest.param(
        "text_models/baseline/Embeddings_LogReg.py",
        id="embeddings-logreg",
        marks=pytest.mark.skipif(
            not _HAS_FASTTEXT,
            reason="gensim not installed or cc.ru.300.bin not found",
        ),
    ),
    pytest.param(
        "text_models/BiLSTM/BiLSTM.py",
        id="bilstm",
        marks=pytest.mark.skipif(
            not (_has_module("torch") and _HAS_FASTTEXT),
            reason="torch not installed or cc.ru.300.bin not found",
        ),
    ),
    pytest.param(
        "text_models/transformers/RuBERT.py",
        id="rubert",
        marks=pytest.mark.skipif(
            not _has_module("transformers"),
            reason="transformers not installed",
        ),
    ),
]


@pytest.fixture(scope="session")
def text_lmdb_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("smoke_text")
    train = create_text_lmdb(tmp / "train", num_samples=80)
    test = create_text_lmdb(tmp / "test", num_samples=40)
    return train, test


@pytest.mark.parametrize("script_rel", TEXT_MODELS)
def test_text_model_smoke(script_rel: str, text_lmdb_pair: tuple[Path, Path]) -> None:
    train_path, test_path = text_lmdb_pair
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
