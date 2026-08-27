"""Tests for utils/pretrained.py path resolution.

Only the FastText path helpers are covered — they are the only ones used by
models (Embeddings_LogReg, BiLSTM). Transformers (RuBERT/wav2vec/HuBERT) are
loaded directly via from_pretrained and are not resolved here.
"""

from __future__ import annotations

from ruintona.my_experiments.utils.pretrained import (
    PRETRAINED_DIR,
    get_fasttext_path,
)


def test_pretrained_dir_is_relative_to_my_experiments() -> None:
    assert PRETRAINED_DIR.parts[-2:] == ("checkpoints", "pretrained")
    assert "my_experiments" in PRETRAINED_DIR.parts


def test_get_fasttext_path_points_into_pretrained_dir() -> None:
    path = get_fasttext_path()
    assert path.name == "cc.ru.300.bin"
    assert path.is_relative_to(PRETRAINED_DIR)
    assert "fasttext" in path.parts
