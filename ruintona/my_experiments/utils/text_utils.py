"""
Общие текстовые утилиты для экспериментов с текстовыми моделями.

extract_text, preprocess_text, load_texts_from_manifest, load_fasttext_model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from ruintona.my_experiments.utils.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb


# ---------------------------------------------------------------------------
# Извлечение текста из LMDB payload
# ---------------------------------------------------------------------------


def extract_text(payload: dict, lowercase: bool = True) -> str:
    """Извлекает текст из LMDB payload по стандартным ключам."""
    text_keys = ("speaker_text", "text", "transcript", "utterance")
    for key in text_keys:
        if key in payload:
            text = str(payload[key]).strip()
            if lowercase:
                text = text.lower()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
    return ""


# ---------------------------------------------------------------------------
# Предобработка текста
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Базовая предобработка: lowercase + удаление лишних пробелов."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Загрузка текстов из LMDB
# ---------------------------------------------------------------------------

def load_texts_from_manifest(
    manifest_path: Path | str,
    preprocess_fn: Callable[[str], str] | None = preprocess_text,
) -> tuple[list[str], np.ndarray]:
    """
    Загружает тексты и метки из LMDB-манифеста.

    Args:
        manifest_path: путь к .lmdb файлу
        preprocess_fn: функция предобработки текста (по умолчанию preprocess_text)

    Returns:
        (texts, labels)
    """
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_fn)


# ---------------------------------------------------------------------------
# FastText
# ---------------------------------------------------------------------------

try:
    from gensim.models.fasttext import load_facebook_model

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


def load_fasttext_model(embeddings_path: Path | str) -> object:
    """
    Загружает предобученную FastText-модель.

    Raises:
        ImportError: если gensim не установлен
        FileNotFoundError: если файл не найден
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Библиотека gensim не установлена!\n"
            "Установите: pip install gensim\n"
            "Или: poetry add gensim"
        )
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Файл с embeddings не найден: {embeddings_path}")

    print(f"Загрузка FastText embeddings из {embeddings_path}...")
    print("Это может занять несколько минут...")
    model = load_facebook_model(str(embeddings_path))
    print("Embeddings загружены!")
    print(f"  - Размерность вектора: {model.wv.vector_size}")
    print(f"  - Количество слов в словаре: {len(model.wv)}")
    return model
