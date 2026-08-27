"""
Централизованное разрешение путей к предобученным моделям.

Сейчас используется только FastText (cc.ru.300.bin) для текстовых моделей
(Embeddings_LogReg, BiLSTM). Transformers-модели (RuBERT, wav2vec, HuBERT)
загружаются напрямую через from_pretrained с собственными HF-id
(напр. DeepPavlov/rubert-base-cased) и здесь не резолвятся.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Путь к каталогу предобученных моделей
# ---------------------------------------------------------------------------

def _get_pretrained_dir() -> Path:
    """Возвращает checkpoints/pretrained/ относительно my_experiments/."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == "my_experiments":
            return parent / "checkpoints" / "pretrained"
    raise RuntimeError("Не удалось определить my_experiments/ directory")


PRETRAINED_DIR: Path = _get_pretrained_dir()


# ---------------------------------------------------------------------------
# FastText
# ---------------------------------------------------------------------------

FASTTEXT_MODEL_NAME = "cc.ru.300.bin"
FASTTEXT_PATH = PRETRAINED_DIR / "fasttext" / FASTTEXT_MODEL_NAME


def get_fasttext_path() -> Path:
    """Возвращает путь к FastText cc.ru.300.bin."""
    return FASTTEXT_PATH


def load_fasttext_model(**kwargs: Any) -> Any:
    """Загружает FastText модель. Если не скачана — скачивает."""
    from gensim.models.fasttext import load_facebook_model

    path = get_fasttext_path()
    if not path.exists():
        print(f"FastText модель не найдена по пути {path}")
        print("Скачиваем с сервера...")
        _download_fasttext()

    return load_facebook_model(str(path), **kwargs)


def _download_fasttext() -> None:
    """Скачивает cc.ru.300.bin если отсутствует."""
    import urllib.request

    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz"
    target_dir = get_fasttext_path().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    gz_path = target_dir / "cc.ru.300.bin.gz"

    print(f"Скачиваем FastText с {url}...")
    urllib.request.urlretrieve(url, str(gz_path))

    import gzip
    import shutil

    with gzip.open(str(gz_path), "rb") as f_in:
        with open(str(get_fasttext_path()), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print(f"FastText сохранён: {get_fasttext_path()}")
