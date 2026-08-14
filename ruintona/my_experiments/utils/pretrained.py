"""
Централизованное разрешение путей к предобученным моделям.

Все модели хранятся в checkpoints/pretrained/ и кешируются через
transformers/huggingface_hub или загружаются вручную.
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


# ---------------------------------------------------------------------------
# HuggingFace Transformers ( текстовые )
# ---------------------------------------------------------------------------

_TEXT_MODELS_DIR = PRETRAINED_DIR / "text"
_RUBERT_MODEL_NAME = "rubert-base-cased"


def get_rubert_path() -> Path:
    """Возвращает локальный путь к RuBERT (или remote name для from_pretrained)."""
    local = _TEXT_MODELS_DIR / _RUBERT_MODEL_NAME
    if local.exists():
        return local
    return Path(f"HuggingFaceTB/{_RUBERT_MODEL_NAME}")


def get_tokenizer_path(model_name: str | None = None) -> str:
    """Возвращает путь/имя токенизатора."""
    name = model_name or _RUBERT_MODEL_NAME
    local = _TEXT_MODELS_DIR / name
    if local.exists():
        return str(local)
    return name


# ---------------------------------------------------------------------------
# HuggingFace Transformers ( аудио )
# ---------------------------------------------------------------------------

_AUDIO_MODELS_DIR = PRETRAINED_DIR / "audio"
_WAV2VEC_MODEL_NAME = "wav2vec2-xls-r-300m"


def get_wav2vec_path() -> Path:
    """Возвращает локальный путь к wav2vec2 (или remote name для from_pretrained)."""
    local = _AUDIO_MODELS_DIR / _WAV2VEC_MODEL_NAME
    if local.exists():
        return local
    return Path(f" facebook/{_WAV2VEC_MODEL_NAME}")


# ---------------------------------------------------------------------------
# Универсальный интерфейс
# ---------------------------------------------------------------------------

def resolve_pretrained(name: str) -> Path:
    """Универсальный резолвер: 'fasttext' → путь, 'rubert' → путь, и т.д."""
    registry = {
        "fasttext": get_fasttext_path,
        "rubert": get_rubert_path,
        "wav2vec": get_wav2vec_path,
        "wav2vec2": get_wav2vec_path,
    }
    factory = registry.get(name.lower())
    if factory is None:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Неизвестная предобученная модель: '{name}'. "
            f"Доступные: {available}"
        )
    return factory()


def list_pretrained() -> list[dict[str, str]]:
    """Возвращает список всех доступных предобученных моделей."""
    models = []
    for name in ("fasttext", "rubert", "wav2vec"):
        path = resolve_pretrained(name)
        models.append({
            "name": name,
            "path": str(path),
            "downloaded": path.exists() if not str(path).startswith("facebook/") and not str(path).startswith("HuggingFaceTB/") else True,
        })
    return models
