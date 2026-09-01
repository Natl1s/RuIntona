"""Реестр моделей RuIntona на Hugging Face и авто-скачивание чекпоинтов.

Приоритет при поиске чекпоинта:
1. Локальные файлы (ruintona/my_experiments/checkpoints/**).
2. Hugging Face: скачивается в ``CHECKPOINTS_DIR / "hf" / <repo_id>``
   (кэш переиспользуется при последующих запусках и работает офлайн).

Чекпоинты на HF самодостаточны: веса и model_params лежат в одном файле
(формат save_pytorch_model), см. load_audio_model / load_text_model.
"""

from __future__ import annotations

from pathlib import Path

from ruintona.my_experiments.utils.config_utils import CHECKPOINTS_DIR

# ---------------------------------------------------------------------------
# Реестр: ключ -> {repo_id на Hugging Face, имя файла внутри репо, путь по умолчанию}
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "audio": {
        "repo_id": "Natlis/cnn-bilstm-emotion-classification-ru",
        "filename": "CNN_BiLSTM_combine_balanced_train_model.pt",
        "local_rel": "audio/CNN_BiLSTM_combine_balanced_train_model.pt",
    },
    "text": {
        "repo_id": "Natlis/rubert-emotion-classification-ru",
        "filename": "RuBERT_dusha_resd_train_model.pt",
        "local_rel": "text/RuBERT_dusha_resd_train_model.pt",
    },
}


def _download(key: str, entry: dict) -> Path:
    repo_id = entry["repo_id"]
    filename = entry["filename"]
    local_dir = CHECKPOINTS_DIR / "hf" / repo_id.replace("/", "__")
    local_path = CHECKPOINTS_DIR / entry["local_rel"]
    cached = local_dir / filename
    try:
        from huggingface_hub import hf_hub_download

        if not cached.exists():
            print(f"Скачивание '{filename}' с Hugging Face ({repo_id})...")
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir))
    except Exception as exc:  # noqa: BLE001 — единое понятное сообщение для CLI
        raise RuntimeError(
            f"Не удалось скачать '{filename}' с Hugging Face ({repo_id}). "
            f"Причина: {exc}. "
            "Проверьте сеть или положите чекпоинт локально: "
            f"{local_path}."
        ) from exc
    if not cached.exists():
        print(f"Готово: {path}")
    return path


def ensure_checkpoint(
    key: str,
    *,
    download: bool = True,
    local_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    """Возвращает путь к чекпоинту ``key``, используя локальный или HF-файл.

    Args:
        key: ключ из MODEL_REGISTRY ("audio" | "text").
        download: скачивать с Hugging Face, если локального файла нет.
        local_paths: кандидаты локальных путей (проверяются по порядку).
            По умолчанию — CHECKPOINTS_DIR/local_rel из MODEL_REGISTRY (см. «Реестр»).

    Returns:
        Path к существующему чекпоинту (локальному или скачанному).

    Raises:
        KeyError: неизвестный ключ.
        FileNotFoundError: ``download=False`` и локального файла нет.
        RuntimeError: не удалось скачать с Hugging Face.
    """
    key = str(key).strip().lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Неизвестный чекпоинт '{key}'. Доступные ключи: {', '.join(MODEL_REGISTRY)}"
        )

    entry = MODEL_REGISTRY[key]
    candidates = [Path(p) for p in (local_paths or []) if p]
    if not candidates:
        candidates = [CHECKPOINTS_DIR / entry["local_rel"]]

    for path in candidates:
        if path.exists():
            return path

    if not download:
        raise FileNotFoundError(
            f"Чекпоинт '{key}' не найден локально: {candidates[0]}. "
            "Скачивание с Hugging Face отключено (--no-download). "
            "Положите файл по этому пути или уберите флаг --no-download."
        )

    return _download(key, entry)