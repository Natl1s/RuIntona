"""
Инференс обученных моделей RuIntona: audio / text / late-fusion.

Единая точка для предсказания эмоции по аудиофайлу и/или тексту.

Примеры:
    # Мультимодально (аудио + текст)
    python ruintona/my_experiments/inference.py --model late-fusion \
        --audio ruintona/DEMO/data/001ce26c07c20eaa0d666b824c6c6924.wav \
        --text "шестьдесят тысяч тенге сколько будет стоить"

    # Только аудио
    python ruintona/my_experiments/inference.py --model audio --audio sample.wav

    # Только текст
    python ruintona/my_experiments/inference.py --model text --text "привет!"
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from pathlib import Path

import librosa

# RuBERT загружается как BertModel без MLM-головы (cls.*) — убираем шумный
# ворнинг про неиспользуемые веса, не трогая остальные сообщения transformers.
class _UnusedWeightsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Some weights of the model checkpoint" not in record.getMessage()


def _silence_unused_weights_warning() -> None:
    """Добавляет фильтр на все логгеры transformers и их хендлеры.

    В Python logging фильтры логгера применяются только к записям,
    созданным на самом этом логгере, поэтому родительского логгера
    мало — фильтруем логгер-источник и хендлеры.
    """
    silence = _UnusedWeightsFilter()
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and (name == "transformers" or name.startswith("transformers.")):
            logger.addFilter(silence)
            for handler in logger.handlers:
                handler.addFilter(silence)
import numpy as np
import torch

from ruintona.my_experiments.audio_models.CNN.CNN_BiLSTM import EmotionCNNBiLSTM
from ruintona.my_experiments.multimodal.late_fusion.Late_Fusion_CNN_BiLSTM_RuBERT import (
    load_audio_model as _load_audio_model,
)
from ruintona.my_experiments.text_models.transformers.RuBERT import EmotionClassifier
from ruintona.my_experiments.utils.config_utils import CHECKPOINTS_DIR, TARGET_NAMES, find_pretrained_model
from ruintona.my_experiments.utils.model_io import load_torch_with_weights

# ---------------------------------------------------------------------------
# Пути к обученным моделям
# ---------------------------------------------------------------------------

AUDIO_MODEL_PATH = (
    find_pretrained_model("audio", "CNN_BiLSTM", "combine_balanced_train")
    or CHECKPOINTS_DIR / "audio" / "CNN_BiLSTM_combine_balanced_train_model.pt"
)
TEXT_MODEL_PATH = (
    find_pretrained_model("text", "RuBERT", "dusha_resd_train")
    or CHECKPOINTS_DIR / "text" / "RuBERT_dusha_resd_train_model.pt"
)
TOKENIZER_DIR = CHECKPOINTS_DIR / "text" / "RuBERT_dusha_resd_train_tokenizer"

DEFAULT_ALPHA = 0.5

MODELS = {
    "audio": "CNN-BiLSTM (аудио, mel-спектрограмма)",
    "text": "RuBERT (текст)",
    "late-fusion": "audio + text (soft-voting, alpha=0.5)",
}

SAMPLE_RATE = 16000
N_MELS = 64


# ---------------------------------------------------------------------------
# Признаки (идентично data_processing/utils/calculate_features.py)
# ---------------------------------------------------------------------------

def extract_mel(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Строит mel-спектрограмму формы (1, 64, T), как при обучении."""
    hop_length = int(sample_rate * 0.01)
    win_length = int(sample_rate * 0.02)
    spec = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=win_length,
        n_mels=N_MELS,
    )
    mel_spec = librosa.power_to_db(spec, ref=np.max)
    return mel_spec[None].astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """Загружает аудио и приводит к 16 кГц, mono, float32."""
    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.asarray(waveform, dtype=np.float32)


# ---------------------------------------------------------------------------
# Загрузка моделей (лениво, с кэшем — один раз на процесс)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_audio_model() -> EmotionCNNBiLSTM:
    print("Загрузка аудио-модели CNN-BiLSTM...")
    device = torch.device("cpu")
    return _load_audio_model(AUDIO_MODEL_PATH, device)


@lru_cache(maxsize=1)
def _get_text_model() -> tuple[EmotionClassifier, object, int]:
    """Возвращает (модель RuBERT, токенизатор, max_len)."""
    print("Загрузка текстовой модели RuBERT (~714 MB)...")
    device = torch.device("cpu")
    if not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Чекпоинт RuBERT не найден: {TEXT_MODEL_PATH}")

    checkpoint = load_torch_with_weights(TEXT_MODEL_PATH, map_location=device)
    model_params = checkpoint.get("model_params", {})
    if not model_params:
        raise ValueError("В чекпоинте RuBERT нет model_params.")

    _silence_unused_weights_warning()

    model = EmotionClassifier(
        model_name=model_params["backbone_name"],
        num_classes=model_params["n_classes"],
        dropout=model_params["dropout"],
        classifier_hidden_size=model_params.get("classifier_hidden_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer_dir = TOKENIZER_DIR if TOKENIZER_DIR.exists() else None
    if tokenizer_dir is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_params["backbone_name"])

    max_len = int(model_params.get("max_len", 128))
    return model, tokenizer, max_len


# ---------------------------------------------------------------------------
# Предсказание
# ---------------------------------------------------------------------------

def _audio_probs(audio_path: str | Path) -> np.ndarray:
    model = _get_audio_model()
    mel = extract_mel(load_audio(audio_path))  # (1, 64, T)
    x = torch.from_numpy(mel).unsqueeze(0)     # (1, 1, 64, T)
    lengths = torch.tensor([mel.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, lengths)
    return torch.softmax(logits, dim=1)[0].numpy().astype(np.float64)


def _text_probs(text: str) -> np.ndarray:
    model, tokenizer, max_len = _get_text_model()
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
    return torch.softmax(logits, dim=1)[0].numpy().astype(np.float64)


def predict(
    audio: str | Path | None = None,
    text: str | None = None,
    model: str = "late-fusion",
    topk: int = 3,
    alpha: float | None = None,
) -> dict:
    """
    Предсказывает эмоцию по аудио и/или тексту.

    Args:
        audio: путь к аудиофайлу (.wav/.mp3/...)
        text: строка текста
        model: "audio" | "text" | "late-fusion"
        topk: сколько лучших классов вернуть в top3
        alpha: вес аудио в late-fusion (по умолчанию 0.5)

    Returns:
        dict с ключами emotion, confidence, top3, probs
    """
    model = str(model).strip().lower()
    if model not in MODELS:
        raise ValueError(
            f"Неизвестная модель: '{model}'. Доступные: {', '.join(MODELS)}"
        )
    if topk < 1:
        raise ValueError(f"topk должен быть >= 1, получено: {topk}")

    probs_audio = None
    probs_text = None

    if model in ("audio", "late-fusion"):
        if audio is None:
            raise ValueError(
                f"Для модели '{model}' нужно аудио: передайте --audio <файл>."
            )
        probs_audio = _audio_probs(audio)

    if model in ("text", "late-fusion"):
        if text is None or not str(text).strip():
            raise ValueError(
                f"Для модели '{model}' нужен текст: передайте --text <строка>."
            )
        probs_text = _text_probs(str(text))

    if model == "late-fusion":
        assert probs_audio is not None and probs_text is not None
        a = float(alpha) if alpha is not None else DEFAULT_ALPHA
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha должен быть в [0, 1], получено: {a}")
        probs = a * probs_audio + (1.0 - a) * probs_text
    elif model == "audio":
        probs = probs_audio
    else:
        probs = probs_text

    probs = probs / probs.sum()

    order = np.argsort(probs)[::-1]
    topk = min(topk, len(order))
    top = [(TARGET_NAMES[i], float(probs[i])) for i in order[:topk]]

    return {
        "emotion": TARGET_NAMES[order[0]],
        "confidence": float(probs[order[0]]),
        "top3": top,
        "probs": {TARGET_NAMES[i]: float(probs[i]) for i in range(len(TARGET_NAMES))},
        "modality": model,
    }


def _format_result(result: dict) -> str:
    lines = [f"Эмоция: {result['emotion']} (уверенность: {result['confidence']:.2f})", ""]
    lines.append("Top-3:")
    for emotion, prob in result["top3"]:
        lines.append(f"  {emotion:<8} {prob:.1%}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Мини-CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Инференс моделей RuIntona (audio / text / late-fusion)."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="late-fusion",
        help="Имя модели из реестра.",
    )
    parser.add_argument("--audio", type=Path, default=None, help="Путь к аудиофайлу.")
    parser.add_argument("--text", type=str, default=None, help="Текст (для text/late-fusion).")
    parser.add_argument("--topk", type=int, default=3, help="Сколько лучших классов вывести.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Вес аудио в late-fusion (по умолчанию 0.5).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать доступные модели и выйти.",
    )
    args = parser.parse_args()

    if args.list:
        print("Доступные модели:")
        for name, desc in MODELS.items():
            print(f"  {name:<14} {desc}")
        return

    result = predict(
        audio=args.audio,
        text=args.text,
        model=args.model,
        topk=args.topk,
        alpha=args.alpha,
    )
    print(_format_result(result))


if __name__ == "__main__":
    main()
