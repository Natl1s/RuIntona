"""Shared helpers for smoke tests: LMDB creation and assertion utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import numpy as np

EMOTIONS = ["angry", "sad", "neutral", "positive"]
SAMPLES_PER_CLASS = 25
TOTAL_SAMPLES = SAMPLES_PER_CLASS * len(EMOTIONS)
AUDIO_FEATURE_SHAPE = (80, 40)
TEXT_SAMPLES = {
    "angry": [
        "я очень злой сегодня",
        "этот день просто кошмар",
        "меня бесит эта ситуация",
        "прекратите меня раздражать",
        "у меня плохое настроение",
    ],
    "sad": [
        "я грущу без причины",
        "мне очень грустно сегодня",
        "жизнь потеряла смысл",
        "мне одиноко и печально",
        "серый день без радости",
    ],
    "neutral": [
        "сегодня обычный день",
        "я просто иду домой",
        "погода сегодня нормальная",
        "ничего особенного не происходит",
        "обычный рабочий день",
    ],
    "positive": [
        "я очень рад сегодня",
        "прекрасный день для прогулки",
        "жизнь прекрасна",
        "у меня отличное настроение",
        "я счастлив и доволен",
    ],
}


def _build_texts() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        base = TEXT_SAMPLES[emotion]
        filled = []
        while len(filled) < SAMPLES_PER_CLASS:
            filled.extend(base)
        result[emotion] = filled[:SAMPLES_PER_CLASS]
    return result


TEXTS_BY_EMOTION = _build_texts()


def create_text_lmdb(path: Path, num_samples: int = TOTAL_SAMPLES) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=1024 * 1024 * 10)
    with env.begin(write=True) as txn:
        for i in range(num_samples):
            emotion = EMOTIONS[i % len(EMOTIONS)]
            texts = TEXTS_BY_EMOTION[emotion]
            text = texts[i // len(EMOTIONS) % len(texts)]
            payload = {
                "speaker_text": text,
                "text": text,
                "emotion": emotion,
                "y": emotion,
            }
            txn.put(str(i).encode("utf-8"), pickle.dumps(payload))
        txn.put(b"__len__", str(num_samples).encode("utf-8"))
    env.close()
    return path


def create_audio_lmdb(path: Path, num_samples: int = TOTAL_SAMPLES) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    env = lmdb.open(str(path), map_size=1024 * 1024 * 50)
    with env.begin(write=True) as txn:
        for i in range(num_samples):
            emotion = EMOTIONS[i % len(EMOTIONS)]
            feat = rng.randn(*AUDIO_FEATURE_SHAPE).astype(np.float32)
            payload = {
                "x": feat,
                "emotion": emotion,
                "y": emotion,
            }
            txn.put(str(i).encode("utf-8"), pickle.dumps(payload))
        txn.put(b"__len__", str(num_samples).encode("utf-8"))
    env.close()
    return path


def create_multimodal_lmdb(path: Path, num_samples: int = TOTAL_SAMPLES) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    env = lmdb.open(str(path), map_size=1024 * 1024 * 50)
    with env.begin(write=True) as txn:
        for i in range(num_samples):
            emotion = EMOTIONS[i % len(EMOTIONS)]
            texts = TEXTS_BY_EMOTION[emotion]
            text = texts[i // len(EMOTIONS) % len(texts)]
            feat = rng.randn(*AUDIO_FEATURE_SHAPE).astype(np.float32)
            payload = {
                "speaker_text": text,
                "text": text,
                "x": feat,
                "emotion": emotion,
                "y": emotion,
            }
            txn.put(str(i).encode("utf-8"), pickle.dumps(payload))
        txn.put(b"__len__", str(num_samples).encode("utf-8"))
    env.close()
    return path


def assert_loss_decreased(losses: list[float]) -> None:
    assert len(losses) >= 2, f"Need at least 2 loss values, got {len(losses)}"
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}"
    )


def assert_metrics_sane(metrics: dict) -> None:
    for key in ["accuracy", "f1_macro", "balanced_accuracy", "precision_macro", "recall_macro"]:
        if key in metrics:
            val = metrics[key]
            assert 0.0 <= val <= 1.0, f"Metric {key}={val} outside [0, 1]"
