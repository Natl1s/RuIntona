import io
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import lmdb
import numpy as np

EMO2LABEL = {"angry": 0, "sad": 1, "neutral": 2, "positive": 3}
LABEL2EMO = {value: key for key, value in EMO2LABEL.items()}
LEN_KEY = b"__len__"


def _is_lmdb_subdir(path: Path) -> bool:
    return path.is_dir() or path.suffix == ""


def open_lmdb_readonly(path: Path) -> lmdb.Environment:
    if not path.exists():
        raise FileNotFoundError(f"LMDB не найден: {path}")
    return lmdb.open(
        str(path),
        subdir=_is_lmdb_subdir(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
    )


def parse_label_to_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Некорректная метка: {value}")
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if index not in LABEL2EMO:
            raise ValueError(f"Некорректный индекс эмоции: {index}")
        return index
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized.isdigit():
            return parse_label_to_index(int(normalized))
        if normalized in EMO2LABEL:
            return EMO2LABEL[normalized]
    raise ValueError(f"Не удалось распарсить метку эмоции: {value}")


def parse_label_to_emotion(value: Any) -> str:
    return LABEL2EMO[parse_label_to_index(value)]


def get_lmdb_length(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw_len = txn.get(LEN_KEY)
    if raw_len is None:
        entries = env.stat()["entries"]
        return max(entries - 1, 0)
    return int(raw_len.decode("utf-8"))


ALLOWED_PICKLE_TYPES = {
    dict, OrderedDict, list, tuple,
    str, int, float, bool, bytes, type(None),
    np.ndarray, np.float32, np.float64, np.int32, np.int64,
    np.int8, np.uint8, np.int16, np.uint16, np.int64, np.float64,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        cls = getattr(__import__(module, fromlist=[name]), name)
        if module.startswith("numpy") or module == "numpy":
            return cls
        if cls not in ALLOWED_PICKLE_TYPES:
            raise pickle.UnpicklingError(
                f"Тип {module}.{name} не входит в whitelist ALLOWED_PICKLE_TYPES"
            )
        return cls


def safe_pickle_loads(data: bytes) -> Any:
    """Безопасная десериализация pickle с whitelist типов."""
    return RestrictedUnpickler(io.BytesIO(data)).load()


def _read_payload(txn: lmdb.Transaction, index: int) -> dict[str, Any]:
    raw = txn.get(str(index).encode("utf-8"))
    if raw is None:
        raise KeyError(f"В LMDB отсутствует ключ {index}")
    payload = safe_pickle_loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Некорректный payload у ключа {index}: ожидается dict")
    return payload


def iter_lmdb_payloads(path: Path):
    env = open_lmdb_readonly(path)
    try:
        total = get_lmdb_length(env)
        with env.begin() as txn:
            for index in range(total):
                yield _read_payload(txn, index)
    finally:
        env.close()


def load_feature_vectors_from_lmdb(
    lmdb_path: Path,
    vectorize_fn: Callable[[np.ndarray], np.ndarray],
    label_kind: str = "emotion",
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    expected_dim = None

    for payload in iter_lmdb_payloads(lmdb_path):
        if "x" not in payload:
            raise KeyError("В payload LMDB отсутствует ключ 'x'")
        array = np.asarray(payload["x"], dtype=np.float32)
        vector = vectorize_fn(array)

        if expected_dim is None:
            expected_dim = vector.shape[0]
        elif vector.shape[0] != expected_dim:
            raise ValueError(
                f"Несовпадение размерности признаков: {vector.shape[0]} != {expected_dim}"
            )

        label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
        if label_kind == "emotion":
            label = parse_label_to_emotion(label_raw)
        elif label_kind == "index":
            label = parse_label_to_index(label_raw)
        else:
            raise ValueError(f"Неизвестный тип метки: {label_kind}")

        features.append(vector)
        labels.append(label)

    if not features:
        raise ValueError(f"LMDB пустой: {lmdb_path}")

    return np.stack(features), np.array(labels)


def load_texts_from_lmdb(
    lmdb_path: Path,
    preprocess_fn: Callable[[str], str] | None = None,
) -> tuple[list[str], np.ndarray]:
    texts = []
    labels = []
    text_keys = ("speaker_text", "text", "transcript", "utterance")

    for payload in iter_lmdb_payloads(lmdb_path):
        text = None
        for key in text_keys:
            if key in payload:
                text = payload[key]
                break
        if text is None:
            raise KeyError(
                "В LMDB payload нет текста. Ожидается один из ключей: "
                "speaker_text/text/transcript/utterance."
            )

        text = str(text)
        if preprocess_fn is not None:
            text = preprocess_fn(text)
        text = text.strip()
        if not text:
            continue

        label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
        labels.append(parse_label_to_emotion(label_raw))
        texts.append(text)

    if not texts:
        raise ValueError(f"В LMDB нет непустых текстов: {lmdb_path}")

    return texts, np.array(labels)


# ---------------------------------------------------------------------------
# Audio feature helpers
# ---------------------------------------------------------------------------


def to_fixed_vector(feat: np.ndarray) -> np.ndarray:
    """Преобразует feature-тензор в вектор фиксированной длины (mean + std)."""
    arr = np.asarray(feat)
    if arr.ndim == 0:
        raise ValueError("Пустой/скалярный feature-тензор")
    if arr.ndim == 1:
        return arr.astype(np.float32)
    mean_part = arr.mean(axis=-1).reshape(-1)
    std_part = arr.std(axis=-1).reshape(-1)
    return np.concatenate([mean_part, std_part]).astype(np.float32)


def load_audio_features_from_lmdb(
    lmdb_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Загружает аудио-признаки из LMDB с vectorize_fn=to_fixed_vector."""
    return load_feature_vectors_from_lmdb(lmdb_path, to_fixed_vector, "emotion")
