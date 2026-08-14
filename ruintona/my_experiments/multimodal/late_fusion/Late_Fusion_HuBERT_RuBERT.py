import argparse
import builtins
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, HubertForSequenceClassification, Wav2Vec2FeatureExtractor

from ruintona.my_experiments.text_models.transformers.RuBERT import EmotionClassifier
from ruintona.my_experiments.utils.config_utils import (
    CHECKPOINTS_DIR,
    add_data_path_args,
    find_pretrained_model,
    resolve_data_paths,
)
from ruintona.my_experiments.utils.lmdb_utils import (
    get_lmdb_length,
    open_lmdb_readonly,
    parse_label_to_index,
    safe_pickle_loads,
)


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)


DEFAULT_AUDIO_MODEL_ID = "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
DEFAULT_FEATURE_EXTRACTOR_ID = "facebook/hubert-large-ls960-ft"
DEFAULT_LABEL_TO_INDEX = {
    "neutral": 0,
    "angry": 1,
    "positive": 2,
    "sad": 3,
    "other": 4,
}
DEFAULT_TEXT_MODEL_PATH = (
    find_pretrained_model("text", "RuBERT", "dusha_resd_train")
    or CHECKPOINTS_DIR / "text" / "RuBERT_dusha_resd_train_model.pt"
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

TARGET_NAMES = ["angry", "sad", "neutral", "positive"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошено устройство cuda, но CUDA недоступна.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def _ensure_transformers_compat() -> None:
    existing = getattr(torch.nn.Module, "all_tied_weights_keys", None)
    if isinstance(existing, property):
        if existing.fset is not None:
            return
    elif existing is not None:
        return

    def _as_keys_dict(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return dict.fromkeys(value)

    def _get_all_tied_weights_keys(self):
        stored = getattr(self, "_all_tied_weights_keys", None)
        if stored is not None:
            return _as_keys_dict(stored)
        keys = getattr(self, "_tied_weights_keys", None)
        return _as_keys_dict(keys)

    def _set_all_tied_weights_keys(self, value):
        setattr(self, "_all_tied_weights_keys", _as_keys_dict(value))

    torch.nn.Module.all_tied_weights_keys = property(
        _get_all_tied_weights_keys, _set_all_tied_weights_keys
    )


def _extract_text(payload: dict) -> str:
    text_keys = ("speaker_text", "text", "transcript", "utterance")
    for key in text_keys:
        if key in payload:
            text = str(payload[key]).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
    return ""


def _normalize_waveform(raw_waveform) -> np.ndarray:
    arr = np.asarray(raw_waveform)
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)

    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
        if peak > 1.0:
            arr = arr / peak
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(arr, -1.0, 1.0)


def _prepare_audio_array(
    waveform: np.ndarray,
    sample_rate: int,
    target_sr: int,
    max_length: int,
) -> np.ndarray:
    tensor = torch.tensor(waveform, dtype=torch.float32)
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=0)
    if sample_rate != target_sr:
        tensor = torchaudio.functional.resample(tensor, orig_freq=sample_rate, new_freq=target_sr)
    if tensor.numel() > max_length:
        tensor = tensor[:max_length]
    elif tensor.numel() < max_length:
        tensor = torch.nn.functional.pad(tensor, (0, max_length - tensor.numel()))
    return tensor.cpu().numpy()


def _prepare_model_inputs(
    feature_extractor: Wav2Vec2FeatureExtractor,
    waveforms: list[np.ndarray],
    sample_rates: list[int],
    max_duration: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    target_sr = int(feature_extractor.sampling_rate)
    max_length = int(target_sr * max_duration)
    arrays = [
        _prepare_audio_array(waveform, sample_rate, target_sr, max_length)
        for waveform, sample_rate in zip(waveforms, sample_rates)
    ]
    inputs = feature_extractor(
        arrays,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _run_audio_model_batch(
    model: HubertForSequenceClassification,
    feature_extractor: Wav2Vec2FeatureExtractor,
    waveforms: list[np.ndarray],
    sample_rates: list[int],
    device: torch.device,
    max_duration: float,
) -> list[np.ndarray]:
    inputs = _prepare_model_inputs(feature_extractor, waveforms, sample_rates, max_duration, device)
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True)
    logits = outputs.logits
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return [logits[idx].detach().cpu().numpy() for idx in range(logits.shape[0])]


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_label_name(value: Any) -> str:
    return str(value).strip().lower()


LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "angry": ("anger", "angr", "mad", "гнев", "злость", "злой", "ярость"),
    "sad": ("sadness", "sad", "грусть", "печаль", "тоска"),
    "neutral": ("neutral", "neutrality", "neu", "нейтр", "нейтрально", "нейтральный"),
    "positive": (
        "positive",
        "pos",
        "happy",
        "happiness",
        "joy",
        "joyful",
        "радость",
        "счастье",
        "позитив",
        "позитивный",
    ),
}


def _resolve_label_key(target_key: str, available: set[str]) -> str | None:
    if target_key in available:
        return target_key
    for alias in LABEL_ALIASES.get(target_key, ()):
        alias_key = _normalize_label_name(alias)
        if alias_key in available:
            return alias_key
    return None


def _resolve_label_to_index(
    label_names: list[str],
    label_to_index: dict[str, int] | None,
) -> tuple[dict[str, int] | None, list[str], set[str]]:
    if not label_to_index:
        return None, list(label_names), set()
    normalized = {_normalize_label_name(key): value for key, value in label_to_index.items()}
    available = set(normalized.keys())
    resolved: dict[str, int] = {}
    missing: list[str] = []
    for name in label_names:
        target_key = _normalize_label_name(name)
        match_key = _resolve_label_key(target_key, available)
        if match_key is None:
            missing.append(name)
            continue
        resolved[target_key] = normalized[match_key]
    if missing:
        return None, missing, available
    return resolved, [], available


def _extract_label_to_index(
    model: HubertForSequenceClassification | None,
) -> dict[str, int] | None:
    if model is None or not hasattr(model, "config"):
        return None
    config = model.config
    id2label = getattr(config, "id2label", None)
    label2id = getattr(config, "label2id", None)

    mapping: dict[str, int] = {}
    if isinstance(id2label, dict):
        for key, label in id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            mapping[_normalize_label_name(label)] = idx
    elif isinstance(id2label, (list, tuple)):
        for idx, label in enumerate(id2label):
            mapping[_normalize_label_name(label)] = idx

    if not mapping and isinstance(label2id, dict):
        for label, idx in label2id.items():
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            mapping[_normalize_label_name(label)] = idx_int

    return mapping or None


def _scores_from_output(
    output: Any,
    label_names: list[str],
    label_to_index: dict[str, int] | None = None,
) -> np.ndarray:
    if isinstance(output, dict):
        for key in ("logits", "scores", "score", "output", "embedding", "features"):
            if key in output:
                output = output[key]
                break

    arr = _as_numpy(output).squeeze()
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(
            f"Неподдерживаемая форма выхода модели: {arr.shape}. "
            f"Ожидается вектор длины {len(label_names)}."
        )

    if label_to_index:
        resolved_map, missing, available = _resolve_label_to_index(label_names, label_to_index)
        if resolved_map is not None:
            aligned = np.zeros(len(label_names), dtype=np.float64)
            for pos, name in enumerate(label_names):
                key = _normalize_label_name(name)
                idx = resolved_map[key]
                if not 0 <= idx < arr.shape[0]:
                    raise ValueError(
                        f"Индекс {idx} для метки '{name}' выходит за границы "
                        f"выхода модели размером {arr.shape[0]}."
                    )
                aligned[pos] = arr[idx]
            return aligned
        if arr.shape[0] != len(label_names):
            available_list = sorted(available)
            raise ValueError(
                "Выход модели не совпадает с TARGET_NAMES и отсутствуют метки: "
                f"{missing}. Размер выхода: {arr.shape[0]}. "
                f"Доступные метки: {available_list}."
            )

    if arr.shape[0] != len(label_names):
        raise ValueError(
            f"Неподдерживаемая форма выхода модели: {arr.shape}. "
            f"Ожидается вектор длины {len(label_names)}."
        )
    return arr.astype(np.float64)


def _scores_to_probs(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if (
        np.all(scores >= 0.0)
        and np.all(scores <= 1.0)
        and np.isclose(scores.sum(), 1.0, atol=1e-3)
    ):
        return scores
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()


class FusionLmdbDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)

        total = get_lmdb_length(self.env)
        valid_indices = []
        valid_labels = []
        with self.env.begin() as txn:
            for idx in tqdm(
                range(total),
                desc=f"Сканирование {self.lmdb_path.name}",
                unit="sample",
            ):
                raw = txn.get(str(idx).encode("utf-8"))
                if raw is None:
                    continue
                payload = safe_pickle_loads(raw)
                if not isinstance(payload, dict):
                    continue

                text = _extract_text(payload)
                if not text:
                    continue

                waveform_raw = payload.get("waveform", payload.get("audio", payload.get("wav")))
                if waveform_raw is None:
                    continue
                waveform = _normalize_waveform(waveform_raw)
                if waveform.size == 0:
                    continue
                sample_rate = int(payload.get("waveform_sr", payload.get("sample_rate", 0)))
                if sample_rate <= 0:
                    continue

                label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
                try:
                    label = parse_label_to_index(label_raw)
                except ValueError:
                    continue

                valid_indices.append(idx)
                valid_labels.append(label)

        if not valid_indices:
            raise ValueError(f"В LMDB нет валидных мультимодальных примеров: {self.lmdb_path}")
        self.indices = np.asarray(valid_indices, dtype=np.int64)
        self.labels = np.asarray(valid_labels, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, item_idx: int):
        lmdb_idx = int(self.indices[item_idx])
        with self.env.begin() as txn:
            raw = txn.get(str(lmdb_idx).encode("utf-8"))
        if raw is None:
            raise KeyError(f"В LMDB отсутствует ключ {lmdb_idx}")

        payload = safe_pickle_loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Некорректный payload у ключа {lmdb_idx}: ожидается dict")

        text = _extract_text(payload)
        if not text:
            raise ValueError(f"Пустой текст в валидном примере (idx={lmdb_idx})")

        waveform_raw = payload.get("waveform", payload.get("audio", payload.get("wav")))
        if waveform_raw is None:
            raise KeyError(f"В payload отсутствует waveform/audio/wav (idx={lmdb_idx})")
        waveform = _normalize_waveform(waveform_raw)
        if waveform.size == 0:
            raise ValueError(f"Пустой waveform в валидном примере (idx={lmdb_idx})")

        sample_rate = int(payload.get("waveform_sr", payload.get("sample_rate", 0)))
        if sample_rate <= 0:
            raise ValueError(f"Некорректная sample_rate={sample_rate} (idx={lmdb_idx})")

        label = parse_label_to_index(payload.get("y", payload.get("label", payload.get("emotion"))))
        return waveform, sample_rate, text, int(label)


def build_collate_fn(tokenizer, max_len: int):
    def _collate(batch):
        waveforms, sample_rates, texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        return (
            list(waveforms),
            list(sample_rates),
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )

    return _collate


def load_audio_model(
    model_name: str,
    feature_extractor_name: str,
    device: torch.device,
) -> tuple[HubertForSequenceClassification, Wav2Vec2FeatureExtractor, dict[str, int]]:
    _ensure_transformers_compat()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_name)
    model = HubertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    label_to_index = _extract_label_to_index(model) or DEFAULT_LABEL_TO_INDEX
    return model, feature_extractor, label_to_index


def load_text_model(
    text_model_path: Path,
    tokenizer_dir: Path | None,
    device: torch.device,
) -> tuple[EmotionClassifier, AutoTokenizer]:
    model_path = Path(text_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден текстовый чекпоинт: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_params" not in checkpoint:
        raise ValueError("Ожидается checkpoint словарь от RuBERT со структурой model_params/model_state_dict.")

    model_params = checkpoint["model_params"]
    model = EmotionClassifier(
        model_name=model_params["backbone_name"],
        num_classes=model_params["n_classes"],
        dropout=model_params["dropout"],
        classifier_hidden_size=model_params.get("classifier_hidden_size"),
    )
    incompat = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompat.missing_keys:
        print(f"WARNING: missing keys in text state_dict: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"WARNING: unexpected keys in text state_dict: {incompat.unexpected_keys}")
    model = model.to(device)
    model.eval()

    if tokenizer_dir is None:
        stem = model_path.stem
        if stem.endswith("_model"):
            tokenizer_dir = model_path.parent / (stem[:-6] + "_tokenizer")
        else:
            tokenizer_dir = model_path.parent / (stem + "_tokenizer")
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_params["backbone_name"])
    return model, tokenizer


def collect_model_probs(
    audio_model: HubertForSequenceClassification,
    feature_extractor: Wav2Vec2FeatureExtractor,
    audio_label_to_index: dict[str, int],
    text_model: EmotionClassifier,
    loader: DataLoader,
    device: torch.device,
    max_duration: float,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_all = []
    probs_audio_all = []
    probs_text_all = []
    with torch.no_grad():
        for waveforms, sample_rates, input_ids, attention_mask, labels in tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            text_logits = text_model(input_ids, attention_mask)
            probs_text = torch.softmax(text_logits, dim=1).detach().cpu().numpy()

            outputs = _run_audio_model_batch(
                audio_model,
                feature_extractor,
                waveforms,
                sample_rates,
                device,
                max_duration,
            )
            probs_audio = []
            for output in outputs:
                scores = _scores_from_output(
                    output,
                    TARGET_NAMES,
                    label_to_index=audio_label_to_index,
                )
                probs_audio.append(_scores_to_probs(scores))

            y_true_all.append(labels.detach().cpu().numpy())
            probs_audio_all.append(np.stack(probs_audio, axis=0))
            probs_text_all.append(probs_text)

    y_true = np.concatenate(y_true_all, axis=0)
    probs_audio = np.concatenate(probs_audio_all, axis=0)
    probs_text = np.concatenate(probs_text_all, axis=0)
    return y_true, probs_audio, probs_text


def evaluate_fusion(y_true: np.ndarray, probs_audio: np.ndarray, probs_text: np.ndarray, alpha: float) -> dict:
    fused_probs = alpha * probs_audio + (1.0 - alpha) * probs_text
    y_pred = np.argmax(fused_probs, axis=1)
    metrics = {
        "alpha": float(alpha),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return {"metrics": metrics, "y_pred": y_pred, "fused_probs": fused_probs}


def print_eval(title: str, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(TARGET_NAMES))),
            target_names=TARGET_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES)))))


def _save_fusion_weights(
    results_dir: Path,
    args,
    best_alpha: float,
    best_val_f1: float,
    val_metrics: dict,
    test_metrics: dict,
    val_confusion_matrix: np.ndarray,
    test_confusion_matrix: np.ndarray,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = results_dir / f"late_fusion_hubert_rubert_weights_{stamp}.json"

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "best_alpha": float(best_alpha),
        "best_val_f1_macro": float(best_val_f1),
        "audio_weight": float(best_alpha),
        "text_weight": float(1.0 - best_alpha),
        "args": {
            "train_lmdb": str(args.train_lmdb),
            "test_lmdb": str(args.test_lmdb),
            "audio_model_name": str(args.audio_model_name),
            "feature_extractor_name": str(args.feature_extractor_name),
            "text_model_path": str(args.text_model_path),
            "text_tokenizer_dir": str(args.text_tokenizer_dir)
            if args.text_tokenizer_dir is not None
            else None,
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
            "val_size": float(args.val_size),
            "alpha_step": float(args.alpha_step),
            "max_duration": float(args.max_duration),
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_confusion_matrix": val_confusion_matrix.tolist(),
        "test_confusion_matrix": test_confusion_matrix.tolist(),
    }
    weights_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return weights_path


def _find_existing_weights(results_dir: Path) -> Path | None:
    candidates = list(results_dir.glob("late_fusion_hubert_rubert_weights_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Late Fusion (soft voting) для HuBERT (audio) + RuBERT (text)."
    )
    add_data_path_args(parser)
    parser.add_argument("--audio-model-name", type=str, default=DEFAULT_AUDIO_MODEL_ID)
    parser.add_argument(
        "--feature-extractor-name",
        type=str,
        default=DEFAULT_FEATURE_EXTRACTOR_ID,
    )
    parser.add_argument("--text-model-path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument(
        "--text-tokenizer-dir",
        type=Path,
        default=None,
        help="Опционально: путь к директории токенайзера. Если не задан, определяется рядом с text-model-path.",
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    args.train_lmdb, args.test_lmdb = resolve_data_paths(args)

    if not args.train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {args.train_lmdb}")
    if not args.test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {args.test_lmdb}")
    if not args.text_model_path.exists():
        raise FileNotFoundError(f"Текстовая модель не найдена: {args.text_model_path}")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError(f"val-size должен быть в (0,1), получено: {args.val_size}")
    if not (0.0 < args.alpha_step <= 1.0):
        raise ValueError(f"alpha-step должен быть в (0,1], получено: {args.alpha_step}")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size должен быть > 0, получено: {args.batch_size}")
    if args.max_duration <= 0:
        raise ValueError(f"max-duration должен быть > 0, получено: {args.max_duration}")

    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Устройство: {device}")
    print(f"Train LMDB: {args.train_lmdb}")
    print(f"Test LMDB:  {args.test_lmdb}")
    print(f"Audio model: {args.audio_model_name}")
    print(f"Feature extractor: {args.feature_extractor_name}")
    print(f"Text model:  {args.text_model_path}")

    audio_model, feature_extractor, audio_label_to_index = load_audio_model(
        args.audio_model_name,
        args.feature_extractor_name,
        device=device,
    )
    text_model, tokenizer = load_text_model(
        args.text_model_path,
        tokenizer_dir=args.text_tokenizer_dir,
        device=device,
    )

    train_ds = FusionLmdbDataset(args.train_lmdb)
    test_ds = FusionLmdbDataset(args.test_lmdb)

    train_indices = np.arange(len(train_ds))
    _, val_indices = train_test_split(
        train_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_ds.labels,
    )
    val_ds = Subset(train_ds, val_indices.tolist())

    collate_fn = build_collate_fn(tokenizer=tokenizer, max_len=args.max_len)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Размер train (валидные мультимодальные): {len(train_ds)}")
    print(f"Размер val: {len(val_ds)}")
    print(f"Размер test (валидные мультимодальные): {len(test_ds)}")

    y_val, p_audio_val, p_text_val = collect_model_probs(
        audio_model=audio_model,
        feature_extractor=feature_extractor,
        audio_label_to_index=audio_label_to_index,
        text_model=text_model,
        loader=val_loader,
        device=device,
        max_duration=args.max_duration,
        desc="Инференс val",
    )
    y_test, p_audio_test, p_text_test = collect_model_probs(
        audio_model=audio_model,
        feature_extractor=feature_extractor,
        audio_label_to_index=audio_label_to_index,
        text_model=text_model,
        loader=test_loader,
        device=device,
        max_duration=args.max_duration,
        desc="Инференс test",
    )

    best_alpha = None
    best_val_f1 = -1.0
    existing_weights = _find_existing_weights(args.results_dir)
    if existing_weights is not None:
        payload = json.loads(existing_weights.read_text(encoding="utf-8"))
        existing_alpha = payload.get("best_alpha")
        if existing_alpha is not None:
            best_alpha = float(existing_alpha)
            best_val_f1 = float(payload.get("best_val_f1_macro", -1.0))
            print(f"\nНайден отчет с коэффициентом: {existing_weights}")
            print("Подбор alpha пропущен, используется сохранённый коэффициент.")
    if best_alpha is None:
        alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.alpha_step), 4)
        print("\nПодбор alpha по val (macro-F1):")
        for alpha in tqdm(alphas, desc="Поиск alpha", unit="alpha"):
            val_eval = evaluate_fusion(y_val, p_audio_val, p_text_val, float(alpha))
            val_f1 = val_eval["metrics"]["f1_macro"]
            tqdm.write(f"alpha={alpha:.2f} -> val_f1_macro={val_f1:.6f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_alpha = float(alpha)

    assert best_alpha is not None
    print(f"\nЛучший alpha: {best_alpha:.2f} (val_f1_macro={best_val_f1:.6f})")

    val_best = evaluate_fusion(y_val, p_audio_val, p_text_val, best_alpha)
    test_best = evaluate_fusion(y_test, p_audio_test, p_text_test, best_alpha)
    print_eval("LATE FUSION @ VAL", y_val, val_best["y_pred"], val_best["metrics"])
    print_eval("LATE FUSION @ TEST", y_test, test_best["y_pred"], test_best["metrics"])

    val_confusion = confusion_matrix(
        y_val, val_best["y_pred"], labels=list(range(len(TARGET_NAMES)))
    )
    test_confusion = confusion_matrix(
        y_test, test_best["y_pred"], labels=list(range(len(TARGET_NAMES)))
    )

    weights_path = _save_fusion_weights(
        results_dir=args.results_dir,
        args=args,
        best_alpha=best_alpha,
        best_val_f1=best_val_f1 if best_val_f1 >= 0 else float(val_best["metrics"]["f1_macro"]),
        val_metrics=val_best["metrics"],
        test_metrics=test_best["metrics"],
        val_confusion_matrix=val_confusion,
        test_confusion_matrix=test_confusion,
    )
    print(f"\nВесовые коэффициенты сохранены: {weights_path}")


if __name__ == "__main__":
    main()
