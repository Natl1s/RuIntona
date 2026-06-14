import argparse
import builtins
import json
import pickle
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

PROJECT_ROOT = None
for parent in Path(__file__).resolve().parents:
    if parent.name == "my_experiments":
        PROJECT_ROOT = parent.parent
        break
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index


TARGET_NAMES = ["angry", "sad", "neutral", "positive"]
TARGET_SAMPLE_RATE = 16000


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            text = str(payload[key]).strip().lower()
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


def _top2_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    top2 = np.argsort(probs, axis=1)[:, -2:]
    hits = np.any(top2 == y_true[:, None], axis=1)
    return float(hits.mean())


def _safe_roc_auc_ovr_macro(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_label_name(value: Any) -> str:
    return str(value).strip().lower()


LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "angry": (
        "anger",
        "angr",
        "mad",
        "гнев",
        "злость",
        "злой",
        "ярость",
    ),
    "sad": (
        "sadness",
        "sad",
        "грусть",
        "печаль",
        "тоска",
    ),
    "neutral": (
        "neutral",
        "neutrality",
        "neu",
        "нейтр",
        "нейтрально",
        "нейтральный",
    ),
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


def _extract_label_to_index(model: AutoModel | None) -> dict[str, int] | None:
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

    if isinstance(output, list) and output and isinstance(output[0], dict):
        score_map_raw = {item.get("label"): item.get("score") for item in output}
        score_map = {
            _normalize_label_name(key): value
            for key, value in score_map_raw.items()
            if key is not None
        }
        if any(v is not None for v in score_map_raw.values()):
            scores = []
            for idx, name in enumerate(label_names):
                name_key = _normalize_label_name(name)
                match_key = _resolve_label_key(name_key, set(score_map.keys()))
                if match_key is None:
                    alt = f"LABEL_{idx}"
                    alt_key = _normalize_label_name(alt)
                    if alt_key in score_map:
                        scores.append(score_map[alt_key])
                        continue
                    raise ValueError(
                        f"В выходе pipeline отсутствует метка '{name}' и '{alt}'. "
                        f"Доступные: {list(score_map_raw.keys())}"
                    )
                scores.append(score_map[match_key])
            return np.asarray(scores, dtype=np.float64)

    arr = _as_numpy(output).squeeze()
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(
            f"Неподдерживаемая форма выхода pipeline: {arr.shape}. "
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
            f"Неподдерживаемая форма выхода pipeline: {arr.shape}. "
            f"Ожидается вектор длины {len(label_names)}."
        )
    return arr.astype(np.float64)


def _scores_to_probs(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if np.all(scores >= 0.0) and np.all(scores <= 1.0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
        probs = scores
        logits = np.log(probs + 1e-12)
        return logits, probs
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return scores, probs


def _prepare_model_inputs(
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    samples: list[dict],
    device: str,
) -> dict[str, torch.Tensor]:
    texts = [sample["text"] for sample in samples]
    audios = [sample["audio"]["array"] for sample in samples]

    text_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    audio_inputs = processor(
        audios,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        "input_ids": text_inputs["input_ids"],
        "text_attention_mask": text_inputs.get("attention_mask"),
        "token_type_ids": text_inputs.get("token_type_ids"),
        "input_values": audio_inputs["input_values"],
        "audio_attention_mask": audio_inputs.get("attention_mask"),
    }
    return {key: value.to(device) for key, value in inputs.items() if value is not None}


def _run_model_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    samples: list[dict],
    device: str,
) -> list[np.ndarray]:
    inputs = _prepare_model_inputs(tokenizer, processor, samples, device)
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True)
    logits = outputs.logits
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return [logits[idx].detach().cpu().numpy() for idx in range(logits.shape[0])]


def _iter_lmdb_samples(lmdb_path: Path, skipped: dict[str, int]):
    env = open_lmdb_readonly(lmdb_path)
    try:
        total = get_lmdb_length(env)
        with env.begin() as txn:
            for idx in tqdm(range(total), desc="Чтение LMDB", unit="sample"):
                raw = txn.get(str(idx).encode("utf-8"))
                if raw is None:
                    skipped["missing_key"] += 1
                    continue
                payload = pickle.loads(raw)
                if not isinstance(payload, dict):
                    skipped["bad_payload"] += 1
                    continue

                text = _extract_text(payload)
                if not text:
                    skipped["empty_text"] += 1
                    continue

                waveform_raw = payload.get("waveform", payload.get("audio", payload.get("wav")))
                if waveform_raw is None:
                    skipped["missing_audio"] += 1
                    continue
                waveform = _normalize_waveform(waveform_raw)
                if waveform.size == 0:
                    skipped["empty_audio"] += 1
                    continue

                sample_rate = int(payload.get("waveform_sr", payload.get("sample_rate", TARGET_SAMPLE_RATE)))
                if sample_rate != TARGET_SAMPLE_RATE:
                    skipped["wrong_sample_rate"] += 1
                    continue

                label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
                try:
                    label = parse_label_to_index(label_raw)
                except ValueError:
                    skipped["bad_label"] += 1
                    continue

                yield waveform, text, label
    finally:
        env.close()


def _format_metrics(metrics: dict) -> str:
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Оценка AniEmore wavlm-bert-fusion по тестовой выборке LMDB."
    )
    parser.add_argument(
        "--lmdb-path",
        type=Path,
        default=PROJECT_ROOT
        / "dusha"
        / "data_processing"
        / "dataset"
        / "processed_dataset_090"
        / "aggregated_dataset"
        / "dusha_resd_test.lmdb",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Aniemore/wavlm-bert-fusion-s-emotion-russian-resd",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"batch-size должен быть > 0, получено: {args.batch_size}")

    set_seed(args.seed)
    _ensure_transformers_compat()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.lmdb_path.exists():
        raise FileNotFoundError(f"LMDB не найден: {args.lmdb_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    label_to_index = _extract_label_to_index(model)

    skipped = defaultdict(int)
    y_true = []
    y_pred = []
    probs_list = []
    losses = []
    samples = []
    labels = []

    for waveform, text, label in _iter_lmdb_samples(args.lmdb_path, skipped):
        samples.append(
            {
                "audio": {"array": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
                "text": text,
            }
        )
        labels.append(label)
        if len(samples) >= args.batch_size:
            outputs = _run_model_batch(model, tokenizer, processor, samples, device)
            for output, target in zip(outputs, labels):
                scores = _scores_from_output(output, TARGET_NAMES, label_to_index=label_to_index)
                logits, probs = _scores_to_probs(scores)
                y_true.append(target)
                probs_list.append(probs)
                losses.append(-float(np.log(probs[target] + 1e-12)))
            samples.clear()
            labels.clear()

    if samples:
        outputs = _run_model_batch(model, tokenizer, processor, samples, device)
        for output, target in zip(outputs, labels):
            scores = _scores_from_output(output, TARGET_NAMES, label_to_index=label_to_index)
            logits, probs = _scores_to_probs(scores)
            y_true.append(target)
            probs_list.append(probs)
            losses.append(-float(np.log(probs[target] + 1e-12)))

    if not y_true:
        raise RuntimeError("Нет валидных примеров для оценки.")

    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.stack(probs_list, axis=0)
    y_pred = np.argmax(probs, axis=1)

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "wa": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "top2_accuracy": float(_top2_accuracy(y_true, probs)),
        "roc_auc_ovr_macro": float(_safe_roc_auc_ovr_macro(y_true, probs)),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=list(range(len(TARGET_NAMES)))))
    except ValueError:
        metrics["log_loss"] = float("nan")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES))))

    print("Метрики:")
    print(_format_metrics(metrics))
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(conf)

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "lmdb_path": str(args.lmdb_path),
        "device": device,
        "metrics": metrics,
        "classification_report_text": report_text,
        "confusion_matrix": conf.tolist(),
        "label_names": TARGET_NAMES,
        "samples_used": int(y_true.shape[0]),
        "samples_skipped": dict(skipped),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.results_dir / f"wavlm_bert_fusion_eval_{stem}.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчет сохранен: {report_path}")


if __name__ == "__main__":
    main()
