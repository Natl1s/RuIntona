import argparse
import builtins
import copy
import json
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.audio_models.CNN.CNN_BiLSTM import EmotionCNNBiLSTM
from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index
from my_experiments.text_models.transformers.RuBERT import EmotionClassifier
from my_experiments.config_utils import (
    MY_EXPERIMENTS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_NAMES,
    load_experiment_config, apply_config_to_args, add_config_arg,
)


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)


DEFAULT_TRAIN_LMDB = TRAIN_DATA_PATH
DEFAULT_TEST_LMDB = TEST_DATA_PATH

DEFAULT_AUDIO_MODEL_PATH = (
    MY_EXPERIMENTS_DIR
    / "audio_models"
    / "CNN"
    / "models_params"
    / "CNN_BiLSTM_combine_balanced_train_model.pt"
)
DEFAULT_TEXT_MODEL_PATH = (
    MY_EXPERIMENTS_DIR
    / "text_models"
    / "transformers"
    / "models_params"
    / "RuBERT_dusha_resd_train_model.pt"
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models_params"
MODEL_NAME = Path(__file__).stem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен --device cuda, но CUDA недоступна.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def _extract_text(payload: dict) -> str:
    text_keys = ("speaker_text", "text", "transcript", "utterance")
    for key in text_keys:
        if key in payload:
            text = str(payload[key]).strip().lower()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
    return ""


def _prepare_audio_tensor(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    elif x.ndim == 3 and x.shape[0] != 1 and x.shape[-1] == 1:
        x = np.transpose(x, (2, 0, 1))
    elif x.ndim != 3:
        raise ValueError(f"Неподдерживаемая форма аудио-тензора: {x.shape}")
    return x


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
                payload = pickle.loads(raw)
                if not isinstance(payload, dict):
                    continue

                text = _extract_text(payload)
                if not text:
                    continue

                if "x" not in payload:
                    continue
                try:
                    _prepare_audio_tensor(np.asarray(payload["x"], dtype=np.float32))
                except (TypeError, ValueError):
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

        payload = pickle.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Некорректный payload у ключа {lmdb_idx}: ожидается dict")

        text = _extract_text(payload)
        if not text:
            raise ValueError(f"Пустой текст в валидном примере (idx={lmdb_idx})")

        if "x" not in payload:
            raise KeyError(f"В payload нет ключа 'x' (idx={lmdb_idx})")
        audio = _prepare_audio_tensor(np.asarray(payload["x"], dtype=np.float32))
        label = parse_label_to_index(payload.get("y", payload.get("label", payload.get("emotion"))))
        return torch.from_numpy(audio), text, int(label)


def build_collate_fn(tokenizer, max_len: int):
    def _collate(batch):
        audios, texts, labels = zip(*batch)

        lengths = torch.tensor([x.shape[-1] for x in audios], dtype=torch.long)
        max_t = int(lengths.max().item())
        padded = []
        for x in audios:
            delta = max_t - x.shape[-1]
            if delta > 0:
                x = nn.functional.pad(x, pad=(0, delta, 0, 0))
            padded.append(x)
        audio_batch = torch.stack(padded, dim=0)

        tokenized = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        label_batch = torch.tensor(labels, dtype=torch.long)
        return audio_batch, lengths, tokenized["input_ids"], tokenized["attention_mask"], label_batch

    return _collate


class FrozenAudioEncoder(nn.Module):
    def __init__(self, model: EmotionCNNBiLSTM):
        super().__init__()
        self.model = model
        self.output_dim = int(model.lstm.hidden_size) * (2 if model.bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feats = self.model.conv(x)
            feats = feats.mean(dim=2)
            feats = feats.permute(0, 2, 1).contiguous()
            out_lengths = self.model._downsample_lengths(lengths.to(feats.device))
            packed = pack_padded_sequence(
                feats,
                lengths=out_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.model.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return lstm_out, out_lengths


class FrozenTextEncoder(nn.Module):
    def __init__(self, model: EmotionClassifier):
        super().__init__()
        self.model = model
        self.output_dim = int(model.bert.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0]
        return cls


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # seq: [B, T, D], lengths: [B]
        scores = self.attn(seq).squeeze(-1)  # [B, T]
        max_t = seq.size(1)
        mask = torch.arange(max_t, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)
        return pooled, weights


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        audio_encoder: FrozenAudioEncoder,
        text_encoder: FrozenTextEncoder,
        projection_dim: int = 256,
        dropout: float = 0.3,
        n_classes: int = 4,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_pool = AttentionPooling(self.audio_encoder.output_dim)
        self.audio_proj = nn.Linear(self.audio_encoder.output_dim, projection_dim)
        self.text_proj = nn.Linear(self.text_encoder.output_dim, projection_dim)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.audio_encoder.eval()
        self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.audio_encoder.eval()
        self.text_encoder.eval()
        return self

    def forward(
        self,
        audio_x: torch.Tensor,
        audio_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_seq, seq_lengths = self.audio_encoder(audio_x, audio_lengths)
        audio_vec, _ = self.audio_pool(audio_seq, seq_lengths)
        text_vec = self.text_encoder(input_ids, attention_mask)

        h_audio = torch.nn.functional.gelu(self.audio_proj(audio_vec))
        h_text = torch.nn.functional.gelu(self.text_proj(text_vec))
        h = torch.cat([h_audio, h_text], dim=1)
        return self.classifier(h)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "wa": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "uar": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_split(model, loader, criterion, device: torch.device, use_cuda: bool, desc: str):
    model.eval()
    running_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for audio_x, audio_lengths, input_ids, attention_mask, labels in tqdm(loader, desc=desc, leave=False):
            audio_x = audio_x.to(device, non_blocking=use_cuda)
            audio_lengths = audio_lengths.to(device, non_blocking=use_cuda)
            input_ids = input_ids.to(device, non_blocking=use_cuda)
            attention_mask = attention_mask.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            logits = model(audio_x, audio_lengths, input_ids, attention_mask)
            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)

            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    mean_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred, mean_loss)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES))))
    return metrics, report_text, conf, y_true, y_pred


def print_eval(title: str, metrics: dict, report_text: str, conf: np.ndarray) -> dict:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(conf)

    return {
        "metrics": metrics,
        "classification_report_text": report_text,
        "classification_report": {},
        "confusion_matrix": conf.tolist(),
    }


def _classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )


def _save_results(
    results_dir: Path,
    args,
    best_epoch: int,
    best_val_f1: float,
    history: list[dict],
    train_export: dict,
    val_export: dict,
    test_export: dict,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"early_fusion_baseline_results_{stamp}.json"

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_val_f1),
        "history": history,
        "args": {
            "train_lmdb": str(args.train_lmdb),
            "test_lmdb": str(args.test_lmdb),
            "audio_model_path": str(args.audio_model_path),
            "text_model_path": str(args.text_model_path),
            "batch_size": int(args.batch_size),
            "val_size": float(args.val_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "projection_dim": int(args.projection_dim),
            "dropout": float(args.dropout),
            "max_len": int(args.max_len) if args.max_len is not None else None,
            "seed": int(args.seed),
            "device": str(args.device),
            "audio_lstm_hidden_size": int(args.audio_lstm_hidden_size),
            "audio_lstm_layers": int(args.audio_lstm_layers),
            "audio_lstm_dropout": float(args.audio_lstm_dropout),
            "audio_unidirectional": bool(args.audio_unidirectional),
            "grad_clip_norm": float(args.grad_clip_norm),
        },
        "train": train_export,
        "val": val_export,
        "test": test_export,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_model(
    model,
    dataset_name: str,
    model_name: str = MODEL_NAME,
    training_params=None,
    test_metrics=None,
) -> tuple[Path, Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    backup_path = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pt"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"

    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), backup_path)

    report_lines = [
        f"model_name: {model_name}",
        f"dataset_name: {dataset_name}",
        f"saved_at: {timestamp}",
        "",
        "training_params:",
        json.dumps(training_params or {}, ensure_ascii=False, indent=2),
        "",
        "test_metrics:",
        json.dumps(test_metrics or {}, ensure_ascii=False, indent=2),
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nМодель сохранена: {model_path.resolve()}")
    print(f"Бэкап: {backup_path.resolve()}")
    print(f"Отчёт: {report_path.resolve()}")
    return model_path, backup_path, report_path


def load_audio_encoder(
    audio_model_path: Path,
    device: torch.device,
    lstm_hidden_size: int,
    lstm_layers: int,
    lstm_dropout: float,
    bidirectional: bool,
) -> FrozenAudioEncoder:
    audio_model = EmotionCNNBiLSTM(
        n_classes=len(TARGET_NAMES),
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        bidirectional=bidirectional,
    ).to(device)
    state_dict = torch.load(audio_model_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError("Некорректный формат checkpoint для аудио модели: ожидается state_dict (dict)")
    audio_model.load_state_dict(state_dict)
    audio_model.eval()
    for param in audio_model.parameters():
        param.requires_grad = False
    return FrozenAudioEncoder(audio_model)


def load_text_encoder(
    text_model_path: Path,
    device: torch.device,
) -> tuple[FrozenTextEncoder, AutoTokenizer, int, dict]:
    try:
        checkpoint = torch.load(text_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(text_model_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Некорректный формат checkpoint для текстовой модели: ожидается dict")
    if "model_state_dict" not in checkpoint or "model_params" not in checkpoint:
        raise KeyError(
            "Checkpoint текстовой модели должен содержать ключи 'model_state_dict' и 'model_params'"
        )

    model_params = checkpoint["model_params"]
    text_model = EmotionClassifier(
        model_name=model_params["backbone_name"],
        num_classes=model_params["n_classes"],
        dropout=model_params["dropout"],
        classifier_hidden_size=model_params.get("classifier_hidden_size"),
    ).to(device)
    text_model.load_state_dict(checkpoint["model_state_dict"])
    text_model.eval()
    for param in text_model.parameters():
        param.requires_grad = False

    tokenizer_dir = Path(str(text_model_path).replace("_model.pt", "_tokenizer"))
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_params["backbone_name"])

    max_len = int(model_params.get("max_len", 128))
    return FrozenTextEncoder(text_model), tokenizer, max_len, model_params


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Early Fusion baseline: frozen CNN+BiLSTM (audio) + frozen RuBERT (text) + trainable MLP."
        )
    )
    parser.add_argument("--train-lmdb", type=Path, default=DEFAULT_TRAIN_LMDB)
    parser.add_argument("--test-lmdb", type=Path, default=DEFAULT_TEST_LMDB)
    parser.add_argument("--audio-model-path", type=Path, default=DEFAULT_AUDIO_MODEL_PATH)
    parser.add_argument("--text-model-path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-len", type=int, default=None, help="Override max_len для токенизации.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--audio-lstm-hidden-size", type=int, default=128)
    parser.add_argument("--audio-lstm-layers", type=int, default=2)
    parser.add_argument("--audio-lstm-dropout", type=float, default=0.2)
    parser.add_argument("--audio-unidirectional", action="store_true")
    add_config_arg(parser)
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config)

    if not args.train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {args.train_lmdb}")
    if not args.test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {args.test_lmdb}")
    if not args.audio_model_path.exists():
        raise FileNotFoundError(f"Аудио модель не найдена: {args.audio_model_path}")
    if not args.text_model_path.exists():
        raise FileNotFoundError(f"Текстовая модель не найдена: {args.text_model_path}")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError(f"val-size должен быть в (0,1), получено: {args.val_size}")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size должен быть > 0, получено: {args.batch_size}")
    if args.epochs <= 0:
        raise ValueError(f"epochs должен быть > 0, получено: {args.epochs}")
    if args.projection_dim <= 0:
        raise ValueError(f"projection-dim должен быть > 0, получено: {args.projection_dim}")
    if args.max_len is not None and args.max_len <= 0:
        raise ValueError(f"max-len должен быть > 0, получено: {args.max_len}")
    if args.grad_clip_norm <= 0:
        raise ValueError(f"grad-clip-norm должен быть > 0, получено: {args.grad_clip_norm}")
    if args.audio_lstm_hidden_size <= 0:
        raise ValueError(f"audio-lstm-hidden-size должен быть > 0, получено: {args.audio_lstm_hidden_size}")
    if args.audio_lstm_layers <= 0:
        raise ValueError(f"audio-lstm-layers должен быть > 0, получено: {args.audio_lstm_layers}")
    if not (0.0 <= args.audio_lstm_dropout < 1.0):
        raise ValueError(
            f"audio-lstm-dropout должен быть в [0, 1), получено: {args.audio_lstm_dropout}"
        )

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"

    print(f"Train LMDB: {args.train_lmdb}")
    print(f"Test LMDB:  {args.test_lmdb}")
    print(f"Audio model: {args.audio_model_path}")
    print(f"Text model:  {args.text_model_path}")
    print(f"Device: {device}")

    audio_encoder = load_audio_encoder(
        audio_model_path=args.audio_model_path,
        device=device,
        lstm_hidden_size=args.audio_lstm_hidden_size,
        lstm_layers=args.audio_lstm_layers,
        lstm_dropout=args.audio_lstm_dropout,
        bidirectional=not args.audio_unidirectional,
    )
    text_encoder, tokenizer, text_max_len, text_model_params = load_text_encoder(
        text_model_path=args.text_model_path,
        device=device,
    )
    max_len = args.max_len if args.max_len is not None else text_max_len

    train_ds = FusionLmdbDataset(args.train_lmdb)
    test_ds = FusionLmdbDataset(args.test_lmdb)

    all_indices = np.arange(len(train_ds), dtype=np.int64)
    tr_idx, val_idx = train_test_split(
        all_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_ds.labels,
    )
    test_indices = np.arange(len(test_ds), dtype=np.int64)

    collate_fn = build_collate_fn(tokenizer=tokenizer, max_len=max_len)
    train_loader = DataLoader(
        Subset(train_ds, tr_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(test_ds, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
    )

    model = EarlyFusionModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        n_classes=len(TARGET_NAMES),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Размер train (валидные мультимодальные): {len(train_ds)}")
    print(f"Размер train-split: {len(tr_idx)}")
    print(f"Размер val: {len(val_idx)}")
    print(f"Размер test (валидные мультимодальные): {len(test_ds)}")
    print(
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, projection_dim={args.projection_dim}, "
        f"dropout={args.dropout}, max_len={max_len}"
    )
    print(
        f"audio_lstm_hidden_size={args.audio_lstm_hidden_size}, audio_lstm_layers={args.audio_lstm_layers}, "
        f"audio_lstm_dropout={args.audio_lstm_dropout}, audio_bidirectional={not args.audio_unidirectional}"
    )
    print(f"text_backbone={text_model_params['backbone_name']}")
    print("epoch | train_loss | train_f1 | val_loss | val_f1")

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        train_preds = []
        train_true = []

        progress = tqdm(train_loader, desc=f"Train {epoch:02d}/{args.epochs}", leave=False)
        for audio_x, audio_lengths, input_ids, attention_mask, labels in progress:
            audio_x = audio_x.to(device, non_blocking=use_cuda)
            audio_lengths = audio_lengths.to(device, non_blocking=use_cuda)
            input_ids = input_ids.to(device, non_blocking=use_cuda)
            attention_mask = attention_mask.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            logits = model(audio_x, audio_lengths, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad], args.grad_clip_norm
            )
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.append(preds.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())
            progress.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

        y_train_pred = np.concatenate(train_preds, axis=0)
        y_train_true = np.concatenate(train_true, axis=0)
        train_loss = running_loss / len(tr_idx)
        train_f1 = f1_score(y_train_true, y_train_pred, average="macro", zero_division=0)

        val_metrics, _, _, _, _ = evaluate_split(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_cuda=use_cuda,
            desc="Eval Val",
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_f1_macro": float(train_f1),
                "val_loss": float(val_metrics["loss"]),
                "val_f1_macro": float(val_metrics["f1_macro"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_uar": float(val_metrics["uar"]),
            }
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"{epoch:02d} | {train_loss:.4f} | {train_f1:.4f} | "
            f"{val_metrics['loss']:.4f} | {val_metrics['f1_macro']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Не удалось сохранить best_state во время обучения.")
    model.load_state_dict(best_state)

    train_metrics, train_report_text, train_cm, y_train_true, y_train_pred = evaluate_split(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        use_cuda=use_cuda,
        desc="Final Train Eval",
    )
    val_metrics, val_report_text, val_cm, y_val_true, y_val_pred = evaluate_split(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_cuda=use_cuda,
        desc="Final Val Eval",
    )
    test_metrics, test_report_text, test_cm, y_test_true, y_test_pred = evaluate_split(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_cuda=use_cuda,
        desc="Final Test Eval",
    )

    train_export = print_eval("EARLY FUSION @ TRAIN", train_metrics, train_report_text, train_cm)
    val_export = print_eval("EARLY FUSION @ VAL", val_metrics, val_report_text, val_cm)
    test_export = print_eval("EARLY FUSION @ TEST", test_metrics, test_report_text, test_cm)

    train_export["classification_report"] = _classification_report_dict(y_train_true, y_train_pred)
    val_export["classification_report"] = _classification_report_dict(y_val_true, y_val_pred)
    test_export["classification_report"] = _classification_report_dict(y_test_true, y_test_pred)

    save_model(
        model=model,
        dataset_name=args.train_lmdb.stem,
        training_params={
            "train_lmdb": str(args.train_lmdb),
            "test_lmdb": str(args.test_lmdb),
            "audio_model_path": str(args.audio_model_path),
            "text_model_path": str(args.text_model_path),
            "batch_size": int(args.batch_size),
            "val_size": float(args.val_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "projection_dim": int(args.projection_dim),
            "dropout": float(args.dropout),
            "max_len": int(max_len),
            "seed": int(args.seed),
            "device": str(device),
            "audio_lstm_hidden_size": int(args.audio_lstm_hidden_size),
            "audio_lstm_layers": int(args.audio_lstm_layers),
            "audio_lstm_dropout": float(args.audio_lstm_dropout),
            "audio_bidirectional": bool(not args.audio_unidirectional),
            "grad_clip_norm": float(args.grad_clip_norm),
            "best_epoch": int(best_epoch),
            "best_val_f1_macro": float(best_val_f1),
            "text_backbone": str(text_model_params.get("backbone_name")),
        },
        test_metrics=test_export.get("metrics", {}),
    )

    results_path = _save_results(
        results_dir=args.results_dir,
        args=args,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        history=history,
        train_export=train_export,
        val_export=val_export,
        test_export=test_export,
    )
    print(f"\nРезультаты сохранены: {results_path}")


if __name__ == "__main__":
    main()
