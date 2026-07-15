"""
BiLSTM для классификации эмоций по тексту из LMDB.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import lmdb
except ImportError as exc:
    raise ImportError("Не найден пакет 'lmdb'. Установите: poetry add lmdb") from exc

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import sys
_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.config_utils import TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_NAMES, EMO2LABEL, get_dataset_name, load_experiment_config, apply_config_to_args, add_config_arg
from my_experiments.torch_utils import set_seed, resolve_device

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem

LABEL2EMO = {v: k for k, v in EMO2LABEL.items()}


@dataclass
class TrainParams:
    embed_dim: int = 256
    hidden_size: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    classifier_dropout: float = 0.3
    bidirectional: bool = True
    max_vocab_size: int = 40000
    min_freq: int = 2
    max_len: int = 256
    epochs: int = 12
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    # Простая токенизация: слова/числа + отдельные символы пунктуации.
    return re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)


def parse_label_to_index(label_raw) -> int:
    if isinstance(label_raw, (np.integer, int)):
        idx = int(label_raw)
        if idx in LABEL2EMO:
            return idx
        raise ValueError(f"Неподдерживаемый числовой label: {label_raw}")

    if isinstance(label_raw, str):
        key = label_raw.strip().lower()
        if key in EMO2LABEL:
            return EMO2LABEL[key]
        if key.isdigit():
            idx = int(key)
            if idx in LABEL2EMO:
                return idx

    raise ValueError(f"Не удалось распарсить label: {label_raw}")


def _decode_lmdb_value(raw: bytes):
    try:
        return pickle.loads(raw)
    except Exception:
        pass

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    raise ValueError("LMDB value не удалось декодировать ни как pickle, ни как JSON")


def _get_text_from_payload(payload: dict) -> str:
    text_candidates = ["speaker_text", "text", "transcript", "utterance", "sentence"]
    for key in text_candidates:
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    return ""


def _get_label_from_payload(payload: dict):
    label_candidates = ["emotion", "y", "label", "target"]
    for key in label_candidates:
        if key in payload:
            return payload[key]
    return None


def load_texts_from_lmdb(lmdb_path: Path, desc: str = "LMDB") -> Tuple[List[str], List[int]]:
    lmdb_path = Path(lmdb_path)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path не найден: {lmdb_path}")

    try:
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=256,
            subdir=lmdb_path.is_dir(),
        )
    except lmdb.Error:
        # Фолбэк на противоположный режим, если формат пути определен неверно.
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=256,
            subdir=not lmdb_path.is_dir(),
        )

    texts: List[str] = []
    labels: List[int] = []
    skipped = 0

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc=f"Чтение {desc}", unit="rec"):
            if key.startswith(b"__"):
                continue
            payload = _decode_lmdb_value(value)
            if not isinstance(payload, dict):
                skipped += 1
                continue

            text = preprocess_text(_get_text_from_payload(payload))
            if not text:
                skipped += 1
                continue

            raw_label = _get_label_from_payload(payload)
            try:
                label = parse_label_to_index(raw_label)
            except Exception:
                skipped += 1
                continue

            texts.append(text)
            labels.append(label)

    env.close()

    if not texts:
        raise ValueError(
            f"Не удалось извлечь тексты/метки из LMDB: {lmdb_path}. "
            f"Проверьте формат payload (ожидаются text/speaker_text и emotion/label/y)."
        )

    print(f"✓ {desc}: прочитано {len(texts)} записей (пропущено {skipped})")
    return texts, labels


def build_vocab(texts: Iterable[str], min_freq: int, max_vocab_size: int) -> Dict[str, int]:
    counter = Counter()
    for text in tqdm(texts, desc="Построение словаря", unit="txt"):
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    tokens = tokenize(text)
    token_ids = [vocab.get(tok, UNK_IDX) for tok in tokens][:max_len]
    if not token_ids:
        token_ids = [UNK_IDX]
    return token_ids


class TextSequenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.sequences = [encode_text(txt, vocab, max_len=max_len) for txt in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(sequences), max_len), PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded, torch.stack(labels), lengths


class TextBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
        classifier_dropout: float,
        bidirectional: bool,
        n_classes: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(out_dim, n_classes),
        )
        self.bidirectional = bidirectional

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(
            emb,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        return self.classifier(last_hidden)


def evaluate_split(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for input_ids, labels, lengths in tqdm(loader, desc="Eval", leave=False):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * input_ids.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    mean_loss = running_loss / len(loader.dataset)

    metrics = {
        "loss": float(mean_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return metrics, y_true, y_pred, report_text, cm


def print_eval_block(title: str, metrics: dict, y_true: np.ndarray, y_pred: np.ndarray):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    for key, value in metrics.items():
        print(f"{key:>15}: {value:.6f}")
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
    print(confusion_matrix(y_true, y_pred))


def save_model(
    model: nn.Module,
    vocab: Dict[str, int],
    dataset_name: str,
    model_config: dict,
    training_params: dict,
    test_metrics: dict,
    model_name: str = MODEL_NAME,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"

    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    backup_path = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pt"
    vocab_path = MODELS_DIR / f"{full_model_name}_vocab.json"
    meta_path = MODELS_DIR / f"{full_model_name}_meta.json"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"

    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), backup_path)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(
        json.dumps({"model_config": model_config, "training_params": training_params}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'=' * 60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Бэкап:  {backup_path.absolute()}")
    print(f"✓ Vocab:  {vocab_path.absolute()}")
    print(f"✓ Meta:   {meta_path.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    print(f"{'=' * 60}")


def load_model(dataset_name: str, model_name: str = MODEL_NAME):
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    vocab_path = MODELS_DIR / f"{full_model_name}_vocab.json"
    meta_path = MODELS_DIR / f"{full_model_name}_meta.json"

    if not model_path.exists() or not vocab_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Модель не найдена! Проверьте наличие файлов:\n"
            f"  {model_path}\n"
            f"  {vocab_path}\n"
            f"  {meta_path}"
        )

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_config = meta["model_config"]

    model = TextBiLSTMClassifier(
        vocab_size=model_config["vocab_size"],
        embed_dim=model_config["embed_dim"],
        hidden_size=model_config["hidden_size"],
        lstm_layers=model_config["lstm_layers"],
        lstm_dropout=model_config["lstm_dropout"],
        classifier_dropout=model_config["classifier_dropout"],
        bidirectional=model_config["bidirectional"],
        n_classes=model_config["n_classes"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print(f"✓ Модель загружена из {model_path}")
    print(f"✓ Vocab загружен из {vocab_path}")
    print(f"✓ Meta загружен из {meta_path}")

    return model, vocab, model_config


def model_exists(dataset_name: str, model_name: str = MODEL_NAME) -> bool:
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    vocab_path = MODELS_DIR / f"{full_model_name}_vocab.json"
    meta_path = MODELS_DIR / f"{full_model_name}_meta.json"
    return model_path.exists() and vocab_path.exists() and meta_path.exists()


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def train_bilstm_text(params: TrainParams, save: bool, device_arg: str):
    set_seed(params.seed)
    device = resolve_device(device_arg)

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    dataset_name = get_dataset_name(train_manifest)

    print(f"📊 Датасет: {dataset_name}")
    print(f"Train LMDB: {train_manifest}")
    print(f"Test LMDB:  {test_manifest}")
    print(f"Устройство: {device}")

    X_train_texts, y_train = load_texts_from_lmdb(train_manifest, desc="train")
    X_test_texts, y_test = load_texts_from_lmdb(test_manifest, desc="test")

    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Количество тестовых примеров: {len(y_test)}")
    train_counts = np.unique(y_train, return_counts=True)
    test_counts = np.unique(y_test, return_counts=True)
    print(f"Распределение классов train: {train_counts}")
    print(f"Распределение классов test:  {test_counts}")
    print(f"Пример текста: '{X_train_texts[0][:160]}'")

    vocab = build_vocab(X_train_texts, min_freq=params.min_freq, max_vocab_size=params.max_vocab_size)
    print(f"✓ Размер словаря: {len(vocab)}")

    train_ds = TextSequenceDataset(X_train_texts, y_train, vocab=vocab, max_len=params.max_len)
    test_ds = TextSequenceDataset(X_test_texts, y_test, vocab=vocab, max_len=params.max_len)

    train_loader = _make_loader(train_ds, params.batch_size, shuffle=True, num_workers=params.num_workers)
    test_loader = _make_loader(test_ds, params.batch_size, shuffle=False, num_workers=params.num_workers)

    model = TextBiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=params.embed_dim,
        hidden_size=params.hidden_size,
        lstm_layers=params.lstm_layers,
        lstm_dropout=params.lstm_dropout,
        classifier_dropout=params.classifier_dropout,
        bidirectional=params.bidirectional,
        n_classes=len(TARGET_NAMES),
    ).to(device)

    # Балансировка классов через веса в CrossEntropy.
    counts = np.bincount(np.asarray(y_train), minlength=len(TARGET_NAMES)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    best_state = None
    best_test_f1 = -1.0

    print(f"\n{'=' * 60}")
    print("СТАРТ ОБУЧЕНИЯ")
    print(f"{'=' * 60}")

    for epoch in range(1, params.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        all_train_preds = []
        all_train_targets = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{params.epochs}", unit="batch")
        for input_ids, labels, lengths in progress:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if params.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.grad_clip)
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            seen_samples += input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_targets.append(labels.detach().cpu().numpy())

            progress.set_postfix(loss=f"{(running_loss / max(1, seen_samples)):.4f}")

        train_pred = np.concatenate(all_train_preds)
        train_true = np.concatenate(all_train_targets)
        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average="macro", zero_division=0)

        test_metrics, _, _, _, _ = evaluate_split(model, test_loader, criterion, device)
        if test_metrics["f1_macro"] > best_test_f1:
            best_test_f1 = test_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d}/{params.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
            f"test_f1={test_metrics['f1_macro']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, y_train_true, y_train_pred, train_report_text, train_cm = evaluate_split(
        model, train_loader, criterion, device
    )
    print_eval_block("ОЦЕНКА НА TRAIN", train_metrics, y_train_true, y_train_pred)

    test_metrics, y_test_true, y_test_pred, test_report_text, test_cm = evaluate_split(
        model, test_loader, criterion, device
    )
    print_eval_block("ОЦЕНКА НА TEST", test_metrics, y_test_true, y_test_pred)

    if save:
        model_config = {
            "vocab_size": len(vocab),
            "embed_dim": params.embed_dim,
            "hidden_size": params.hidden_size,
            "lstm_layers": params.lstm_layers,
            "lstm_dropout": params.lstm_dropout,
            "classifier_dropout": params.classifier_dropout,
            "bidirectional": params.bidirectional,
            "n_classes": len(TARGET_NAMES),
        }
        training_params = {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "lr": params.lr,
            "weight_decay": params.weight_decay,
            "grad_clip": params.grad_clip,
            "seed": params.seed,
            "device": str(device),
            "max_vocab_size": params.max_vocab_size,
            "min_freq": params.min_freq,
            "max_len": params.max_len,
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
        }
        metrics_payload = {
            **test_metrics,
            "test_classification_report_text": test_report_text,
            "test_confusion_matrix": test_cm,
            "train_metrics": train_metrics,
            "train_classification_report_text": train_report_text,
            "train_confusion_matrix": train_cm,
        }
        save_model(
            model=model,
            vocab=vocab,
            dataset_name=dataset_name,
            model_config=model_config,
            training_params=training_params,
            test_metrics=metrics_payload,
        )

    return model, vocab, dataset_name


def load_and_evaluate(device_arg: str, batch_size: int, num_workers: int, max_len: int):
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    dataset_name = get_dataset_name(train_manifest)

    model, vocab, _ = load_model(dataset_name)
    device = resolve_device(device_arg)
    model = model.to(device)

    X_train_texts, y_train = load_texts_from_lmdb(train_manifest, desc="train")
    X_test_texts, y_test = load_texts_from_lmdb(test_manifest, desc="test")

    train_ds = TextSequenceDataset(X_train_texts, y_train, vocab=vocab, max_len=max_len)
    test_ds = TextSequenceDataset(X_test_texts, y_test, vocab=vocab, max_len=max_len)
    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = _make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()

    train_metrics, y_train_true, y_train_pred, _, _ = evaluate_split(model, train_loader, criterion, device)
    print_eval_block("ОЦЕНКА НА TRAIN", train_metrics, y_train_true, y_train_pred)

    test_metrics, y_test_true, y_test_pred, _, _ = evaluate_split(model, test_loader, criterion, device)
    print_eval_block("ОЦЕНКА НА TEST", test_metrics, y_test_true, y_test_pred)

    return model, vocab


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiLSTM модель для классификации 4 эмоций по тексту из LMDB."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load", "auto"],
        default="auto",
        help="Режим работы: train/load/auto",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--classifier-dropout", type=float, default=0.3)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--max-vocab-size", type=int, default=40000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Устройство обучения",
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять модель")
    add_config_arg(parser)
    return parser.parse_args()


def main():
    args = build_args()
    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config)
    dataset_name = get_dataset_name(TRAIN_DATA_PATH)

    params = TrainParams(
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        classifier_dropout=args.classifier_dropout,
        bidirectional=not args.unidirectional,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.mode == "train":
        print("🎯 Режим: Обучение новой модели\n")
        train_bilstm_text(params=params, save=not args.no_save, device_arg=args.device)
    elif args.mode == "load":
        print("📂 Режим: Загрузка существующей модели\n")
        load_and_evaluate(
            device_arg=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_len=args.max_len,
        )
    else:
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            load_and_evaluate(
                device_arg=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_len=args.max_len,
            )
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            train_bilstm_text(params=params, save=not args.no_save, device_arg=args.device)


if __name__ == "__main__":
    main()
