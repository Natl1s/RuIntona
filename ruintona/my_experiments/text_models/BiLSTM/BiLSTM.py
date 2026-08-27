"""
FastText Embeddings + BiLSTM для классификации эмоций по тексту.
"""

import argparse
import builtins
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, apply_config_to_args, add_config_arg, add_data_path_args, resolve_data_paths
from ruintona.my_experiments.utils.model_io import save_pytorch_model, load_pytorch_model, pytorch_model_exists, save_metrics_report
from ruintona.my_experiments.utils.pretrained import get_fasttext_path
from ruintona.my_experiments.utils.torch_utils import set_seed, resolve_device, EarlyStopping, compute_classification_metrics, _eval_collect
from ruintona.my_experiments.utils.text_utils import preprocess_text, load_fasttext_model, GENSIM_AVAILABLE
from ruintona.my_experiments.utils.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem
DEFAULT_EMBEDDINGS_PATH = get_fasttext_path()

EMO2IDX = {name: i for i, name in enumerate(TARGET_NAMES)}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def load_texts_from_manifest(manifest_path: Path):
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_text)


def build_vocab(texts: list[str], max_vocab_size: int, min_freq: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in word2idx:
            continue
        word2idx[token] = len(word2idx)
        if len(word2idx) >= max_vocab_size:
            break
    return word2idx


def build_embedding_matrix(word2idx: dict[str, int], fasttext_model) -> np.ndarray:
    dim = int(fasttext_model.wv.vector_size)
    matrix = np.zeros((len(word2idx), dim), dtype=np.float32)
    for token, idx in tqdm(word2idx.items(), desc="Инициализация embedding matrix"):
        if idx == PAD_IDX:
            continue
        matrix[idx] = fasttext_model.wv[token]
    return matrix


def encode_text(text: str, word2idx: dict[str, int], max_len: int) -> tuple[np.ndarray, int]:
    tokens = text.split()
    ids = [word2idx.get(tok, UNK_IDX) for tok in tokens[:max_len]]
    length = max(1, len(ids))
    if len(ids) < max_len:
        ids.extend([PAD_IDX] * (max_len - len(ids)))
    return np.asarray(ids, dtype=np.int64), length


class TextSequenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, word2idx: dict[str, int], max_len: int):
        self.input_ids = []
        self.lengths = []
        self.labels = []

        for text, label in zip(texts, labels):
            ids, length = encode_text(text, word2idx, max_len=max_len)
            self.input_ids.append(ids)
            self.lengths.append(length)
            if label not in EMO2IDX:
                raise ValueError(f"Неизвестная метка эмоции: {label}")
            self.labels.append(EMO2IDX[label])

        self.input_ids = np.stack(self.input_ids)
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BiLSTMEmotionClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        freeze_embeddings: bool,
        n_classes: int,
        pooling_mode: str = "mean_max",
    ):
        super().__init__()
        if pooling_mode not in {"mean_max", "last_hidden"}:
            raise ValueError(f"Неподдерживаемый pooling_mode: {pooling_mode}")
        self.pooling_mode = pooling_mode
        emb_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor, freeze=freeze_embeddings, padding_idx=PAD_IDX
        )
        emb_dim = emb_tensor.shape[1]
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_size * 4 if pooling_mode == "mean_max" else hidden_size * 2
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        if self.pooling_mode == "last_hidden":
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            feats = torch.cat([forward_last, backward_last], dim=1)
        else:
            sequence_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1)
            )
            max_steps = sequence_out.size(1)
            time_idx = torch.arange(max_steps, device=lengths.device).unsqueeze(0)
            mask = (time_idx < lengths.unsqueeze(1)).unsqueeze(-1)
            sum_pool = (sequence_out * mask).sum(dim=1)
            mean_pool = sum_pool / lengths.clamp(min=1).unsqueeze(1)
            min_value = torch.finfo(sequence_out.dtype).min
            max_pool = sequence_out.masked_fill(~mask, min_value).max(dim=1).values
            feats = torch.cat([mean_pool, max_pool], dim=1)
        feats = self.dropout(feats)
        return self.classifier(feats)


def evaluate_split(model, loader, criterion, device: torch.device, desc: str):
    def forward_fn(m, batch):
        input_ids, lengths, labels = batch
        return m(input_ids.to(device), lengths.to(device))

    logits_arr, probs_arr, y_pred, y_true, mean_loss = _eval_collect(
        model, loader, criterion, device,
        forward_fn=forward_fn,
        unpack_y_fn=lambda b: b[2],
    )

    metrics = compute_classification_metrics(y_true, y_pred, probs_arr)
    metrics["loss"] = float(mean_loss)
    return metrics, y_true, y_pred, probs_arr, logits_arr


def save_model(
    model: nn.Module,
    dataset_name: str,
    checkpoint_payload: dict,
    training_params: dict,
    test_metrics: dict,
    model_name: str = MODEL_NAME,
):
    extra_artifacts = {
        "embedding_matrix.pkl": checkpoint_payload.get("embedding_matrix"),
        "word2idx.json": checkpoint_payload.get("word2idx"),
    }
    save_pytorch_model(
        checkpoint_payload["model_state_dict"],
        dataset_name,
        models_dir=MODELS_DIR,
        model_name=model_name,
        training_params=training_params,
        test_metrics=test_metrics,
        extra_artifacts=extra_artifacts,
        model_class="BiLSTMEmotionClassifier",
        model_params=checkpoint_payload.get("model_params", {}),
    )


def load_model(dataset_name: str, model_name: str = MODEL_NAME, map_location: str | torch.device = "cpu"):
    checkpoint = load_pytorch_model(dataset_name, models_dir=MODELS_DIR, model_name=model_name, map_location=map_location)

    # Обратная совместимость: поддержка старого формата (embedding_matrix в checkpoint)
    if "embedding_matrix" in checkpoint:
        embedding_matrix = checkpoint["embedding_matrix"]
        model_params = checkpoint["model_params"]
        word2idx = checkpoint.get("word2idx", {})
    elif "extra_artifacts" in checkpoint:
        embedding_matrix = checkpoint["extra_artifacts"].get("embedding_matrix.pkl")
        word2idx = checkpoint["extra_artifacts"].get("word2idx.json", {})
        model_params = checkpoint.get("model_params", {})
    else:
        raise KeyError("Не найдены embedding_matrix или extra_artifacts в checkpoint")

    model = BiLSTMEmotionClassifier(
        embedding_matrix=embedding_matrix,
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_layers"],
        dropout=model_params["dropout"],
        freeze_embeddings=model_params["freeze_embeddings"],
        n_classes=model_params["n_classes"],
        pooling_mode=model_params.get("pooling_mode", "last_hidden"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Модель загружена из {MODELS_DIR}")
    return model, checkpoint


def model_exists(dataset_name: str, model_name: str = MODEL_NAME) -> bool:
    return pytorch_model_exists(dataset_name, models_dir=MODELS_DIR, model_name=model_name)


def _build_loader(dataset, batch_size: int, shuffle: bool, use_cuda: bool):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )


def train_bilstm(
    embeddings_path: Path | None = None,
    save: bool = True,
    epochs: int = 30,
    batch_size: int = 64,
    max_len: int = 64,
    max_vocab_size: int = 50000,
    min_freq: int = 1,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    freeze_embeddings: bool = False,
    lr: float = 1e-3,
    lr_embeddings: float | None = None,
    weight_decay: float = 1e-5,
    val_size: float = 0.1,
    seed: int = 42,
    device_arg: str = "auto",
    train_path=None,
    test_path=None,
    patience: int = 5,
):
    if embeddings_path is None:
        embeddings_path = DEFAULT_EMBEDDINGS_PATH

    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    print(f"Обучение запущено на устройстве: {device}")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА FASTTEXT EMBEDDINGS")
    print(f"{'=' * 60}")
    fasttext_model = load_fasttext_model(embeddings_path)

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"\nДатасет: {dataset_name}\n")

    print(f"{'=' * 60}")
    print("ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ")
    print(f"{'=' * 60}")
    train_texts, y_train_raw = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train_raw)}")
    print(f"Распределение классов в train: {np.unique(y_train_raw, return_counts=True)}")

    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size должен быть в интервале (0, 1), получено: {val_size}")

    train_texts, val_texts, y_train_raw, y_val_raw = train_test_split(
        train_texts,
        y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw,
    )
    print(f"Train после split: {len(y_train_raw)}")
    print(f"Val после split:   {len(y_val_raw)}")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print(f"{'=' * 60}")
    test_texts, y_test_raw = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test_raw)}")
    print(f"Распределение классов в test: {np.unique(y_test_raw, return_counts=True)}")

    print(f"\n{'=' * 60}")
    print("ПОСТРОЕНИЕ СЛОВАРЯ И EMBEDDING MATRIX")
    print(f"{'=' * 60}")
    word2idx = build_vocab(train_texts, max_vocab_size=max_vocab_size, min_freq=min_freq)
    embedding_matrix = build_embedding_matrix(word2idx, fasttext_model)
    print(f"Размер словаря: {len(word2idx)}")
    print(f"Размер embedding matrix: {embedding_matrix.shape}")

    train_ds = TextSequenceDataset(train_texts, y_train_raw, word2idx, max_len=max_len)
    val_ds = TextSequenceDataset(val_texts, y_val_raw, word2idx, max_len=max_len)
    test_ds = TextSequenceDataset(test_texts, y_test_raw, word2idx, max_len=max_len)
    train_loader = _build_loader(train_ds, batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(val_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)
    test_loader = _build_loader(test_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    model = BiLSTMEmotionClassifier(
        embedding_matrix=embedding_matrix,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
        n_classes=len(TARGET_NAMES),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if lr_embeddings is None:
        lr_embeddings = lr * 0.1
    if freeze_embeddings:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        non_embedding_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("embedding.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.embedding.parameters(), "lr": lr_embeddings},
                {"params": non_embedding_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер val:   {len(val_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(
        f"Эпох: {epochs}, batch_size: {batch_size}, max_len: {max_len}, "
        f"hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}"
    )
    print(
        f"freeze_embeddings: {freeze_embeddings}, lr: {lr}, "
        f"lr_embeddings: {0.0 if freeze_embeddings else lr_embeddings}"
    )
    print("epoch | train_loss | train_f1 | val_loss | val_f1")

    best_state = None
    best_val_f1 = -1.0
    history = []
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        all_train_preds = []
        all_train_targets = []

        progress = tqdm(train_loader, desc=f"Train {epoch:02d}/{epochs}", leave=False)
        for input_ids, lengths, labels in progress:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            seen_samples += input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_targets.append(labels.detach().cpu().numpy())
            progress.set_postfix(loss=f"{running_loss / max(1, seen_samples):.4f}")

        train_preds = np.concatenate(all_train_preds, axis=0)
        train_true = np.concatenate(all_train_targets, axis=0)
        train_loss = running_loss / len(train_ds)
        train_f1 = f1_score(train_true, train_preds, average="macro", zero_division=0)

        val_metrics, _, _, _, _ = evaluate_split(model, val_loader, criterion, device, desc="Eval Val")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1_macro": float(train_f1),
                "val_loss": val_metrics["loss"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        early_stopping.step(val_metrics["f1_macro"])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        print(
            f"{epoch:02d} | {train_loss:.4f} | {train_f1:.4f} | "
            f"{val_metrics['loss']:.4f} | {val_metrics['f1_macro']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TRAIN")
    print(f"{'=' * 60}")
    train_metrics, y_train_true, y_train_pred, _, _ = evaluate_split(
        model, train_loader, criterion, device, desc="Final Train Eval"
    )
    train_report_text = classification_report(
        y_train_true, y_train_pred,
        labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES,
        zero_division=0,
    )
    train_cm = confusion_matrix(y_train_true, y_train_pred, labels=list(range(len(TARGET_NAMES))))
    print(train_report_text)
    print("Матрица ошибок (train):")
    print(train_cm)

    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНАЯ ОЦЕНКА НА VAL")
    print(f"{'=' * 60}")
    val_metrics, y_val_true, y_val_pred, _, _ = evaluate_split(
        model, val_loader, criterion, device, desc="Final Val Eval"
    )
    val_report_text = classification_report(
        y_val_true, y_val_pred,
        labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES,
        zero_division=0,
    )
    val_cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(len(TARGET_NAMES))))
    print(val_report_text)
    print("Матрица ошибок (val):")
    print(val_cm)

    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST")
    print(f"{'=' * 60}")
    test_metrics, y_test_true, y_test_pred, _, _ = evaluate_split(
        model, test_loader, criterion, device, desc="Final Test Eval"
    )
    test_report_text = classification_report(
        y_test_true, y_test_pred,
        labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES,
        zero_division=0,
    )
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(len(TARGET_NAMES))))
    print(test_report_text)
    print("Матрица ошибок (test):")
    print(test_cm)

    if save:
        model_params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "freeze_embeddings": freeze_embeddings,
            "n_classes": len(TARGET_NAMES),
            "max_len": max_len,
            "pooling_mode": "mean_max",
        }
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "embedding_matrix": embedding_matrix,
            "word2idx": word2idx,
            "target_names": TARGET_NAMES,
            "model_params": model_params,
        }
        training_params = {
            "embeddings_path": str(embeddings_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "max_len": max_len,
            "max_vocab_size": max_vocab_size,
            "min_freq": min_freq,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "freeze_embeddings": freeze_embeddings,
            "lr": lr,
            "lr_embeddings": 0.0 if freeze_embeddings else float(lr_embeddings),
            "weight_decay": weight_decay,
            "val_size": val_size,
            "patience": patience,
            "seed": seed,
            "device": str(device),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "history": history,
            "best_val_f1_macro": float(best_val_f1),
        }
        export_metrics = {
            **test_metrics,
            "test_classification_report_text": test_report_text,
            "test_confusion_matrix": test_cm.tolist(),
            "train_metrics": train_metrics,
            "train_classification_report_text": train_report_text,
            "train_confusion_matrix": train_cm.tolist(),
            "val_metrics": val_metrics,
            "val_classification_report_text": val_report_text,
            "val_confusion_matrix": val_cm.tolist(),
        }
        save_model(
            model=model,
            dataset_name=dataset_name,
            checkpoint_payload=checkpoint_payload,
            training_params=training_params,
            test_metrics=export_metrics,
        )

    return model, dataset_name


def load_and_evaluate(device_arg: str = "auto", train_path=None, test_path=None):
    device = resolve_device(device_arg)
    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)

    model, checkpoint = load_model(dataset_name, map_location=device)
    model = model.to(device)
    model.eval()

    word2idx = checkpoint["word2idx"]
    model_params = checkpoint["model_params"]
    max_len = int(model_params["max_len"])

    train_texts, y_train_raw = load_texts_from_manifest(train_manifest)
    test_texts, y_test_raw = load_texts_from_manifest(test_manifest)

    train_ds = TextSequenceDataset(train_texts, y_train_raw, word2idx, max_len=max_len)
    test_ds = TextSequenceDataset(test_texts, y_test_raw, word2idx, max_len=max_len)
    train_loader = _build_loader(train_ds, batch_size=64, shuffle=False, use_cuda=device.type == "cuda")
    test_loader = _build_loader(test_ds, batch_size=64, shuffle=False, use_cuda=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    train_metrics, y_train_true, y_train_pred, _, _ = evaluate_split(
        model, train_loader, criterion, device, desc="Final Train Eval"
    )
    train_report_text = classification_report(
        y_train_true, y_train_pred,
        labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES,
        zero_division=0,
    )
    train_cm = confusion_matrix(y_train_true, y_train_pred, labels=list(range(len(TARGET_NAMES))))
    test_metrics, y_test_true, y_test_pred, _, _ = evaluate_split(
        model, test_loader, criterion, device, desc="Final Test Eval"
    )
    test_report_text = classification_report(
        y_test_true, y_test_pred,
        labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES,
        zero_division=0,
    )
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(len(TARGET_NAMES))))

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    print(train_report_text)
    print("Матрица ошибок (train):")
    print(train_cm)
    print(train_metrics)

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    print(test_report_text)
    print("Матрица ошибок (test):")
    print(test_cm)
    print(test_metrics)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение или загрузка модели FastText Embeddings + BiLSTM для классификации эмоций"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load", "auto", "smoke"],
        default="auto",
        help="train - обучить новую модель, load - загрузить существующую, auto - загрузить если есть, smoke - быстрая проверка",
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять модель после обучения")
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=None,
        help=f"Путь к FastText embeddings (.bin). По умолчанию: {DEFAULT_EMBEDDINGS_PATH}",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-embeddings",
        type=float,
        default=None,
        help="LR для embedding-слоя (по умолчанию lr*0.1 при разморозке).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Число эпох без улучшения val_f1 до ранней остановки.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-embeddings",
        dest="freeze_embeddings",
        action="store_true",
        help="Заморозить embedding слой.",
    )
    parser.add_argument(
        "--no-freeze-embeddings",
        dest="freeze_embeddings",
        action="store_false",
        help="Разрешить дообучение embedding слоя.",
    )
    parser.set_defaults(freeze_embeddings=False)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Устройство обучения.",
    )
    add_data_path_args(parser)
    add_config_arg(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config, parser)

    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Требуется библиотека gensim. Установите: pip install gensim или poetry add gensim"
        )

    dataset_name = get_dataset_name(train_path)
    if args.mode == "smoke":
        print("Режим: Smoke-тест\n")
        train_bilstm(
            embeddings_path=None,
            save=False,
            epochs=2,
            batch_size=8,
            max_len=32,
            max_vocab_size=500,
            min_freq=1,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            freeze_embeddings=True,
            lr=1e-3,
            weight_decay=1e-5,
            val_size=0.1,
            seed=42,
            device_arg="cpu",
            train_path=train_path,
            test_path=test_path,
        )
    elif args.mode == "train":
        train_bilstm(
            embeddings_path=args.embeddings_path,
            save=not args.no_save,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            freeze_embeddings=args.freeze_embeddings,
            lr=args.lr,
            lr_embeddings=args.lr_embeddings,
            weight_decay=args.weight_decay,
            val_size=args.val_size,
            patience=args.patience,
            seed=args.seed,
            device_arg=args.device,
            train_path=train_path,
            test_path=test_path,
        )
    elif args.mode == "load":
        load_and_evaluate(device_arg=args.device, train_path=train_path, test_path=test_path)
    else:
        if model_exists(dataset_name):
            print("Режим: AUTO - найдена существующая модель, загружаем...\n")
            load_and_evaluate(device_arg=args.device, train_path=train_path, test_path=test_path)
        else:
            print("Режим: AUTO - модель не найдена, начинаем обучение...\n")
            train_bilstm(
                embeddings_path=args.embeddings_path,
                save=not args.no_save,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
                max_vocab_size=args.max_vocab_size,
                min_freq=args.min_freq,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                freeze_embeddings=args.freeze_embeddings,
                lr=args.lr,
                lr_embeddings=args.lr_embeddings,
                weight_decay=args.weight_decay,
                val_size=args.val_size,
                patience=args.patience,
                seed=args.seed,
                device_arg=args.device,
                train_path=train_path,
                test_path=test_path,
            )
