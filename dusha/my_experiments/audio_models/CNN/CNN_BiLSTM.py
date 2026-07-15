import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import sys
_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.config_utils import TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_NAMES, get_dataset_name, load_experiment_config, apply_config_to_args, add_config_arg
from my_experiments.model_io import save_pytorch_model
from my_experiments.metrics import weighted_accuracy
from my_experiments.torch_utils import set_seed, resolve_device
from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index

MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem


class LmdbFeaturesDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)
        self.length = get_lmdb_length(self.env)
        if self.length <= 0:
            raise ValueError(f"Пустой LMDB: {self.lmdb_path}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            raw = txn.get(str(int(idx)).encode("utf-8"))
        if raw is None:
            raise KeyError(f"В LMDB отсутствует ключ {idx}")
        payload = pickle.loads(raw)
        if "x" not in payload:
            raise KeyError(f"В payload LMDB отсутствует ключ 'x' (idx={idx})")
        arr = np.asarray(payload["x"], dtype=np.float32)
        label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
        label = parse_label_to_index(label_raw)
        x = torch.from_numpy(arr)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[0] != 1 and x.shape[-1] == 1:
            x = x.permute(2, 0, 1)
        elif x.ndim != 3:
            raise ValueError(f"Неподдерживаемая форма тензора {x.shape} в idx={idx}")

        return x, torch.tensor(label, dtype=torch.long)


def pad_collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    max_t = int(lengths.max().item())
    padded = []
    for x in xs:
        delta = max_t - x.shape[-1]
        if delta > 0:
            x = nn.functional.pad(x, pad=(0, delta, 0, 0))
        padded.append(x)
    return torch.stack(padded), torch.stack(ys), lengths


class EmotionCNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.lstm_input_size = 64
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(lstm_out_size, n_classes),
        )

    @staticmethod
    def _downsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.div(lengths, 4, rounding_mode="floor"), min=1)

    def forward(self, x, lengths):
        feats = self.conv(x)
        feats = feats.mean(dim=2)
        feats = feats.permute(0, 2, 1).contiguous()
        out_lengths = self._downsample_lengths(lengths.to(feats.device))
        packed = pack_padded_sequence(feats, lengths=out_lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (out_lengths - 1).view(-1, 1, 1).expand(-1, 1, lstm_out.size(-1))
        last_valid = lstm_out.gather(dim=1, index=idx).squeeze(1)
        return self.classifier(last_valid)


def evaluate_split(model, loader, criterion, device):
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            running_loss += loss.item() * x.size(0)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    mean_loss = running_loss / len(loader.dataset)

    per_class_recall = recall_score(y_true, y_pred, labels=list(range(len(TARGET_NAMES))), average=None, zero_division=0)

    metrics = {
        "loss": float(mean_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "WA": float(weighted_accuracy(y_true, y_pred)),
        "UAR": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa_qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "top2_accuracy": float(top_k_accuracy_score(y_true, probs, k=2, labels=list(range(len(TARGET_NAMES))))),
    }
    for i, class_name in enumerate(TARGET_NAMES):
        metrics[f"recall_{class_name}"] = float(per_class_recall[i])

    try:
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=list(range(len(TARGET_NAMES)))))
    except ValueError:
        metrics["log_loss"] = float("nan")

    return metrics, y_true, y_pred, probs, logits


def print_metrics(title, metrics, y_true, y_pred):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.6f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def train_cnn_bilstm(
    train_lmdb: Path,
    test_lmdb: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    save: bool,
    device_arg: str,
    lstm_hidden_size: int,
    lstm_layers: int,
    lstm_dropout: float,
    bidirectional: bool,
):
    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    print(f"Обучение запущено на устройстве: {device}")
    if use_cuda:
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(gpu_index)} (cuda:{gpu_index})")
    print(f"Train LMDB: {train_lmdb}")
    print(f"Test LMDB:  {test_lmdb}")

    train_ds = LmdbFeaturesDataset(train_lmdb)
    test_ds = LmdbFeaturesDataset(test_lmdb)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)

    model = EmotionCNNBiLSTM(
        n_classes=len(TARGET_NAMES), lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers, lstm_dropout=lstm_dropout, bidirectional=bidirectional,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(f"Эпох: {epochs}, batch_size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}, "
          f"lstm_hidden_size: {lstm_hidden_size}, lstm_layers: {lstm_layers}, "
          f"lstm_dropout: {lstm_dropout}, bidirectional: {bidirectional}")

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        running_preds = []
        running_targets = []
        num_batches = len(train_loader)

        for batch_idx, (x, y, lengths) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            lengths = lengths.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            seen_samples += x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_preds.append(preds.detach().cpu().numpy())
            running_targets.append(y.detach().cpu().numpy())
            bar_width = 30
            filled = int(bar_width * batch_idx / num_batches)
            bar = "#" * filled + "-" * (bar_width - filled)
            mean_batch_loss = running_loss / max(seen_samples, 1)
            print(f"\rEpoch {epoch:03d}/{epochs} [{bar}] {batch_idx}/{num_batches} loss={mean_batch_loss:.4f}", end="", flush=True)
        print()

        train_pred = np.concatenate(running_preds, axis=0)
        train_true = np.concatenate(running_targets, axis=0)
        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average="macro")

        test_metrics, _, _, _, _ = evaluate_split(model, test_loader, criterion, device)
        test_f1 = test_metrics["f1_macro"]
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} test_f1={test_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test_metrics, y_true, y_pred, _, _ = evaluate_split(model, test_loader, criterion, device)
    print_metrics("ФИНАЛЬНАЯ ОЦЕНКА НА TEST", final_test_metrics, y_true, y_pred)

    dataset_name = get_dataset_name(train_lmdb)
    if save:
        final_report_text = classification_report(y_true, y_pred, labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES, digits=4, zero_division=0)
        final_confusion_matrix = confusion_matrix(y_true, y_pred)
        training_params = {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "seed": seed, "device": str(device),
            "lstm_hidden_size": lstm_hidden_size, "lstm_layers": lstm_layers,
            "lstm_dropout": lstm_dropout, "bidirectional": bidirectional,
            "train_lmdb": str(train_lmdb), "test_lmdb": str(test_lmdb),
        }
        test_metrics = {
            **final_test_metrics,
            "test_classification_report_text": final_report_text,
            "test_confusion_matrix": final_confusion_matrix.tolist(),
        }
        save_pytorch_model(
            model.state_dict(), dataset_name=dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            training_params=training_params, test_metrics=test_metrics,
        )

    return model


def main():
    parser = argparse.ArgumentParser(description="CNN + BiLSTM baseline для классификации эмоций из LMDB.")
    parser.add_argument("--aggregated-dir", type=Path, default=TRAIN_DATA_PATH.parent)
    parser.add_argument("--train-lmdb-name", type=str, default=TRAIN_DATA_PATH.name)
    parser.add_argument("--test-lmdb-name", type=str, default=TEST_DATA_PATH.name)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--no-save", action="store_true")
    add_config_arg(parser)
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config)

    train_lmdb = args.aggregated_dir / args.train_lmdb_name
    test_lmdb = args.aggregated_dir / args.test_lmdb_name

    if not train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {train_lmdb}")
    if not test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {test_lmdb}")

    train_cnn_bilstm(
        train_lmdb=train_lmdb, test_lmdb=test_lmdb,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        save=not args.no_save, device_arg=args.device,
        lstm_hidden_size=args.lstm_hidden_size, lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout, bidirectional=not args.unidirectional,
    )


if __name__ == "__main__":
    main()
