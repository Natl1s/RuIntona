import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys
_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, add_data_path_args, resolve_data_paths
from my_experiments.utils.model_io import save_pytorch_model, load_pytorch_model, pytorch_model_exists
from my_experiments.utils.metrics import weighted_accuracy
from my_experiments.utils.torch_utils import set_seed, resolve_device
from my_experiments.utils.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index

MODELS_DIR = models_dir_for(__file__)
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
    max_t = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        delta = max_t - x.shape[-1]
        if delta > 0:
            x = nn.functional.pad(x, pad=(0, delta, 0, 0))
        padded.append(x)
    return torch.stack(padded), torch.stack(ys)


class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 4, conv_channels: list[int] | None = None, classifier_dropout: float = 0.2):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32, 64]
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_ch = out_ch
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(conv_channels[-1], n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate_split(model, loader, criterion, device):
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
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

    metrics = {
        "loss": float(mean_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "WA": float(weighted_accuracy(y_true, y_pred)),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

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


def train_cnn(
    train_lmdb: Path,
    test_lmdb: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    save: bool,
    device_arg: str,
    conv_channels: list[int] | None = None,
    classifier_dropout: float = 0.2,
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

    model = EmotionCNN(n_classes=len(TARGET_NAMES), conv_channels=conv_channels, classifier_dropout=classifier_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(f"Эпох: {epochs}, batch_size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}")

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        running_preds = []
        running_targets = []
        num_batches = len(train_loader)

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()
            logits = model(x)
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


CNN_DEFAULTS = {
    "conv_channels": [16, 32, 64],
    "kernel_size": 3,
    "classifier_dropout": 0.2,
    "epochs": 5,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "seed": 42,
}


def main():
    parser = argparse.ArgumentParser(description="CNN baseline для классификации эмоций из LMDB.")
    add_data_path_args(parser)
    parser.add_argument("--mode", type=str, choices=["train", "load", "auto", "smoke"], default="auto")
    parser.add_argument("--conv-channels", type=int, nargs="+", default=None, help="Каналы свёрточных слоёв (по умолчанию: 16 32 64)")
    parser.add_argument("--classifier-dropout", type=float, default=None, help="Dropout в классификаторе (по умолчанию: 0.2)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="Путь к JSON-конфигу (относительно configs/ или абсолютный)")
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    cfg = {**CNN_DEFAULTS, **(load_experiment_config(args.config) or {})}

    # CLI-флаги перезаписывают конфиг
    for key in ["epochs", "batch_size", "lr", "weight_decay", "seed", "device", "classifier_dropout"]:
        cli_val = getattr(args, key.replace("-", "_"), None) if "-" not in key else getattr(args, key.replace("-", "_"), None)
        # getattr с kebab→snake
        attr = key.replace("-", "_")
        cli_val = getattr(args, attr, None)
        if cli_val is not None:
            cfg[attr] = cli_val
    if args.conv_channels is not None:
        cfg["conv_channels"] = args.conv_channels

    train_lmdb = train_path
    test_lmdb = test_path

    if not train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {train_lmdb}")
    if not test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {test_lmdb}")

    if args.mode == "smoke":
        print("💨 Режим: Smoke-тест\n")
        train_cnn(
            train_lmdb=train_lmdb, test_lmdb=test_lmdb,
            epochs=2, batch_size=8, lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], seed=cfg["seed"],
            save=False, device_arg="cpu",
            conv_channels=[8, 16], classifier_dropout=0.1,
        )
    else:
        train_cnn(
            train_lmdb=train_lmdb, test_lmdb=test_lmdb,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            weight_decay=cfg["weight_decay"], seed=cfg["seed"],
            save=not args.no_save, device_arg=cfg.get("device", "cuda"),
            conv_channels=cfg["conv_channels"], classifier_dropout=cfg["classifier_dropout"],
        )


if __name__ == "__main__":
    main()
