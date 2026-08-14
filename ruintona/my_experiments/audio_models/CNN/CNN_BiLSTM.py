import argparse
import json
from copy import deepcopy
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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, apply_config_to_args, add_config_arg, add_data_path_args, resolve_data_paths, CONFIGS_DIR
from ruintona.my_experiments.utils.model_io import save_pytorch_model, load_pytorch_model, pytorch_model_exists
from ruintona.my_experiments.utils.torch_utils import set_seed, resolve_device, compute_classification_metrics, EarlyStopping, _eval_collect
from ruintona.my_experiments.utils.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index, safe_pickle_loads

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem


class LmdbFeaturesDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)
        self.length = get_lmdb_length(self.env)
        if self.length <= 0:
            raise ValueError(f"Пустой LMDB: {self.lmdb_path}")
        self.labels = self._load_labels()

    def _load_labels(self):
        labels = []
        with self.env.begin() as txn:
            for idx in range(self.length):
                raw = txn.get(str(int(idx)).encode("utf-8"))
                if raw is None:
                    continue
                payload = safe_pickle_loads(raw)
                label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
                labels.append(parse_label_to_index(label_raw))
        return np.array(labels, dtype=np.int64)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            raw = txn.get(str(int(idx)).encode("utf-8"))
        if raw is None:
            raise KeyError(f"В LMDB отсутствует ключ {idx}")
        payload = safe_pickle_loads(raw)
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
        conv_channels: list[int] | None = None,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        classifier_dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32, 64]
        self.conv_channels = list(conv_channels)
        self.classifier_dropout = classifier_dropout

        conv_layers = []
        in_ch = 1
        n_blocks = len(self.conv_channels)
        for i, out_ch in enumerate(self.conv_channels):
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if i < n_blocks - 1:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.lstm_input_size = self.conv_channels[-1]
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
            nn.Dropout(p=classifier_dropout),
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
    def forward_fn(m, batch):
        x, y, lengths = batch
        return m(x.to(device), lengths.to(device))

    logits_arr, probs_arr, y_pred, y_true, mean_loss = _eval_collect(
        model, loader, criterion, device,
        forward_fn=forward_fn,
        unpack_y_fn=lambda b: b[1],
    )

    per_class_recall = recall_score(y_true, y_pred, labels=list(range(len(TARGET_NAMES))), average=None, zero_division=0)

    metrics = compute_classification_metrics(y_true, y_pred, probs_arr)
    metrics["loss"] = float(mean_loss)
    metrics["UAR"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["cohen_kappa_qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    metrics["top2_accuracy"] = float(top_k_accuracy_score(y_true, probs_arr, k=2, labels=list(range(len(TARGET_NAMES)))))
    for i, class_name in enumerate(TARGET_NAMES):
        metrics[f"recall_{class_name}"] = float(per_class_recall[i])

    try:
        metrics["log_loss"] = float(log_loss(y_true, probs_arr, labels=list(range(len(TARGET_NAMES)))))
    except ValueError:
        metrics["log_loss"] = float("nan")

    return metrics, y_true, y_pred, probs_arr, logits_arr


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
    conv_channels: list[int] | None = None,
    classifier_dropout: float = 0.3,
    val_size: float = 0.15,
    early_stopping_patience: int = 5,
    train_indices: np.ndarray | None = None,
    model_name: str = MODEL_NAME,
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

    if conv_channels is None:
        conv_channels = [16, 32, 64]
    conv_channels = list(conv_channels)

    train_ds = LmdbFeaturesDataset(train_lmdb)
    test_ds = LmdbFeaturesDataset(test_lmdb)

    if train_indices is None:
        train_indices = np.arange(len(train_ds))
    train_indices = np.asarray(train_indices)

    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_size,
        random_state=seed,
        stratify=train_ds.labels[train_indices],
    )
    train_split = Subset(train_ds, train_idx.tolist())
    val_split = Subset(train_ds, val_idx.tolist())

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)

    model = EmotionCNNBiLSTM(
        n_classes=len(TARGET_NAMES), conv_channels=conv_channels,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers, lstm_dropout=lstm_dropout,
        classifier_dropout=classifier_dropout, bidirectional=bidirectional,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nРазмер train: {len(train_split)}")
    print(f"Размер val:   {len(val_split)}")
    print(f"Размер test:  {len(test_ds)}")
    print(f"Эпох: {epochs}, batch_size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}, "
          f"lstm_hidden_size: {lstm_hidden_size}, lstm_layers: {lstm_layers}, "
          f"lstm_dropout: {lstm_dropout}, bidirectional: {bidirectional}")
    print(f"conv_channels: {conv_channels}, classifier_dropout: {classifier_dropout}")
    print(f"val_size: {val_size}, early_stopping_patience: {early_stopping_patience}")

    best_state = None
    best_val_f1 = -1.0
    early_stopping = EarlyStopping(patience=early_stopping_patience)

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
        train_loss = running_loss / len(train_split)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average="macro")

        val_metrics, _, _, _, _ = evaluate_split(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        early_stopping.step(val_f1)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_f1={val_f1:.4f}")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test_metrics, y_true, y_pred, _, _ = evaluate_split(model, test_loader, criterion, device)
    print_metrics("ФИНАЛЬНАЯ ОЦЕНКА НА TEST", final_test_metrics, y_true, y_pred)

    dataset_name = get_dataset_name(train_lmdb)
    if save:
        final_report_text = classification_report(y_true, y_pred, labels=list(range(len(TARGET_NAMES))), target_names=TARGET_NAMES, digits=4, zero_division=0)
        final_confusion_matrix = confusion_matrix(y_true, y_pred)
        training_params = {
            "epochs": epoch, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "seed": seed, "device": str(device),
            "lstm_hidden_size": lstm_hidden_size, "lstm_layers": lstm_layers,
            "lstm_dropout": lstm_dropout, "bidirectional": bidirectional,
            "conv_channels": list(conv_channels), "classifier_dropout": classifier_dropout,
            "train_lmdb": str(train_lmdb), "test_lmdb": str(test_lmdb),
            "val_size": val_size, "early_stopping_patience": early_stopping_patience,
            "train_size": int(len(train_split)),
        }
        model_params = {
            "n_classes": len(TARGET_NAMES),
            "conv_channels": list(conv_channels),
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_layers": lstm_layers,
            "lstm_dropout": lstm_dropout,
            "classifier_dropout": classifier_dropout,
            "bidirectional": bidirectional,
        }
        test_metrics = {
            **final_test_metrics,
            "val_f1_macro": float(best_val_f1),
            "test_classification_report_text": final_report_text,
            "test_confusion_matrix": final_confusion_matrix.tolist(),
        }
        save_pytorch_model(
            model.state_dict(), dataset_name=dataset_name,
            models_dir=MODELS_DIR, model_name=model_name,
            training_params=training_params, test_metrics=test_metrics,
            model_class=model.__class__.__name__,
            model_params=model_params,
        )

    return model


def _stratified_subsample(train_ds, max_samples, random_state=42):
    """Стратифицированная подвыборка индексов для ускорения подбора гиперпараметров."""
    if max_samples and max_samples < len(train_ds):
        idx, _ = train_test_split(
            np.arange(len(train_ds)),
            train_size=int(max_samples),
            random_state=random_state,
            stratify=train_ds.labels,
        )
        return idx
    return np.arange(len(train_ds))


def _run_trial(
    model: EmotionCNNBiLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    trial=None,
) -> tuple[float, float, int]:
    """Обучает одну конфигурацию гиперпараметров, возвращает (best_val_f1, last_val_f1, n_epochs).

    Использует EarlyStopping по val_f1_macro. Если передан trial Optuna —
    сообщает прогресс и позволяет pruner'у отсечь слабые триалы.
    """
    set_seed(42)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    best_val_f1 = -1.0
    val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, lengths in train_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_metrics, _, _, _, _ = evaluate_split(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1_macro"]
        best_val_f1 = max(best_val_f1, val_f1)

        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        early_stopping.step(val_f1)
        if early_stopping.early_stop:
            break
    return best_val_f1, val_f1, epoch


def _write_tuned_config(
    best_params: dict,
    *,
    conv_channels: list[int],
    seed: int,
    epochs: int,
) -> None:
    """Пишет подобранные Optuna гиперпараметры в configs/audio/cnn_bilstm_tuned.json.

    Вызывается из --mode tune при save=True; при --no-save конфиг не трогаем.
    """
    config_path = CONFIGS_DIR / "audio" / "cnn_bilstm_tuned.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    updated = {
        "_description": (
            "CNN + BiLSTM с гиперпараметрами, подобранными Optuna (TPE) в "
            "model_analise/cnn_bilstm_hyperparameter_tuning.ipynb / --mode tune. "
            "Файл автоматически перезаписывается при каждом запуске --mode tune "
            "(см. checkpoints/audio/*_optuna_results.json)."
        ),
        "_script": "ruintona/my_experiments/audio_models/CNN/CNN_BiLSTM.py",
        "conv_channels": list(conv_channels),
        "classifier_dropout": best_params["classifier_dropout"],
        "lstm_hidden_size": best_params["lstm_hidden_size"],
        "lstm_layers": best_params["lstm_layers"],
        "lstm_dropout": best_params["lstm_dropout"],
        "unidirectional": False,
        "epochs": epochs,
        "batch_size": best_params["batch_size"],
        "lr": best_params["lr"],
        "weight_decay": best_params["weight_decay"],
        "seed": seed,
    }
    config_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=4) + "\n", encoding="utf-8",
    )
    print(f"✓ Подобранные гиперпараметры сохранены: {config_path.absolute()}")


def _save_tune_results(study, dataset_name, max_samples, tune_epochs):
    """Сохраняет результаты Optuna-поиска в JSON + CSV с историей триалов."""
    import optuna

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    results = {
        "dataset_name": dataset_name,
        "sampler": type(study.sampler).__name__,
        "pruner": type(study.pruner).__name__,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(complete),
        "n_trials_pruned": len(pruned),
        "search_max_samples": int(max_samples) if max_samples else None,
        "tune_epochs": tune_epochs,
        "best_params": study.best_params,
        "best_val_f1_macro": float(study.best_value),
        "top_trials": [
            {
                "number": t.number,
                "value": float(t.value),
                "state": t.state.name,
                "params": dict(t.params),
            }
            for t in sorted(
                (tr for tr in study.trials if tr.value is not None),
                key=lambda tr: tr.value,
                reverse=True,
            )[:10]
        ],
    }
    results_dir = MODELS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{MODEL_NAME}_{dataset_name}_optuna_results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    trials_path = results_dir / f"{MODEL_NAME}_{dataset_name}_optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    print(f"\n✓ Результаты поиска сохранены: {results_path.absolute()}")
    print(f"✓ История триалов сохранена:  {trials_path.absolute()}")


def tune_cnn_bilstm(
    train_lmdb: Path,
    test_lmdb: Path,
    n_trials: int = 20,
    timeout: int | None = None,
    max_samples: int | None = None,
    tune_epochs: int = 10,
    retrain_epochs: int = 30,
    retrain_max_samples: int | None = None,
    val_size: float = 0.15,
    trial_patience: int = 3,
    seed: int = 42,
    save: bool = True,
    device_arg: str = "cuda",
):
    """Optuna TPE-поиск гиперпараметров CNN-BiLSTM + переобучение на полных данных.

    Поиск ведётся на стратифицированной подвыборке train (для скорости).
    Метрика Optuna — best val f1-macro (test в поиске не участвует).
    После поиска лучшие параметры используются для обучения на полном train
    и оценки на test.
    """
    import optuna

    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    dataset_name = get_dataset_name(train_lmdb)
    print(f"📊 Датасет: {dataset_name}\n")

    print("Загрузка данных...")
    train_ds = LmdbFeaturesDataset(train_lmdb)
    test_ds = LmdbFeaturesDataset(test_lmdb)

    search_idx = _stratified_subsample(train_ds, max_samples, random_state=seed)
    if len(search_idx) < len(train_ds):
        print(f"Используем подвыборку для поиска: {len(search_idx)} примеров "
              f"(стратифицировано по классам)")
    search_labels = train_ds.labels[search_idx]
    search_tr_idx, val_idx = train_test_split(
        np.arange(len(search_idx)),
        test_size=val_size,
        random_state=seed,
        stratify=search_labels,
    )
    search_tr = Subset(train_ds, search_idx[search_tr_idx].tolist())
    val_split = Subset(train_ds, search_idx[val_idx].tolist())
    print(f"Поисковый train: {len(search_tr)}, val: {len(val_split)}")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        lstm_hidden_size = trial.suggest_categorical(
            "lstm_hidden_size", [64, 96, 128, 160, 192, 256],
        )
        lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2, 3])
        lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5)
        classifier_dropout = trial.suggest_float("classifier_dropout", 0.0, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = EmotionCNNBiLSTM(
            n_classes=len(TARGET_NAMES), conv_channels=[16, 32, 64],
            lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout, classifier_dropout=classifier_dropout,
            bidirectional=True,
        ).to(device)

        train_loader = DataLoader(search_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda, collate_fn=pad_collate_fn)

        best_val_f1, _, _ = _run_trial(
            model, train_loader, val_loader,
            epochs=tune_epochs, lr=lr, weight_decay=weight_decay,
            patience=trial_patience, device=device, trial=trial,
        )
        return best_val_f1

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{MODEL_NAME}_{dataset_name}",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )

    print(f"\n{'=' * 60}")
    print("ПОДБОР ГИПЕРПАРАМЕТРОВ (Optuna, TPE)")
    print(f"{'=' * 60}")
    print(f"n_trials={n_trials}, timeout={timeout}, метрика: best val f1-macro")
    print("Search space: lr, weight_decay, lstm_hidden_size, lstm_layers, "
          "lstm_dropout, classifier_dropout, batch_size")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    print(f"\n✓ Поиск завершён. Лучший val f1-macro: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")
    print("\nТоп-5 триалов:")
    for t in sorted(
        (tr for tr in study.trials if tr.value is not None),
        key=lambda tr: tr.value, reverse=True,
    )[:5]:
        print(f"  trial #{t.number:>3} f1={t.value:.4f} [{t.state.name}] {t.params}")

    if save:
        _save_tune_results(study, dataset_name, max_samples, tune_epochs)

    best = study.best_params
    if save:
        _write_tuned_config(
            best,
            conv_channels=[16, 32, 64],
            seed=seed,
            epochs=retrain_epochs,
        )
    print(f"\n{'=' * 60}")
    print("ПЕРЕОБУЧЕНИЕ ЛУЧШЕЙ КОНФИГУРАЦИИ НА ПОЛНЫХ ДАННЫХ")
    print(f"{'=' * 60}")

    retrain_indices = None
    if retrain_max_samples and retrain_max_samples < len(train_ds):
        retrain_indices = _stratified_subsample(train_ds, retrain_max_samples, random_state=seed)
        print(f"Retrain на подвыборке: {len(retrain_indices)} примеров "
              f"(стратифицировано; задано --retrain-max-samples)")

    model = train_cnn_bilstm(
        train_lmdb=train_lmdb, test_lmdb=test_lmdb,
        epochs=retrain_epochs, batch_size=best["batch_size"], lr=best["lr"],
        weight_decay=best["weight_decay"], seed=seed,
        save=save, device_arg=device_arg,
        lstm_hidden_size=best["lstm_hidden_size"], lstm_layers=best["lstm_layers"],
        lstm_dropout=best["lstm_dropout"], bidirectional=True,
        conv_channels=[16, 32, 64], classifier_dropout=best["classifier_dropout"],
        val_size=val_size, early_stopping_patience=max(trial_patience * 2, 5),
        train_indices=retrain_indices,
        model_name=f"{MODEL_NAME}_tuned",
    )
    return model, study


def main():
    parser = argparse.ArgumentParser(description="CNN + BiLSTM baseline для классификации эмоций из LMDB.")
    add_data_path_args(parser)
    parser.add_argument("--mode", type=str, choices=["train", "load", "auto", "smoke", "tune"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Число триалов Optuna (только --mode tune)")
    parser.add_argument("--tune-timeout", type=int, default=None,
                        help="Лимит времени на поиск в секундах (--mode tune)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Стратифицированная подвыборка train для поиска (--mode tune)")
    parser.add_argument("--tune-epochs", type=int, default=10,
                        help="Максимум эпох на один триал (--mode tune)")
    parser.add_argument("--retrain-epochs", type=int, default=None,
                        help="Эпохи для переобучения лучшей конфигурации (по умолчанию --epochs)")
    parser.add_argument("--retrain-max-samples", type=int, default=None,
                        help="Стратифицированная подвыборка train для финального переобучения "
                             "лучшей конфигурации (--mode tune); None = весь train")
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[16, 32, 64],
                        help="Каналы свёрточных слоёв (по умолчанию: 16 32 64)")
    parser.add_argument("--classifier-dropout", type=float, default=0.3,
                        help="Dropout в классификаторе")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    add_config_arg(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config, parser)

    train_lmdb = train_path
    test_lmdb = test_path

    if not train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {train_lmdb}")
    if not test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {test_lmdb}")

    if args.mode == "smoke":
        print("💨 Режим: Smoke-тест\n")
        train_cnn_bilstm(
            train_lmdb=train_lmdb, test_lmdb=test_lmdb,
            epochs=2, batch_size=8, lr=args.lr,
            weight_decay=args.weight_decay, seed=args.seed,
            save=False, device_arg="cpu",
            lstm_hidden_size=32, lstm_layers=1,
            lstm_dropout=0.0, bidirectional=True,
            conv_channels=[8, 16], classifier_dropout=0.1,
        )
    elif args.mode == "tune":
        print("🔍 Режим: Подбор гиперпараметров (Optuna)\n")
        tune_cnn_bilstm(
            train_lmdb=train_lmdb, test_lmdb=test_lmdb,
            n_trials=args.n_trials, timeout=args.tune_timeout,
            max_samples=args.max_samples, tune_epochs=args.tune_epochs,
            retrain_epochs=args.retrain_epochs or args.epochs,
            retrain_max_samples=args.retrain_max_samples,
            val_size=args.val_size,
            save=not args.no_save, device_arg=args.device,
        )
    else:
        train_cnn_bilstm(
            train_lmdb=train_lmdb, test_lmdb=test_lmdb,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, seed=args.seed,
            save=not args.no_save, device_arg=args.device,
            lstm_hidden_size=args.lstm_hidden_size, lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout, bidirectional=not args.unidirectional,
            conv_channels=args.conv_channels, classifier_dropout=args.classifier_dropout,
            val_size=args.val_size, early_stopping_patience=args.early_stopping_patience,
        )


if __name__ == "__main__":
    main()
