"""
Общие PyTorch-утилиты для всех экспериментов.

set_seed, resolve_device, build_loader, ensure_transformers_compat,
EarlyStopping, compute_classification_metrics, evaluate_split.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES
from ruintona.my_experiments.utils.metrics import weighted_accuracy


def set_seed(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    """Выбирает устройство по строковому аргументу ('cuda' / 'cpu' / 'auto')."""
    device_arg = device_arg.lower()
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Запрошено обучение на GPU (--device cuda), но CUDA недоступна."
            )
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    use_cuda: bool,
    collate_fn=None,
) -> DataLoader:
    """Обёртка над DataLoader с едиными настройками по умолчанию."""
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(**kwargs)


class EarlyStopping:
    """Ранняя остановка при отсутствии улучшения метрики."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.early_stop = False

    def step(self, score: float) -> bool:
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Единый расчёт метрик классификации."""
    if target_names is None:
        target_names = TARGET_NAMES

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "WA": float(weighted_accuracy(y_true, y_pred)),
    }
    if probs is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


def _eval_collect(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    forward_fn,
    unpack_y_fn,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Единый цикл сбора предсказаний.

    Args:
        forward_fn: ``callable(model, batch) -> logits`` (тензор уже на device).
        unpack_y_fn: ``callable(batch) -> y`` (тензор меток, будет отправлен на device).
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            y = unpack_y_fn(batch).to(device)
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    logits = forward_fn(model, batch)
                    loss = criterion(logits, y)
            else:
                logits = forward_fn(model, batch)
                loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            running_loss += loss.item() * y.size(0)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    mean_loss = running_loss / max(len(loader.dataset), 1)

    return logits_arr, probs_arr, y_pred, y_true, mean_loss


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    desc: str = "Eval",
    use_amp: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None, list[np.ndarray] | None]:
    """Функция оценки для моделей с batch=(x, y) и model(x)."""
    logits_arr, probs_arr, y_pred, y_true, mean_loss = _eval_collect(
        model, loader, criterion, device,
        forward_fn=lambda m, b: m(b[0].to(device)),
        unpack_y_fn=lambda b: b[1],
        use_amp=use_amp,
    )
    metrics = compute_classification_metrics(y_true, y_pred, probs_arr)
    metrics["loss"] = float(mean_loss)
    return metrics, y_true, y_pred, probs_arr, logits_arr


def ensure_transformers_compat() -> None:
    """Гарантирует наличие свойства ``all_tied_weights_keys`` на ``torch.nn.Module``.

    Нужно для совместимости со старыми версиями transformers.
    """
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
