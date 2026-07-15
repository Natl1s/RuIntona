"""
Общие утилиты вычисления и вывода метрик классификации.
"""

from __future__ import annotations

import numpy as np
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

from my_experiments.config_utils import EMO2LABEL, TARGET_NAMES


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """Вычисляет стандартный набор метрик для классификации."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


def weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> float:
    """Взвешенная точность (средний recall по классам)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for cls_idx in range(n_classes):
        cls_mask = y_true == cls_idx
        if cls_mask.sum() == 0:
            continue
        recalls.append((y_pred[cls_mask] == cls_idx).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def print_eval_block(
    title: str,
    metrics: dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> None:
    """Выводит блок с метриками, classification report и confusion matrix."""
    if target_names is None:
        target_names = TARGET_NAMES

    labels = list(range(len(target_names))) if isinstance(target_names[0], str) and target_names == TARGET_NAMES else None
    if labels is None:
        labels = list(range(len(target_names)))

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
