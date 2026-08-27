"""
Общие утилиты вычисления и вывода метрик классификации.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

try:
    from IPython.display import display as ipy_display
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False


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
    labels: list | None = None,
) -> float:
    """Взвешенная точность (средний recall по классам).

    Args:
        y_true: истинные метки (int индексы или строки)
        y_pred: предсказанные метки (int индексы или строки)
        n_classes: количество классов (используется если labels=None)
        labels: список меток для итерации (строки или int)
               если передан — используется вместо range(n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    items = labels if labels is not None else range(n_classes)
    recalls = []
    for item in items:
        mask = y_true == item
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == item).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def _resolve_target_names(target_names: list[str] | None) -> tuple[list[str], list[int]]:
    if target_names is None:
        target_names = TARGET_NAMES
    labels = list(range(len(target_names)))
    return target_names, labels


def _overview_df(metrics: dict[str, float]) -> pd.DataFrame:
    nice_names = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "f1_weighted": "F1 (weighted)",
        "mcc": "MCC",
        "roc_auc_ovr_macro": "ROC-AUC (OvR macro)",
    }
    rows = []
    for k, v in metrics.items():
        name = nice_names.get(k, k.replace("_", " ").title())
        val = "N/A" if np.isnan(v) else f"{v:.4f}"
        rows.append({"Metric": name, "Value": val})
    return pd.DataFrame(rows)


def _per_class_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    labels: list[int],
) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred,
        labels=labels, target_names=target_names,
        digits=4, zero_division=0, output_dict=True,
    )
    rows = []
    for name in target_names:
        if name in report and isinstance(report[name], dict):
            r = report[name]
            rows.append({
                "Class": name,
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1-Score": f"{r['f1-score']:.4f}",
                "Support": int(r["support"]),
            })
    return pd.DataFrame(rows), report


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    labels: list[int],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, title_text, cmap in [
        (axes[0], cm, "d", "Counts", "Blues"),
        (axes[1], cm_norm, ".2f", "Normalized (by true class)", "Blues"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=target_names, yticklabels=target_names,
            ax=ax, linewidths=0.5, linecolor="white",
            vmin=0, vmax=(1.0 if fmt == ".2f" else None),
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title_text, fontsize=12)

    plt.suptitle("Confusion Matrix", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def _plot_per_class_metrics(report: dict, target_names: list[str]) -> None:
    classes = [n for n in target_names if n in report and isinstance(report[n], dict)]
    prec = [report[c]["precision"] for c in classes]
    rec = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.8), 5))
    ax.bar(x - width, prec, width, label="Precision", color="#4a86c8")
    ax.bar(x, rec, width, label="Recall", color="#f4a261")
    ax.bar(x + width, f1, width, label="F1-Score", color="#2a9d8f")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-Class Metrics", fontsize=13)

    for bars_group in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        ax.bar_label(bars_group, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.show()


def _fallback_text_output(
    title: str,
    metrics: dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    labels: list[int],
) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.6f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true, y_pred,
            labels=labels, target_names=target_names,
            digits=4, zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))


def print_eval_block(
    title: str,
    metrics: dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> None:
    """Вывод метрик классификации.

    В notebook: DataFrame-таблицы + confusion matrix heatmap + bar chart.
    Вне notebook: plain-text fallback.
    """
    target_names, labels = _resolve_target_names(target_names)

    if not _IN_NOTEBOOK or not _HAS_PLOTTING:
        _fallback_text_output(title, metrics, y_true, y_pred, target_names, labels)
        return

    # 1. Title
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    # 2. Overview metrics
    print("\nOverall Metrics:")
    ipy_display(_overview_df(metrics))

    # 3. Per-class report
    per_class, report_dict = _per_class_df(y_true, y_pred, target_names, labels)
    print("Per-Class Report:")
    ipy_display(per_class)

    # 4. Confusion matrix
    _plot_confusion_matrix(y_true, y_pred, target_names, labels)

    # 5. Per-class bar chart
    _plot_per_class_metrics(report_dict, target_names)
