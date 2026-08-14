"""
Общие утилиты для sklearn-классификаторов.

evaluate_sklearn_classifier — единая функция оценки.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES


def evaluate_sklearn_classifier(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    print_params_fn: Callable[[object], None] | None = None,
) -> dict:
    """Общая функция оценки sklearn-классификатора.

    Args:
        model: обученная sklearn-модель (должна иметь ``.predict()``).
        X_train, y_train: обучающая выборка (до трансформации).
        X_test, y_test: тестовая выборка (до трансформации).
        transform_fn: функция преобразования признаков
            (``scaler.transform``, ``vectorizer.transform`` и т.д.).
        print_params_fn: callback для вывода параметров модели
            (вызывается ``print_params_fn(model)``).

    Returns:
        dict с ключами ``test_accuracy``, ``test_classification_report_text``,
        ``test_classification_report``, ``test_confusion_matrix``,
        а также метриками переобучения: ``train_accuracy``, ``train_f1_macro``,
        ``test_f1_macro``, ``overfit_gap_accuracy``, ``overfit_gap_f1_macro``.
    """
    if transform_fn is not None:
        X_train = transform_fn(X_train)
        X_test = transform_fn(X_test)

    if print_params_fn is not None:
        print_params_fn(model)

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    train_pred = model.predict(X_train)
    train_report_text = classification_report(
        y_train, train_pred,
        labels=TARGET_NAMES, target_names=TARGET_NAMES,
        zero_division=0,
    )
    print(train_report_text)

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    test_pred = model.predict(X_test)
    test_report_text = classification_report(
        y_test, test_pred,
        labels=TARGET_NAMES, target_names=TARGET_NAMES,
        zero_division=0,
    )
    print(test_report_text)

    print("\nМатрица ошибок:")
    test_cm = confusion_matrix(y_test, test_pred)
    print(test_cm)

    test_report_dict = classification_report(
        y_test, test_pred,
        labels=TARGET_NAMES, target_names=TARGET_NAMES,
        zero_division=0, output_dict=True,
    )

    train_accuracy = float(accuracy_score(y_train, train_pred))
    test_accuracy = float(accuracy_score(y_test, test_pred))
    train_f1_macro = float(
        f1_score(y_train, train_pred, average="macro", zero_division=0)
    )
    test_f1_macro = float(
        f1_score(y_test, test_pred, average="macro", zero_division=0)
    )

    print(f"\n{'=' * 60}")
    print("ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ (train vs test)")
    print(f"{'=' * 60}")
    print(f"Train accuracy: {train_accuracy:.4f} | Test accuracy: {test_accuracy:.4f} | "
          f"Gap: {train_accuracy - test_accuracy:.4f}")
    print(f"Train F1-macro: {train_f1_macro:.4f} | Test F1-macro: {test_f1_macro:.4f} | "
          f"Gap: {train_f1_macro - test_f1_macro:.4f}")
    if train_accuracy - test_accuracy > 0.15:
        print("⚠️  Разрыв train/test значительный (>0.15) — модель склонна к переобучению. "
              "Рекомендуется усилить регуляризацию (max_depth, min_samples_leaf, ccp_alpha) "
              "или подобрать гиперпараметры.")
    elif train_accuracy - test_accuracy < 0.05:
        print("✓ Разрыв train/test небольшой (<0.05) — признаков переобучения нет.")
    else:
        print("Разрыв train/test умеренный — при желании можно усилить регуляризацию.")

    return {
        "test_accuracy": test_accuracy,
        "test_classification_report_text": test_report_text,
        "test_classification_report": test_report_dict,
        "test_confusion_matrix": test_cm.tolist(),
        "train_accuracy": train_accuracy,
        "train_f1_macro": train_f1_macro,
        "test_f1_macro": test_f1_macro,
        "overfit_gap_accuracy": train_accuracy - test_accuracy,
        "overfit_gap_f1_macro": train_f1_macro - test_f1_macro,
    }
