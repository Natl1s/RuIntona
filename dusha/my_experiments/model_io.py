"""
Общие утилиты сохранения/загрузки моделей.

Поддерживает sklearn (joblib) и PyTorch (state_dict) форматы.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


# ---------------------------------------------------------------------------
# sklearn (joblib)
# ---------------------------------------------------------------------------

def save_sklearn_model(
    model: Any,
    artifact: Any,
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    artifact_name: str = "scaler",
    training_params: dict | None = None,
    test_metrics: dict | None = None,
) -> None:
    """
    Сохраняет sklearn-модель + артефакт (scaler / vectorizer) в формате joblib.

    Args:
        model: обученная sklearn-модель
        artifact: scaler / vectorizer / другой артефакт
        dataset_name: имя датасета (из имени LMDB-файла)
        models_dir: папка для сохранения
        model_name: имя модели (по умолчанию — имя скрипта)
        artifact_name: имя артефакта ("scaler" / "vectorizer")
        training_params: параметры обучения (для отчёта)
        test_metrics: метрики (для отчёта)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{dataset_name}"

    model_path = models_dir / f"{full_name}_model.pkl"
    artifact_path = models_dir / f"{full_name}_{artifact_name}.pkl"
    backup_path = models_dir / f"{full_name}_model_{timestamp}.pkl"
    report_path = models_dir / f"{full_name}_training_report.txt"

    joblib.dump(model, model_path)
    joblib.dump(artifact, artifact_path)
    joblib.dump({"model": model, artifact_name: artifact}, backup_path)

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
    print(f"✓ {artifact_name.title()}: {artifact_path.absolute()}")
    print(f"✓ Бэкап:  {backup_path.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    print(f"{'=' * 60}")


def load_sklearn_model(
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    artifact_name: str = "scaler",
) -> tuple[Any, Any]:
    """
    Загружает sklearn-модель + артефакт.

    Returns:
        (model, artifact)
    """
    full_name = f"{model_name}_{dataset_name}"
    model_path = models_dir / f"{full_name}_model.pkl"
    artifact_path = models_dir / f"{full_name}_{artifact_name}.pkl"

    if not model_path.exists() or not artifact_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена! Проверьте наличие файлов:\n"
            f"  {model_path}\n"
            f"  {artifact_path}"
        )

    model = joblib.load(model_path)
    artifact = joblib.load(artifact_path)

    print(f"✓ Модель загружена из {model_path}")
    print(f"✓ {artifact_name.title()} загружен из {artifact_path}")

    return model, artifact


def sklearn_model_exists(
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    artifact_name: str = "scaler",
) -> bool:
    """Проверяет существование сохранённой sklearn-модели."""
    full_name = f"{model_name}_{dataset_name}"
    model_path = models_dir / f"{full_name}_model.pkl"
    artifact_path = models_dir / f"{full_name}_{artifact_name}.pkl"
    return model_path.exists() and artifact_path.exists()


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

def save_pytorch_model(
    model_state: dict,
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    training_params: dict | None = None,
    test_metrics: dict | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> None:
    """
    Сохраняет PyTorch-модель (state_dict) с бэкапом и отчётом.

    Args:
        model_state: state_dict модели (или checkpoint dict)
        dataset_name: имя датасета
        models_dir: папка для сохранения
        model_name: имя модели
        training_params: параметры обучения
        test_metrics: метрики
        extra_artifacts: дополнительные артефакты для сохранения
            {расширение: содержимое} — например {".vocab.json": {...}}
    """
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{dataset_name}"

    model_path = models_dir / f"{full_name}_model.pt"
    backup_path = models_dir / f"{full_name}_model_{timestamp}.pt"
    report_path = models_dir / f"{full_name}_training_report.txt"

    torch.save(model_state, model_path)
    torch.save(model_state, backup_path)

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
    print(f"✓ Отчёт:  {report_path.absolute()}")
    print(f"{'=' * 60}")


def pytorch_model_exists(
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
) -> bool:
    """Проверяет существование сохранённой PyTorch-модели."""
    full_name = f"{model_name}_{dataset_name}"
    return (models_dir / f"{full_name}_model.pt").exists()


# ---------------------------------------------------------------------------
# Общее
# ---------------------------------------------------------------------------

def save_metrics_report(
    report_lines: list[str],
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
) -> Path:
    """Сохраняет текстовый отчёт с метриками."""
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{dataset_name}"
    metrics_path = models_dir / f"{full_name}_metrics_{timestamp}.txt"

    full_report_lines = [
        f"model_name: {model_name}",
        f"dataset_name: {dataset_name}",
        f"saved_at: {timestamp}",
        "",
        *report_lines,
        "",
    ]
    metrics_path.write_text("\n".join(full_report_lines), encoding="utf-8")
    print(f"✓ Метрики сохранены: {metrics_path.absolute()}")
    return metrics_path
