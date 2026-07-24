"""
Общие утилиты сохранения/загрузки моделей.

Поддерживает sklearn (joblib) и PyTorch (state_dict) форматы.
Единый формат checkpoint для PyTorch моделей.
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

    _write_report(report_path, model_name, dataset_name, timestamp,
                  training_params, test_metrics)

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
# PyTorch — единый формат checkpoint
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
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    model_class: str = "",
    model_params: dict | None = None,
) -> None:
    """
    Сохраняет PyTorch-модель в едином формате checkpoint.

    Формат checkpoint:
        model_state_dict, model_class, model_params,
        dataset_name, created_at,
        optimizer_state_dict (опционально), scheduler_state_dict (опционально),
        training_params, test_metrics,
        extra_artifacts (опционально).

    Args:
        model_state: state_dict модели
        dataset_name: имя датасета
        models_dir: папка для сохранения
        model_name: имя модели
        training_params: параметры обучения
        test_metrics: метрики
        extra_artifacts: доп. артефакты {имя: данные}
        optimizer_state: state_dict оптимизатора
        scheduler_state: state_dict планировщика
        model_class: имя класса модели (для воссоздания)
        model_params: гиперпараметры модели (для воссоздания)
    """
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{dataset_name}"

    model_path = models_dir / f"{full_name}_model.pt"
    backup_path = models_dir / f"{full_name}_model_{timestamp}.pt"
    report_path = models_dir / f"{full_name}_training_report.txt"

    checkpoint = {
        "model_state_dict": model_state,
        "model_class": model_class,
        "model_params": model_params or {},
        "dataset_name": dataset_name,
        "model_name": model_name,
        "created_at": timestamp,
        "training_params": training_params or {},
        "test_metrics": test_metrics or {},
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if extra_artifacts:
        checkpoint["extra_artifacts"] = extra_artifacts

    torch.save(checkpoint, model_path)
    torch.save(checkpoint, backup_path)

    # Сохраняем extra_artifacts как отдельные файлы для обратной совместимости
    if extra_artifacts:
        for name, data in extra_artifacts.items():
            artifact_path = models_dir / f"{full_name}_{name}"
            if name.endswith(".json"):
                with open(artifact_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                joblib.dump(data, artifact_path)

    _write_report(report_path, model_name, dataset_name, timestamp,
                  training_params, test_metrics)

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'=' * 60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Бэкап:  {backup_path.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    if extra_artifacts:
        print(f"✓ Артефакты: {list(extra_artifacts.keys())}")
    print(f"{'=' * 60}")


def load_pytorch_model(
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    map_location: str = "cpu",
) -> dict[str, Any]:
    """
    Загружает PyTorch checkpoint и возвращает весь dict.

    Обратная совместимость: поддерживает как единый формат checkpoint,
    так и старый формат (простой state_dict без обёртки).

    Returns:
        dict с ключами checkpoint или {'model_state_dict': state_dict}
    """
    import torch

    full_name = f"{model_name}_{dataset_name}"
    model_path = models_dir / f"{full_name}_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}"
        )

    checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)

    # Обратная совместимость: если это просто state_dict (старый формат)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"model_state_dict": checkpoint}

    print(f"✓ Checkpoint загружен из {model_path}")
    return checkpoint


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
# Единый checkpoint API (sklearn + pytorch)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Any,
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    framework: str = "sklearn",
    training_params: dict | None = None,
    test_metrics: dict | None = None,
    artifact: Any = None,
    artifact_name: str = "scaler",
    extra_artifacts: dict[str, Any] | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    model_class: str = "",
    model_params: dict | None = None,
) -> Path:
    """
    Универсальная функция сохранения.

    Автоматически определяет формат по framework и вызывает
    соответствующую функцию. Возвращает путь к checkpoint.

    Args:
        model: обученная модель (sklearn) или state_dict (pytorch)
        dataset_name: имя датасета
        models_dir: папка для сохранения
        model_name: имя модели
        framework: 'sklearn' или 'pytorch'
        training_params: гиперпараметры обучения
        test_metrics: метрики на тесте
        artifact: scaler / vectorizer (для sklearn)
        artifact_name: имя артефакта
        extra_artifacts: доп. артефакты (vocab.json и т.д.)
        optimizer_state: state_dict оптимизатора
        scheduler_state: state_dict планировщика
        model_class: имя класса модели
        model_params: гиперпараметры модели

    Returns:
        Path к сохранённому checkpoint
    """
    full_name = f"{model_name}_{dataset_name}"

    if framework == "sklearn":
        save_sklearn_model(
            model, artifact, dataset_name,
            models_dir=models_dir, model_name=model_name,
            artifact_name=artifact_name,
            training_params=training_params, test_metrics=test_metrics,
        )
        return models_dir / f"{full_name}_model.pkl"

    elif framework == "pytorch":
        save_pytorch_model(
            model, dataset_name,
            models_dir=models_dir, model_name=model_name,
            training_params=training_params, test_metrics=test_metrics,
            extra_artifacts=extra_artifacts,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            model_class=model_class,
            model_params=model_params,
        )
        return models_dir / f"{full_name}_model.pt"

    else:
        raise ValueError(f"Неизвестный framework: '{framework}'. Доступные: sklearn, pytorch")


def load_checkpoint(
    dataset_name: str,
    *,
    models_dir: Path,
    model_name: str = "",
    framework: str = "sklearn",
    artifact_name: str = "scaler",
    map_location: str = "cpu",
) -> Any:
    """
    Универсальная функция загрузки.

    Возвращает:
        - для sklearn: (model, artifact)
        - для pytorch: checkpoint dict
    """
    if framework == "sklearn":
        return load_sklearn_model(
            dataset_name,
            models_dir=models_dir, model_name=model_name,
            artifact_name=artifact_name,
        )
    elif framework == "pytorch":
        return load_pytorch_model(
            dataset_name,
            models_dir=models_dir, model_name=model_name,
            map_location=map_location,
        )
    else:
        raise ValueError(f"Неизвестный framework: '{framework}'. Доступные: sklearn, pytorch")


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

def _write_report(
    report_path: Path,
    model_name: str,
    dataset_name: str,
    timestamp: str,
    training_params: dict | None,
    test_metrics: dict | None,
) -> None:
    """Записывает training report."""
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
