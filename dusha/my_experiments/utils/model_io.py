"""
Общие утилиты сохранения/загрузки моделей.

Поддерживает sklearn (joblib) и PyTorch (state_dict) форматы.
Единый формат checkpoint для PyTorch моделей.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import torch

from dusha.my_experiments.utils.config_utils import MY_EXPERIMENTS_DIR


# ---------------------------------------------------------------------------
# CSV experiment log
# ---------------------------------------------------------------------------

def log_experiment_to_csv(
    model_name: str,
    dataset_name: str,
    framework: str,
    training_params: dict | None = None,
    test_metrics: dict | None = None,
) -> None:
    """Записывает строку эксперимента в CSV-лог (experiments.csv)."""
    records_dir = MY_EXPERIMENTS_DIR / "checkpoints"
    records_dir.mkdir(parents=True, exist_ok=True)
    csv_path = records_dir / "experiments.csv"

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
        ).decode().strip()
    except Exception:
        git_hash = ""

    metrics = test_metrics or {}
    # DL-модели (evaluate_split / compute_classification_metrics) используют ключи
    # "f1_macro"/"accuracy", sklearn-модели — "test_f1_macro"/"test_accuracy".
    # Нормализуем, чтобы колонки experiments.csv не оставались пустыми.
    row = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "framework": framework,
        "val_f1_macro": metrics.get("val_f1_macro", ""),
        "test_f1_macro": metrics.get("test_f1_macro", metrics.get("f1_macro", "")),
        "test_accuracy": metrics.get("test_accuracy", metrics.get("accuracy", "")),
        "training_params": json.dumps(training_params, default=str) if training_params else "",
        "git_commit_hash": git_hash,
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Резолвер чекпоинтов
# ---------------------------------------------------------------------------

_TS_PATTERN = re.compile(r"_(\d{8}_\d{6})\.(?:pt|pkl)$")


def _checkpoint_timestamp(name: str) -> str | None:
    """Извлекает timestamp вида ``YYYYMMDD_HHMMSS`` из имени чекпоинта."""
    match = _TS_PATTERN.search(name)
    return match.group(1) if match else None


def resolve_latest_checkpoint(
    models_dir: Path,
    full_name: str,
    ext: str,
) -> Path | None:
    """
    Возвращает последний чекпоинт ``{full_name}_model_{ts}.{ext}``.

    Если чекпоинтов с timestamp нет — legacy-файл ``{full_name}_model.{ext}``
    (старая конвенция без времени). Возвращает ``None``, если ничего нет.
    """
    models_dir = Path(models_dir)
    candidates = [
        p for p in models_dir.glob(f"{full_name}_model_*.{ext}")
        if _checkpoint_timestamp(p.name) is not None
    ]
    if candidates:
        return max(candidates, key=lambda p: _checkpoint_timestamp(p.name))
    legacy = models_dir / f"{full_name}_model.{ext}"
    return legacy if legacy.exists() else None


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
) -> Path:
    """
    Сохраняет sklearn-модель + артефакт (scaler / vectorizer) в формате joblib.

    Имена файлов содержат timestamp: ``{model_name}_{dataset_name}_model_{ts}.pkl``
    и ``{model_name}_{dataset_name}_{artifact_name}_{ts}.pkl`` (общий timestamp),
    поэтому каждая тренировка сохраняется в отдельный файл.

    Args:
        model: обученная sklearn-модель
        artifact: scaler / vectorizer / другой артефакт
        dataset_name: имя датасета (из имени LMDB-файла)
        models_dir: папка для сохранения
        model_name: имя модели (по умолчанию — имя скрипта)
        artifact_name: имя артефакта ("scaler" / "vectorizer")
        training_params: параметры обучения (для отчёта)
        test_metrics: метрики (для отчёта)

    Returns:
        Path к сохранённой модели
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{dataset_name}"

    model_path = models_dir / f"{full_name}_model_{timestamp}.pkl"
    artifact_path = models_dir / f"{full_name}_{artifact_name}_{timestamp}.pkl"
    report_path = models_dir / f"{full_name}_training_report.txt"

    joblib.dump(model, model_path)
    joblib.dump(artifact, artifact_path)

    _write_report(report_path, model_name, dataset_name, timestamp,
                  training_params, test_metrics)

    log_experiment_to_csv(
        model_name=model_name or Path(sys.argv[0]).stem,
        dataset_name=dataset_name,
        framework="sklearn",
        training_params=training_params,
        test_metrics=test_metrics,
    )

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'=' * 60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ {artifact_name.title()}: {artifact_path.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    print(f"{'=' * 60}")

    return model_path


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
    model_path = resolve_latest_checkpoint(models_dir, full_name, "pkl")
    if model_path is None:
        raise FileNotFoundError(
            f"Модель не найдена! Проверьте наличие файлов:\n"
            f"  {models_dir / f'{full_name}_model.pkl'}\n"
            f"  (или версии с timestamp вида {full_name}_model_YYYYMMDD_HHMMSS.pkl)"
        )

    ts = _checkpoint_timestamp(model_path.name)
    artifact_path = (
        models_dir / f"{full_name}_{artifact_name}_{ts}.pkl"
        if ts else models_dir / f"{full_name}_{artifact_name}.pkl"
    )
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Артефакт не найден!\n"
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
    model_path = resolve_latest_checkpoint(models_dir, full_name, "pkl")
    if model_path is None:
        return False
    ts = _checkpoint_timestamp(model_path.name)
    artifact_path = (
        models_dir / f"{full_name}_{artifact_name}_{ts}.pkl"
        if ts else models_dir / f"{full_name}_{artifact_name}.pkl"
    )
    return artifact_path.exists()


# ---------------------------------------------------------------------------
# Safe torch.load helper (weights_only for security, fallback for older PyTorch)
# ---------------------------------------------------------------------------

def load_torch_with_weights(path, map_location=None):
    """
    Загружает torch-чекпоинт, безопасно работая с разными версиями PyTorch.

    Сначала пробует ``torch.load(..., weights_only=True)`` (безопасный режим,
    используемый по умолчанию с PyTorch 2.6+). Если аргумент неизвестен
    (PyTorch < 1.13) либо чекпоинт содержит не-тензорные глобалы — например,
    numpy-матрицу ``embedding_matrix`` из старого формата BiLSTM —
    ``weights_only=True`` падает с ``UnpicklingError``. В таких случаях
    откатываемся к полной загрузке: чекпоинты создаются кодом этого
    репозитория, поэтому источник доверенный.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # PyTorch 2.6+ default weights_only=True blocks non-tensor globals
        # (e.g. numpy arrays saved by save_pytorch_model). Checkpoints come
        # from this repo's own training, so trust them and load fully.
        return torch.load(path, map_location=map_location, weights_only=False)


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
) -> Path:
    """
    Сохраняет PyTorch-модель в едином формате checkpoint.

    Имя файла содержит timestamp: ``{model_name}_{dataset_name}_model_{ts}.pt``,
    поэтому каждая тренировка сохраняется в отдельный файл.

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

    Returns:
        Path к сохранённому checkpoint
    """
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"{model_name}_{dataset_name}"

    model_path = models_dir / f"{full_name}_model_{timestamp}.pt"
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

    log_experiment_to_csv(
        model_name=model_name or Path(sys.argv[0]).stem,
        dataset_name=dataset_name,
        framework="pytorch",
        training_params=training_params,
        test_metrics=test_metrics,
    )

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'=' * 60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    if extra_artifacts:
        print(f"✓ Артефакты: {list(extra_artifacts.keys())}")
    print(f"{'=' * 60}")

    return model_path


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
    full_name = f"{model_name}_{dataset_name}"
    model_path = resolve_latest_checkpoint(models_dir, full_name, "pt")

    if model_path is None:
        raise FileNotFoundError(
            f"Модель не найдена: {models_dir / f'{full_name}_model.pt'}\n"
            f"  (или версии с timestamp вида {full_name}_model_YYYYMMDD_HHMMSS.pt)"
        )

    checkpoint = load_torch_with_weights(model_path, map_location=map_location)

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
    return resolve_latest_checkpoint(models_dir, full_name, "pt") is not None


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
    if framework == "sklearn":
        return save_sklearn_model(
            model, artifact, dataset_name,
            models_dir=models_dir, model_name=model_name,
            artifact_name=artifact_name,
            training_params=training_params, test_metrics=test_metrics,
        )

    elif framework == "pytorch":
        return save_pytorch_model(
            model, dataset_name,
            models_dir=models_dir, model_name=model_name,
            training_params=training_params, test_metrics=test_metrics,
            extra_artifacts=extra_artifacts,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            model_class=model_class,
            model_params=model_params,
        )

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
