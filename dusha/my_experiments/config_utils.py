"""
Общие утилиты конфигурации для всех экспериментов.

Централизованно загружает data.config / train_data.config / test_data.config,
предоставляет константы (EMO2LABEL, TARGET_NAMES, MODELS_DIR, MODEL_NAME)
и вспомогательные функции.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PROJECT_ROOT: корень репозитория (dusha_new/)
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Ищет корень проекта, поднимаясь от текущего модуля до папки my_experiments."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == "my_experiments":
            return parent.parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# MY_EXPERIMENTS_DIR
# ---------------------------------------------------------------------------

MY_EXPERIMENTS_DIR: Path = PROJECT_ROOT / "my_experiments"


# ---------------------------------------------------------------------------
# Загрузка конфигов
# ---------------------------------------------------------------------------

def _exec_config(config_path: Path) -> dict:
    config_ns: dict[str, Any] = {"__file__": str(config_path)}
    exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns


def _load_configs() -> tuple[Path, Path, Path]:
    """Загружает data.config, train_data.config, test_data.config."""
    data_cfg = MY_EXPERIMENTS_DIR / "data.config"
    train_cfg = MY_EXPERIMENTS_DIR / "train_data.config"
    test_cfg = MY_EXPERIMENTS_DIR / "test_data.config"

    if not data_cfg.exists():
        raise FileNotFoundError(f"data.config не найден: {data_cfg}")
    if not train_cfg.exists():
        raise FileNotFoundError(f"train_data.config не найден: {train_cfg}")
    if not test_cfg.exists():
        raise FileNotFoundError(f"test_data.config не найден: {test_cfg}")

    data_ns = _exec_config(data_cfg)
    train_ns = _exec_config(train_cfg)
    test_ns = _exec_config(test_cfg)

    return (
        data_ns["base_path"],
        Path(train_ns["train_data_path"]),
        Path(test_ns["test_data_path"]),
    )


DATASET_PATH: Path
TRAIN_DATA_PATH: Path
TEST_DATA_PATH: Path

DATASET_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH = _load_configs()


# ---------------------------------------------------------------------------
# Модельные константы
# ---------------------------------------------------------------------------

EMO2LABEL: dict[str, int] = {"angry": 0, "sad": 1, "neutral": 2, "positive": 3}
LABEL2EMO: dict[int, str] = {v: k for k, v in EMO2LABEL.items()}
TARGET_NAMES: list[str] = [LABEL2EMO[i] for i in range(len(LABEL2EMO))]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_dataset_name(train_manifest_path: Path | str) -> str:
    """Извлекает имя датасета из пути к манифесту."""
    return Path(train_manifest_path).stem


def models_dir_for(caller_file: Path | str) -> Path:
    """Возвращает папку models_params рядом с вызывающим скриптом."""
    return Path(caller_file).parent / "models_params"


def resolve_aggregated_dir(dataset_path: Path) -> Path:
    """Ищет aggregated_dataset в стандартных расположениях."""
    candidates = [
        dataset_path / "processed_dataset_090" / "aggregated_dataset",
        dataset_path / "aggregated_dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Загрузка конфигов экспериментов (configs/*.json)
# ---------------------------------------------------------------------------

import json
import argparse


CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


def load_experiment_config(config_path: Path | str | None) -> dict:
    """
    Загружает JSON-конфиг эксперимента.

    Если config_path — None, возвращает пустой dict (используются
    дефолты из argparse / hardcoded значения).
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Убираем служебные ключи (начинаются с _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def apply_config_to_args(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """
    Применяет значения из JSON-конфига к argparse.Namespace.

    Конфиг применяется ТОЛЬКО к тем аргументам, которые ещё не были
    переданы через CLI (т.е. имеют значение по умолчанию).

    Важный нюанс: argparse не различает "передано явно" vs "дефолт".
    Поэтому применяется правило: если значение в config не None и
    атрибут существует — перезаписываем. Это даёт приоритет конфигу
    перед дефолтами argparse, но CLI-флаги нужно передавать ПОСЛЕ --config.
    """
    for key, value in config.items():
        # Преобразуем kebab-case → snake_case
        attr_name = key.replace("-", "_")
        if hasattr(args, attr_name) and value is not None:
            setattr(args, attr_name, value)
    return args


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргумент --config во все экспериментальные скрипты."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Путь к JSON-конфигу с гиперпараметрами. "
            "Может быть абсолютным или относительно configs/. "
            "Пример: --config text/bilstm_text.json"
        ),
    )
