"""
Общие утилиты конфигурации для всех экспериментов.

Централизованно загружает data.config / train_data.config / test_data.config,
предоставляет константы (EMO2LABEL, TARGET_NAMES, CHECKPOINTS_DIR)
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
# CHECKPOINTS_DIR
# ---------------------------------------------------------------------------

CHECKPOINTS_DIR: Path = MY_EXPERIMENTS_DIR / "checkpoints"


def checkpoints_dir_for(caller_file: Path | str, subfolder: str = "") -> Path:
    """
    Возвращает подпапку checkpoints/{subfolder}/ для данного типа моделей.

    Args:
        caller_file: __file__ вызывающего скрипта
        subfolder: 'text', 'audio', 'multimodal' или '' (корневая)

    Returns:
        Path к подпапке checkpoints
    """
    folder = CHECKPOINTS_DIR / subfolder if subfolder else CHECKPOINTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _detect_subfolder(caller_file: Path | str) -> str:
    """Авто-определение subfolder по пути скрипта."""
    path_str = str(Path(caller_file).resolve())
    if "/text_models/" in path_str:
        return "text"
    elif "/audio_models/" in path_str:
        return "audio"
    elif "/multimodal/" in path_str:
        return "multimodal"
    return ""


def models_dir_for(caller_file: Path | str) -> Path:
    """
    Возвращает checkpoints/{subfolder}/ для вызывающего скрипта.

    Автоматически определяет subfolder по пути к скрипту.
    """
    subfolder = _detect_subfolder(caller_file)
    return checkpoints_dir_for(caller_file, subfolder)


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


def resolve_model_path(
    subfolder: str,
    model_name: str,
    dataset_name: str,
) -> Path:
    """
    Резолвит путь к чекпоинту модели по конвенции.

    Пример:
        resolve_model_path('audio', 'CNN', 'dusha_balanced')
        → checkpoints/audio/CNN_dusha_balanced_model.pt
    """
    models_dir = CHECKPOINTS_DIR / subfolder
    # Сначала ищем .pt (PyTorch), потом .pkl (sklearn)
    pt_path = models_dir / f"{model_name}_{dataset_name}_model.pt"
    if pt_path.exists():
        return pt_path
    pkl_path = models_dir / f"{model_name}_{dataset_name}_model.pkl"
    if pkl_path.exists():
        return pkl_path
    # Возвращаем .pt по умолчанию
    return pt_path


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


def add_data_path_args(parser: argparse.ArgumentParser) -> None:
    """Добавляет --train-data-path и --test-data-path аргументы."""
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=None,
        help="Путь к train LMDB (по умолчанию из train_data.config)",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=None,
        help="Путь к test LMDB (по умолчанию из test_data.config)",
    )


def resolve_data_paths(args) -> tuple[Path, Path]:
    """Возвращает (train_path, test_path) — из CLI аргументов или из конфигов."""
    train_path = getattr(args, "train_data_path", None) or TRAIN_DATA_PATH
    test_path = getattr(args, "test_data_path", None) or TEST_DATA_PATH
    return train_path, test_path


# ---------------------------------------------------------------------------
# Multimodal: поиск предобученных моделей
# ---------------------------------------------------------------------------

def find_pretrained_model(
    subfolder: str,
    model_name: str,
    dataset_name: str | None = None,
) -> Path | None:
    """
    Ищет сохранённую модель в checkpoints/{subfolder}/.

    Если dataset_name не указан, ищет любой чекпоинт для данного model_name.
    Возвращает Path к checkpoint или None.
    """
    models_dir = CHECKPOINTS_DIR / subfolder
    if not models_dir.exists():
        return None

    if dataset_name:
        # Ищем конкретный чекпоинт
        pt_path = models_dir / f"{model_name}_{dataset_name}_model.pt"
        if pt_path.exists():
            return pt_path
        pkl_path = models_dir / f"{model_name}_{dataset_name}_model.pkl"
        if pkl_path.exists():
            return pkl_path
        return None

    # Ищем любой чекпоинт для данной модели
    for ext in ("*.pt", "*.pkl"):
        for p in sorted(models_dir.glob(f"{model_name}_*{ext}"), reverse=True):
            return p
    return None
