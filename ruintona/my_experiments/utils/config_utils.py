"""
Общие утилиты конфигурации для всех экспериментов.

Централизованно загружает data.json,
предоставляет константы (EMO2LABEL, TARGET_NAMES, CHECKPOINTS_DIR)
и вспомогательные функции.
"""

from __future__ import annotations

import json
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
# Загрузка конфигов (JSON вместо exec)
# ---------------------------------------------------------------------------

def _resolve_path(template: str, base_path: Path) -> Path:
    """Подставляет {base_path} в шаблон пути."""
    return Path(str(template).replace("{base_path}", str(base_path)))


def _load_configs() -> tuple[Path | None, Path | None, Path | None]:
    """Загружает data.json (JSON, безопасный).

    Если data.json отсутствует — возвращает (None, None, None), чтобы
    импорт модуля не падал (smoke-тесты и запуски с явными CLI-флагами
    работают без конфига). Ошибка с понятным сообщением возникает позже,
    в resolve_data_paths/get_dataset_path, если путь реально понадобился.
    """
    data_json = MY_EXPERIMENTS_DIR / "data.json"
    if not data_json.exists():
        return None, None, None

    with open(data_json, encoding="utf-8") as f:
        cfg = json.load(f)

    base_path = Path(cfg["base_path"])
    train_path = _resolve_path(cfg["train_lmdb"], base_path)
    test_path = _resolve_path(cfg["test_lmdb"], base_path)
    return base_path, train_path, test_path


DATASET_PATH: Path | None
TRAIN_DATA_PATH: Path | None
TEST_DATA_PATH: Path | None

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

    Ищет последний чекпоинт с timestamp (``{Model}_{dataset}_model_{ts}.{pt|pkl}``);
    если таковых нет — legacy ``{Model}_{dataset}_model.{pt|pkl}``.

    Пример:
        resolve_model_path('audio', 'CNN', 'dusha_balanced')
        → checkpoints/audio/CNN_dusha_balanced_model_{ts}.pt
    """
    from ruintona.my_experiments.utils.model_io import resolve_latest_checkpoint

    models_dir = CHECKPOINTS_DIR / subfolder
    full_name = f"{model_name}_{dataset_name}"
    for ext in ("pt", "pkl"):
        path = resolve_latest_checkpoint(models_dir, full_name, ext)
        if path is not None:
            return path
    # Возвращаем .pt по умолчанию
    return models_dir / f"{full_name}_model.pt"


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


def apply_config_to_args(
    args: argparse.Namespace,
    config: dict,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """
    Применяет значения из JSON-конфига к argparse.Namespace.

    Приоритет: явные CLI-флаги > конфиг > значения по умолчанию.

    Конфиг применяется только к аргументам, которые НЕ были заданы явно
    через CLI: argparse не различает "передано явно" vs "дефолт", поэтому
    сравниваем текущее значение с дефолтом парсера. Если значение
    отличается от дефолта — аргумент задан через CLI, конфиг его не
    перезаписывает.

    Если parser передан, дополнительно проверяются неизвестные ключи
    конфига (не имеющие соответствующего argparse-аргумента) — о них
    выводится предупреждение, а не тихий пропуск. Вложенные группы
    (dict-значения, например `tfidf.ngram_range`) пропускаются — их
    применяют сами скрипты.

    Args:
        args: argparse.Namespace после parse_args().
        config: конфиг из load_experiment_config (ключи без префикса `_`).
        parser: argparse.ArgumentParser, использованный для parse_args().
            Если None — конфиг применяется безусловно (старое поведение,
            сохраняется для обратной совместимости).

    Returns:
        обновлённый args.
    """
    defaults: dict[str, object] = {}
    known_dests: set[str] = set()
    if parser is not None:
        defaults = {action.dest: action.default for action in parser._actions}
        known_dests = set(defaults)

    for key, value in config.items():
        # Вложенные группы (dict-значения) скрипты применяют сами
        # (например, TF-IDF_LogReg читает config['tfidf'] напрямую) —
        # через argparse их не прокинуть.
        if isinstance(value, dict):
            continue

        # Преобразуем kebab-case → snake_case
        attr_name = key.replace("-", "_")

        if parser is not None:
            if attr_name not in known_dests:
                print(
                    f"⚠️  Конфиг: ключ '{key}' не соответствует ни одному аргументу "
                    f"скрипта и будет проигнорирован (возможно, опечатка).",
                    file=sys.stderr,
                )
                continue
            # Значение задано явно в CLI (отличается от дефолта) — конфиг не трогаем
            if getattr(args, attr_name, None) != defaults.get(attr_name):
                continue

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


MISSING_DATA_JSON_MSG = (
    f"data.json не найден: {MY_EXPERIMENTS_DIR / 'data.json'}. "
    "Создайте его из data.json.example или передайте пути через CLI-флаги."
)


def resolve_data_paths(args) -> tuple[Path, Path]:
    """Возвращает (train_path, test_path) — из CLI аргументов или из конфигов."""
    train_path = getattr(args, "train_data_path", None) or TRAIN_DATA_PATH
    test_path = getattr(args, "test_data_path", None) or TEST_DATA_PATH
    if train_path is None:
        raise FileNotFoundError(f"train LMDB не задан. {MISSING_DATA_JSON_MSG}")
    if test_path is None:
        raise FileNotFoundError(f"test LMDB не задан. {MISSING_DATA_JSON_MSG}")
    return train_path, test_path


def get_dataset_path() -> Path:
    """Возвращает DATASET_PATH (корень датасета) или падает с понятной ошибкой."""
    if DATASET_PATH is None:
        raise FileNotFoundError(f"Путь к датасету не задан. {MISSING_DATA_JSON_MSG}")
    return DATASET_PATH


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
    from ruintona.my_experiments.utils.model_io import (
        _checkpoint_timestamp,
        resolve_latest_checkpoint,
    )

    models_dir = CHECKPOINTS_DIR / subfolder
    if not models_dir.exists():
        return None

    if dataset_name:
        return (
            resolve_latest_checkpoint(models_dir, f"{model_name}_{dataset_name}", "pt")
            or resolve_latest_checkpoint(models_dir, f"{model_name}_{dataset_name}", "pkl")
        )

    # Ищем любой чекпоинт для данной модели (с timestamp, потом legacy)
    for ext in ("pt", "pkl"):
        ts_candidates = [
            p for p in models_dir.glob(f"{model_name}_*_model_*.{ext}")
            if _checkpoint_timestamp(p.name) is not None
        ]
        if ts_candidates:
            return max(ts_candidates, key=lambda p: _checkpoint_timestamp(p.name))

    for ext in ("pt", "pkl"):
        legacy = sorted(models_dir.glob(f"{model_name}_*_model.{ext}"))
        if legacy:
            return legacy[-1]
    return None


def resolve_sklearn_artifact_path(
    subfolder: str,
    model_name: str,
    dataset_name: str,
    artifact_name: str,
) -> Path:
    """
    Путь к артефакту (scaler / vectorizer) для последнего чекпоинта модели.

    Если чекпоинт сохранён с timestamp, артефакт подбирается с тем же
    timestamp; иначе — legacy ``{full_name}_{artifact_name}.pkl``.
    """
    from ruintona.my_experiments.utils.model_io import (
        _checkpoint_timestamp,
        resolve_latest_checkpoint,
    )

    models_dir = CHECKPOINTS_DIR / subfolder
    full_name = f"{model_name}_{dataset_name}"
    model_path = resolve_latest_checkpoint(models_dir, full_name, "pkl")
    if model_path is None:
        return models_dir / f"{full_name}_{artifact_name}.pkl"
    ts = _checkpoint_timestamp(model_path.name)
    return models_dir / (
        f"{full_name}_{artifact_name}_{ts}.pkl" if ts
        else f"{full_name}_{artifact_name}.pkl"
    )
