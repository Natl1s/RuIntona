"""
Общие утилиты CLI для всех экспериментов.

Централизует argparse (--mode, --no-save) и логику диспетчеризации
(auto / train / load).
"""

from __future__ import annotations

import argparse
from typing import Any, Callable


def add_mode_args(parser: argparse.ArgumentParser) -> None:
    """Добавляет стандартные аргументы --mode и --no-save."""
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load", "auto"],
        default="auto",
        help=(
            "Режим работы: train — обучить новую модель, "
            "load — загрузить существующую, "
            "auto — загрузить если есть, иначе обучить"
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять модель после обучения",
    )


def dispatch_mode(
    args: argparse.Namespace,
    *,
    train_fn: Callable[..., Any],
    load_fn: Callable[..., Any],
    model_exists_fn: Callable[[str], bool],
    dataset_name: str,
) -> Any:
    """
    Диспетчеризует режим работы (auto / train / load).

    Args:
        args: распарсенные аргументы (должен содержать .mode и .no_save)
        train_fn: функция обучения (принимает save=bool)
        load_fn: функция загрузки (без аргументов)
        model_exists_fn: проверка существования модели (принимает dataset_name)
        dataset_name: имя датасета
    """
    if args.mode == "train":
        print("🎯 Режим: Обучение новой модели\n")
        return train_fn(save=not args.no_save)

    if args.mode == "load":
        print("📂 Режим: Загрузка существующей модели\n")
        return load_fn()

    # auto
    if model_exists_fn(dataset_name):
        print("📂 Режим: AUTO — найдена существующая модель, загружаем...\n")
        return load_fn()

    print("🎯 Режим: AUTO — модель не найдена, начинаем обучение...\n")
    return train_fn(save=not args.no_save)
