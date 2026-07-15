"""
Общие PyTorch-утилиты для всех экспериментов.

set_seed, resolve_device, build_loader.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    """Выбирает устройство по строковому аргументу ('cuda' / 'cpu' / 'auto')."""
    device_arg = device_arg.lower()
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Запрошено обучение на GPU (--device cuda), но CUDA недоступна."
            )
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    use_cuda: bool,
    collate_fn=None,
) -> DataLoader:
    """Обёртка над DataLoader с едиными настройками по умолчанию."""
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(**kwargs)
