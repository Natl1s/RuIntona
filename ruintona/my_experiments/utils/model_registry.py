"""
Реестр моделей — CRUD-операции с checkpoints/registry.json.

Хранит метаданные обо всех обученных моделях:
  - model_name, dataset_name, model_class, framework
  - checkpoint path, created_at, test_metrics, training_params
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Потокобезопасный реестр моделей в формате JSON."""

    def __init__(self, registry_path: Path):
        self._path = registry_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _entry_key(model_name: str, dataset_name: str) -> str:
        return f"{model_name}/{dataset_name}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        model_name: str,
        dataset_name: str,
        model_class: str,
        framework: str,
        checkpoint_path: Path,
        model_params: dict[str, Any] | None = None,
        training_params: dict[str, Any] | None = None,
        test_metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Регистрирует новую модель (или перезаписывает существующую)."""
        key = self._entry_key(model_name, dataset_name)
        entry = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "model_class": model_class,
            "framework": framework,
            "checkpoint_path": str(checkpoint_path),
            "model_params": model_params or {},
            "training_params": training_params or {},
            "test_metrics": test_metrics or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry["extra"] = extra

        data = self._read()
        data[key] = entry
        self._write(data)
        return entry

    def get(
        self, model_name: str, dataset_name: str
    ) -> dict[str, Any] | None:
        """Возвращает запись или None, если не найдена."""
        key = self._entry_key(model_name, dataset_name)
        return self._read().get(key)

    def find(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        framework: str | None = None,
        model_class: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет записи по комбинации фильтров (все параметры опциональны)."""
        results = []
        for entry in self._read().values():
            if model_name and entry.get("model_name") != model_name:
                continue
            if dataset_name and entry.get("dataset_name") != dataset_name:
                continue
            if framework and entry.get("framework") != framework:
                continue
            if model_class and entry.get("model_class") != model_class:
                continue
            results.append(entry)
        return results

    def latest(
        self, model_name: str, dataset_name: str | None = None
    ) -> dict[str, Any] | None:
        """Возвращает последнюю по времени запись для model_name (+ опц. dataset)."""
        entries = self.find(model_name=model_name, dataset_name=dataset_name)
        if not entries:
            return None
        return max(entries, key=lambda e: e.get("created_at", ""))

    def remove(self, model_name: str, dataset_name: str) -> bool:
        """Удаляет запись. Возвращает True если запись была найдена."""
        key = self._entry_key(model_name, dataset_name)
        data = self._read()
        if key in data:
            del data[key]
            self._write(data)
            return True
        return False

    def list_all(self) -> list[dict[str, Any]]:
        """Возвращает все записи."""
        return list(self._read().values())

    def update_metrics(
        self,
        model_name: str,
        dataset_name: str,
        test_metrics: dict[str, Any],
    ) -> bool:
        """Обновляет test_metrics для существующей записи."""
        key = self._entry_key(model_name, dataset_name)
        data = self._read()
        if key not in data:
            return False
        data[key]["test_metrics"] = test_metrics
        data[key]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(data)
        return True
