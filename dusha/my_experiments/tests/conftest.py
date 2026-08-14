"""Shared fixtures for smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from smoke_helpers import create_audio_lmdb, create_multimodal_lmdb, create_text_lmdb


@pytest.fixture(scope="session")
def tiny_text_lmdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("smoke_text")
    return create_text_lmdb(tmp / "text_lmdb")


@pytest.fixture(scope="session")
def tiny_audio_lmdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("smoke_audio")
    return create_audio_lmdb(tmp / "audio_lmdb")


@pytest.fixture(scope="session")
def tiny_multimodal_lmdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("smoke_multi")
    return create_multimodal_lmdb(tmp / "multimodal_lmdb")
