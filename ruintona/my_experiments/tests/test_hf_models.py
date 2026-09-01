"""Tests for Hugging Face model registry and auto-download (utils/hf_models.py).

All tests are hermetic: Hugging Face downloads are mocked, and CHECKPOINTS_DIR
is redirected to a tmp dir, so the suite never hits the network or the real
checkpoints/ directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from ruintona.my_experiments.utils import hf_models  # noqa: E402


def _checkpoints(tmp_path: Path) -> None:
    hf_models.CHECKPOINTS_DIR = tmp_path


def test_registry_has_required_fields() -> None:
    for key in ("audio", "text"):
        entry = hf_models.MODEL_REGISTRY[key]
        assert entry["repo_id"].count("/") == 1
        assert entry["filename"].endswith(".pt")
        assert entry["local_rel"].startswith(("audio/", "text/"))


def test_returns_existing_local_file(tmp_path) -> None:
    local = tmp_path / "mymodel.pt"
    local.write_bytes(b"not-a-real-model")
    path = hf_models.ensure_checkpoint("audio", local_paths=[local])
    assert path == local


def test_returns_registry_local_path_without_download(tmp_path) -> None:
    _checkpoints(tmp_path)
    local = tmp_path / hf_models.MODEL_REGISTRY["text"]["local_rel"]
    local.parent.mkdir(parents=True)
    local.write_bytes(b"model")
    path = hf_models.ensure_checkpoint("text", download=False)
    assert path == local


def test_missing_without_download_raises(tmp_path) -> None:
    _checkpoints(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        hf_models.ensure_checkpoint("audio", download=False, local_paths=[tmp_path / "nope.pt"])
    message = str(excinfo.value)
    assert "--no-download" in message
    assert "audio" in message


def test_unknown_key_raises() -> None:
    with pytest.raises(KeyError) as excinfo:
        hf_models.ensure_checkpoint("video")
    assert "video" in str(excinfo.value)


def test_downloads_missing_checkpoint_from_hf(tmp_path, monkeypatch) -> None:
    _checkpoints(tmp_path)
    downloaded_path = (
        tmp_path / "hf" / "Natlis__cnn-bilstm-emotion-classification-ru" / "CNN_BiLSTM_combine_balanced_train_model.pt"
    )
    downloaded_path.parent.mkdir(parents=True)
    downloaded_path.write_bytes(b"weights")

    calls: list[dict] = []

    def fake_download(repo_id, filename, local_dir, **kwargs) -> Path:
        calls.append({"repo_id": repo_id, "filename": filename, "local_dir": str(local_dir)})
        return downloaded_path

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    result = hf_models.ensure_checkpoint("audio", local_paths=[tmp_path / "nope.pt"])
    assert result == downloaded_path
    assert calls == [
        {
            "repo_id": "Natlis/cnn-bilstm-emotion-classification-ru",
            "filename": "CNN_BiLSTM_combine_balanced_train_model.pt",
            "local_dir": str(tmp_path / "hf" / "Natlis__cnn-bilstm-emotion-classification-ru"),
        }
    ]


def test_download_failure_raises_readable_error(tmp_path, monkeypatch) -> None:
    _checkpoints(tmp_path)

    def failing_download(*args, **kwargs) -> Path:
        raise ConnectionError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failing_download)

    with pytest.raises(RuntimeError) as excinfo:
        hf_models.ensure_checkpoint("audio", local_paths=[tmp_path / "nope.pt"])
    message = str(excinfo.value)
    assert "offline" in message
    assert "Hugging Face" in message
    assert "локально" in message