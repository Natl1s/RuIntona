"""Regression tests for PyTorch checkpoint loading.

Covers the case where a checkpoint contains non-tensor globals (numpy arrays),
which PyTorch 2.6+ blocks under the default ``weights_only=True`` in
``torch.load``. ``model_io.load_torch_with_weights`` must fall back to a full
unpickle for such own-repo checkpoints (old BiLSTM format stores
``embedding_matrix`` as a numpy array).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ruintona.my_experiments.utils.model_io import (  # noqa: E402
    load_pytorch_model,
    load_torch_with_weights,
)

pytestmark = pytest.mark.skipif(
    torch.__version__ < "1.13",
    reason="weights_only argument requires torch >= 1.13",
)


def _save_torch_checkpoint(path, payload) -> None:
    torch.save(payload, path)


def test_load_torch_with_weights_numpy_global(tmp_path) -> None:
    """Checkpoint with a numpy array loads despite weights_only=True blocking it."""
    checkpoint_path = tmp_path / "old_bilstm.pt"
    payload = {
        "model_state_dict": {"embedding.weight": torch.randn(4, 8)},
        "embedding_matrix": np.arange(32, dtype=np.float32).reshape(4, 8),
        "word2idx": {"<pad>": 0, "hello": 1},
        "target_names": ["angry", "sad", "neutral", "positive"],
        "model_params": {"hidden_size": 8, "num_layers": 1},
    }
    _save_torch_checkpoint(checkpoint_path, payload)

    with pytest.raises(Exception):
        torch.load(checkpoint_path, weights_only=True)

    loaded = load_torch_with_weights(checkpoint_path, map_location="cpu")
    assert set(loaded.keys()) == set(payload.keys())
    assert isinstance(loaded["embedding_matrix"], np.ndarray)
    np.testing.assert_allclose(loaded["embedding_matrix"], payload["embedding_matrix"])


def test_load_pytorch_model_unified_format(tmp_path) -> None:
    """load_pytorch_model returns the unified checkpoint dict with numpy artifacts."""
    dataset_name = "combine_balanced_train"
    model_name = "TestModel"
    checkpoint_path = tmp_path / f"{model_name}_{dataset_name}_model.pt"
    payload = {
        "model_state_dict": {"fc.weight": torch.randn(2, 2)},
        "model_class": "SomeClassifier",
        "model_params": {"n_classes": 4},
        "dataset_name": dataset_name,
        "model_name": model_name,
        "created_at": "20260101_000000",
        "training_params": {},
        "test_metrics": {},
        "extra_artifacts": {"embedding_matrix.pkl": np.zeros((2, 2), dtype=np.float32)},
    }
    _save_torch_checkpoint(checkpoint_path, payload)

    loaded = load_pytorch_model(
        dataset_name,
        models_dir=tmp_path,
        model_name=model_name,
        map_location="cpu",
    )
    assert loaded["model_class"] == "SomeClassifier"
    np.testing.assert_allclose(
        loaded["extra_artifacts"]["embedding_matrix.pkl"],
        payload["extra_artifacts"]["embedding_matrix.pkl"],
    )


def test_load_pytorch_model_legacy_state_dict(tmp_path) -> None:
    """Raw state_dict (old format) is wrapped as model_state_dict."""
    dataset_name = "legacy"
    model_name = "LegacyModel"
    checkpoint_path = tmp_path / f"{model_name}_{dataset_name}_model.pt"
    state_dict = {"features.0.weight": torch.randn(4, 8)}
    _save_torch_checkpoint(checkpoint_path, state_dict)

    loaded = load_pytorch_model(
        dataset_name,
        models_dir=tmp_path,
        model_name=model_name,
        map_location="cpu",
    )
    assert "model_state_dict" in loaded
    assert set(loaded["model_state_dict"].keys()) == set(state_dict.keys())


def test_save_pytorch_model_timestamp_roundtrip(tmp_path) -> None:
    """save_pytorch_model writes {name}_model_{ts}.pt and load reads it back."""
    from ruintona.my_experiments.utils.model_io import (
        pytorch_model_exists,
        save_pytorch_model,
    )

    dataset_name = "combine_balanced_train"
    model_name = "RoundTripModel"
    state = {"fc.weight": torch.randn(2, 2)}

    model_path = save_pytorch_model(
        state, dataset_name,
        models_dir=tmp_path, model_name=model_name,
        model_class="SomeClassifier", model_params={"n_classes": 4},
    )

    assert model_path.name.startswith(f"{model_name}_{dataset_name}_model_")
    assert model_path.name.endswith(".pt")
    assert model_path != tmp_path / f"{model_name}_{dataset_name}_model.pt"
    assert model_path.exists()

    loaded = load_pytorch_model(
        dataset_name, models_dir=tmp_path, model_name=model_name,
    )
    assert set(loaded["model_state_dict"].keys()) == set(state.keys())
    assert pytorch_model_exists(
        dataset_name, models_dir=tmp_path, model_name=model_name
    )


def test_load_pytorch_model_prefers_latest_timestamp(tmp_path) -> None:
    """load_pytorch_model picks the most recent timestamped checkpoint."""
    from ruintona.my_experiments.utils.model_io import save_pytorch_model

    dataset_name = "dataset"
    model_name = "TsModel"

    save_pytorch_model(
        {"w": torch.ones(1)}, dataset_name,
        models_dir=tmp_path, model_name=model_name,
    )
    old = tmp_path / f"{model_name}_{dataset_name}_model_20200101_000000.pt"
    torch.save(
        {"model_state_dict": {"w": torch.zeros(1)}, "created_at": "20200101_000000"},
        old,
    )

    loaded = load_pytorch_model(
        dataset_name, models_dir=tmp_path, model_name=model_name,
    )
    assert loaded["model_state_dict"]["w"].item() == 1.0


def test_save_sklearn_model_timestamp_roundtrip(tmp_path) -> None:
    """save_sklearn_model writes timestamped model+artifact; load matches them."""
    from ruintona.my_experiments.utils.model_io import (
        load_sklearn_model,
        save_sklearn_model,
        sklearn_model_exists,
    )

    dataset_name = "dusha_resd_train"
    model_name = "SkModel"

    model = {"type": "dummy", "coef": [1, 2, 3]}
    artifact = {"scale": 2.0}

    model_path = save_sklearn_model(
        model, artifact, dataset_name,
        models_dir=tmp_path, model_name=model_name, artifact_name="scaler",
    )
    assert model_path.name.startswith(f"{model_name}_{dataset_name}_model_")
    assert model_path.name.endswith(".pkl")

    artifact_paths = list(tmp_path.glob(f"{model_name}_{dataset_name}_scaler_*.pkl"))
    assert len(artifact_paths) == 1

    loaded_model, loaded_artifact = load_sklearn_model(
        dataset_name, models_dir=tmp_path, model_name=model_name,
        artifact_name="scaler",
    )
    assert loaded_model == model
    assert loaded_artifact == artifact
    assert sklearn_model_exists(
        dataset_name, models_dir=tmp_path, model_name=model_name,
    )
