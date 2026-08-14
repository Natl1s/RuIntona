"""Smoke tests for multimodal models: run --mode smoke on tiny synthetic LMDBs.

Early Fusion needs pretrained audio (CNN-BiLSTM) and text (RuBERT) backbones,
so the test builds tiny synthetic checkpoints: a random-weight EmotionCNNBiLSTM
and a small EmotionClassifier on a light HuggingFace backbone (bert-tiny).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from smoke_helpers import create_multimodal_lmdb  # noqa: E402

from dusha.my_experiments.audio_models.CNN.CNN_BiLSTM import EmotionCNNBiLSTM  # noqa: E402
from dusha.my_experiments.text_models.transformers.RuBERT import EmotionClassifier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MULTIMODAL_CHK = PROJECT_ROOT / "dusha" / "my_experiments" / "checkpoints" / "multimodal"
TEXT_BACKBONE = "prajjwal1/bert-tiny"
TIMEOUT = 600

AUDIO_ARCH = {
    "lstm_hidden_size": 128,
    "lstm_layers": 2,
    "lstm_dropout": 0.2,
    "bidirectional": True,
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_audio_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    model = EmotionCNNBiLSTM(n_classes=4, **AUDIO_ARCH)
    model.eval()
    torch.save(model.state_dict(), path)
    return path


def _build_text_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    model = EmotionClassifier(model_name=TEXT_BACKBONE, num_classes=4, dropout=0.2)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_params": {
            "backbone_name": TEXT_BACKBONE,
            "n_classes": 4,
            "dropout": 0.2,
            "classifier_hidden_size": model.classifier_hidden_size,
            "max_len": 64,
        },
    }
    torch.save(checkpoint, path)
    return path


@pytest.fixture(scope="module")
def early_fusion_backbones(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("smoke_ef_backbones")
    audio_path = _build_audio_checkpoint(tmp / "CNN_BiLSTM_combine_balanced_train_model.pt")
    text_path = _build_text_checkpoint(tmp / "RuBERT_dusha_resd_train_model.pt")
    return audio_path, text_path


def test_early_fusion_smoke(
    tmp_path_factory: pytest.TempPathFactory,
    early_fusion_backbones: tuple[Path, Path],
) -> None:
    audio_path, text_path = early_fusion_backbones
    tmp = tmp_path_factory.mktemp("smoke_ef_data")
    train_path = create_multimodal_lmdb(tmp / "train", num_samples=80)
    test_path = create_multimodal_lmdb(tmp / "test", num_samples=40)

    script = (
        PROJECT_ROOT
        / "dusha"
        / "my_experiments"
        / "multimodal"
        / "early_fusion"
        / "Early_Fusion_Baseline.py"
    )
    assert script.exists(), f"Script not found: {script}"

    result = subprocess.run(
        [
            sys.executable, str(script),
            "--mode", "smoke",
            "--train-data-path", str(train_path),
            "--test-data-path", str(test_path),
            "--audio-model-path", str(audio_path),
            "--text-model-path", str(text_path),
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    combined = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, (
        f"Early_Fusion_Baseline exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "accuracy" in combined.lower(), (
        "Early_Fusion_Baseline: no 'accuracy' in output"
    )

    checkpoints = sorted(MULTIMODAL_CHK.glob("Early_Fusion_Baseline_train_model_*.pt"))
    checkpoint_path = checkpoints[-1] if checkpoints else (
        MULTIMODAL_CHK / "Early_Fusion_Baseline_train_model.pt"
    )
    assert checkpoint_path.exists(), (
        f"Unified checkpoint not created: {checkpoint_path}"
    )

    ef = _load_module("Early_Fusion_Baseline", script)
    model, tokenizer = ef.load_model("train")
    model.eval()
    with torch.no_grad():
        audio = torch.randn(1, 1, 80, 40)
        tokenized = tokenizer(["привет мир"], return_tensors="pt")
        logits = model(
            audio,
            torch.tensor([40], dtype=torch.long),
            tokenized["input_ids"],
            tokenized["attention_mask"],
        )
        assert logits.shape == (1, 4), f"Unexpected logits shape: {logits.shape}"
