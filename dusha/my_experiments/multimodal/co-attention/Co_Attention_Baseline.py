import argparse
import builtins
import json
import math
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, get_linear_schedule_with_warmup

_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index
from my_experiments.config_utils import (
    MY_EXPERIMENTS_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_NAMES,
    load_experiment_config, apply_config_to_args, add_config_arg,
)


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)


DEFAULT_TRAIN_LMDB = TRAIN_DATA_PATH
DEFAULT_TEST_LMDB = TEST_DATA_PATH

DEFAULT_TEXT_MODEL_PATH = (
    MY_EXPERIMENTS_DIR
    / "text_models"
    / "transformers"
    / "models_params"
    / "RuBERT_dusha_resd_train_model.pt"
)

DEFAULT_AUDIO_WARM_START_PATH = (
    MY_EXPERIMENTS_DIR
    / "audio_models"
    / "transformers"
    / "models_params"
    / "wav2vec2_xlsr300m_self_attention_combine_balanced_train_small_model.pt"
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models_params"
MODEL_NAME = Path(__file__).stem
TARGET_SAMPLE_RATE = 16000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен --device cuda, но CUDA недоступна.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def _extract_text(payload: dict) -> str:
    text_keys = ("speaker_text", "text", "transcript", "utterance")
    for key in text_keys:
        if key in payload:
            text = str(payload[key]).strip().lower()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
    return ""


def _normalize_waveform(raw_waveform) -> np.ndarray:
    arr = np.asarray(raw_waveform)
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)

    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
        if peak > 1.0:
            arr = arr / peak
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(arr, -1.0, 1.0)


def _crop_or_pad(
    waveform: np.ndarray,
    is_train: bool,
    min_crop_sec: float,
    max_crop_sec: float,
) -> tuple[np.ndarray, int]:
    if max_crop_sec < min_crop_sec:
        raise ValueError(f"max_crop_sec < min_crop_sec: {max_crop_sec} < {min_crop_sec}")

    target_sec = random.uniform(min_crop_sec, max_crop_sec) if is_train else max_crop_sec
    target_len = int(round(target_sec * TARGET_SAMPLE_RATE))
    if target_len <= 0:
        raise ValueError(f"Некорректная целевая длина: {target_len}")

    src_len = int(waveform.shape[0])
    if src_len >= target_len:
        if is_train:
            start = random.randint(0, src_len - target_len)
        else:
            start = max((src_len - target_len) // 2, 0)
        cropped = waveform[start : start + target_len]
        return cropped.astype(np.float32), target_len

    padded = np.zeros((target_len,), dtype=np.float32)
    padded[:src_len] = waveform
    return padded, src_len


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"lengths должен быть 1D, получено: {lengths.shape}")
    if max_len is None:
        max_len = int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def ensure_non_empty_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"mask должен быть 2D [B, T], получено: {mask.shape}")
    if mask.size(1) == 0:
        raise ValueError("mask не должен быть пустым по временной оси (T=0).")
    valid = mask.any(dim=1)
    if bool(valid.all()):
        return mask
    mask = mask.clone()
    mask[~valid, 0] = True
    return mask


def _tensor_finite_summary(name: str, tensor: torch.Tensor) -> str:
    if tensor is None:
        return f"{name}: none"
    t = tensor.detach()
    if t.numel() == 0:
        return f"{name}: empty"
    n_total = int(t.numel())
    if not (torch.is_floating_point(t) or torch.is_complex(t)):
        min_val = float(t.min().item())
        max_val = float(t.max().item())
        return f"{name}: finite={n_total}/{n_total}, nan=0, inf=0, min={min_val:.6g}, max={max_val:.6g}"
    finite = torch.isfinite(t)
    n_finite = int(finite.sum().item())
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    if n_finite == 0:
        return f"{name}: finite=0/{n_total}, nan={n_nan}, inf={n_inf}"
    finite_vals = t[finite]
    min_val = float(finite_vals.min().item())
    max_val = float(finite_vals.max().item())
    return (
        f"{name}: finite={n_finite}/{n_total}, nan={n_nan}, inf={n_inf}, "
        f"min={min_val:.6g}, max={max_val:.6g}"
    )


def _build_non_finite_report(prefix: str, **tensors: torch.Tensor) -> str:
    lines = [prefix]
    for name, tensor in tensors.items():
        lines.append(_tensor_finite_summary(name, tensor))
    return "\n".join(lines)


def _forward_logits_loss(
    model: nn.Module,
    criterion: nn.Module,
    waves: torch.Tensor,
    valid_lens: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    autocast_device_type: str,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(autocast_device_type, enabled=use_amp):
        logits, _ = model(
            waves=waves,
            valid_lens=valid_lens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_aux=False,
        )
        logits = torch.clamp(logits, -20, 20)
        loss = criterion(logits, labels)
    return logits, loss


def _top2_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    top2 = np.argsort(probs, axis=1)[:, -2:]
    hits = np.any(top2 == y_true[:, None], axis=1)
    return float(hits.mean())


def _safe_roc_auc_ovr_macro(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _attention_entropy(
    weights: torch.Tensor,
    query_mask: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    # weights: [B, H, Q, K] or [B, Q, K]
    if weights.ndim == 3:
        weights = weights.unsqueeze(1)
    if weights.ndim != 4:
        raise ValueError(f"Ожидаются attention weights [B,H,Q,K], получено {weights.shape}")

    p = weights.clamp_min(1e-9)
    entropy = -(p * torch.log(p)).sum(dim=-1)  # [B, H, Q]

    key_lengths = key_mask.sum(dim=1).clamp_min(1).to(dtype=entropy.dtype)  # [B]
    normalizer = torch.log(key_lengths).clamp_min(1e-9).view(-1, 1, 1)
    entropy_norm = entropy / normalizer

    query_mask_f = query_mask.to(dtype=entropy_norm.dtype).unsqueeze(1)  # [B,1,Q]
    denom = query_mask_f.sum().clamp_min(1.0)
    return (entropy_norm * query_mask_f).sum() / denom


class FusionWaveTextDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)

        total = get_lmdb_length(self.env)
        valid_indices = []
        valid_labels = []
        with self.env.begin() as txn:
            for idx in tqdm(range(total), desc=f"Сканирование {self.lmdb_path.name}", unit="sample"):
                raw = txn.get(str(idx).encode("utf-8"))
                if raw is None:
                    continue
                payload = pickle.loads(raw)
                if not isinstance(payload, dict):
                    continue

                text = _extract_text(payload)
                if not text:
                    continue

                waveform_raw = payload.get("waveform", payload.get("audio", payload.get("wav")))
                if waveform_raw is None:
                    continue
                waveform = _normalize_waveform(waveform_raw)
                if waveform.size == 0:
                    continue

                sample_rate = int(payload.get("waveform_sr", payload.get("sample_rate", TARGET_SAMPLE_RATE)))
                if sample_rate != TARGET_SAMPLE_RATE:
                    continue

                label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
                try:
                    label = parse_label_to_index(label_raw)
                except ValueError:
                    continue

                valid_indices.append(idx)
                valid_labels.append(label)

        if not valid_indices:
            raise ValueError(f"В LMDB нет валидных мультимодальных примеров: {self.lmdb_path}")
        self.indices = np.asarray(valid_indices, dtype=np.int64)
        self.labels = np.asarray(valid_labels, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, item_idx: int):
        lmdb_idx = int(self.indices[item_idx])
        with self.env.begin() as txn:
            raw = txn.get(str(lmdb_idx).encode("utf-8"))
        if raw is None:
            raise KeyError(f"В LMDB отсутствует ключ {lmdb_idx}")

        payload = pickle.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Некорректный payload у ключа {lmdb_idx}: ожидается dict")

        text = _extract_text(payload)
        if not text:
            raise ValueError(f"Пустой текст в валидном примере (idx={lmdb_idx})")

        waveform_raw = payload.get("waveform", payload.get("audio", payload.get("wav")))
        if waveform_raw is None:
            raise KeyError(
                f"В payload отсутствует waveform/audio/wav (idx={lmdb_idx}); "
                "для co-attention требуется raw waveform"
            )
        waveform = _normalize_waveform(waveform_raw)
        if waveform.size == 0:
            raise ValueError(f"Пустой waveform в валидном примере (idx={lmdb_idx})")

        sample_rate = int(payload.get("waveform_sr", payload.get("sample_rate", TARGET_SAMPLE_RATE)))
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Некорректная sample_rate={sample_rate} в idx={lmdb_idx}; "
                f"ожидается {TARGET_SAMPLE_RATE}"
            )

        label = parse_label_to_index(payload.get("y", payload.get("label", payload.get("emotion"))))
        return torch.from_numpy(waveform), text, int(label)


def build_collate_fn(
    tokenizer,
    max_len: int,
    min_crop_sec: float,
    max_crop_sec: float,
    is_train: bool,
):
    def _collate(batch):
        waves, texts, labels = zip(*batch)

        cropped = []
        valid_lens = []
        for wave in waves:
            wave_np = wave.detach().cpu().numpy()
            wave_np, valid_len = _crop_or_pad(
                waveform=wave_np,
                is_train=is_train,
                min_crop_sec=min_crop_sec,
                max_crop_sec=max_crop_sec,
            )
            cropped.append(torch.from_numpy(wave_np))
            valid_lens.append(valid_len)

        max_wave_len = max(x.size(0) for x in cropped)
        wave_batch = torch.zeros((len(cropped), max_wave_len), dtype=torch.float32)
        for i, x in enumerate(cropped):
            wave_batch[i, : x.size(0)] = x

        valid_lens_t = torch.tensor(valid_lens, dtype=torch.long)
        tokenized = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        label_batch = torch.tensor(labels, dtype=torch.long)
        return (
            wave_batch,
            valid_lens_t,
            tokenized["input_ids"],
            tokenized["attention_mask"],
            label_batch,
        )

    return _collate


class MaskedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # seq: [B, T, D], mask: [B, T]
        scores = self.attn(seq).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)
        return pooled, weights


class AudioSequenceEncoder(nn.Module):
    def __init__(self, pretrained_name: str, gradient_checkpointing: bool):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(pretrained_name)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.output_dim = int(self.model.config.hidden_size)

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        layers = self.model.encoder.layers
        trainable = min(int(n_layers), len(layers))
        for layer in layers[-trainable:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, waves: torch.Tensor, valid_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = lengths_to_mask(valid_lens, max_len=waves.size(1)).to(dtype=torch.long)
        encoder_has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(encoder_has_trainable_params):
            outputs = self.model(
                input_values=waves,
                attention_mask=attention_mask,
                return_dict=True,
            )
        seq = outputs.last_hidden_state
        feat_lens = self.model._get_feat_extract_output_lengths(valid_lens).to(torch.long)
        feat_lens = torch.clamp(feat_lens, min=1, max=seq.size(1))
        feat_mask = lengths_to_mask(feat_lens, max_len=seq.size(1))
        feat_mask = ensure_non_empty_mask(feat_mask)
        return seq, feat_mask


class TextSequenceEncoder(nn.Module):
    def __init__(self, backbone: AutoModel):
        super().__init__()
        self.model = backbone
        self.output_dim = int(self.model.config.hidden_size)

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layer"):
            return
        layers = self.model.encoder.layer
        trainable = min(int(n_layers), len(layers))
        for layer in layers[-trainable:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(encoder_has_trainable_params):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        seq = outputs.last_hidden_state
        mask = attention_mask.to(dtype=torch.bool)
        mask = ensure_non_empty_mask(mask)
        return seq, mask


class CoAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_mult: int):
        super().__init__()
        self.t2a_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.a2t_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_t = nn.Dropout(dropout)
        self.drop_a = nn.Dropout(dropout)
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * ffn_mult)
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm_t2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)
        self.ffn_drop_t = nn.Dropout(dropout)
        self.ffn_drop_a = nn.Dropout(dropout)

    def forward(
        self,
        text_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        t2a, w_t2a = self.t2a_attn(
            query=text_seq,
            key=audio_seq,
            value=audio_seq,
            key_padding_mask=~audio_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        a2t, w_a2t = self.a2t_attn(
            query=audio_seq,
            key=text_seq,
            value=text_seq,
            key_padding_mask=~text_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        text_ctx = self.norm_t1(text_seq + self.drop_t(t2a))
        audio_ctx = self.norm_a1(audio_seq + self.drop_a(a2t))

        text_ctx = self.norm_t2(text_ctx + self.ffn_drop_t(self.ffn_t(text_ctx)))
        audio_ctx = self.norm_a2(audio_ctx + self.ffn_drop_a(self.ffn_a(audio_ctx)))
        return text_ctx, audio_ctx, w_t2a, w_a2t


class CrossModalCoAttentionModel(nn.Module):
    def __init__(
        self,
        audio_encoder: AudioSequenceEncoder,
        text_encoder: TextSequenceEncoder,
        d_model: int,
        num_heads: int,
        n_blocks: int,
        dropout: float,
        ffn_mult: int,
        n_classes: int,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_proj = nn.Linear(self.audio_encoder.output_dim, d_model)
        self.text_proj = nn.Linear(self.text_encoder.output_dim, d_model)
        self.audio_in_drop = nn.Dropout(dropout)
        self.text_in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CoAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult) for _ in range(n_blocks)]
        )
        self.text_pool = MaskedAttentionPooling(d_model)
        self.audio_pool = MaskedAttentionPooling(d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def freeze_backbones(self) -> None:
        self.audio_encoder.freeze_all()
        self.text_encoder.freeze_all()

    def unfreeze_last_layers(self, audio_layers: int, text_layers: int) -> None:
        self.audio_encoder.unfreeze_last_layers(audio_layers)
        self.text_encoder.unfreeze_last_layers(text_layers)

    def forward(
        self,
        waves: torch.Tensor,
        valid_lens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        audio_seq, audio_mask = self.audio_encoder(waves, valid_lens)
        text_seq, text_mask = self.text_encoder(input_ids, attention_mask)

        audio_seq = self.audio_in_drop(self.audio_proj(audio_seq))
        text_seq = self.text_in_drop(self.text_proj(text_seq))

        attn_t2a = None
        attn_a2t = None
        for block in self.blocks:
            text_seq, audio_seq, cur_t2a, cur_a2t = block(
                text_seq=text_seq,
                audio_seq=audio_seq,
                text_mask=text_mask,
                audio_mask=audio_mask,
                need_weights=return_aux,
            )
            if return_aux:
                attn_t2a = cur_t2a
                attn_a2t = cur_a2t

        text_vec, _ = self.text_pool(text_seq, text_mask)
        audio_vec, _ = self.audio_pool(audio_seq, audio_mask)
        gate = torch.sigmoid(self.gate(torch.cat([text_vec, audio_vec], dim=1)))
        fused = gate * text_vec + (1.0 - gate) * audio_vec
        logits = self.classifier(fused)

        aux = {}
        if return_aux:
            aux["gate_mean"] = gate.mean()
            aux["gate_std"] = gate.std(unbiased=False)
            if attn_t2a is not None and attn_a2t is not None:
                aux["text_to_audio_attn_entropy"] = _attention_entropy(attn_t2a, text_mask, audio_mask)
                aux["audio_to_text_attn_entropy"] = _attention_entropy(attn_a2t, audio_mask, text_mask)
        return logits, aux


def _build_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(TARGET_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        missing = [TARGET_NAMES[i] for i, c in enumerate(counts) if c == 0]
        raise ValueError(
            "В train-части после split отсутствуют некоторые классы, "
            f"невозможно рассчитать class weights. Missing={missing}"
        )
    n_samples = counts.sum()
    n_classes = len(TARGET_NAMES)
    weights = n_samples / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _load_text_backbone(
    text_model_path: Path,
) -> tuple[AutoModel, AutoTokenizer, int, dict]:
    try:
        checkpoint = torch.load(text_model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(text_model_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError("Некорректный формат checkpoint текстовой модели: ожидается dict")
    if "model_state_dict" not in checkpoint or "model_params" not in checkpoint:
        raise KeyError(
            "Checkpoint текстовой модели должен содержать ключи "
            "'model_state_dict' и 'model_params'"
        )

    model_params = checkpoint["model_params"]
    if "backbone_name" not in model_params:
        raise KeyError("В model_params отсутствует ключ 'backbone_name'")
    backbone_name = model_params["backbone_name"]

    text_backbone = AutoModel.from_pretrained(backbone_name)
    state_dict = checkpoint["model_state_dict"]
    bert_state = {}
    for k, v in state_dict.items():
        if k.startswith("bert."):
            bert_state[k[len("bert.") :]] = v
    if not bert_state:
        raise ValueError("В model_state_dict не найдены ключи с префиксом 'bert.'")
    text_backbone.load_state_dict(bert_state, strict=False)

    tokenizer_dir = Path(str(text_model_path).replace("_model.pt", "_tokenizer"))
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    max_len = int(model_params.get("max_len", 128))
    return text_backbone, tokenizer, max_len, model_params


def _load_audio_warm_start(audio_encoder: AudioSequenceEncoder, warm_start_path: Path) -> None:
    try:
        checkpoint = torch.load(warm_start_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(warm_start_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError("Некорректный формат checkpoint аудио warm start: ожидается dict")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Некорректный state_dict в audio warm start checkpoint")

    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder.") :]] = v
        elif k.startswith("model."):
            encoder_state[k[len("model.") :]] = v
    if not encoder_state:
        print(
            "Audio warm start: подходящие ключи не найдены "
            "(ожидается префикс 'encoder.' или 'model.'); пропускаю загрузку."
        )
        return
    missing, unexpected = audio_encoder.model.load_state_dict(encoder_state, strict=False)
    print(
        "Audio warm start загружен: "
        f"{warm_start_path.name}, missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
    )


def _trainable_param_groups(
    model: CrossModalCoAttentionModel,
    lr_head: float,
    lr_encoder: float,
    weight_decay: float,
) -> list[dict]:
    head_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("audio_encoder.") or name.startswith("text_encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay})
    return groups


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    loss: float,
    aux_means: dict[str, float],
) -> dict:
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "wa": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "uar": recall_macro,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": recall_macro,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "top2_accuracy": float(_top2_accuracy(y_true, probs)),
        "roc_auc_ovr_macro": float(_safe_roc_auc_ovr_macro(y_true, probs)),
    }
    metrics.update(aux_means)
    return metrics


def evaluate_split(model, loader, criterion, device: torch.device, use_cuda: bool, desc: str):
    model.eval()
    running_loss = 0.0
    all_true = []
    all_pred = []
    all_probs = []
    aux_sums = {
        "gate_mean": 0.0,
        "gate_std": 0.0,
        "text_to_audio_attn_entropy": 0.0,
        "audio_to_text_attn_entropy": 0.0,
    }
    aux_count = 0

    with torch.no_grad():
        for waves, valid_lens, input_ids, attention_mask, labels in tqdm(loader, desc=desc, leave=False):
            waves = waves.to(device, non_blocking=use_cuda)
            valid_lens = valid_lens.to(device, non_blocking=use_cuda)
            input_ids = input_ids.to(device, non_blocking=use_cuda)
            attention_mask = attention_mask.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            logits, aux = model(
                waves=waves,
                valid_lens=valid_lens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_aux=True,
            )
            logits = torch.clamp(logits, -20, 20)
            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

            batch_size = labels.size(0)
            aux_count += batch_size
            for k in aux_sums.keys():
                if k in aux:
                    aux_sums[k] += float(aux[k].detach().cpu().item()) * batch_size

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    mean_loss = running_loss / len(loader.dataset)
    aux_means = {k: float(v / max(1, aux_count)) for k, v in aux_sums.items()}
    metrics = compute_metrics(y_true, y_pred, probs, mean_loss, aux_means)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES))))
    return metrics, report_text, conf, y_true, y_pred


def print_eval(title: str, metrics: dict, report_text: str, conf: np.ndarray) -> dict:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(conf)

    return {
        "metrics": metrics,
        "classification_report_text": report_text,
        "classification_report": {},
        "confusion_matrix": conf.tolist(),
    }


def _classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )


def _save_artifacts(
    results_dir: Path,
    args,
    model: CrossModalCoAttentionModel,
    best_epoch: int,
    best_val_f1: float,
    history: list[dict],
    train_export: dict,
    val_export: dict,
    test_export: dict,
    text_model_params: dict,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"co_attention_baseline_model_{stamp}.pt"
    report_path = results_dir / f"co_attention_baseline_results_{stamp}.json"
    dataset_name = Path(args.train_lmdb).stem
    full_model_name = f"{MODEL_NAME}_{dataset_name}"
    weights_path = MODELS_DIR / f"{full_model_name}_model.pt"
    weights_backup_path = MODELS_DIR / f"{full_model_name}_model_{stamp}.pt"
    weights_report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"

    model_payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "CrossModalCoAttentionModel",
        "state_dict": _state_dict_to_cpu(model),
        "model_params": {
            "audio_pretrained_name": args.audio_pretrained_name,
            "d_model": int(args.d_model),
            "num_heads": int(args.num_heads),
            "num_coattn_blocks": int(args.num_coattn_blocks),
            "dropout": float(args.dropout),
            "ffn_mult": int(args.ffn_mult),
            "n_classes": len(TARGET_NAMES),
            "text_model_path": str(args.text_model_path),
            "text_backbone_name": text_model_params["backbone_name"],
        },
    }
    torch.save(model_payload, model_path)

    torch.save(model.state_dict(), weights_path)
    torch.save(model.state_dict(), weights_backup_path)

    args_payload = {
        "train_lmdb": str(args.train_lmdb),
        "test_lmdb": str(args.test_lmdb),
        "text_model_path": str(args.text_model_path),
        "audio_pretrained_name": str(args.audio_pretrained_name),
        "audio_warm_start_path": (
            str(args.audio_warm_start_path) if args.audio_warm_start_path is not None else None
        ),
        "batch_size": int(args.batch_size),
        "val_size": float(args.val_size),
        "epochs": int(args.epochs),
        "stage1_epochs": int(args.stage1_epochs),
        "lr_head": float(args.lr_head),
        "lr_encoder": float(args.lr_encoder),
        "weight_decay": float(args.weight_decay),
        "d_model": int(args.d_model),
        "num_heads": int(args.num_heads),
        "num_coattn_blocks": int(args.num_coattn_blocks),
        "dropout": float(args.dropout),
        "ffn_mult": int(args.ffn_mult),
        "max_len": int(args.max_len) if args.max_len is not None else None,
        "min_crop_sec": float(args.min_crop_sec),
        "max_crop_sec": float(args.max_crop_sec),
        "use_class_weights": bool(args.use_class_weights),
        "seed": int(args.seed),
        "device": str(args.device),
        "unfreeze_audio_layers": int(args.unfreeze_audio_layers),
        "unfreeze_text_layers": int(args.unfreeze_text_layers),
        "grad_clip_norm": float(args.grad_clip_norm),
        "warmup_ratio": float(args.warmup_ratio),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
    }
    training_params = dict(args_payload)
    training_params.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_f1_macro": float(best_val_f1),
            "text_backbone_name": text_model_params["backbone_name"],
        }
    )
    weights_report_lines = [
        f"model_name: {MODEL_NAME}",
        f"dataset_name: {dataset_name}",
        f"saved_at: {stamp}",
        "",
        "training_params:",
        json.dumps(training_params, ensure_ascii=False, indent=2),
        "",
        "test_metrics:",
        json.dumps(test_export.get("metrics", {}), ensure_ascii=False, indent=2),
        "",
    ]
    weights_report_path.write_text("\n".join(weights_report_lines), encoding="utf-8")

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_val_f1),
        "weights_path": str(weights_path),
        "weights_backup_path": str(weights_backup_path),
        "weights_report_path": str(weights_report_path),
        "saved_model_path": str(model_path),
        "history": history,
        "args": args_payload,
        "train": train_export,
        "val": val_export,
        "test": test_export,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, report_path


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _save_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_state: dict[str, torch.Tensor] | None,
    best_val_f1: float,
    best_epoch: int,
    no_improve_epochs: int,
    history: list[dict],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "model_state_dict": _state_dict_to_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_state_dict": best_state,
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "no_improve_epochs": int(no_improve_epochs),
        "history": history,
    }
    torch.save(payload, checkpoint_path)


def _load_resume_checkpoint(checkpoint_path: Path) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Некорректный формат resume checkpoint: ожидается dict.")
    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(f"Resume checkpoint неполный, отсутствуют ключи: {missing}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Co-Attention baseline: wav2vec2 (audio sequence) + RuBERT (token sequence) + "
            "bidirectional cross-modal attention + gated fusion."
        )
    )
    parser.add_argument("--train-lmdb", type=Path, default=DEFAULT_TRAIN_LMDB)
    parser.add_argument("--test-lmdb", type=Path, default=DEFAULT_TEST_LMDB)
    parser.add_argument("--text-model-path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument("--audio-pretrained-name", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--audio-warm-start-path", type=Path, default=DEFAULT_AUDIO_WARM_START_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--resume-checkpoint-path",
        "--resume-epoch2-path",
        dest="resume_checkpoint_path",
        type=Path,
        default=None,
        help="Путь к resume-чекпоинту для продолжения с последней завершенной эпохи.",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Размер батча для val/test. По умолчанию равен --batch-size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Количество worker-процессов DataLoader.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Сколько батчей заранее загружает каждый worker (актуально при num-workers > 0).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Держать worker-процессы DataLoader между эпохами.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=1e-5)
    parser.add_argument("--lr-encoder", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Остановить обучение, если val_f1_macro не улучшается N эпох подряд (>=3).",
    )

    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-coattn-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--unfreeze-audio-layers", type=int, default=4)
    parser.add_argument("--unfreeze-text-layers", type=int, default=4)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Включить gradient checkpointing в wav2vec2.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Отключить gradient checkpointing.",
    )
    parser.add_argument(
        "--stage2-micro-batch-size",
        type=int,
        default=1,
        help="Размер микробатча в Stage 2 (после разморозки энкодеров).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Отключить mixed precision (AMP) на CUDA.",
    )

    parser.add_argument("--max-len", type=int, default=None, help="Override max_len для токенизации.")
    parser.add_argument("--min-crop-sec", type=float, default=4.0)
    parser.add_argument("--max-crop-sec", type=float, default=6.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto")
    parser.set_defaults(gradient_checkpointing=True)
    add_config_arg(parser)
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config)

    if not args.train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {args.train_lmdb}")
    if not args.test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {args.test_lmdb}")
    if not args.text_model_path.exists():
        raise FileNotFoundError(f"Текстовая модель не найдена: {args.text_model_path}")
    if args.audio_warm_start_path is not None and not args.audio_warm_start_path.exists():
        print(f"Audio warm start не найден и будет пропущен: {args.audio_warm_start_path}")
        args.audio_warm_start_path = None
    if not (0.0 < args.val_size < 1.0):
        raise ValueError(f"val-size должен быть в (0,1), получено: {args.val_size}")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size должен быть > 0, получено: {args.batch_size}")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError(f"eval-batch-size должен быть > 0, получено: {args.eval_batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num-workers должен быть >= 0, получено: {args.num_workers}")
    if args.prefetch_factor <= 0:
        raise ValueError(f"prefetch-factor должен быть > 0, получено: {args.prefetch_factor}")
    if args.epochs <= 0:
        raise ValueError(f"epochs должен быть > 0, получено: {args.epochs}")
    if args.stage1_epochs < 0 or args.stage1_epochs > args.epochs:
        raise ValueError(
            f"stage1-epochs должен быть в [0, epochs], получено: {args.stage1_epochs} при epochs={args.epochs}"
        )
    if args.d_model <= 0:
        raise ValueError(f"d-model должен быть > 0, получено: {args.d_model}")
    if args.num_heads <= 0 or args.d_model % args.num_heads != 0:
        raise ValueError(
            f"num-heads должен быть > 0 и делить d-model без остатка: d_model={args.d_model}, num_heads={args.num_heads}"
        )
    if args.num_coattn_blocks <= 0:
        raise ValueError(f"num-coattn-blocks должен быть > 0, получено: {args.num_coattn_blocks}")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError(f"dropout должен быть в [0,1), получено: {args.dropout}")
    if args.ffn_mult <= 0:
        raise ValueError(f"ffn-mult должен быть > 0, получено: {args.ffn_mult}")
    if args.max_len is not None and args.max_len <= 0:
        raise ValueError(f"max-len должен быть > 0, получено: {args.max_len}")
    if not (0.0 < args.min_crop_sec <= args.max_crop_sec):
        raise ValueError(
            f"Ожидается 0 < min-crop-sec <= max-crop-sec, получено: {args.min_crop_sec}, {args.max_crop_sec}"
        )
    if args.unfreeze_audio_layers < 0:
        raise ValueError(f"unfreeze-audio-layers должен быть >= 0, получено: {args.unfreeze_audio_layers}")
    if args.unfreeze_text_layers < 0:
        raise ValueError(f"unfreeze-text-layers должен быть >= 0, получено: {args.unfreeze_text_layers}")
    if args.grad_clip_norm <= 0:
        raise ValueError(f"grad-clip-norm должен быть > 0, получено: {args.grad_clip_norm}")
    if args.early_stopping_patience < 3:
        raise ValueError(
            "early-stopping-patience должен быть >= 3, "
            f"получено: {args.early_stopping_patience}"
        )
    if args.stage2_micro_batch_size <= 0:
        raise ValueError(
            f"stage2-micro-batch-size должен быть > 0, получено: {args.stage2_micro_batch_size}"
        )
    if args.stage2_micro_batch_size > args.batch_size:
        raise ValueError(
            "stage2-micro-batch-size не должен превышать batch-size: "
            f"{args.stage2_micro_batch_size} > {args.batch_size}"
        )
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError(f"warmup-ratio должен быть в [0,1), получено: {args.warmup_ratio}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = use_cuda and not args.no_amp
    autocast_device_type = "cuda" if use_cuda else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    resume_checkpoint_path = (
        args.resume_checkpoint_path
        if args.resume_checkpoint_path is not None
        else args.results_dir / "co_attention_baseline_resume.pt"
    )

    print(f"Train LMDB: {args.train_lmdb}")
    print(f"Test LMDB:  {args.test_lmdb}")
    print(f"Text model: {args.text_model_path}")
    print(f"Audio pretrained: {args.audio_pretrained_name}")
    print(f"Audio warm start: {args.audio_warm_start_path}")
    print(f"Device: {device}")

    text_backbone, tokenizer, text_max_len, text_model_params = _load_text_backbone(args.text_model_path)
    max_len = args.max_len if args.max_len is not None else text_max_len
    audio_encoder = AudioSequenceEncoder(
        pretrained_name=args.audio_pretrained_name,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.audio_warm_start_path is not None:
        _load_audio_warm_start(audio_encoder, args.audio_warm_start_path)
    text_encoder = TextSequenceEncoder(text_backbone)

    train_ds = FusionWaveTextDataset(args.train_lmdb)
    test_ds = FusionWaveTextDataset(args.test_lmdb)

    all_indices = np.arange(len(train_ds), dtype=np.int64)
    tr_idx, val_idx = train_test_split(
        all_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_ds.labels,
    )
    test_indices = np.arange(len(test_ds), dtype=np.int64)

    train_collate = build_collate_fn(
        tokenizer=tokenizer,
        max_len=max_len,
        min_crop_sec=args.min_crop_sec,
        max_crop_sec=args.max_crop_sec,
        is_train=True,
    )
    eval_collate = build_collate_fn(
        tokenizer=tokenizer,
        max_len=max_len,
        min_crop_sec=args.min_crop_sec,
        max_crop_sec=args.max_crop_sec,
        is_train=False,
    )

    train_loader_kwargs = {
        "dataset": Subset(train_ds, tr_idx.tolist()),
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        "collate_fn": train_collate,
    }
    eval_loader_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        "collate_fn": eval_collate,
    }
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader_kwargs["persistent_workers"] = args.persistent_workers
        eval_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        eval_loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(dataset=Subset(train_ds, val_idx.tolist()), **eval_loader_kwargs)
    test_loader = DataLoader(dataset=Subset(test_ds, test_indices.tolist()), **eval_loader_kwargs)

    model = CrossModalCoAttentionModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        d_model=args.d_model,
        num_heads=args.num_heads,
        n_blocks=args.num_coattn_blocks,
        dropout=args.dropout,
        ffn_mult=args.ffn_mult,
        n_classes=len(TARGET_NAMES),
    ).to(device)
    model.freeze_backbones()

    class_weights = None
    if args.use_class_weights:
        class_weights = _build_class_weights(train_ds.labels[tr_idx]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    param_groups = _trainable_param_groups(
        model=model,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
    )
    if not param_groups:
        raise RuntimeError("Не найдено trainable параметров для оптимизатора.")
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Размер train (валидные мультимодальные): {len(train_ds)}")
    print(f"Размер train-split: {len(tr_idx)}")
    print(f"Размер val: {len(val_idx)}")
    print(f"Размер test (валидные мультимодальные): {len(test_ds)}")
    print(
        f"epochs={args.epochs}, stage1_epochs={args.stage1_epochs}, batch_size={args.batch_size}, "
        f"eval_batch_size={eval_batch_size}, num_workers={args.num_workers}, "
        f"lr_head={args.lr_head}, lr_encoder={args.lr_encoder}, warmup_ratio={args.warmup_ratio}, "
        f"weight_decay={args.weight_decay}, max_len={max_len}, crop_sec=[{args.min_crop_sec},{args.max_crop_sec}]"
    )
    print(
        f"d_model={args.d_model}, num_heads={args.num_heads}, num_coattn_blocks={args.num_coattn_blocks}, "
        f"dropout={args.dropout}, ffn_mult={args.ffn_mult}, "
        f"unfreeze_audio_layers={args.unfreeze_audio_layers}, unfreeze_text_layers={args.unfreeze_text_layers}, "
        f"gradient_checkpointing={args.gradient_checkpointing}, amp={use_amp}, "
        f"stage2_micro_batch_size={args.stage2_micro_batch_size}, "
        f"early_stopping_patience={args.early_stopping_patience}"
    )
    print(f"text_backbone={text_model_params['backbone_name']}")
    print(f"class_weights={'enabled' if args.use_class_weights else 'disabled'}")
    print("epoch | train_loss | train_f1 | val_loss | val_f1 | val_uar | val_top2 | val_gate")

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    history = []
    start_epoch = 1

    if resume_checkpoint_path.exists():
        resume_payload = _load_resume_checkpoint(resume_checkpoint_path)
        resume_epoch = int(resume_payload["epoch"])
        if resume_epoch >= 1:
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            if use_amp and "scaler_state_dict" in resume_payload:
                scaler.load_state_dict(resume_payload["scaler_state_dict"])
            best_state = resume_payload.get("best_state_dict")
            best_val_f1 = float(resume_payload.get("best_val_f1", best_val_f1))
            best_epoch = int(resume_payload.get("best_epoch", best_epoch))
            no_improve_epochs = int(
                resume_payload.get("no_improve_epochs", max(0, resume_epoch - best_epoch))
            )
            history = list(resume_payload.get("history", history))
            start_epoch = max(1, resume_epoch + 1)
            print(
                f"Найден resume checkpoint: {resume_checkpoint_path}. "
                f"Продолжаем обучение с эпохи {start_epoch}."
            )
            if start_epoch > args.stage1_epochs + 1:
                model.unfreeze_last_layers(
                    audio_layers=args.unfreeze_audio_layers,
                    text_layers=args.unfreeze_text_layers,
                )
                if use_cuda:
                    torch.cuda.empty_cache()

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.stage1_epochs + 1:
            model.unfreeze_last_layers(
                audio_layers=args.unfreeze_audio_layers,
                text_layers=args.unfreeze_text_layers,
            )
            if use_cuda:
                torch.cuda.empty_cache()
            print(
                "Stage 2: разморожены последние слои энкодеров: "
                f"audio={args.unfreeze_audio_layers}, text={args.unfreeze_text_layers}"
            )

        model.train()
        running_loss = 0.0
        seen = 0
        train_preds = []
        train_true = []
        skipped_non_finite_batches = 0
        first_non_finite_report = None

        progress = tqdm(train_loader, desc=f"Train {epoch:02d}/{args.epochs}", leave=False)
        for waves, valid_lens, input_ids, attention_mask, labels in progress:
            waves = waves.to(device, non_blocking=use_cuda)
            valid_lens = valid_lens.to(device, non_blocking=use_cuda)
            input_ids = input_ids.to(device, non_blocking=use_cuda)
            attention_mask = attention_mask.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            is_stage2 = epoch > args.stage1_epochs
            micro_batch_size = args.stage2_micro_batch_size if is_stage2 else labels.size(0)
            full_batch_size = labels.size(0)

            batch_logits = []
            batch_loss_sum = 0.0
            batch_has_non_finite = False
            batch_oom = False
            batch_has_backward = False

            for start in range(0, full_batch_size, micro_batch_size):
                end = min(start + micro_batch_size, full_batch_size)
                mb_waves = waves[start:end]
                mb_valid_lens = valid_lens[start:end]
                mb_input_ids = input_ids[start:end]
                mb_attention_mask = attention_mask[start:end]
                mb_labels = labels[start:end]

                try:
                    logits, loss = _forward_logits_loss(
                        model=model,
                        criterion=criterion,
                        waves=mb_waves,
                        valid_lens=mb_valid_lens,
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                        labels=mb_labels,
                        autocast_device_type=autocast_device_type,
                        use_amp=use_amp,
                    )
                except torch.OutOfMemoryError:
                    batch_oom = True
                    break

                if not torch.isfinite(loss).item():
                    if use_amp:
                        logits_fp32, loss_fp32 = _forward_logits_loss(
                            model=model,
                            criterion=criterion,
                            waves=mb_waves,
                            valid_lens=mb_valid_lens,
                            input_ids=mb_input_ids,
                            attention_mask=mb_attention_mask,
                            labels=mb_labels,
                            autocast_device_type=autocast_device_type,
                            use_amp=False,
                        )
                        if torch.isfinite(loss_fp32).item():
                            if batch_has_backward:
                                print(
                                    "Обнаружен NaN/Inf loss с AMP после частичного backward. "
                                    "Отключаю AMP и пропускаю этот batch."
                                )
                                use_amp = False
                                scaler = torch.cuda.amp.GradScaler(enabled=False)
                                batch_has_non_finite = True
                                break
                            print(
                                "Обнаружен NaN/Inf loss с AMP. "
                                "Пробую продолжить без mixed precision."
                            )
                            use_amp = False
                            scaler = torch.cuda.amp.GradScaler(enabled=False)
                            logits, loss = logits_fp32, loss_fp32
                        else:
                            if first_non_finite_report is None:
                                first_non_finite_report = _build_non_finite_report(
                                    "NaN/Inf loss в train batch (даже без AMP):",
                                    waves=mb_waves,
                                    valid_lens=mb_valid_lens,
                                    input_ids=mb_input_ids,
                                    attention_mask=mb_attention_mask,
                                    labels=mb_labels,
                                    logits=logits_fp32,
                                )
                            batch_has_non_finite = True
                            break
                    else:
                        if first_non_finite_report is None:
                            first_non_finite_report = _build_non_finite_report(
                                "NaN/Inf loss в train batch:",
                                waves=mb_waves,
                                valid_lens=mb_valid_lens,
                                input_ids=mb_input_ids,
                                attention_mask=mb_attention_mask,
                                labels=mb_labels,
                                logits=logits,
                            )
                        batch_has_non_finite = True
                        break

                loss_weight = mb_labels.size(0) / full_batch_size
                if use_amp:
                    scaler.scale(loss * loss_weight).backward()
                else:
                    (loss * loss_weight).backward()
                batch_has_backward = True

                batch_loss_sum += loss.item() * mb_labels.size(0)
                batch_logits.append(logits.detach())

            if batch_oom:
                optimizer.zero_grad(set_to_none=True)
                if use_cuda:
                    torch.cuda.empty_cache()
                continue
            if batch_has_non_finite:
                optimizer.zero_grad(set_to_none=True)
                skipped_non_finite_batches += 1
                continue

            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad], args.grad_clip_norm
                )
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scheduler_should_step = scaler.get_scale() >= prev_scale
            else:
                nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad], args.grad_clip_norm
                )
                optimizer.step()
                scheduler_should_step = True

            if scheduler_should_step:
                scheduler.step()

            running_loss += batch_loss_sum
            seen += full_batch_size
            preds = torch.argmax(torch.cat(batch_logits, dim=0), dim=1)
            train_preds.append(preds.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())
            progress.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

        if not train_preds:
            detail = f"\n{first_non_finite_report}" if first_non_finite_report else ""
            raise RuntimeError(
                "Во всех train-батчах получены нечисловые loss (NaN/Inf). "
                "Проверьте входные данные/гиперпараметры."
                f"{detail}"
            )
        if skipped_non_finite_batches > 0:
            print(f"Пропущено батчей с NaN/Inf loss: {skipped_non_finite_batches}")

        y_train_pred = np.concatenate(train_preds, axis=0)
        y_train_true = np.concatenate(train_true, axis=0)
        train_loss = running_loss / max(1, seen)
        train_f1 = f1_score(y_train_true, y_train_pred, average="macro", zero_division=0)

        val_metrics, _, _, _, _ = evaluate_split(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_cuda=use_cuda,
            desc="Eval Val",
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_f1_macro": float(train_f1),
                "val_loss": float(val_metrics["loss"]),
                "val_f1_macro": float(val_metrics["f1_macro"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_uar": float(val_metrics["uar"]),
                "val_top2_accuracy": float(val_metrics["top2_accuracy"]),
                "val_gate_mean": float(val_metrics["gate_mean"]),
                "val_gate_std": float(val_metrics["gate_std"]),
                "val_t2a_entropy": float(val_metrics["text_to_audio_attn_entropy"]),
                "val_a2t_entropy": float(val_metrics["audio_to_text_attn_entropy"]),
            }
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = int(epoch)
            best_state = _state_dict_to_cpu(model)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(
            f"{epoch:02d} | {train_loss:.4f} | {train_f1:.4f} | "
            f"{val_metrics['loss']:.4f} | {val_metrics['f1_macro']:.4f} | "
            f"{val_metrics['uar']:.4f} | {val_metrics['top2_accuracy']:.4f} | "
            f"{val_metrics['gate_mean']:.4f}"
        )

        _save_resume_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_state=best_state,
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            no_improve_epochs=no_improve_epochs,
            history=history,
        )
        print(f"Сохранен resume checkpoint после эпохи {epoch}: {resume_checkpoint_path}")

        if no_improve_epochs >= args.early_stopping_patience:
            print(
                "Early stopping: val_f1_macro не улучшается "
                f"{no_improve_epochs} эпох подряд (patience={args.early_stopping_patience})."
            )
            break

    if best_state is None:
        raise RuntimeError("Не удалось сохранить best_state во время обучения.")
    model.load_state_dict(best_state)

    train_metrics, train_report_text, train_cm, y_train_true, y_train_pred = evaluate_split(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        use_cuda=use_cuda,
        desc="Final Train Eval",
    )
    val_metrics, val_report_text, val_cm, y_val_true, y_val_pred = evaluate_split(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_cuda=use_cuda,
        desc="Final Val Eval",
    )
    test_metrics, test_report_text, test_cm, y_test_true, y_test_pred = evaluate_split(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_cuda=use_cuda,
        desc="Final Test Eval",
    )

    train_export = print_eval("CO-ATTENTION @ TRAIN", train_metrics, train_report_text, train_cm)
    val_export = print_eval("CO-ATTENTION @ VAL", val_metrics, val_report_text, val_cm)
    test_export = print_eval("CO-ATTENTION @ TEST", test_metrics, test_report_text, test_cm)

    train_export["classification_report"] = _classification_report_dict(y_train_true, y_train_pred)
    val_export["classification_report"] = _classification_report_dict(y_val_true, y_val_pred)
    test_export["classification_report"] = _classification_report_dict(y_test_true, y_test_pred)

    model_path, report_path = _save_artifacts(
        results_dir=args.results_dir,
        args=args,
        model=model,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        history=history,
        train_export=train_export,
        val_export=val_export,
        test_export=test_export,
        text_model_params=text_model_params,
    )
    print(f"\nЛучшая эпоха: {best_epoch}, val_f1_macro={best_val_f1:.6f}")
    print(f"Модель сохранена: {model_path}")
    print(f"Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
