# Multimodal models (`my_experiments/multimodal/`)

Joint emotion classification from audio and text. Best project result: **late fusion HuBERT + RuBERT (α = 0.5) → Test Acc 0.822, F1-macro 0.830 on `dusha_resd_test`** (the α=0.5 weight was tuned on `combine_balanced`).

## Models

| Script | Approach | Components | Output |
|---|---|---|---|
| `late_fusion/Late_Fusion_CNN_BiLSTM_RuBERT.py` | **Late fusion** — weighted sum of probabilities | CNN-BiLSTM (audio) + RuBERT (text) | search over weight α on validation |
| `late_fusion/Late_Fusion_Baseline_SVM_TF_IDF.py` | Late fusion (baseline) | SVM (audio) + TF-IDF LogReg (text) | search over weight α |
| `early_fusion/Early_Fusion_Baseline.py` | **Early fusion** — feature concatenation + LSTM | CNN-BiLSTM (audio) + RuBERT (text) → projection → LSTM → Linear | single network |
| `co-attention/Co_Attention_Baseline.py` | Co-attention | Wav2Vec2 XLS-R 300M (audio) + RuBERT (text) | cross-attention |

## Quick start

```bash
# Late fusion CNN-BiLSTM + RuBERT (needs CNN-BiLSTM and RuBERT checkpoints)
poetry run python dusha/my_experiments/multimodal/late_fusion/Late_Fusion_CNN_BiLSTM_RuBERT.py --mode auto \
    --audio-model-path checkpoints/audio/CNN_BiLSTM_combine_balanced_train_model.pt \
    --text-model-path checkpoints/text/RuBERT_dusha_resd_train_model.pt

# Late fusion baseline (SVM + TF-IDF LogReg)
poetry run python dusha/my_experiments/multimodal/late_fusion/Late_Fusion_Baseline_SVM_TF_IDF.py --mode auto

# Early fusion baseline
poetry run python dusha/my_experiments/multimodal/early_fusion/Early_Fusion_Baseline.py --mode train

# Co-attention (needs large GPU memory, batch-size 4)
poetry run python dusha/my_experiments/multimodal/co-attention/Co_Attention_Baseline.py --mode train
```

Results are saved into `results/*.json` of each subfolder.

## Key flags

- Common: `--mode {train,load,auto,smoke}`, `--audio-model-path`, `--text-model-path`, `--results-dir`, `--batch-size`, `--val-size`, `--seed`, `--device`.
- Late fusion (weight α): `--alpha-step` (search step, default 0.05); `Late_Fusion_CNN_BiLSTM_RuBERT.py` also `--audio-model-path`, `--text-model-path`.
- `Late_Fusion_Baseline_SVM_TF_IDF.py`: `--audio-scaler-path`, `--text-vectorizer-path`.
- `Early_Fusion_Baseline.py`: `--epochs`, `--lr`, `--weight-decay`, `--projection-dim`, `--dropout`, `--max-len`, `--audio-lstm-hidden-size`, `--audio-lstm-layers`, `--audio-unidirectional`, `--grad-clip-norm`.
- `Co_Attention_Baseline.py`: `--audio-pretrained-name`, `--audio-warm-start-path`, `--text-max-len`, `--audio-max-length`.

## Results

| Model | Dataset | Test Acc | F1-macro |
|---|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | dusha_resd | 0.795 | 0.795 |
| Late fusion CNN-BiLSTM + RuBERT (α=0.5) | dusha_resd | 0.790 | 0.786 |
| **Late fusion HuBERT + RuBERT (α=0.5)** | **dusha_resd** | **0.822** | **0.830** |
| Late fusion baseline (SVM + TF-IDF LogReg, α=0.35) | dusha_resd | 0.621 | 0.629 |

All 4 multimodal models are evaluated on `dusha_resd_test` (6616 samples) in the `multimodal_models_analise.ipynb` notebook, so they are directly comparable. The audio/text backbones of CNN-BiLSTM and HuBERT+RuBERT (CNN-BiLSTM, HuBERT, RuBERT) and the α=0.5 weights were trained/tuned on `combine_balanced`. HuBERT is a pretrained foundation model (description and source — [`model_analise/README.md`](../model_analise/README.md)). Corpus composition and building rules — [`CORPUS.md`](../../../CORPUS.md). Full reports — in `results/` (val/test, confusion matrices, per-epoch history).
