# RuIntona — Speech Emotion Recognition (Russian speech)

Research project on **Speech Emotion Recognition (SER) for Russian speech** built on the open-source [Dusha](./DUSHA.md) dataset. Utterances are classified into 4 emotions:

| Emotion | Label |
|---|---|
| angry | 0 |
| sad | 1 |
| neutral | 2 |
| positive | 3 |

The project contains full training and evaluation pipelines for three modalities: **text**, **audio**, and **multimodal (audio + text)** — from raw data processing to inference and a demo notebook.

> **Experiment corpus.** Most models are trained and evaluated on the combined corpus **Dusha (Sber) + [RESD](./RESD.md) (Aniemore, Hugging Face)** (`dusha_resd`). The corpora and the rules for building them from `data_processing/dataset/processed_dataset_090/aggregated_dataset` are described in [CORPUS.md](./CORPUS.md).

## Quick start

### 1. Installation

```bash
poetry install
```

Optional dependency groups:

```bash
# PyTorch + HuggingFace Transformers (BiLSTM, RuBERT, Wav2Vec2)
poetry install --with ml

# Data analysis (data_analise/ notebooks)
poetry install --with analysis

# Audio models: openSMILE + XGBoost
poetry install --with audio-extra

# All optional dependencies at once
poetry install --with all
```

### 2. Configure data paths

Dataset paths and the train/test LMDB locations are stored in `ruintona/my_experiments/data.json` (gitignored, as it contains absolute paths).

```bash
cp ruintona/my_experiments/data.json.example ruintona/my_experiments/data.json
# then edit "base_path" to point to your dataset location
```

### 3. Demo and inference

```bash
# Multimodal (audio + text, late-fusion)
poetry run python ruintona/my_experiments/inference.py --model late-fusion \
    --audio ruintona/DEMO/data/001ce26c07c20eaa0d666b824c6c6924.wav \
    --text "шестьдесят тысяч тенге сколько будет стоить"

# Audio only
poetry run python ruintona/my_experiments/inference.py --model audio --audio sample.wav

# Text only
poetry run python ruintona/my_experiments/inference.py --model text --text "я очень рад сегодня"
```

> **Note.** Inference and the demo require trained model checkpoints in `ruintona/my_experiments/checkpoints/`. Weights are not included in the repository (see `.gitignore`): before running the demo you need to obtain them — train the models following [`ruintona/my_experiments/README.md`](./ruintona/my_experiments/README.md), or download separately published weights.

Interactive version — the notebook [`ruintona/DEMO/demo.ipynb`](./ruintona/DEMO/demo.ipynb).

## Repository structure

```
dusha_new/
├── DUSHA.md                   # Dusha dataset description and attribution (Salute Developers)
├── RESD.md                    # RESD dataset description and attribution (Aniemore, Hugging Face)
├── CORPUS.md                  # Data corpora and their building rules (Dusha + RESD)
├── LICENSE                    # License of the project's own code (MIT)
├── license/                   # Dusha/Golos dataset license (EN/RU)
├── NOTICE                     # Third-party licenses summary (mixed licensing)
├── pyproject.toml             # Poetry project and dependencies
└── ruintona/
    ├── data_processing/       # Raw data processing pipeline (adapted from Golos/Dusha)
    ├── configs/               # Experiment JSON configs
    ├── DEMO/                  # Demo inference notebook
    └── my_experiments/        # Experiments: models, utilities, analysis, tests
        ├── audio_models/      # Audio models (baseline, CNN, transformers)
        ├── text_models/       # Text models (baseline, BiLSTM, RuBERT)
        ├── multimodal/        # Multimodal models (late/early fusion, co-attention)
        ├── utils/             # Shared utilities (LMDB, metrics, model registry)
        ├── data_analise/      # Data analysis (notebooks)
        ├── model_analise/     # Model analysis (notebooks)
        ├── tests/             # Smoke tests
        ├── checkpoints/       # Trained models (gitignored)
        └── inference.py       # Unified inference entry point (audio / text / late-fusion)
```

## Documentation index

| Section | README |
|---|---|
| Project | [`README.md`](./README.md) |
| Dusha dataset & attribution | [`DUSHA.md`](./DUSHA.md) |
| RESD dataset & attribution | [`RESD.md`](./RESD.md) |
| Data corpora & building rules | [`CORPUS.md`](./CORPUS.md) |
| Raw data processing | [`ruintona/data_processing/README.md`](./ruintona/data_processing/README.md) |
| Experiments (overview) | [`ruintona/my_experiments/README.md`](./ruintona/my_experiments/README.md) |
| Experiment configs | [`ruintona/configs/README.md`](./ruintona/configs/README.md) |
| Utilities | [`ruintona/my_experiments/utils/README.md`](./ruintona/my_experiments/utils/README.md) |
| Text models | [`ruintona/my_experiments/text_models/README.md`](./ruintona/my_experiments/text_models/README.md) |
| Audio models | [`ruintona/my_experiments/audio_models/README.md`](./ruintona/my_experiments/audio_models/README.md) |
| Multimodal models | [`ruintona/my_experiments/multimodal/README.md`](./ruintona/my_experiments/multimodal/README.md) |
| Data analysis | [`ruintona/my_experiments/data_analise/README.md`](./ruintona/my_experiments/data_analise/README.md) |
| Model analysis | [`ruintona/my_experiments/model_analise/README.md`](./ruintona/my_experiments/model_analise/README.md) |
| Demo | [`ruintona/DEMO/README.md`](./ruintona/DEMO/README.md) |
| Tests | [`ruintona/my_experiments/tests/README.md`](./ruintona/my_experiments/tests/README.md) |

## Experiment results

### Ready-made solutions (pretrained, no tuning)

Open models from Hugging Face, evaluated zero-shot on `dusha_resd_test` (6616 samples) — they were not trained by us. Detailed description and sources — [`model_analise/README.md`](ruintona/my_experiments/model_analise/README.md), results — `checkpoints/pretrained/*_eval_*.json`.

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Whisper-large-v3 | — (pretrained) | dusha_resd | 0.435 | 0.345 |
| WavLM-BERT fusion | — (pretrained) | dusha_resd | 0.552 | 0.503 |
| HuBERT-large (Dusha-finetuned) | — (pretrained) | dusha_resd | **0.805** | **0.815** |

### Audio models

| Model | Train | Test | Test Acc | F1-macro | Source |
|---|---|---|---|---|---|
| Logistic Regression | dusha_resd | dusha_resd | 0.474 | 0.465 | `model_analise/audio_models_analise.ipynb` |
| Random Forest | dusha_resd | dusha_resd | 0.471 | 0.458 | `model_analise/audio_models_analise.ipynb` |
| Random Forest (tuned) | dusha_resd | dusha_resd | 0.485 | 0.465 | `model_analise/audio_models_analise.ipynb` |
| SVM (RBF) | dusha_resd | dusha_resd | 0.510 | 0.499 | `model_analise/audio_models_analise.ipynb` |
| openSMILE+XGBoost | dusha_resd | dusha_resd | 0.600 | 0.590 | `model_analise/audio_models_analise.ipynb` |
| CNN | combine_balanced | dusha_resd | 0.571 | 0.564 | `model_analise/audio_models_analise.ipynb` |
| CNN-BiLSTM | combine_balanced | dusha_resd | **0.740** | **0.732** | `model_analise/audio_models_analise.ipynb` |
| Wav2Vec2 XLS-R 300M + Self-Attention | combine_balanced_small (warm-start) | dusha_resd | 0.644 | 0.648 | `model_analise/audio_models_analise.ipynb` |

> Note: Wav2Vec2 XLS-R 300M + Self-Attention was trained under tight compute constraints: warm-start from the pretrained XLS-R 300M, only the last 4 layers unfrozen, training on the reduced `combine_balanced_small` subset (fp16 + gradient checkpointing), evaluation on CPU (~98 min). Hence its metrics are below what full training would likely achieve.

### Text models

| Model | Train | Test | Test Acc | F1-macro | Source |
|---|---|---|---|---|---|
| TF-IDF + LogReg | combine_balanced | dusha_resd | 0.540 | 0.556 | `model_analise/text_models_analise.ipynb` |
| Embeddings (FastText) + LogReg | combine_balanced | dusha_resd | 0.531 | 0.541 | `model_analise/text_models_analise.ipynb` |
| BiLSTM | combine_balanced | dusha_resd | 0.560 | 0.580 | `model_analise/text_models_analise.ipynb` |
| RuBERT | dusha_resd | dusha_resd | **0.586** | **0.601** | `model_analise/text_models_analise.ipynb` |

### Multimodal models

| Model | Train | Test | Test Acc | F1-macro | Source |
|---|---|---|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | dusha_resd | dusha_resd | 0.795 | 0.795 | `model_analise/multimodal_models_analise.ipynb` |
| Late-fusion CNN-BiLSTM + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | 0.790 | 0.786 | `model_analise/multimodal_models_analise.ipynb` |
| Late-fusion HuBERT + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | **0.822** | **0.830** | `model_analise/multimodal_models_analise.ipynb` |
| Late-fusion baseline SVM + TF-IDF LogReg (α=0.35) | combine_balanced | dusha_resd | 0.621 | 0.629 | `model_analise/multimodal_models_analise.ipynb` |

Model-to-corpus mapping and corpus composition — see [`CORPUS.md`](./CORPUS.md). Note: the α=0.5 weight was tuned on `combine_balanced`; the text backbones of the multimodal models (RuBERT) and the early fusion were trained on `dusha_resd`, while the audio backbones (CNN-BiLSTM, HuBERT) were trained on `combine_balanced`. The text baselines TF-IDF/Embeddings/BiLSTM are evaluated on `dusha_resd_test` even though their checkpoints were trained on `combine_balanced`. All models are evaluated on `dusha_resd_test` (notebooks `text_models_analise.ipynb`, `multimodal_models_analise.ipynb`).

## Tests and CI

```bash
poetry run pytest ruintona/my_experiments/tests/ -v
poetry run ruff check ruintona/my_experiments/tests/
```

GitHub Actions CI is configured (`.github/workflows/ci.yml`): ruff lint + smoke tests; `.github/workflows/semgrep.yml` — Semgrep security scanning.

## Dataset and license

- **Datasets**: Dusha — [`DUSHA.md`](./DUSHA.md); RESD (Aniemore) — [`RESD.md`](./RESD.md); corpora and building rules — [`CORPUS.md`](./CORPUS.md).
- **Dusha dataset license and `data_processing/` code**: Dusha/Golos (attribution + share-alike), license text in [`license/`](./license/). This code and the dataset are adapted from the [Salute Developers — Golos](https://github.com/salute-developers/golos) project.
- **RESD dataset**: MIT license, attribution in [`RESD.md`](./RESD.md).
- **Project's own code** (`my_experiments/`, `DEMO/`, `configs/`): MIT — [`LICENSE`](./LICENSE).
- **Summary table of licenses for all project parts** — [`NOTICE`](./NOTICE).
