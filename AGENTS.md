# RuIntona Repository - Agent Instructions

## Project Overview

Speech emotion recognition (SER) research project with two datasets:
- **Golos**: Russian speech corpus (~1240 hours)
- **Dusha**: Bi-modal SER dataset (~300k recordings, 4 emotions: angry, sad, neutral, positive)

## Project Structure

```
ruintona/
├── data_processing/    # Raw data processing pipeline
└── my_experiments/     # Custom experiments (text, audio, multimodal, data analysis)
```

## Critical Configuration

### Data Path

The dataset paths are configured in `ruintona/my_experiments/data.json` (copy from `ruintona/my_experiments/data.json.example`):
```json
{
    "base_path": "/path/to/ruintona/data_processing/dataset",
    "train_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/combine_balanced_train.lmdb",
    "test_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/combine_balanced_test.lmdb"
}
```

**Update `base_path`** to match your local dataset location before running any experiments.

### Environment Setup

```bash
# Poetry virtualenv is in-project (.venv)
poetry install

# Or for custom experiments with additional deps
pip install lmdb gensim transformers
```

## Key Commands

### Data Processing

```bash
# Process raw dataset (requires crowd.tar, podcast.tar in DATASET_PATH)
poetry run python ruintona/data_processing/processing.py -dataset_path /path/to/dataset

# Build balanced JSONL datasets for custom experiments
poetry run python ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/build_balanced_aggregated_jsonl.py
```

### Custom Experiments (ruintona/my_experiments/)

```bash
# Text models
poetry run python ruintona/my_experiments/text_models/baseline/TF-IDF_LogReg.py --mode train
poetry run python ruintona/my_experiments/text_models/BiLSTM/BiLSTM.py --mode train

# Audio models
poetry run python ruintona/my_experiments/audio_models/baseline/logistic_regression.py --mode train
poetry run python ruintona/my_experiments/audio_models/CNN/CNN.py --mode train

# Multimodal models
poetry run python ruintona/my_experiments/multimodal/late_fusion/Late_Fusion_CNN_BiLSTM_RuBERT.py --mode train
```

## Architecture Notes

### Emotion Labels

Four emotions with consistent mapping across codebase:
```python
EMO2LABEL = {'angry': 0, 'sad': 1, 'neutral': 2, 'positive': 3}
```

### Corpora

Two LMDB corpora are built from `aggregated_dataset` (full rules in `CORPUS.md`):
- `combine_balanced` — Dusha only (balanced Dusha + Golos samples).
- `dusha_resd` — combined corpus: `combine_balanced` + RESD (`Aniemore/resd_annotated` from Hugging Face, mapped `happiness→positive`, `anger→angry`, `sadness→sad`, `neutral→neutral`). **Most experiments (text models, audio baselines LogReg/RandomForest/SVM/openSMILE+XGBoost, multimodal baselines, foundation-model eval) are trained/evaluated on `dusha_resd`.** Exceptions: CNN/CNN-BiLSTM and wav2vec2 use `combine_balanced`/`combine_balanced_small`; multimodal CNN-BiLSTM/HuBERT backbones are trained on `combine_balanced` but evaluated on `dusha_resd_test` (see `model_analise/multimodal_models_analise.ipynb`).

### Data Formats

- **Custom experiments**: LMDB databases (see `ruintona/my_experiments/utils/lmdb_utils.py`)
- **Data analysis**: JSONL manifests + CSV files

### LMDB Structure

LMDB records contain:
- `x`: feature array
- `y` / `label` / `emotion`: emotion label
- `speaker_text` / `text` / `transcript` / `utterance`: text (for text models)

## Common Pitfalls

1. **Dataset path**: Update `data.json` (see `data.json.example`) before training. Missing `data.json` does not break imports — only commands that need a real LMDB path (without CLI flags) fail, with a clear error
2. **CUDA version**: Docker image uses CUDA 10.1 (old). Local setup may need different CUDA
3. **Reproducibility**: Training scripts fix random seeds by default (remove seed fixing for speed)
4. **Filename typo**: `logistic_regression.py` — legacy name was `logictic_regressoin.py`, now fixed

## Testing

Smoke tests for the model scripts run in `--mode smoke` on tiny synthetic LMDBs (see `ruintona/my_experiments/tests/README.md`):

```bash
poetry install --with dev --with ml   # pytest/ruff + torch models
poetry run pytest ruintona/my_experiments/tests/ -v
```

Notes:
- The package must be installed (`poetry install`) so `ruintona` is importable from anywhere.
- `config_utils` loads `data.json` tolerantly: missing `data.json` does NOT break import — scripts that get explicit `--train-data-path`/`--test-data-path` run fine without it. A clear error is raised only when a path is actually needed (`resolve_data_paths`/`get_dataset_path`).
- Tests requiring `cc.ru.300.bin` (embeddings-logreg, bilstm) or missing optional deps are skipped, not failed.

Additionally, use `--mode auto` or `--mode load` flags to test trained models on real data.

## Installation Verification (Quick Start)

Two levels; level 0 needs no dataset.

**Level 0 — no real data (pipeline check):**
```bash
poetry install --with all        # or: --with dev --with ml (torch models)
poetry run pytest ruintona/my_experiments/tests/ -v   # each model in --mode smoke
```
Works without `data.json`. Heavy downloads to expect: torch (~2GB), RuBERT backbone (~700MB, downloaded by rubert smoke test). FastText `cc.ru.300.bin` (2.3GB) is NOT downloaded automatically — embedding tests will skip.

**Level 1 — real data (`data.json` + real LMDBs):**
- Create `ruintona/my_experiments/data.json` from `data.json.example` with a correct `base_path`.
- Run one representative per family with `--mode auto`: `audio_models/baseline/logistic_regression.py`, `audio_models/CNN/CNN.py` (needs `--with ml`), `text_models/baseline/TF-IDF_LogReg.py`.
- Heavy/optional (skip in a quick check): `openSmile_XGBoost.py` (needs waveform LMDB + `--with audio-extra`), wav2vec/HuBERT (GPU + downloads), multimodal `Late_Fusion_CNN_BiLSTM_RuBERT.py` (needs trained checkpoints, use `--mode load`), `data_processing/processing.py` (needs raw `crowd.tar`/`podcast.tar`).

## CI/CD

- `.github/workflows/ci.yml`: lint (`ruff check ruintona/my_experiments/tests/`) + `pytest ruintona/my_experiments/tests/` on push/PR to `main`/`master`. The test job installs `--with dev --with ml` so torch model smoke tests also run. `data.json` is NOT created in CI — tolerant config loading keeps tests green on a fresh checkout.
- `.github/workflows/semgrep.yml`: Semgrep security scanning.
