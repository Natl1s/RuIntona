# Dusha Repository - Agent Instructions

## Project Overview

Speech emotion recognition (SER) research project with two datasets:
- **Golos**: Russian speech corpus (~1240 hours)
- **Dusha**: Bi-modal SER dataset (~300k recordings, 4 emotions: angry, sad, neutral, positive)

## Project Structure

```
dusha/
├── data_processing/    # Raw data processing pipeline
└── my_experiments/     # Custom experiments (text, audio, multimodal, data analysis)
```

## Critical Configuration

### Data Path

The dataset path is configured in `dusha/my_experiments/data.config`:
```python
base_path = Path('/home/natlis/PycharmProjects/dusha_new/dusha/data_processing/dataset')
```

**Update this path** to match your local dataset location before running any experiments.

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
poetry run python dusha/data_processing/processing.py -dataset_path /path/to/dataset

# Build balanced JSONL datasets for custom experiments
poetry run python dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/build_balanced_aggregated_jsonl.py
```

### Custom Experiments (dusha/my_experiments/)

```bash
# Text models
poetry run python dusha/my_experiments/text_models/baseline/TF-IDF_LogReg.py --mode train
poetry run python dusha/my_experiments/text_models/RNN/BiLSTM_Text.py --mode train

# Audio models
poetry run python dusha/my_experiments/audio_models/baseline/logistic_regression.py --mode train
poetry run python dusha/my_experiments/audio_models/CNN/CNN.py --mode train

# Multimodal models
poetry run python dusha/my_experiments/multimodal/late_fusion/Late_Fusion.py --mode train
```

## Architecture Notes

### Emotion Labels

Four emotions with consistent mapping across codebase:
```python
EMO2LABEL = {'angry': 0, 'sad': 1, 'neutral': 2, 'positive': 3}
```

### Data Formats

- **Custom experiments**: LMDB databases (see `dusha/my_experiments/lmdb_utils.py`)
- **Data analysis**: JSONL manifests + CSV files

### LMDB Structure

LMDB records contain:
- `x`: feature array
- `y` / `label` / `emotion`: emotion label
- `speaker_text` / `text` / `transcript` / `utterance`: text (for text models)

## Common Pitfalls

1. **Dataset path**: Must update `data.config` before running experiments
2. **CUDA version**: Docker image uses CUDA 10.1 (old). Local setup may need different CUDA
3. **Reproducibility**: Training scripts fix random seeds by default (remove seed fixing for speed)
4. **Filename typo**: `logistic_regression.py` — legacy name was `logictic_regressoin.py`, now fixed

## Testing

No formal test suite exists. Use `--mode auto` or `--mode load` flags to test trained models.

## CI/CD

Only Semgrep security scanning configured (`.github/workflows/semgrep.yml`). No linting or test workflows.
