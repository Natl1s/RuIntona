# RuIntona Experiments (`my_experiments/`)

Speech emotion recognition experiments: **text**, **audio** and **multimodal** models, plus inference, data analysis and smoke tests.

> Sources and licenses of third-party pretrained models and datasets — [`SOURCES.en.md`](../../SOURCES.en.md).

## Table of contents

- [Data](#data)
- [Common CLI](#common-cli)
- [Configs](#configs)
- [Hyperparameters](#hyperparameters)
- [Checkpoints](#checkpoints)
- [Inference and demo](#inference-and-demo)
- [Results](#results)
- [Tests](#tests)
- [Documentation index](#documentation-index)

## Data

Experiments use LMDB databases from `data_processing/dataset/processed_dataset_090/aggregated_dataset/`. Data paths are stored in `ruintona/my_experiments/data.json` (gitignored, since it contains absolute paths):

```json
{
    "base_path": "/path/to/ruintona/data_processing/dataset",
    "train_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/combine_balanced_train.lmdb",
    "test_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/combine_balanced_test.lmdb"
}
```

Template: `ruintona/my_experiments/data.json.example`. Copy it and set your own path:

```bash
cp ruintona/my_experiments/data.json.example ruintona/my_experiments/data.json
```

> **Which corpus to use.** `data.json.example` points to `combine_balanced` (Dusha only). Most models in this repository (RuBERT, tuned Random Forest, early/late-fusion baselines) are trained on the combined corpus **`dusha_resd` = combine_balanced + RESD (Aniemore, Hugging Face)**. To reproduce their results, point `data.json` to:
>
> ```json
> "train_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/dusha_resd_train.lmdb",
> "test_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/dusha_resd_test.lmdb"
> ```

### Corpora

| Corpus | Composition | Train | Test | Used by |
|---|---|---|---|---|
| `combine_balanced` | Dusha only, 4 emotions, balanced | 68203 | 6392 | CNN/CNN-BiLSTM, wav2vec2 |
| `combine_balanced_small` | 30% of the full set | 20474 | 1863 | fast runs, wav2vec2 warm-start |
| `dusha_resd` | combine_balanced + RESD | 69119 | 6616 | **most models** (RuBERT, LogReg, Random Forest default+tuned, SVM, openSMILE+XGBoost, early/late fusion, HuBERT+RuBERT late fusion, foundation models) |

More on the corpora, their composition and building rules from `aggregated_dataset` — [`CORPUS.en.md`](../../CORPUS.en.md) (in the repository root).

### LMDB record format

Each LMDB record is a pickled dict:

| Key | Type | Description |
|---|---|---|
| `x` | `np.ndarray(float32)`, shape `(1, 64, T)` | mel-spectrogram (acoustic features) |
| `y` | `int` | emotion label (0–3) |
| `id` | `str` | `hash_id` of the record |
| `waveform` | `np.ndarray(float32)` | raw waveform, 16 kHz |
| `waveform_sr` | `int` | sample rate (`16000`) |
| `text` | `str` | transcript (also `speaker_text`, `transcript`, `utterance` are accepted) |

## Common CLI

All experiment scripts share a common interface (`utils/cli_utils.py`):

| Flag | Description |
|---|---|
| `--mode {train,load,auto,smoke}` | `train` — retrain; `load` — load existing; `auto` — load if exists, else train (default); `smoke` — quick check on 1–2 epochs without saving |
| `--no-save` | train without saving |
| `--config PATH` | JSON hyperparameter config (absolute path or relative to `configs/`) |
| `--train-data-path PATH` | path to train LMDB (default: from `data.json`) |
| `--test-data-path PATH` | path to test LMDB |
| `--device {cuda,cpu,auto}` | device (for PyTorch models) |

Examples:

```bash
# Train an audio baseline
poetry run python ruintona/my_experiments/audio_models/baseline/logistic_regression.py --mode train

# Train RuBERT with a config
poetry run python ruintona/my_experiments/text_models/transformers/RuBERT.py --mode train \
    --config text/rubert.json

# Load an existing model and evaluate
poetry run python ruintona/my_experiments/audio_models/baseline/svm.py --mode load
```

## Configs

JSON hyperparameter configs live in [`ruintona/configs/`](../configs/README.en.md). A config is applied to arguments that still hold their defaults; explicit CLI flags take precedence.

## Hyperparameters

A summary of hyperparameters for all models, the tuning methodology (RandomizedSearchCV for Random Forest, Optuna TPE for CNN-BiLSTM, α sweep for late fusion) and references to configs — in [`HYPERPARAMETERS.md`](./HYPERPARAMETERS.md).

## Checkpoints

- Models are saved into `ruintona/my_experiments/checkpoints/{text,audio,multimodal}/`.
- Every training run is saved as a separate file with a timestamp: `{Model}_{dataset}_model_{YYYYMMDD_HHMMSS}.{pt|pkl}` (e.g. `CNN_BiLSTM_combine_balanced_train_model_20260813_120000.pt`). The timestamp makes the name unique — separate "backups" are no longer created.
- When hyperparameter tuning retrains the model (`--mode tune`), it is saved as `{Model}_tuned_{dataset}_model_{ts}.{pt|pkl}` (e.g. `CNN_BiLSTM_tuned_combine_balanced_train_model_{ts}.pt`).
- sklearn models additionally save artifacts with the same timestamp: `{...}_scaler_{ts}.pkl` / `{...}_vectorizer_{ts}.pkl`.
- Loading looks for the **latest** timestamped checkpoint; if none exist, it falls back to the legacy file `{Model}_{dataset}_model.{pt|pkl}` (the old convention without a timestamp). So old models and models restored under the old name keep loading unchanged.
- Training metadata is written to `checkpoints/experiments.csv`, text reports to `{...}_training_report.txt`.
- **The `checkpoints/` folder is gitignored** — weights are published separately.
- **Models on Hugging Face**: trained weights are published in the collection **RuIntona SER** (<https://huggingface.co/collections/Natlis/ruintona-ser-6a96d7cd6f9979ac35d1b505>):

  | Model | HF repo | License |
  |---|---|---|
  | RuBERT (text) | <https://huggingface.co/Natlis/rubert-emotion-classification-ru> | Apache-2.0 |
  | CNN-BiLSTM (audio) | <https://huggingface.co/Natlis/cnn-bilstm-emotion-classification-ru> | CC BY-SA 4.0 |
  | CNN (audio baseline) | <https://huggingface.co/Natlis/cnn-emotion-classification-ru> | CC BY-SA 4.0 |
  | openSMILE + XGBoost (audio) | <https://huggingface.co/Natlis/opensmile-xgboost-emotion-classification-ru> | CC BY-SA 4.0 |
  | SVM RBF (audio baseline) | <https://huggingface.co/Natlis/svm-emotion-classification-ru> | CC BY-SA 4.0 |
  | Late fusion HuBERT+RuBERT | <https://huggingface.co/Natlis/hubert-rubert-late-fusion-emotion-classification-ru> | CC BY-SA 4.0 |

  Metrics, hyperparameters and loading instructions — in each repo's model card (`README.md`, `LICENSE`, `NOTICE`).

Checkpoint path resolution conventions — in `utils/config_utils.py` (`resolve_model_path`, `checkpoints_dir_for`).

## Inference and demo

Unified inference entry point — `ruintona/my_experiments/inference.py` (models: `audio`, `text`, `late-fusion`):

```bash
poetry run python ruintona/my_experiments/inference.py --model late-fusion \
    --audio ruintona/DEMO/data/001ce26c07c20eaa0d666b824c6c6924.wav \
    --text "шестьдесят тысяч тенге сколько будет стоить"
```

Demo notebook: [`ruintona/DEMO/`](../DEMO/README.en.md).

## Results

### Ready-made solutions (pretrained, no tuning)

Open models from Hugging Face, evaluated zero-shot on `dusha_resd_test` (6616 samples) — they were not trained by us. Detailed description and sources — [`model_analise/README.en.md`](./model_analise/README.en.md), results — `checkpoints/pretrained/*_eval_*.json`.

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Whisper-large-v3 | — (pretrained) | dusha_resd | 0.435 | 0.345 |
| WavLM-BERT fusion | — (pretrained) | dusha_resd | 0.552 | 0.503 |
| HuBERT-large (Dusha-finetuned) | — (pretrained) | dusha_resd | **0.805** | **0.815** |

### Audio models

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Logistic Regression | dusha_resd | dusha_resd | 0.474 | 0.465 |
| Random Forest | dusha_resd | dusha_resd | 0.471 | 0.458 |
| Random Forest (tuned) | dusha_resd | dusha_resd | 0.485 | 0.465 |
| SVM (RBF) | dusha_resd | dusha_resd | 0.510 | 0.499 |
| openSMILE+XGBoost | dusha_resd | dusha_resd | 0.600 | 0.590 |
| CNN | combine_balanced | dusha_resd | 0.571 | 0.564 |
| CNN-BiLSTM | combine_balanced | dusha_resd | **0.740** | **0.732** |
| Wav2Vec2 XLS-R 300M + Self-Attention | combine_balanced_small (warm-start) | dusha_resd | 0.644 | 0.648 |

> Note: Wav2Vec2 XLS-R 300M + Self-Attention was trained under tight compute constraints: warm-start from the pretrained XLS-R 300M, only the last 4 layers unfrozen, training on the reduced `combine_balanced_small` subset (fp16 + gradient checkpointing), evaluation on CPU (~98 min). Hence its metrics are below what full training would likely achieve.

### Text models

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| TF-IDF + LogReg | combine_balanced | dusha_resd | 0.540 | 0.556 |
| Embeddings (FastText) + LogReg | combine_balanced | dusha_resd | 0.531 | 0.541 |
| BiLSTM | combine_balanced | dusha_resd | 0.560 | 0.580 |
| RuBERT | dusha_resd | dusha_resd | **0.586** | **0.601** |

### Multimodal models

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | dusha_resd | dusha_resd | 0.795 | 0.795 |
| Late-fusion CNN-BiLSTM + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | 0.790 | 0.786 |
| Late-fusion HuBERT + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | **0.822** | **0.830** |
| Late-fusion baseline SVM + TF-IDF LogReg (α=0.35) | combine_balanced | dusha_resd | 0.621 | 0.629 |

## Tests

```bash
poetry run pytest ruintona/my_experiments/tests/ -v
```

Smoke tests run every model in `--mode smoke` on synthetic LMDBs. Details: [`tests/README.en.md`](./tests/README.en.md).

## Documentation index

| Section | README |
|---|---|
| Text models | [`text_models/README.en.md`](./text_models/README.en.md) |
| Audio models | [`audio_models/README.en.md`](./audio_models/README.en.md) |
| Multimodal models | [`multimodal/README.en.md`](./multimodal/README.en.md) |
| Utilities | [`utils/README.en.md`](./utils/README.en.md) |
| Configs | [`../configs/README.en.md`](../configs/README.en.md) |
| Hyperparameters | [`HYPERPARAMETERS.md`](./HYPERPARAMETERS.md) |
| Data analysis | [`data_analise/README.en.md`](./data_analise/README.en.md) |
| Model analysis | [`model_analise/README.en.md`](./model_analise/README.en.md) |
| Demo | [`../DEMO/README.en.md`](../DEMO/README.en.md) |
| Tests | [`tests/README.en.md`](./tests/README.en.md) |
