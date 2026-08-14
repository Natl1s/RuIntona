# Audio models (`my_experiments/audio_models/`)

Emotion classification from audio. Features are read from LMDB (`x` — mel-spectrogram of shape `(1, 64, T)`), raw `waveform` — for wav2vec2.

## Models

| Folder / script | Model | Input features | Framework |
|---|---|---|---|
| `baseline/logistic_regression.py` | Logistic Regression | mel mean+std (fixed vector) | sklearn |
| `baseline/svm.py` | SVM (RBF) | mel mean+std | sklearn |
| `baseline/random_forest.py` | Random Forest | mel mean+std | sklearn |
| `baseline/openSmile_XGBoost.py` | OpenSMILE + XGBoost / LightGBM | openSMILE features | sklearn |
| `CNN/CNN.py` | CNN | mel / MFCC `(1, C, T)` | PyTorch |
| `CNN/CNN_BiLSTM.py` | CNN + BiLSTM | mel `(1, 64, T)` | PyTorch |
| `transformers/wav2vec_self_attention.py` | Wav2Vec2 XLS-R 300M + Self-Attention | raw waveform | PyTorch + HF |

## Quick start

```bash
# Baselines (sklearn)
poetry run python ruintona/my_experiments/audio_models/baseline/logistic_regression.py --mode train
poetry run python ruintona/my_experiments/audio_models/baseline/svm.py --mode train
poetry run python ruintona/my_experiments/audio_models/baseline/random_forest.py --mode train

# OpenSMILE + XGBoost (needs opensmile, xgboost: poetry install --with audio-extra)
poetry run python ruintona/my_experiments/audio_models/baseline/openSmile_XGBoost.py --mode train

# CNN / CNN-BiLSTM
poetry run python ruintona/my_experiments/audio_models/CNN/CNN.py --mode train --config audio/cnn.json
poetry run python ruintona/my_experiments/audio_models/CNN/CNN_BiLSTM.py --mode train --config audio/cnn_bilstm.json

# Wav2Vec2 XLS-R 300M + Self-Attention
poetry run python ruintona/my_experiments/audio_models/transformers/wav2vec_self_attention.py \
    --mode train --config audio/wav2vec_self_attention.json --device cuda
```

## Key flags

- Common: `--mode {train,load,auto,smoke}`, `--config`, `--train-data-path`, `--test-data-path`, `--no-save`, `--device`.
- `CNN.py`: `--conv-channels`, `--classifier-dropout`, `--epochs`, `--batch-size`, `--lr`, `--val-size`, `--early-stopping-patience`.
- `CNN_BiLSTM.py`: `--lstm-hidden-size`, `--lstm-layers`, `--lstm-dropout`, `--unidirectional`, `--epochs`, `--batch-size`, `--lr`.
- `openSmile_XGBoost.py`: `--model-type {auto,xgboost,lightgbm}`, `--n-estimators`, `--max-depth`, `--max-train-samples`, `--no-cache`.
- `wav2vec_self_attention.py`: `--lr-encoder`, `--lr-head`, `--min-crop-sec`, `--max-crop-sec`, `--num-workers`, `--use-amp`.
- `random_forest.py` (fighting overfitting): `--max-depth`, `--min-samples-leaf`, `--min-samples-split`, `--ccp-alpha`, `--class-weight`, `--oob-score`, plus `--mode tune` with `--max-samples` (subsample for search), `--search-iterations`, `--cv-folds` (RandomizedSearchCV).

## Overfitting and hyperparameter tuning (Random Forest)

By default Random Forest (`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`) memorizes the training set: `train acc ≈ 1.0` vs `test acc ≈ 0.47`. The `train − test` gap is now printed in every sklearn model report (block `ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ` in `sklearn_utils.evaluate_sklearn_classifier`).

The full process is in [`model_analise/random_forest_hyperparameter_tuning.ipynb`](../model_analise/random_forest_hyperparameter_tuning.ipynb):

1. Diagnostics: OOB curve and the train/val gap.
2. Step-by-step tuning: `n_estimators` (OOB saturation), `max_depth`, `min_samples_leaf`, `ccp_alpha`.
3. `RandomizedSearchCV` (cv=3, scoring `f1_macro`) on a narrowed grid.
4. Final training on the full train set, evaluation on test (test is not used during tuning).

```bash
# Full hyperparameter search (on a 20k subsample, then train on the full train set)
poetry run python ruintona/my_experiments/audio_models/baseline/random_forest.py \
    --mode tune --max-samples 20000 --search-iterations 12 --cv-folds 3

# Training with the tuned hyperparameters
poetry run python ruintona/my_experiments/audio_models/baseline/random_forest.py \
    --mode train --config audio/random_forest_tuned.json
```

Result (dataset `dusha_resd`): the train/test gap shrinks and the test metric improves. Corpus composition (`dusha_resd` / `combine_balanced`) and building rules — [`CORPUS.md`](../../../CORPUS.md).

| Model | Train acc | Test acc | F1-macro | train−test gap |
|---|---|---|---|---|
| RF default (max_depth=None) | 1.0000 | 0.4714 | 0.4579 | 0.5286 |
| RF tuned (max_depth=15, min_samples_leaf=2, n_estimators=300) | 0.9202 | 0.4846 | 0.4646 | 0.4356 |

## Artifacts

- sklearn models are saved into `checkpoints/audio/`: `{Model}_{dataset}_model.pkl` + `{...}_scaler.pkl`, report `{...}_training_report.txt`.
- PyTorch: `{Model}_{dataset}_model.pt` (+ timestamped backup).
- Metadata of all experiments: `checkpoints/experiments.csv`.

## Results (audio models)

Current checkpoints and reports — in `checkpoints/audio/*_training_report.txt`, `checkpoints/experiments.csv` and the notebook [`model_analise/audio_models_analise.ipynb`](../model_analise/audio_models_analise.ipynb). Logistic Regression, Random Forest (default + tuned), SVM and openSMILE+XGBoost were **retrained on the combined `dusha_resd` corpus** (06–08 Aug); CNN, CNN-BiLSTM and Wav2Vec2 were trained on `combine_balanced` / `combine_balanced_small` and evaluated on `dusha_resd_test`. Corpus composition — [`CORPUS.md`](../../../CORPUS.md).

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Logistic Regression | `dusha_resd` | `dusha_resd` | 0.474 | 0.465 |
| Random Forest | `dusha_resd` | `dusha_resd` | 0.471 | 0.458 |
| Random Forest (tuned) | `dusha_resd` | `dusha_resd` | 0.485 | 0.465 |
| SVM (RBF) | `dusha_resd` | `dusha_resd` | 0.510 | 0.499 |
| openSMILE+XGBoost | `dusha_resd` | `dusha_resd` | 0.600 | 0.590 |
| CNN | `combine_balanced` | `dusha_resd` | 0.571 | 0.564 |
| CNN-BiLSTM | `combine_balanced` | `dusha_resd` | **0.740** | **0.732** |
| Wav2Vec2 XLS-R 300M + Self-Attention | `combine_balanced_small` (warm-start) | `dusha_resd` | 0.644 | 0.648 |

> Note: Wav2Vec2 XLS-R 300M + Self-Attention was trained under tight compute constraints: warm-start from the pretrained XLS-R 300M, only the last 4 layers unfrozen, training on the reduced `combine_balanced_small` subset (fp16 + gradient checkpointing), evaluation on CPU (~98 min). Hence its metrics are below what full training would likely achieve.

Evaluation of pretrained foundation models (HuBERT, WavLM-BERT, Whisper) — see [`model_analise/README.md`](../model_analise/README.md) and `checkpoints/pretrained/*_eval_*.json`.

## Note

Legacy artifacts in `baseline/models_params/` are outdated; current models are saved into `checkpoints/audio/` (the folder is detected automatically from the script path via `config_utils.models_dir_for()`).
