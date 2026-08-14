# Experiment configs (`dusha/configs/`)

JSON hyperparameter configs for all models. Loaded via `--config PATH`:

```bash
poetry run python dusha/my_experiments/text_models/baseline/TF-IDF_LogReg.py \
    --mode train --config text/tfidf_logreg.json
```

The path may be absolute or **relative to `dusha/configs/`**.

## Format

```json
{
    "_description": "TF-IDF + Logistic Regression for text emotion classification",
    "_script": "dusha/my_experiments/text_models/baseline/TF-IDF_LogReg.py",
    "tfidf": {
        "ngram_range": [1, 2],
        "max_features": 10000
    },
    "logreg": {
        "solver": "lbfgs",
        "max_iter": 1000
    }
}
```

- `_description` / `_script` keys are metadata and are ignored when applying the config.
- Nested keys (`tfidf.*`, `logreg.*`) and kebab-case (`--batch-size` → `batch_size`) are converted to snake_case automatically.
- **Precedence**: explicit CLI flags > config > defaults.
- **Unknown keys** (with no matching argparse argument) print a warning and are ignored — so typos in a config do not go unnoticed. Nested groups (dict values) are skipped — scripts apply them themselves (e.g. TF-IDF_LogReg reads `config['tfidf']` directly).

## Audio models

| Config | Model | Key parameters |
|---|---|---|
| `audio/logreg_baseline.json` | Logistic Regression (mel/MFCC) | `C`, `solver`, `class_weight` |
| `audio/svm.json` | SVM (mel) | `kernel`, `C`, `gamma` |
| `audio/random_forest.json` | Random Forest (mel) | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `ccp_alpha`, `class_weight` |
| `audio/random_forest_tuned.json` | Random Forest (mel, tuned) | Hyperparameters from `model_analise/random_forest_hyperparameter_tuning.ipynb` |
| `audio/opensmile_xgboost.json` | OpenSMILE + XGBoost/LightGBM | `opensmile.feature_set`, `xgboost.n_estimators/learning_rate/max_depth`, `lightgbm.*` |
| `audio/cnn.json` | CNN (MFCC) | `conv_channels`, `classifier_dropout`, `epochs`, `batch_size`, `lr` |
| `audio/cnn_bilstm.json` | CNN + BiLSTM (mel) | `conv_channels`, `classifier_dropout`, `lstm_hidden_size`, `lstm_layers`, `lstm_dropout`, `unidirectional` |
| `audio/cnn_bilstm_tuned.json` | CNN + BiLSTM (mel, tuned) | Hyperparameters found by Optuna in `cnn_bilstm_hyperparameter_tuning.ipynb` |
| `audio/wav2vec_self_attention.json` | Wav2Vec2 XLS-R 300M + Self-Attention | `model_name`, `max_duration`, `batch_size` |

## Text models

| Config | Model | Key parameters |
|---|---|---|
| `text/tfidf_logreg.json` | TF-IDF + LogReg | `tfidf.ngram_range`, `max_features`, `min_df`, `max_df`, `logreg.solver` |
| `text/fasttext_logreg.json` | FastText + LogReg | `embeddings_path`, `logreg.solver` |
| `text/bilstm_fasttext.json` | FastText + BiLSTM | `hidden_size`, `num_layers`, `dropout`, `max_len`, `freeze_embeddings`, `patience` |
| `text/rubert.json` | RuBERT (DeepPavlov) | `backbone_name`, `max_len`, `dropout`, `classifier_hidden_size`, `stage1_epochs`, `grad_accum_steps`, `warmup_ratio`, `early_stopping_patience` |

## Multimodal models

| Config | Model | Key parameters |
|---|---|---|
| `multimodal/late_fusion_baseline.json` | Late Fusion (sklearn): SVM + TF-IDF LogReg, alpha sweep | `alpha_step`, `val_size` |
| `multimodal/late_fusion.json` | Late Fusion (PyTorch): CNN-BiLSTM + RuBERT | `alpha_step`, `batch_size`, `max_len`, `val_size`, `seed` |
| `multimodal/early_fusion_baseline.json` | Early Fusion: text + audio at an early stage | `alpha_step`, `val_size` |
| `multimodal/co_attention_baseline.json` | Co-Attention: Wav2Vec2 + RuBERT, cross-attention | `max_len`, `batch_size`, `val_size`, `seed` |

## `--config` in scripts

The `--config` argument is added to every experiment script via `config_utils.add_config_arg()`. Application logic — `config_utils.apply_config_to_args()`: the config overwrites argparse attributes that were not set explicitly via CLI. Precedence: **CLI > config > defaults**.

## See also

- [`my_experiments/README.md`](../my_experiments/README.md) — experiments overview
- [`my_experiments/HYPERPARAMETERS.md`](../my_experiments/HYPERPARAMETERS.md) — per-model hyperparameter map and tuning methodology
- [`my_experiments/model_analise/random_forest_hyperparameter_tuning.ipynb`](../my_experiments/model_analise/random_forest_hyperparameter_tuning.ipynb) — RF tuning (RandomizedSearchCV)
- [`my_experiments/model_analise/cnn_bilstm_hyperparameter_tuning.ipynb`](../my_experiments/model_analise/cnn_bilstm_hyperparameter_tuning.ipynb) — CNN-BiLSTM tuning (Optuna TPE)
