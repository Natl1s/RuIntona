# Конфиги экспериментов (`ruintona/configs/`)

> Источники и лицензии сторонних предобученных моделей и датасетов — [`SOURCES.md`](../../SOURCES.md).

JSON-конфиги с гиперпараметрами для всех моделей. Загружаются через `--config PATH`:

```bash
poetry run python ruintona/my_experiments/text_models/baseline/TF-IDF_LogReg.py \
    --mode train --config text/tfidf_logreg.json
```

Путь может быть абсолютным или **относительно `ruintona/configs/`**.

## Формат

```json
{
    "_description": "TF-IDF + Logistic Regression для классификации эмоций по тексту",
    "_script": "ruintona/my_experiments/text_models/baseline/TF-IDF_LogReg.py",
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

- Ключи `_description` / `_script` — служебные, игнорируются при применении конфига.
- Вложенные ключи (`tfidf.*`, `logreg.*`) и kebab-case (`--batch-size` → `batch_size`) конвертируются в snake_case автоматически.
- **Приоритет**: явные CLI-флаги > конфиг > значения по умолчанию.
- **Неизвестные ключи** (без соответствующего argparse-аргумента) выводят предупреждение и игнорируются — так опечатки в конфиге не остаются незамеченными.

## Аудио-модели

| Конфиг | Модель | Ключевые параметры |
|---|---|---|
| `audio/logreg_baseline.json` | Logistic Regression (mel/MFCC) | `C`, `solver`, `class_weight` |
| `audio/svm.json` | SVM (mel) | `kernel`, `C`, `gamma` |
| `audio/random_forest.json` | Random Forest (mel) | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `ccp_alpha`, `class_weight` |
| `audio/random_forest_tuned.json` | Random Forest (mel, tuned) | Гиперпараметры из `model_analise/random_forest_hyperparameter_tuning.ipynb` |
| `audio/opensmile_xgboost.json` | OpenSMILE + XGBoost/LightGBM | `opensmile.feature_set`, `xgboost.n_estimators/learning_rate/max_depth`, `lightgbm.*` |
| `audio/cnn.json` | CNN (MFCC) | `conv_channels`, `classifier_dropout`, `epochs`, `batch_size`, `lr` |
| `audio/cnn_bilstm.json` | CNN + BiLSTM (mel) | `conv_channels`, `classifier_dropout`, `lstm_hidden_size`, `lstm_layers`, `lstm_dropout`, `unidirectional` |
| `audio/cnn_bilstm_tuned.json` | CNN + BiLSTM (mel, tuned) | Гиперпараметры, подобранные Optuna в `cnn_bilstm_hyperparameter_tuning.ipynb` |
| `audio/wav2vec_self_attention.json` | Wav2Vec2 XLS-R 300M + Self-Attention | `model_name`, `max_duration`, `batch_size` |

## Текстовые модели

| Конфиг | Модель | Ключевые параметры |
|---|---|---|
| `text/tfidf_logreg.json` | TF-IDF + LogReg | `tfidf.ngram_range`, `max_features`, `min_df`, `max_df`, `logreg.solver` |
| `text/fasttext_logreg.json` | FastText + LogReg | `embeddings_path`, `logreg.solver` |
| `text/bilstm_fasttext.json` | FastText + BiLSTM | `hidden_size`, `num_layers`, `dropout`, `max_len`, `freeze_embeddings`, `patience` |
| `text/rubert.json` | RuBERT (DeepPavlov) | `backbone_name`, `max_len`, `dropout`, `classifier_hidden_size`, `stage1_epochs`, `grad_accum_steps`, `warmup_ratio`, `early_stopping_patience` |

## Мультимодальные модели

| Конфиг | Модель | Ключевые параметры |
|---|---|---|
| `multimodal/late_fusion_baseline.json` | Late Fusion (sklearn): SVM + TF-IDF LogReg, alpha-перебор | `alpha_step`, `val_size` |
| `multimodal/late_fusion.json` | Late Fusion (PyTorch): CNN-BiLSTM + RuBERT | `alpha_step`, `batch_size`, `max_len`, `val_size`, `seed` |
| `multimodal/early_fusion_baseline.json` | Early Fusion: текст + аудио на ранней стадии | `alpha_step`, `val_size` |
| `multimodal/co_attention_baseline.json` | Co-Attention: Wav2Vec2 + RuBERT, перекрёстное внимание | `max_len`, `batch_size`, `val_size`, `seed` |

## Описание `--config` в скриптах

Аргумент `--config` добавляется в каждый экспериментальный скрипт через `config_utils.add_config_arg()`. Логика применения — `config_utils.apply_config_to_args()`: конфиг перезаписывает атрибуты argparse, у которых значение ещё не задано явно из CLI (аргументы, не переданные пользователем, удаляются из `args` перед применением конфига). Приоритет: **CLI > конфиг > дефолты**.

## См. также

- [`my_experiments/README.md`](../my_experiments/README.md) — обзор экспериментов
- [`my_experiments/HYPERPARAMETERS.md`](../my_experiments/HYPERPARAMETERS.md) — карта гиперпараметров по моделям и методика подбора
- [`my_experiments/model_analise/random_forest_hyperparameter_tuning.ipynb`](../my_experiments/model_analise/random_forest_hyperparameter_tuning.ipynb) — подбор RF (RandomizedSearchCV)
- [`my_experiments/model_analise/cnn_bilstm_hyperparameter_tuning.ipynb`](../my_experiments/model_analise/cnn_bilstm_hyperparameter_tuning.ipynb) — подбор CNN-BiLSTM (Optuna TPE)
