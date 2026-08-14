# Гиперпараметры моделей Dusha

Сводка по гиперпараметрам всех моделей проекта, способам их задания и методологии подбора.

## 1. Как задаются гиперпараметры

Единый подход для всех экспериментальных скриптов:

1. **Значения по умолчанию** — зашиты в argparse, в сигнатуры функций обучения либо в модульные `DEFAULTS`-dict (TF-IDF/Embeddings, CNN).
2. **JSON-конфиг** (`--config PATH`) — переопределяет значения по умолчанию. Файлы лежат в [`dusha/configs/`](../configs/README.md).
3. **CLI-флаги** — имеют наивысший приоритет (`CLI > конфиг > дефолты`).

Конфиг применяется двумя способами:

- **`config_utils.apply_config_to_args()`** — большинство скриптов (BiLSTM, RuBERT, svm, random_forest, openSmile_XGBoost, CNN-BiLSTM, wav2vec, все multimodal). Конфиг применяется только к аргументам, значение которых ещё не задано явно из CLI; kebab-case ключи конвертируются в snake_case; неизвестные ключи выводят предупреждение и игнорируются. Вложенные группы (dict-значения) этим путём не прокидываются.
- **Ручной merge `{**DEFAULTS, **(config)}`** — `TF-IDF_LogReg.py`, `Embeddings_LogReg.py`, `logistic_regression.py`, `CNN.py`. Здесь вложенные группы конфига (`tfidf.*`, `logreg.*`) читаются самим скриптом напрямую.

Вложенные ключи конфига (например, `tfidf.ngram_range`, `opensmile.feature_set`, `xgboost.subsample`) применяют **сами скрипты**:
`TF-IDF_LogReg`/`Embeddings_LogReg` читают `config["tfidf"]`/`config["logreg"]`;
`openSmile_XGBoost` — `config["opensmile"]`/`config["xgboost"]`/`config["lightgbm"]`.

## 2. Что сохраняется в чекпоинтах

| Где | Что |
|---|---|
| `*_training_report.txt` | `training_params` (все гиперпараметры обучения), `test_metrics` |
| `*_model.pt` (PyTorch) | `model_class`, `model_params` (для воссоздания архитектуры), `training_params`, `test_metrics` |
| `checkpoints/experiments.csv` | сводка экспериментов: `val_f1_macro`, `test_f1_macro`, `test_accuracy`, полный JSON `training_params`, git hash |
| `*_optuna_results.json` | результаты Optuna-поиска (лучшие параметры, история) |
| `*_search_results.json` | результаты RandomizedSearchCV для sklearn-моделей |

## 3. Карта гиперпараметров по моделям

### Текстовые модели

| Модель | Ключевые гиперпараметры | Дефолты | Конфиг |
|---|---|---|---|
| TF-IDF + LogReg | `ngram_range`, `max_features`, `min_df`, `max_df`, `sublinear_tf` / `solver`, `max_iter`, `class_weight` | `[1,2]`, `10000` / `solver=lbfgs` | `text/tfidf_logreg.json` |
| Embeddings (FastText) + LogReg | `embeddings_path` / `solver`, `max_iter`, `class_weight` | `solver=lbfgs` | `text/fasttext_logreg.json` |
| BiLSTM | `hidden_size`, `num_layers`, `dropout`, `max_len`, `freeze_embeddings`, `lr`, `lr_embeddings`, `weight_decay`, `patience` | `256`, `2`, `0.3`, `64`, `lr=1e-3` | `text/bilstm_fasttext.json` |
| RuBERT | `backbone_name`, `stage1_epochs`, `batch_size`, `grad_accum_steps`, `max_len`, `dropout`, `classifier_hidden_size`, `lr`, `weight_decay`, `warmup_ratio`, `loss_name`, `label_smoothing`, `use_class_weights`, `early_stopping_patience` | `batch=16`, `grad_accum=8`, `lr=2e-5`, `warmup=0.1` | `text/rubert.json` |

### Аудио-модели

| Модель | Ключевые гиперпараметры | Дефолты | Конфиг |
|---|---|---|---|
| Logistic Regression | `solver`, `max_iter`, `class_weight` | `solver=lbfgs` | `audio/logreg_baseline.json` |
| SVM | `kernel`, `C`, `gamma` | `rbf`, `C=1.0` | `audio/svm.json` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `ccp_alpha`, `class_weight`, `oob_score` | `n_estimators=100` | `audio/random_forest.json` / `random_forest_tuned.json` |
| openSMILE + XGBoost/LightGBM | `feature_set`, `feature_level`, `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha` | `eGeMAPSv02`, `n_estimators=500` | `audio/opensmile_xgboost.json` |
| CNN | `conv_channels`, `classifier_dropout`, `epochs`, `batch_size`, `lr`, `weight_decay`, `val_size`, `early_stopping_patience` | `[16,32,64]`, `dropout=0.2` | `audio/cnn.json` |
| CNN-BiLSTM | `conv_channels`, `classifier_dropout`, `lstm_hidden_size`, `lstm_layers`, `lstm_dropout`, `unidirectional`, `epochs`, `batch_size`, `lr`, `weight_decay` | `[16,32,64]`, `dropout=0.3` | `audio/cnn_bilstm.json` / `cnn_bilstm_tuned.json` |
| Wav2Vec2 XLS-R 300M + Self-Attention | `pretrained_name`, `min_crop_sec`, `max_crop_sec`, `lr_encoder`, `lr_head`, `batch_size`, `epochs` | XLS-R 300M | `audio/wav2vec_self_attention.json` |

### Мультимодальные модели

| Модель | Ключевые гиперпараметры | Конфиг |
|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | `projection_dim`, `dropout`, `audio_lstm_hidden_size`, `lr`, `batch_size`, `val_size`, `max_len`, `seed` | `multimodal/early_fusion_baseline.json` |
| Late fusion (CNN-BiLSTM + RuBERT) | `alpha_step` (шаг перебора веса аудио α на валидации), `max_len`, `batch_size`, `val_size` | `multimodal/late_fusion.json` |
| Late fusion (SVM + TF-IDF LogReg) | `alpha_step`, `batch_size`, `val_size`, `seed` (гиперпараметры компонентов — в их конфигах: `audio/svm.json`, `text/tfidf_logreg.json`) | `multimodal/late_fusion_baseline.json` |
| Co-Attention (Wav2Vec2 + RuBERT) | `d_model`, `num_heads`, `num_coattn_blocks`, `ffn_mult`, `dropout`, `lr_head`, `lr_encoder`, `batch_size`, `val_size`, `seed` | `multimodal/co_attention_baseline.json` |

## 4. Методология подбора

### Random Forest — RandomizedSearchCV + пошаговый подбор

[`model_analise/random_forest_hyperparameter_tuning.ipynb`](./model_analise/random_forest_hyperparameter_tuning.ipynb)

- **Диагностика переобучения**: сравнение train/val f1, OOB-кривая по `n_estimators`, кривые обучения.
- **Пошаговый подбор**: сначала глубина/разбиения, затем `min_samples_leaf` как регуляризатор, затем cost-complexity pruning (`ccp_alpha`).
- Итог — конфиг `audio/random_forest_tuned.json` (best cv f1-macro 0.4636).

### CNN-BiLSTM — Optuna (TPE)

Режим `--mode tune` в `audio_models/CNN/CNN_BiLSTM.py`:

```bash
poetry run python dusha/my_experiments/audio_models/CNN/CNN_BiLSTM.py \
    --mode tune --train-data-path <train.lmdb> --test-data-path <test.lmdb>
```

- Поиск по пространству: `lr` (лог-равномерно), `weight_decay`, `lstm_hidden_size`, `lstm_layers`, `lstm_dropout`, `classifier_dropout`, `batch_size`.
- Каждый триал обучается на стратифицированной подвыборке train (для скорости), валидация — отдельный stratify-сплит.
- Early stopping по `val_f1_macro`, лучший триал переобучается на полном train и оценивается на test.
- Результаты: `*_optuna_results.json`, `*_optuna_trials.csv`; лучшие параметры автоматически пишутся в `audio/cnn_bilstm_tuned.json`.
- Разбор результатов — [`model_analise/cnn_bilstm_hyperparameter_tuning.ipynb`](./model_analise/cnn_bilstm_hyperparameter_tuning.ipynb).

### Late fusion — перебор веса модальности α

Вес аудио-модальности α перебирается в самом скрипте на валидации (шаг `alpha_step` по диапазону `[0, 1]`,
лучший α — по `val_f1_macro`), затем применяется к тесту. Разбор результатов — в `model_analise/multimodal_models_analise.ipynb`.

## 5. Воспроизводимость

- Фиксируются `seed` и (для DL) device; обучение детерминировано.
- Полный набор гиперпараметров каждого эксперимента пишется в `experiments.csv` и `*_training_report.txt`.
- Smoke-тесты (`--mode smoke`) проверяют, что все гиперпараметры корректно прокидываются (см. [`tests/`](./tests/README.md)).
