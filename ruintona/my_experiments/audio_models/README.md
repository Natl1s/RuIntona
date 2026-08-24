# Аудио-модели (`my_experiments/audio_models/`)

Классификация эмоций по аудио. Признаки читаются из LMDB (`x` — mel-спектрограмма формы `(1, 64, T)`), сырой сигнал `waveform` — для wav2vec2.

> Источники и лицензии сторонних предобученных моделей и датасетов — [`SOURCES.md`](../../../SOURCES.md).

## Модели

| Папка / скрипт | Модель | Входные признаки | Фреймворк |
|---|---|---|---|
| `baseline/logistic_regression.py` | Logistic Regression | mel mean+std (фиксированный вектор) | sklearn |
| `baseline/svm.py` | SVM (RBF) | mel mean+std | sklearn |
| `baseline/random_forest.py` | Random Forest | mel mean+std | sklearn |
| `baseline/openSmile_XGBoost.py` | OpenSMILE + XGBoost / LightGBM | openSMILE-признаки | sklearn |
| `CNN/CNN.py` | CNN | mel / MFCC `(1, C, T)` | PyTorch |
| `CNN/CNN_BiLSTM.py` | CNN + BiLSTM | mel `(1, 64, T)` | PyTorch |
| `transformers/wav2vec_self_attention.py` | Wav2Vec2 XLS-R 300M + Self-Attention | сырой waveform | PyTorch + HF |

## Быстрый запуск

```bash
# Базлайны (sklearn)
poetry run python ruintona/my_experiments/audio_models/baseline/logistic_regression.py --mode train
poetry run python ruintona/my_experiments/audio_models/baseline/svm.py --mode train
poetry run python ruintona/my_experiments/audio_models/baseline/random_forest.py --mode train

# OpenSMILE + XGBoost (нужны пакеты opensmile, xgboost: poetry install --with audio-extra)
poetry run python ruintona/my_experiments/audio_models/baseline/openSmile_XGBoost.py --mode train

# CNN / CNN-BiLSTM
poetry run python ruintona/my_experiments/audio_models/CNN/CNN.py --mode train --config audio/cnn.json
poetry run python ruintona/my_experiments/audio_models/CNN/CNN_BiLSTM.py --mode train --config audio/cnn_bilstm.json

# Wav2Vec2 XLS-R 300M + Self-Attention
poetry run python ruintona/my_experiments/audio_models/transformers/wav2vec_self_attention.py \
    --mode train --config audio/wav2vec_self_attention.json --device cuda
```

## Ключевые флаги

- Общие: `--mode {train,load,auto,smoke}`, `--config`, `--train-data-path`, `--test-data-path`, `--no-save`, `--device`.
- `CNN.py`: `--conv-channels`, `--classifier-dropout`, `--epochs`, `--batch-size`, `--lr`, `--val-size`, `--early-stopping-patience`.
- `CNN_BiLSTM.py`: `--lstm-hidden-size`, `--lstm-layers`, `--lstm-dropout`, `--unidirectional`, `--epochs`, `--batch-size`, `--lr`.
- `openSmile_XGBoost.py`: `--model-type {auto,xgboost,lightgbm}`, `--n-estimators`, `--max-depth`, `--max-train-samples`, `--no-cache`.
- `wav2vec_self_attention.py`: `--lr-encoder`, `--lr-head`, `--min-crop-sec`, `--max-crop-sec`, `--num-workers`, `--use-amp`.
- `random_forest.py` (борьба с переобучением): `--max-depth`, `--min-samples-leaf`, `--min-samples-split`, `--ccp-alpha`, `--class-weight`, `--oob-score`, а также `--mode tune` с флагами `--max-samples` (подвыборка для поиска), `--search-iterations`, `--cv-folds` (RandomizedSearchCV).

## Переобучение и подбор гиперпараметров (Random Forest)

По умолчанию Random Forest (`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`) запоминает обучающую выборку: `train acc ≈ 1.0` при `test acc ≈ 0.47`. Разрыв `train − test` теперь выводится в отчёте каждой sklearn-модели (блок `ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ` в `sklearn_utils.evaluate_sklearn_classifier`).

Полный процесс — в [`model_analise/random_forest_hyperparameter_tuning.ipynb`](../model_analise/random_forest_hyperparameter_tuning.ipynb):

1. Диагностика: OOB-кривая и разрыв train/val.
2. Пошаговый подбор: `n_estimators` (насыщение OOB), `max_depth`, `min_samples_leaf`, `ccp_alpha`.
3. `RandomizedSearchCV` (cv=3, scoring `f1_macro`) по суженной сетке.
4. Финальное обучение на полном train, оценка на test (test в подборе не участвует).

```bash
# Полный подбор гиперпараметров (на подвыборке 20k, затем обучение на полном train)
poetry run python ruintona/my_experiments/audio_models/baseline/random_forest.py \
    --mode tune --max-samples 20000 --search-iterations 12 --cv-folds 3

# Обучение с отобранными гиперпараметрами
poetry run python ruintona/my_experiments/audio_models/baseline/random_forest.py \
    --mode train --config audio/random_forest_tuned.json
```

Результат (датасет `dusha_resd`, dusha_resd_test): разрыв train/test сократился, тестовая метрика выросла. Состав корпусов (`dusha_resd` / `combine_balanced`) и правила их сборки — [`CORPUS.md`](../../../CORPUS.md).

| Модель | Train acc | Test acc | F1-macro | Разрыв train−test |
|---|---|---|---|---|
| RF default (max_depth=None) | 1.0000 | 0.4714 | 0.4579 | 0.5286 |
| RF tuned (max_depth=15, min_samples_leaf=2, n_estimators=300) | 0.9202 | 0.4846 | 0.4646 | 0.4356 |

## Артефакты

- sklearn-модели сохраняются в `checkpoints/audio/`: `{Модель}_{датасет}_model.pkl` + `{...}_scaler.pkl`, отчёт `{...}_training_report.txt`.
- PyTorch: `{Модель}_{датасет}_model.pt` (+ бэкап с меткой времени).
- Метаданные всех экспериментов: `checkpoints/experiments.csv`.

## Результаты (аудио-модели)

Актуальные чекпоинты и отчёты — в `checkpoints/audio/*_training_report.txt`, `checkpoints/experiments.csv` и ноутбуке [`model_analise/audio_models_analise.ipynb`](../model_analise/audio_models_analise.ipynb). Logistic Regression, Random Forest (default + tuned), SVM и openSMILE+XGBoost **переобучены на совмещённом корпусе `dusha_resd`** (06–08.08); CNN, CNN-BiLSTM и Wav2Vec2 обучены на `combine_balanced` / `combine_balanced_small` и оценены на `dusha_resd_test`. Состав корпусов — [`CORPUS.md`](../../../CORPUS.md).

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Logistic Regression | `dusha_resd` | `dusha_resd` | 0.474 | 0.465 |
| Random Forest | `dusha_resd` | `dusha_resd` | 0.471 | 0.458 |
| Random Forest (tuned) | `dusha_resd` | `dusha_resd` | 0.485 | 0.465 |
| SVM (RBF) | `dusha_resd` | `dusha_resd` | 0.510 | 0.499 |
| openSMILE+XGBoost | `dusha_resd` | `dusha_resd` | 0.600 | 0.590 |
| CNN | `combine_balanced` | `dusha_resd` | 0.571 | 0.564 |
| CNN-BiLSTM | `combine_balanced` | `dusha_resd` | **0.740** | **0.732** |
| Wav2Vec2 XLS-R 300M + Self-Attention | `combine_balanced_small` (warm-start) | `dusha_resd` | 0.644 | 0.648 |

> Примечание: Wav2Vec2 XLS-R 300M + Self-Attention обучен ограниченно из-за ограниченности вычислительных ресурсов: warm-start с предобученного XLS-R 300M, разморожены только 4 последних слоя, обучение на уменьшенной подвыборке `combine_balanced_small` (fp16 + gradient checkpointing), оценка на CPU (~98 мин). Поэтому его метрики ниже потенциально достижимых при полном обучении.

Оценка предобученных фондационных моделей (HuBERT, WavLM-BERT, Whisper) — см. [`model_analise/README.md`](../model_analise/README.md) и `checkpoints/pretrained/*_eval_*.json`.

## Примечание

Старые артефакты в `baseline/models_params/` — устаревшие; актуальные модели сохраняются в `checkpoints/audio/` (папка определяется автоматически по пути скрипта через `config_utils.models_dir_for()`).
