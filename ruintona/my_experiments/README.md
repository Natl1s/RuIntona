# Эксперименты RuIntona (`my_experiments/`)

Эксперименты по распознаванию эмоций: **текстовые**, **аудио** и **мультимодальные** модели, а также инференс, анализ данных и smoke-тесты.

## Оглавление

- [Данные](#данные)
- [Единый CLI](#единый-cli)
- [Конфиги](#конфиги)
- [Гиперпараметры](#гиперпараметры)
- [Чекпоинты](#чекпоинты)
- [Инференс и демо](#инференс-и-демо)
- [Результаты](#результаты)
- [Тесты](#тесты)
- [Индекс документации](#индекс-документации)

## Данные

Эксперименты работают с LMDB-базами из `data_processing/dataset/processed_dataset_090/aggregated_dataset/`. Пути хранятся в `ruintona/my_experiments/data.json` (файл в `.gitignore`, т.к. содержит абсолютные пути):

```json
{
    "base_path": "/path/to/ruintona/data_processing/dataset",
    "train_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/combine_balanced_train.lmdb",
    "test_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/combine_balanced_test.lmdb"
}
```

Шаблон: `ruintona/my_experiments/data.json.example`. Скопируйте и укажите свой путь:

```bash
cp ruintona/my_experiments/data.json.example ruintona/my_experiments/data.json
```

> **Какой корпус использовать.** `data.json.example` указывает на `combine_balanced` (только Dusha). Большинство моделей в этом репозитории (RuBERT, Random Forest tuned, базлайны early/late fusion) обучены на совмещённом корпусе **`dusha_resd` = combine_balanced + RESD (Aniemore, Hugging Face)**. Для воспроизведения их результатов укажите в `data.json`:
>
> ```json
> "train_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/dusha_resd_train.lmdb",
> "test_lmdb": "{base_path}/processed_dataset_090/aggregated_dataset/dusha_resd_test.lmdb"
> ```

### Корпусы

| Корпус | Состав | Train | Test | Используется |
|---|---|---|---|---|
| `combine_balanced` | Только Dusha, 4 эмоции, сбалансирован | 68203 | 6392 | CNN/CNN-BiLSTM, wav2vec2 |
| `combine_balanced_small` | 30% от полного | 20474 | 1863 | быстрые прогоны, warm-start wav2vec2 |
| `dusha_resd` | combine_balanced + RESD | 69119 | 6616 | **большинство моделей** (RuBERT, LogReg, Random Forest default+tuned, SVM, openSMILE+XGBoost, early/late fusion, HuBERT+RuBERT late fusion, foundation-модели) |

Подробнее о корпусах, их составе и правилах сборки из `aggregated_dataset` — [`CORPUS.md`](../../CORPUS.md) (в корне репозитория).

### Формат LMDB

Каждая запись LMDB — pickle-словарь:

| Ключ | Тип | Описание |
|---|---|---|
| `x` | `np.ndarray(float32)`, shape `(1, 64, T)` | mel-спектрограмма (акустические признаки) |
| `y` | `int` | метка эмоции (0–3) |
| `id` | `str` | `hash_id` записи |
| `waveform` | `np.ndarray(float32)` | сырой сигнал, 16 кГц |
| `waveform_sr` | `int` | частота дискретизации (`16000`) |
| `text` | `str` | транскрипт (также принимаются `speaker_text`, `transcript`, `utterance`) |

## Единый CLI

Все экспериментальные скрипты используют общий интерфейс (`utils/cli_utils.py`):

| Флаг | Описание |
|---|---|
| `--mode {train,load,auto,smoke}` | `train` — обучить заново; `load` — загрузить; `auto` — загрузить если есть, иначе обучить (по умолчанию); `smoke` — быстрая проверка на 1–2 эпохах без сохранения |
| `--no-save` | обучить, но не сохранять |
| `--config PATH` | JSON-конфиг гиперпараметров (абсолютный путь или относительно `configs/`) |
| `--train-data-path PATH` | путь к train LMDB (по умолчанию — из `data.json`) |
| `--test-data-path PATH` | путь к test LMDB |
| `--device {cuda,cpu,auto}` | устройство (для PyTorch-моделей) |

Примеры:

```bash
# Обучить аудио-базлайн
poetry run python ruintona/my_experiments/audio_models/baseline/logistic_regression.py --mode train

# Обучить RuBERT с конфигом
poetry run python ruintona/my_experiments/text_models/transformers/RuBERT.py --mode train \
    --config text/rubert.json

# Загрузить существующую модель и оценить
poetry run python ruintona/my_experiments/audio_models/baseline/svm.py --mode load
```

## Конфиги

JSON-конфиги гиперпараметров лежат в [`ruintona/configs/`](../configs/README.md). Конфиг применяется к аргументам со значениями по умолчанию; явные CLI-флаги имеют приоритет.

## Гиперпараметры

Сводка гиперпараметров по всем моделям, методология подбора (RandomizedSearchCV для Random Forest, Optuna TPE для CNN-BiLSTM, перебор α для late fusion) и ссылки на конфиги — в [`HYPERPARAMETERS.md`](./HYPERPARAMETERS.md).

## Чекпоинты

- Модели сохраняются в `ruintona/my_experiments/checkpoints/{text,audio,multimodal}/`.
- Каждая тренировка сохраняется в отдельный файл с временной меткой: `{Model}_{dataset}_model_{YYYYMMDD_HHMMSS}.{pt|pkl}` (например `CNN_BiLSTM_combine_balanced_train_model_20260813_120000.pt`). Временная метка делает имя уникальным — отдельные «бэкапы» больше не создаются.
- При подборе гиперпараметров с переобучением (`--mode tune`) модель сохраняется как `{Model}_tuned_{dataset}_model_{ts}.{pt|pkl}` (например `CNN_BiLSTM_tuned_combine_balanced_train_model_{ts}.pt`).
- sklearn-модели дополнительно сохраняют артефакты с тем же timestamp: `{...}_scaler_{ts}.pkl` / `{...}_vectorizer_{ts}.pkl`.
- Загрузка ищет **последний** чекпоинт с timestamp; если таковых нет — legacy-файл `{Model}_{dataset}_model.{pt|pkl}` (старая конвенция без времени). Т.е. старые модели и модели, восстановленные под старым именем, продолжают загружаться без изменений.
- Метаданные обучения пишутся в `checkpoints/experiments.csv`, текстовые отчёты — `{...}_training_report.txt`.
- **Папка `checkpoints/` в `.gitignore`** — веса публикуются отдельно.

Конвенции разрешения путей чекпоинтов — в `utils/config_utils.py` (`resolve_model_path`, `checkpoints_dir_for`).

## Инференс и демо

Единая точка инференса — `ruintona/my_experiments/inference.py` (модели: `audio`, `text`, `late-fusion`):

```bash
poetry run python ruintona/my_experiments/inference.py --model late-fusion \
    --audio ruintona/DEMO/data/001ce26c07c20eaa0d666b824c6c6924.wav \
    --text "шестьдесят тысяч тенге сколько будет стоить"
```

Демо-ноутбук: [`ruintona/DEMO/`](../DEMO/README.md).

## Результаты

### Готовые решения (pretrained, без тюнинга)

Открытые модели с Hugging Face, оценённые zero-shot на `dusha_resd_test` (6616 сэмплов) — они не обучались нами. Подробное описание и источники — [`model_analise/README.md`](./model_analise/README.md), результаты — `checkpoints/pretrained/*_eval_*.json`.

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Whisper-large-v3 | — (pretrained) | dusha_resd | 0.435 | 0.345 |
| WavLM-BERT fusion | — (pretrained) | dusha_resd | 0.552 | 0.503 |
| HuBERT-large (Dusha-finetuned) | — (pretrained) | dusha_resd | **0.805** | **0.815** |

### Аудио модели

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Logistic Regression | dusha_resd | dusha_resd | 0.474 | 0.465 |
| Random Forest | dusha_resd | dusha_resd | 0.471 | 0.458 |
| Random Forest (tuned) | dusha_resd | dusha_resd | 0.485 | 0.465 |
| SVM (RBF) | dusha_resd | dusha_resd | 0.510 | 0.499 |
| openSMILE+XGBoost | dusha_resd | dusha_resd | 0.600 | 0.590 |
| CNN | combine_balanced | dusha_resd | 0.571 | 0.564 |
| CNN-BiLSTM | combine_balanced | dusha_resd | **0.740** | **0.732** |
| Wav2Vec2 XLS-R 300M + Self-Attention | combine_balanced_small (warm-start) | dusha_resd | 0.644 | 0.648 |

> Примечание: Wav2Vec2 XLS-R 300M + Self-Attention обучен ограниченно из-за ограниченности вычислительных ресурсов: warm-start с предобученного XLS-R 300M, разморожены только 4 последних слоя, обучение на уменьшенной подвыборке `combine_balanced_small` (fp16 + gradient checkpointing), оценка на CPU (~98 мин). Поэтому его метрики ниже потенциально достижимых при полном обучении.

### Текстовые модели

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| TF-IDF + LogReg | combine_balanced | dusha_resd | 0.540 | 0.556 |
| Embeddings (FastText) + LogReg | combine_balanced | dusha_resd | 0.531 | 0.541 |
| BiLSTM | combine_balanced | dusha_resd | 0.560 | 0.580 |
| RuBERT | dusha_resd | dusha_resd | **0.586** | **0.601** |

### Мультимодальные модели

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | dusha_resd | dusha_resd | 0.795 | 0.795 |
| Late-fusion CNN-BiLSTM + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | 0.790 | 0.786 |
| Late-fusion HuBERT + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | **0.822** | **0.830** |
| Late-fusion базлайн SVM + TF-IDF LogReg (α=0.35) | combine_balanced | dusha_resd | 0.621 | 0.629 |

## Тесты

```bash
poetry run pytest ruintona/my_experiments/tests/ -v
```

Smoke-тесты прогоняют каждую модель в режиме `--mode smoke` на синтетических LMDB. Подробнее: [`tests/README.md`](./tests/README.md).

## Индекс документации

| Раздел | README |
|---|---|
| Текстовые модели | [`text_models/README.md`](./text_models/README.md) |
| Аудио-модели | [`audio_models/README.md`](./audio_models/README.md) |
| Мультимодальные модели | [`multimodal/README.md`](./multimodal/README.md) |
| Утилиты | [`utils/README.md`](./utils/README.md) |
| Конфиги | [`../configs/README.md`](../configs/README.md) |
| Гиперпараметры | [`HYPERPARAMETERS.md`](./HYPERPARAMETERS.md) |
| Анализ данных | [`data_analise/README.md`](./data_analise/README.md) |
| Анализ моделей | [`model_analise/README.md`](./model_analise/README.md) |
| Демо | [`../DEMO/README.md`](../DEMO/README.md) |
| Тесты | [`tests/README.md`](./tests/README.md) |
