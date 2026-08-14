# Dusha — Speech Emotion Recognition (русская речь)

Исследовательский проект по **распознаванию эмоций в русской речи** (Speech Emotion Recognition, SER) на основе открытого датасета [Dusha](./DUSHA.md). Классификация реплик на 4 эмоции:

| Эмоция | Метка |
|---|---|
| angry | 0 |
| sad | 1 |
| neutral | 2 |
| positive | 3 |

В проекте реализованы полные пайплайны обучения и оценки моделей трёх модальностей: **текст**, **аудио** и **мультимодальная (аудио + текст)** — от обработки сырых данных до инференса и демо.

> **Корпус экспериментов.** Большинство моделей обучено и протестировано на совмещённом корпусе **Dusha (Сбер) + [RESD](./RESD.md) (Aniemore, Hugging Face)** (`dusha_resd`). Описание корпусов и правила их сборки из `data_processing/dataset/processed_dataset_090/aggregated_dataset` — в [CORPUS.md](./CORPUS.md).

## Быстрый старт

### 1. Установка

```bash
poetry install
```

Опциональные группы зависимостей:

```bash
# PyTorch + HuggingFace Transformers (BiLSTM, RuBERT, Wav2Vec2)
poetry install --with ml

# Анализ данных (ноутбуки data_analise/)
poetry install --with analysis

# Аудио-модели: openSMILE + XGBoost
poetry install --with audio-extra

# Все опциональные зависимости сразу
poetry install --with all
```

### 2. Настройка путей к данным

Пути к датасету и train/test LMDB хранятся в `dusha/my_experiments/data.json` (файл в `.gitignore`, так как содержит абсолютные пути).

```bash
cp dusha/my_experiments/data.json.example dusha/my_experiments/data.json
# и отредактируйте "base_path" под ваше расположение датасета
```

### 3. Демо и инференс

```bash
# Мультимодально (аудио + текст, late-fusion)
poetry run python dusha/my_experiments/inference.py --model late-fusion \
    --audio dusha/DEMO/data/001ce26c07c20eaa0d666b824c6c6924.wav \
    --text "шестьдесят тысяч тенге сколько будет стоить"

# Только аудио
poetry run python dusha/my_experiments/inference.py --model audio --audio sample.wav

# Только текст
poetry run python dusha/my_experiments/inference.py --model text --text "я очень рад сегодня"
```

> **Примечание.** Инференс и демо требуют обученных чекпоинтов моделей в `dusha/my_experiments/checkpoints/`. Веса не включаются в репозиторий (см. `.gitignore`): перед демо их нужно получить — обучить модели по инструкциям в [`dusha/my_experiments/README.md`](./dusha/my_experiments/README.md) либо загрузить отдельно публикуемые веса.

Интерактивная версия — ноутбук [`dusha/DEMO/demo.ipynb`](./dusha/DEMO/demo.ipynb).

## Структура репозитория

```
dusha_new/
├── DUSHA.md                   # Описание и атрибуция датасета Dusha (Salute Developers)
├── RESD.md                    # Описание и атрибуция датасета RESD (Aniemore, Hugging Face)
├── CORPUS.md                  # Корпусы данных и правила их сборки (Dusha + RESD)
├── LICENSE                    # Лицензия собственного кода (MIT)
├── license/                   # Лицензия датасета Dusha/Golos (RU/EN)
├── pyproject.toml             # Poetry-проект и зависимости
└── dusha/
    ├── data_processing/       # Пайплайн обработки сырых данных (адаптирован из Golos/Dusha)
    ├── configs/               # JSON-конфиги экспериментов
    ├── DEMO/                  # Демо-ноутбук инференса
    └── my_experiments/        # Эксперименты: модели, утилиты, анализ, тесты
        ├── audio_models/      # Аудио-модели (baseline, CNN, трансформеры)
        ├── text_models/       # Текстовые модели (baseline, BiLSTM, RuBERT)
        ├── multimodal/        # Мультимодальные модели (late/early fusion, co-attention)
        ├── utils/             # Общие утилиты (LMDB, метрики, реестр моделей)
        ├── data_analise/      # Анализ данных (ноутбуки)
        ├── model_analise/     # Анализ моделей (ноутбуки)
        ├── tests/             # Smoke-тесты
        ├── checkpoints/       # Обученные модели (в .gitignore)
        └── inference.py       # Единая точка инференса (audio / text / late-fusion)
```

## Индекс документации

| Раздел | README |
|---|---|
| Проект | [`README.md`](./README.md) |
| Датасет Dusha и атрибуция | [`DUSHA.md`](./DUSHA.md) |
| Датасет RESD и атрибуция | [`RESD.md`](./RESD.md) |
| Корпусы данных и правила сборки | [`CORPUS.md`](./CORPUS.md) |
| Обработка сырых данных | [`dusha/data_processing/README.md`](./dusha/data_processing/README.md) |
| Эксперименты (обзор) | [`dusha/my_experiments/README.md`](./dusha/my_experiments/README.md) |
| Конфиги экспериментов | [`dusha/configs/README.md`](./dusha/configs/README.md) |
| Утилиты | [`dusha/my_experiments/utils/README.md`](./dusha/my_experiments/utils/README.md) |
| Текстовые модели | [`dusha/my_experiments/text_models/README.md`](./dusha/my_experiments/text_models/README.md) |
| Аудио-модели | [`dusha/my_experiments/audio_models/README.md`](./dusha/my_experiments/audio_models/README.md) |
| Мультимодальные модели | [`dusha/my_experiments/multimodal/README.md`](./dusha/my_experiments/multimodal/README.md) |
| Анализ данных | [`dusha/my_experiments/data_analise/README.md`](./dusha/my_experiments/data_analise/README.md) |
| Анализ моделей | [`dusha/my_experiments/model_analise/README.md`](./dusha/my_experiments/model_analise/README.md) |
| Демо | [`dusha/DEMO/README.md`](./dusha/DEMO/README.md) |
| Тесты | [`dusha/my_experiments/tests/README.md`](./dusha/my_experiments/tests/README.md) |

## Результаты экспериментов

### Готовые решения (pretrained, без тюнинга)

Открытые модели с Hugging Face, оценённые zero-shot на `dusha_resd_test` (6616 сэмплов) — они не обучались нами. Подробное описание и источники — [`model_analise/README.md`](dusha/my_experiments/model_analise/README.md), результаты — `checkpoints/pretrained/*_eval_*.json`.

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| Whisper-large-v3 | — (pretrained) | dusha_resd | 0.435 | 0.345 |
| WavLM-BERT fusion | — (pretrained) | dusha_resd | 0.552 | 0.503 |
| HuBERT-large (Dusha-finetuned) | — (pretrained) | dusha_resd | **0.805** | **0.815** |

### Аудио модели

| Модель | Train | Test | Test Acc | F1-macro | Источник |
|---|---|---|---|---|---|
| Logistic Regression | dusha_resd | dusha_resd | 0.474 | 0.465 | `model_analise/audio_models_analise.ipynb` |
| Random Forest | dusha_resd | dusha_resd | 0.471 | 0.458 | `model_analise/audio_models_analise.ipynb` |
| Random Forest (tuned) | dusha_resd | dusha_resd | 0.485 | 0.465 | `model_analise/audio_models_analise.ipynb` |
| SVM (RBF) | dusha_resd | dusha_resd | 0.510 | 0.499 | `model_analise/audio_models_analise.ipynb` |
| openSMILE+XGBoost | dusha_resd | dusha_resd | 0.600 | 0.590 | `model_analise/audio_models_analise.ipynb` |
| CNN | combine_balanced | dusha_resd | 0.571 | 0.564 | `model_analise/audio_models_analise.ipynb` |
| CNN-BiLSTM | combine_balanced | dusha_resd | **0.740** | **0.732** | `model_analise/audio_models_analise.ipynb` |
| Wav2Vec2 XLS-R 300M + Self-Attention | combine_balanced_small (warm-start) | dusha_resd | 0.644 | 0.648 | `model_analise/audio_models_analise.ipynb` |

> Примечание: Wav2Vec2 XLS-R 300M + Self-Attention обучен ограниченно из-за ограниченности вычислительных ресурсов: warm-start с предобученного XLS-R 300M, разморожены только 4 последних слоя, обучение на уменьшенной подвыборке `combine_balanced_small` (fp16 + gradient checkpointing), оценка на CPU (~98 мин). Поэтому его метрики ниже потенциально достижимых при полном обучении.

### Текстовые модели

| Модель | Train | Test | Test Acc | F1-macro | Источник |
|---|---|---|---|---|---|
| TF-IDF + LogReg | combine_balanced | dusha_resd | 0.540 | 0.556 | `model_analise/text_models_analise.ipynb` |
| Embeddings (FastText) + LogReg | combine_balanced | dusha_resd | 0.531 | 0.541 | `model_analise/text_models_analise.ipynb` |
| BiLSTM | combine_balanced | dusha_resd | 0.560 | 0.580 | `model_analise/text_models_analise.ipynb` |
| RuBERT | dusha_resd | dusha_resd | **0.586** | **0.601** | `model_analise/text_models_analise.ipynb` |

### Мультимодальные модели

| Модель | Train | Test | Test Acc | F1-macro | Источник |
|---|---|---|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | dusha_resd | dusha_resd | 0.795 | 0.795 | `model_analise/multimodal_models_analise.ipynb` |
| Late-fusion CNN-BiLSTM + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | 0.790 | 0.786 | `model_analise/multimodal_models_analise.ipynb` |
| Late-fusion HuBERT + RuBERT (α=0.5) | combine_balanced (audio) + dusha_resd (text) | dusha_resd | **0.822** | **0.830** | `model_analise/multimodal_models_analise.ipynb` |
| Late-fusion базлайн SVM + TF-IDF LogReg (α=0.35) | combine_balanced | dusha_resd | 0.621 | 0.629 | `model_analise/multimodal_models_analise.ipynb` |

Соответствие моделей корпусам и состав корпусов — см. [`CORPUS.md`](./CORPUS.md). Примечание: вес α=0.5 подбирался на `combine_balanced`; текст-бэкбоны мультимодальных моделей (RuBERT) и ранний fusion обучены на `dusha_resd`, аудио-бэкбоны (CNN-BiLSTM, HuBERT) — на `combine_balanced`. Текстовые базлайны TF-IDF/Embeddings/BiLSTM оценены на `dusha_resd_test`, хотя их чекпоинты обучены на `combine_balanced`. Оценка всех моделей выполнена на `dusha_resd_test` (ноутбуки `text_models_analise.ipynb`, `multimodal_models_analise.ipynb`).

## Тесты и CI

```bash
poetry run pytest dusha/my_experiments/tests/ -v
poetry run ruff check dusha/my_experiments/tests/
```

Настроен GitHub Actions CI (`.github/workflows/ci.yml`): ruff-линт + smoke-тесты; `.github/workflows/semgrep.yml` — сканирование безопасности Semgrep.

## Датасет и лицензия

- **Датасеты**: Dusha — [`DUSHA.md`](./DUSHA.md); RESD (Aniemore) — [`RESD.md`](./RESD.md); корпусы и правила сборки — [`CORPUS.md`](./CORPUS.md).
- **Лицензия датасета Dusha и кода `data_processing/`**: Dusha/Golos (attribution + share-alike), текст в [`license/`](./license/). Этот код и датасет адаптированы из проекта [Salute Developers — Golos](https://github.com/salute-developers/golos).
- **Датасет RESD**: лицензия MIT, атрибуция в [`RESD.md`](./RESD.md).
- **Собственный код проекта** (`my_experiments/`, `DEMO/`, `configs/`): MIT — [`LICENSE`](./LICENSE).
