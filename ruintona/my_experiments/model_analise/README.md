# Анализ моделей (`my_experiments/model_analise/`)

Jupyter-ноутбуки с анализом результатов моделей + результаты оценки предобученных (foundation) моделей.

## Ноутбуки

| Ноутбук | Что анализирует |
|---|---|
| `audio_models_analise.ipynb` | Метрики, confusion matrices, разбор ошибок аудио-моделей |
| `random_forest_hyperparameter_tuning.ipynb` | Диагностика и борьба с переобучением: OOB-кривая, `max_depth`/`min_samples_leaf`/`ccp_alpha`, RandomizedSearchCV, сравнение baseline vs tuned |
| `text_models_analise.ipynb` | Метрики и ошибки текстовых моделей |
| `multimodal_models_analise.ipynb` | Анализ late/early fusion, влияние веса α и т.д. |
| `pretrained_models_analise.ipynb` | Описание и метрики предобученных (foundation) моделей на `dusha_resd_test`: HuBERT-large, Whisper-large-v3, WavLM-BERT fusion |

## Метрики текстовых моделей

Текстовые модели оценены на `dusha_resd_test` (6616 сэмплов) в ноутбуке `text_models_analise.ipynb`:

| Модель | Train | Test Acc | F1-macro |
|---|---|---|---|
| TF-IDF + LogReg | `combine_balanced` | 0.540 | 0.556 |
| Embeddings (FastText) + LogReg | `combine_balanced` | 0.531 | 0.541 |
| BiLSTM | `combine_balanced` | 0.560 | 0.580 |
| **RuBERT** | **`dusha_resd`** | **0.586** | **0.601** |

## Оценка предобученных моделей

Скрипты оценки и чекпоинты (`checkpoints/pretrained/`) не входят в репозиторий (`.gitignore`), но агрегированные результаты оценки (`*_eval_*.json`) закоммичены. Оценка выполнена на `dusha_resd_test` (6616 сэмплов, аудио-модальность; см. состав корпусов в [`CORPUS.md`](../../../CORPUS.md)). Подробное описание каждой модели и её метрик — в ноутбуке `pretrained_models_analise.ipynb`:

| Модель | Источник | Test Acc | F1-macro |
|---|---|---|---|
| Whisper-large-v3 | `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` | 0.435 | 0.345 |
| WavLM-BERT fusion | `Aniemore/wavlm-bert-fusion-s-emotion-russian-resd` | 0.552 | 0.503 |
| **HuBERT-large (дообучен на Dusha)** | `xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned` | **0.805** | **0.815** |

> Ссылки на источники и лицензии предобученных моделей — [`SOURCES.md`](../../../SOURCES.md).

HuBERT-эмбеддинги далее используются в late fusion (см. [`multimodal/README.md`](../multimodal/README.md)).

## Запуск

```bash
poetry run jupyter notebook ruintona/my_experiments/model_analise/
```

Ноутбуки читают метрики из `results/`-файлов групп моделей и `checkpoints/pretrained/*_eval_*.json`.
