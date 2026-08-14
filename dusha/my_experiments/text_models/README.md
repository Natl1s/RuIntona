# Текстовые модели (`my_experiments/text_models/`)

Классификация эмоций по тексту (транскриптам). Данные читаются из LMDB (ключи текста: `speaker_text` / `text` / `transcript` / `utterance`).

## Модели

| Модель | Признаки | Архитектура | README |
|---|---|---|---|
| `TF-IDF_LogReg.py` | TF-IDF (1–2 граммы) | Logistic Regression | [baseline/README.md](./baseline/README.md) |
| `Embeddings_LogReg.py` | FastText `cc.ru.300.bin` (средний вектор) | Logistic Regression | [baseline/README.md](./baseline/README.md), [EMBEDDINGS_SETUP.md](./baseline/EMBEDDINGS_SETUP.md) |
| `BiLSTM/BiLSTM.py` | FastText embeddings (замороженная матрица) | Embedding → BiLSTM → Linear | — |
| `transformers/RuBERT.py` | токены DeepPavlov `rubert-base-cased` | RuBERT + классификатор | — |
| `baseline/example_usage.py` | — | пример использования baseline | — |

## Быстрый запуск

```bash
# TF-IDF + Logistic Regression
poetry run python dusha/my_experiments/text_models/baseline/TF-IDF_LogReg.py --mode train

# FastText + Logistic Regression
poetry run python dusha/my_experiments/text_models/baseline/Embeddings_LogReg.py --mode train

# FastText + BiLSTM
poetry run python dusha/my_experiments/text_models/BiLSTM/BiLSTM.py --mode train

# RuBERT
poetry run python dusha/my_experiments/text_models/transformers/RuBERT.py --mode train --config text/rubert.json
```

## Ключевые флаги

- Общие: `--mode {train,load,auto,smoke}`, `--config`, `--train-data-path`, `--test-data-path`, `--no-save`.
- `BiLSTM.py`: `--epochs`, `--batch-size`, `--max-len`, `--hidden-size`, `--num-layers`, `--max-vocab-size`, `--min-freq`.
- `RuBERT.py`: `--backbone-name`, `--epochs`, `--stage1-epochs`, `--batch-size`, `--max-len`, `--lr`, `--loss-name {ce,focal}`, `--label-smoothing`, `--fp16`.

## Предобученные FastText

Для `Embeddings_LogReg.py` и `BiLSTM.py` нужна модель `cc.ru.300.bin` (2.3 GB). Скачивается автоматически через `utils/pretrained.load_fasttext_model()` в `checkpoints/pretrained/fasttext/` либо вручную — см. [baseline/EMBEDDINGS_SETUP.md](./baseline/EMBEDDINGS_SETUP.md).

## Артефакты

- sklearn: `{Модель}_{датасет}_model.pkl` + `{...}_vectorizer.pkl` / `{...}_scaler.pkl`.
- PyTorch: `{Модель}_{датасет}_model.pt` (+ бэкапы с меткой времени, словарь, meta).
- Папка сохранения — `checkpoints/text/`.

## Результаты

Метрики текстовых моделей зафиксированы в ноутбуке [`model_analise/text_models_analise.ipynb`](../model_analise/text_models_analise.ipynb). Все модели оценены на `dusha_resd_test` (6616 сэмплов):

| Модель | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| TF-IDF + LogReg | `combine_balanced` | `dusha_resd` | 0.540 | 0.556 |
| Embeddings (FastText) + LogReg | `combine_balanced` | `dusha_resd` | 0.531 | 0.541 |
| BiLSTM | `combine_balanced` | `dusha_resd` | 0.560 | 0.580 |
| **RuBERT** | **`dusha_resd`** | **`dusha_resd`** | **0.586** | **0.601** |

Состав корпусов — [`CORPUS.md`](../../../CORPUS.md). Аудио/текст-базлайны других модальностей и мультимодальные результаты — см. [`audio_models/README.md`](../audio_models/README.md) и [`multimodal/README.md`](../multimodal/README.md).

Проанализированные чекпоинты TF-IDF/Embeddings/BiLSTM обучены на `combine_balanced`, RuBERT — на `dusha_resd` (см. ноутбук). По умолчанию скрипты текстовых моделей (в т.ч. RuBERT, чекпоинт `RuBERT_dusha_resd_train_model.pt`) обучаются на совмещённом корпусе `dusha_resd` (Dusha + RESD).
