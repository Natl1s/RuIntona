# RuIntona — Hugging Face Hub collection

Публикация обученных весов проекта **RuIntona** (Speech Emotion Recognition,
русский язык) на Hugging Face Hub. Здесь живут model cards, лицензии и NOTICE;
сами файлы весов заливаются скриптом `upload.py` из `checkpoints/`.

**Статус: все 6 репозиториев опубликованы.** Коллекция «RuIntona SER»:

https://huggingface.co/collections/Natlis/ruintona-ser-6a96d7cd6f9979ac35d1b505

## Коллекция моделей

| Модель | Репозиторий HF | Лицензия весов | Файлы |
|---|---|---|---|
| RuBERT (текст) | <https://huggingface.co/Natlis/rubert-emotion-classification-ru> | Apache-2.0 | `.pt` + токенизатор |
| CNN-BiLSTM (аудио) | <https://huggingface.co/Natlis/cnn-bilstm-emotion-classification-ru> | CC BY-SA 4.0 | `.pt` |
| CNN (аудио, baseline) | <https://huggingface.co/Natlis/cnn-emotion-classification-ru> | CC BY-SA 4.0 | `.pt` |
| openSMILE + XGBoost (аудио) | <https://huggingface.co/Natlis/opensmile-xgboost-emotion-classification-ru> | CC BY-SA 4.0 | `model.pkl` + `scaler.pkl` |
| SVM RBF (аудио, baseline) | <https://huggingface.co/Natlis/svm-emotion-classification-ru> | CC BY-SA 4.0 | `model.pkl` + `scaler.pkl` |
| Late fusion HuBERT+RuBERT | <https://huggingface.co/Natlis/hubert-rubert-late-fusion-emotion-classification-ru> | CC BY-SA 4.0 | отчёт JSON (α) |

Каждая карточка модели (`hf/cards/<NN>_*/README.md`) содержит метрики,
гиперпараметры, примеры загрузки и файлы `LICENSE` / `NOTICE`.

## Почему такие лицензии

- **Группа A (Apache-2.0)** — чекпоинты содержат дообученный бэкбон
  (`rubert-base-cased`, `wav2vec2-xls-r-300m`, HuBERT — все Apache-2.0):
  производные веса остаются под лицензией бэкбона.
- **Группа B (CC BY-SA 4.0)** — модели, обученные с нуля на корпусе Dusha:
  лицензия Dusha — указание авторства + Share-Alike (переработанный CC BY-SA 4.0).
  Подробнее: `DUSHA.md`, `RESD.md`, `SOURCES.md`, `NOTICE` в корне проекта.

## Использование

Любой файл репозитория HF скачивается так:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("Natlis/<repo>", "<file>")   # см. таблицу выше
```

Загрузка весов в архитект-модель проекта — штатными утилитами `utils/model_io.py`
(`load_pytorch_model` / `load_sklearn_model`). Инструкции per-model — в README карточек.

## Публикация

```bash
poetry run huggingface-cli login           # однократно
poetry run python ruintona/my_experiments/hf/upload.py --dry-run   # проверить
poetry run python ruintona/my_experiments/hf/upload.py             # залить всё
```

Карточки (`cards/`) находятся под контролем git; веса (`.pt`/`.pkl`) — нет
(см. `.gitignore` в этой папке и в корне проекта).