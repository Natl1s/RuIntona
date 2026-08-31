# Тесты (`my_experiments/tests/`)

Smoke-тесты моделей: запуск каждого скрипта в режиме `--mode smoke` на маленьких синтетических LMDB. Прогоняются в CI (`.github/workflows/ci.yml`: `ruff` + `pytest`).

> Источники и лицензии сторонних предобученных моделей и датасетов — [`SOURCES.md`](../../../SOURCES.md).

## Структура

| Файл | Назначение |
|---|---|
| `conftest.py` | Фикстуры: tiny-датасеты audio/text/multimodal |
| `smoke_helpers.py` | Генераторы синтетических LMDB и вспомогательные функции |
| `test_smoke_audio.py` | Smoke-тесты аудио-моделей |
| `test_smoke_text.py` | Smoke-тесты текстовых моделей |
| `test_smoke_multimodal.py` | Smoke-тесты мультимодальных моделей (early fusion) |
| `test_model_io.py` | Загрузка PyTorch-чекпоинтов (numpy-глобалы / fallback при `weights_only`) |
| `test_pretrained.py` | Пути `utils/pretrained.py` (FastText) |

Покрытие `test_smoke_audio.py`: `logreg`, `svm`, `random-forest`, `cnn`, `cnn-bilstm`.
Покрытие `test_smoke_text.py`: `tfidf-logreg`, `embeddings-logreg`, `bilstm`, `rubert`.
Покрытие `test_smoke_multimodal.py`: `early-fusion` (требуются `torch` + `transformers`).

## Запуск

```bash
poetry run pytest ruintona/my_experiments/tests/ -v

# локальный линт (как в CI)
poetry run ruff check ruintona/my_experiments/tests/
```

Примечание: тесты не проверяют качество моделей, а лишь гарантируют, что пайплайн (обучение/инференс) отрабатывает без ошибок end-to-end.
