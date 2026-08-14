# Анализ данных (`my_experiments/data_analise/`)

Jupyter-ноутбуки с разведочным анализом (EDA) датасета Dusha.

| Ноутбук | Что анализирует |
|---|---|
| `main_statistic.ipynb` | Общая статистика всех датасетов и **обоснование корпусов**: источники (crowd/podcast), распределение эмоций, качество разметки (аннотаторы → Dawid-Skene), воспроизведение пайплайна сборки (балансировка, small 30%, RESD, LMDB), качество и честность разбиений train/test |
| `text_analise.ipynb` | Текстовые признаки: длина транскриптов, лексика по эмоциям, типичные слова |
| `audio_analise.ipynb` | Аудио признаки: длительности, спектральные характеристики, визуализация waveform/спектрограмм |

## Запуск

```bash
poetry run jupyter notebook dusha/my_experiments/data_analise/
```

Ноутбуки читают данные из JSONL/CSV-манифестов (см. `data_processing/dataset/processed_dataset_090/aggregated_dataset/`). `DATASET_PATH` задаётся в начале каждого ноутбука и корректируется под локальное расположение датасета.

`main_statistic.ipynb` дополнительно читает длины баз LMDB (`__len__`) и выборку записей для проверки форматов (нужен модуль `lmdb`, входит в состав проекта).

## Выходные файлы

При выполнении `main_statistic.ipynb` рядом с ноутбуком сохраняются:

- `data_statistics.csv` — сводная таблица датасетов (строки, часы, длительности, доля текстов, спикеры);
- `data_statistics.json` — полный отчёт (датасеты, длины LMDB, сравнение train/test).

## Зависимости

`pandas`, `matplotlib`, `seaborn`, `librosa` (для `audio_analise.ipynb`). Устанавливаются через `poetry install`.
