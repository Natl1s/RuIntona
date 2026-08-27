# Анализ данных (`my_experiments/data_analise/`)

Jupyter-ноутбуки с разведочным анализом (EDA) датасета Dusha и исследованиями корпусов.

| Ноутбук | Что анализирует |
|---|---|
| `corpus_statistics.ipynb` | Статистика ключевых LMDB-корпусов (`combine_balanced`, `combine_balanced_small`, `dusha_resd`) и **обоснование корпусов**: распределение эмоций, пересечение корпусов по ID, качество и честность разбиений train/test (χ² по эмоциям, KS по длительностям) |
| `dusha_resd_analyse.ipynb` | Исследование корпуса `dusha_resd` (Dusha + RESD): состав, источники, распределение эмоций, пересечение с Dusha |
| `dusha_resd_leak_audio_check.ipynb` | Проверка утечки (leak) между train/test в корпусе `dusha_resd` по аудио-идентификаторам |
| `text_analise.ipynb` | Текстовые признаки: длина транскриптов, лексика по эмоциям, типичные слова |
| `audio_analise.ipynb` | Аудио признаки: длительности, спектральные характеристики, визуализация waveform/спектрограмм |

## Запуск

```bash
poetry run jupyter notebook ruintona/my_experiments/data_analise/
```

Ноутбуки читают данные из JSONL/CSV-манифестов (см. `data_processing/dataset/processed_dataset_090/aggregated_dataset/`) и/или LMDB-корпусов (см. `CORPUS.md`). Пути к данным задаются в начале каждого ноутбука и корректируются под локальное расположение датасета (см. `my_experiments/data.json`).

## Выходные файлы

Современные ноутбуки в этом каталоге носят интерактивный характер и **не пишут файлов** при выполнении — результаты выводятся в ячейках.

В каталоге лежат `data_statistics.csv` и `data_statistics.json` — сводная таблица датасетов (строки, часы, длительности, доля текстов, спикеры, источники, распределение эмоций) и полный отчёт (датасеты, длины LMDB, сравнение train/test). Это **устаревшие артефакты** от прежней версии ноутбука (ранее `main_statistic.ipynb`): текущий `corpus_statistics.ipynb` их не обновляет. Если нужен актуальный дамп — доработайте ноутбук или удалите файлы.

## Зависимости

`pandas`, `matplotlib`, `seaborn`, `scipy`, `librosa` (для `audio_analise.ipynb`), `lmdb` (для чтения `.lmdb`-корпусов в `corpus_statistics.ipynb`). Устанавливаются через `poetry install`.
