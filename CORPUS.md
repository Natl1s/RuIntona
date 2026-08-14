# Корпусы данных (CORPUS)

Модели в этом репозитории обучаются и оцениваются на корпусах, построенных из двух исходных датасетов:

- **Dusha** (Сбер / Salute Developers) — см. [DUSHA.md](DUSHA.md)
- **RESD** (Aniemore, Hugging Face) — см. [RESD.md](RESD.md)

Все артефакты находятся в `dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/`.

## Корпусы

| Корпус | Состав | Train | Test | Используется |
|---|---|---|---|---|
| `combine_balanced` | Только Dusha (crowd + podcast), 4 эмоции, сбалансирован | 68203 | 6392 | CNN/CNN-BiLSTM, late fusion HuBERT+RuBERT, оценка foundation-моделей |
| `combine_balanced_small` | 30% от `combine_balanced` | 20474 | 1863 | быстрые прогоны, warm-start чекпоинты wav2vec2 |
| `dusha_resd` | `combine_balanced` + RESD (916 / 224 записей) | 69119 | 6616 | **большинство моделей**: RuBERT, Logistic Regression, Random Forest (default + tuned), SVM, openSMILE+XGBoost, базлайны early/late fusion |

Размеры указаны в **записях LMDB** (после конвертации). Они могут отличаться от числа строк JSONL — см. [Известные особенности](#известные-особенности).

## Правила построения

Цепочка сборки от сырых данных до LMDB:

```
raw (crowd.tar / podcast.tar)
  → data_processing/processing.py            (mel-фичи, Dawid-Skene, порог 0.9)
  → aggregated_dataset/crowd_*.jsonl, podcast_*.jsonl
  → make_data_scripts/build_balanced_aggregated_jsonl.py
  → combine_balanced_{train,test}.jsonl (+ *_small.jsonl)
  → make_data_scripts/lmdb_convert.py        (JSONL + WAV → LMDB)
  → hug_dataset/add_missing_spectrograms.py  (досчёт x для записей без него)

RESD parquet (Aniemore/resd_annotated)
  → hug_dataset/make_raw.py                  (parquet → raw_*.jsonl + wavs, эмоция из названия)
  → объединение с combine_balanced_*.jsonl   → dusha_resd_{train,test}.jsonl
  → make_data_scripts/lmdb_convert.py        → dusha_resd_{train,test}.lmdb
  → hug_dataset/add_missing_spectrograms.py  (у RESD-строк после конвертации нет x)
```

### Шаг 0. Агрегированные манифесты (только Dusha)

`dusha/data_processing/processing.py` считает mel-спектрограммы (`features/*.npy`), агрегирует разметку **Dawid-Skene** (crowdkit, порог 0.9 → `processed_dataset_090`) и формирует агрегированные манифесты `crowd_{train,test}.jsonl` / `podcast_{train,test}.jsonl` в `aggregated_dataset/` (поля: `hash_id`, `audio_path`, `duration`, `emotion`, `golden_emo`, `speaker_text`, `speaker_emo`, `source_id`).

### Шаг 1. Балансировка → `combine_balanced`

`make_data_scripts/build_balanced_aggregated_jsonl.py`:

1. Источник train: `crowd_train.jsonl + podcast_train.jsonl`; источник test: `crowd_test.jsonl + podcast_test.jsonl`.
2. Остаются только целевые эмоции: `angry`, `sad`, `neutral`, `positive`.
3. Полные наборы: `neutral <= 2 * min(число записей не-нейтральных классов)`; не-нейтральные классы берутся целиком.
4. Small-наборы: размер = 30% (`--small-ratio`) от полного, пропорции классов сохраняются максимально близко (`_scaled_targets_with_same_ratio`).
5. `--seed 42` (по умолчанию) обеспечивает воспроизводимость сэмплинга.

Выход: `combine_balanced_train.jsonl`, `combine_balanced_test.jsonl`, `combine_balanced_train_small.jsonl`, `combine_balanced_test_small.jsonl`.

### Шаг 2. RESD → raw-манифесты

`hug_dataset/make_raw.py` конвертирует parquet-файлы `Aniemore/resd_annotated` (train 1116 / test 280 строк) в `raw_*.jsonl` + `wavs/*.wav`:

- эмоция читается из названия клипа (например, `32_happiness_enthusiasm_h_120`);
- маппинг `happiness → positive`, `anger → angry`, `sadness → sad`, `neutral → neutral`;
- строки, у которых первые два токена названия не входят в целевое множество (`disgust`, `fear`, `enthusiasm`, …), пропускаются → train 916 / test 224 строк.

### Шаг 3. Объединение → `dusha_resd`

`dusha_resd_train.jsonl = combine_balanced_train.jsonl + raw_train_*.jsonl (RESD)`; аналогично для test. (Отдельного скрипта слияния нет — манифесты конкатенируются.)

### Шаг 4. Конвертация в LMDB

`make_data_scripts/lmdb_convert.py` (JSONL + WAV → LMDB). Формат записи: ключ `<index>` (bytes), значение — pickle-словарь:

| Ключ | Тип | Описание |
|---|---|---|
| `y` | `int` | метка эмоции (0–3) |
| `id` | `str` | идентификатор записи |
| `waveform` | `np.ndarray(float32)` | моно-сигнал, ресемплирован в 16 кГц |
| `waveform_sr` | `int` | частота дискретизации (`16000`) |
| `text` | `str` | транскрипт |

Плюс служебный ключ `b"__len__"` с количеством записей.

> **Фильтр podcast.** `lmdb_convert.py` пропускает строки, у которых путь к аудио содержит `podcast` (они исключаются из LMDB). Поэтому длина LMDB меньше числа строк JSONL.

### Шаг 5. Досчёт спектрограмм

`hug_dataset/add_missing_spectrograms.py` считает mel-спектрограмму `x` (форма `(1, 64, T)`, hop=160, n_fft=320, n_mels=64, `power_to_db(ref=np.max)`) из `waveform` для записей, у которых `x` отсутствует. Это необходимо для `dusha_resd`, т.к. `lmdb_convert.py` не записывает `x` для RESD-строк.

### Альтернатива: только спектрограммы

`make_data_scripts/lmdb_convert_only_spectr.py` (JSONL + NPY → LMDB с полями только `x`, `y`, `id`) — быстрее и компактнее, не требует WAV.

## Какие модели на каком корпусе

| Модель | Корпус |
|---|---|
| SVM (RBF) | `dusha_resd` (переобучен 08.08, тест acc 0.510) |
| openSMILE+XGBoost | `dusha_resd` (переобучен 08.08, тест acc 0.600) |
| Logistic Regression | `dusha_resd` (переобучена с `combine_balanced` 06.08, тест acc 0.474) |
| Random Forest | `dusha_resd` (переобучен с `combine_balanced` 06.08, тест acc 0.471) |
| CNN / CNN-BiLSTM | `combine_balanced` |
| Wav2Vec2 XLS-R 300M + Self-Attention | `combine_balanced_small` (warm-start) |
| RuBERT | `dusha_resd` |
| Random Forest (tuned) | `dusha_resd` |
| Early fusion (CNN-BiLSTM + RuBERT) | `dusha_resd` |
| Late fusion базлайн (SVM + TF-IDF LogReg) | `dusha_resd` |
| Late fusion HuBERT + RuBERT (α=0.5) | `combine_balanced` |
| Foundation-модели (Whisper, WavLM-BERT, HuBERT) | `combine_balanced` |

Подробное описание и источники foundation-моделей (Whisper, WavLM-BERT, HuBERT) — [`model_analise/README.md`](./dusha/my_experiments/model_analise/README.md).

## Известные особенности

- **Расхождение JSONL ↔ LMDB.** Из-за фильтра podcast в `lmdb_convert.py` длины LMDB меньше числа строк JSONL. Например, в `combine_balanced_train.jsonl` 89943 строк, а в LMDB — 68203 записи (см. [`make_data_scripts/README.md`](dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/README.md) — фильтр podcast и размеры корпусов).
- **Частота дискретизации RESD неоднородна** (16 кГц / 44.1 кГц) — `lmdb_convert.py` ресемплит всё в 16 кГц.

## Ссылки

- Скрипты сборки: [`make_data_scripts/README.md`](dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/README.md)
- HuggingFace-датасеты в `dusha/data_processing/dataset/hug_dataset/`: `make_raw.py`, `dataset_stats.py`, `add_missing_spectrograms.py`
- Пайплайн обработки: [`data_processing/README.md`](dusha/data_processing/README.md)
