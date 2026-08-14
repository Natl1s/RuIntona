# Скрипты сборки датасетов (`make_data_scripts/`)

Скрипты для подготовки датасетов из обработанных манифестов `data_processing`. Все команды ниже можно запускать из корня репозитория (или используя путь, соответствующий вашему расположению).

Скрипты расположены в:
`dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/`

Полное описание корпусов и общая схема их построения (включая совмещённый корпус Dusha + RESD) — в [`CORPUS.md`](../../../../../../CORPUS.md) (в корне репозитория).

## Общая схема

```
raw (crowd.tar / podcast.tar)
  → data_processing/processing.py            (mel-фичи, Dawid-Skene, порог 0.9)
  → aggregated_dataset/crowd_*.jsonl, podcast_*.jsonl
  → build_balanced_aggregated_jsonl.py
  → combine_balanced_{train,test}.jsonl (+ *_small.jsonl)
  → lmdb_convert.py                          (JSONL + WAV → LMDB)
  → hug_dataset/add_missing_spectrograms.py  (досчёт x для записей без него)

RESD parquet (Aniemore/resd_annotated)
  → hug_dataset/make_raw.py                  (parquet → raw_*.jsonl + wavs)
  → объединение с combine_balanced_*.jsonl   → dusha_resd_{train,test}.jsonl
  → lmdb_convert.py                          → dusha_resd_{train,test}.lmdb
  → hug_dataset/add_missing_spectrograms.py  (у RESD-строк после конвертации нет x)
```

## `build_balanced_aggregated_jsonl.py`

Создаёт 4 сбалансированных JSONL-датасета внутри `aggregated_dataset`:

- `combine_balanced_train.jsonl`
- `combine_balanced_test.jsonl`
- `combine_balanced_train_small.jsonl`
- `combine_balanced_test_small.jsonl`

Правила:

1. Источник train: `crowd_train.jsonl + podcast_train.jsonl`
2. Источник test: `crowd_test.jsonl + podcast_test.jsonl`
3. Используются только целевые эмоции: `angry`, `sad`, `neutral`, `positive`
4. Для полных наборов: `neutral <= 2 * min(число записей не-нейтральных классов)`; не-нейтральные классы берутся целиком
5. Для small-наборов: размер = 30% (настраивается `--small-ratio`) от полного набора, классовое соотношение сохраняется максимально близко
6. `--seed` (по умолчанию 42) обеспечивает воспроизводимость сэмплинга

```bash
python dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/build_balanced_aggregated_jsonl.py

# с настройками
python dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/build_balanced_aggregated_jsonl.py \
  --aggregated-dir dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset \
  --small-ratio 0.3 \
  --seed 42
```

Именно эти JSONL используются для построения LMDB и экспериментов (`dusha/my_experiments/`).

## `lmdb_convert.py`

Конвертер **JSONL + WAV → LMDB** (полные записи: waveform, текст, метка).

```bash
python dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/lmdb_convert.py \
  --manifest dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train.jsonl \
  --data-root dusha/data_processing/dataset/processed_dataset_090 \
  --output dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train.lmdb
```

Поддерживает в манифесте поля `audio_path` / `wav_path` / `wav` (аудио), `hash_id` / `id` (id) и метки из полей `label` / `emotion` / `annotator_emo` / `speaker_emo` / `target` / `class` / `emo`. Формат записи LMDB:

- ключ: `<index>` (bytes)
- значение: pickle:
  - `y`: метка (int, angry=0, sad=1, neutral=2, positive=3)
  - `id`: sample_id
  - `waveform`: сырой mono-сигнал (`np.float32`, ресемплирован в 16 кГц)
  - `waveform_sr`: частота дискретизации (`16000`)
  - `text`: транскрипт (`speaker_text` / `text` / `transcript` / `utterance`)
- служебный ключ `b"__len__"`: число записей

> **Фильтр podcast.** Скрипт пропускает строки, у которых путь к аудио содержит `podcast` (в поля `audio_path`/`wav_path`/`wav`/`tensor`/`feature_path`/`id`/`hash_id`). Поэтому длина LMDB **меньше** числа строк JSONL. Например, `combine_balanced_train.jsonl` → `combine_balanced_train.lmdb`.
>
> **Замечание про спектрограммы.** `lmdb_convert.py` **не записывает** mel-спектрограмму `x`. Для записей без `x` её можно досчитать из `waveform` скриптом `dusha/data_processing/dataset/hug_dataset/add_missing_spectrograms.py` (нужно для `dusha_resd`). Существующие на диске LMDB `combine_balanced*` содержат `x` — они построены старой версией конвертера.

Флаги: `--manifest`, `--output`, `--data-root` (по умолчанию — директория манифеста), `--commit-interval` (записей на транзакцию, по умолчанию 1024).

## `lmdb_convert_only_spectr.py`

Конвертер **JSONL + NPY → LMDB (только спектрограммы)** — без waveform и текста, только `x`, `y` и `id`. Быстрее и компактнее, чем `lmdb_convert.py`; не требует аудиофайлов.

```bash
python dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/lmdb_convert_only_spectr.py \
  --manifest dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train.jsonl \
  --data-root dusha/data_processing/dataset/processed_dataset_090 \
  --output dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train_spectr.lmdb
```

Признаки берутся из полей `tensor` / `feature_path` или `features/<hash_id>.npy`. Флаги: те же, что у `lmdb_convert.py`.

## Сборка совмещённого корпуса `dusha_resd` (Dusha + RESD)

Большинство моделей (`my_experiments/`) обучено на корпусе `dusha_resd` — объединении `combine_balanced` и датасета RESD (Aniemore, Hugging Face). Последовательность сборки:

```bash
# 1. RESD parquet → raw jsonl + wavs (train 916 / test 224 строк, эмоции из названия)
python dusha/data_processing/dataset/hug_dataset/make_raw.py \
  --input-dir dusha/data_processing/dataset/hug_dataset/data \
  --wavs-dir dusha/data_processing/dataset/hug_dataset/wavs

# 2. Объединить (конкатенация) raw_train + combine_balanced_train → dusha_resd_train.jsonl;
#    аналогично для test. (Отдельного скрипта merge нет.)

# 3. Конвертация в LMDB
python dusha/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/lmdb_convert.py \
  --manifest .../dusha_resd_train.jsonl --data-root .../processed_dataset_090 \
  --output .../aggregated_dataset/dusha_resd_train.lmdb
# (аналогично dusha_resd_test.lmdb)

# 4. Досчёт mel-спектрограмм x для записей без них (нужно для RESD-строк)
python dusha/data_processing/dataset/hug_dataset/add_missing_spectrograms.py \
  --lmdb .../aggregated_dataset/dusha_resd_train.lmdb \
  --lmdb .../aggregated_dataset/dusha_resd_test.lmdb
```

Итоговые размеры: `dusha_resd_train.lmdb` — 69119 записей (68203 combine_balanced + 916 RESD), `dusha_resd_test.lmdb` — 6616 (6392 + 224).

## Скрипты `dusha/data_processing/dataset/hug_dataset/`

| Скрипт | Назначение |
|---|---|
| `make_raw.py` | Конвертация parquet (RESD / прочие HF-датасеты) в `raw_*.jsonl` + `wavs/*.wav`; эмоция читается из названия клипа, маппинг `happiness→positive`, `anger→angry`, `sadness→sad`, `neutral→neutral` |
| `dataset_stats.py` | Статистики по raw JSONL (распределение эмоций, длительности, длина текстов, дубликаты) |
| `add_missing_spectrograms.py` | Досчёт mel-спектрограммы `x` из `waveform` для записей без `x` (по умолчанию работает с `dusha_resd_train/test.lmdb`) |

## `make_manifest.py`

Пустой файл-заглушка (0 строк) — не используется.

## См. также

- [`CORPUS.md`](../../../../../../CORPUS.md) — состав корпусов и полные правила построения
- [`dusha/data_processing/README.md`](../../../../README.md) — пайплайн первичной обработки (Dawid-Skene, фичи)
- [`dusha/data_processing/dataset/hug_dataset/`](../../../hug_dataset/) — скрипты для внешних HF-датасетов
