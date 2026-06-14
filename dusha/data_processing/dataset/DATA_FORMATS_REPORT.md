# Отчет по форматам данных в `dusha/data_processing/dataset`

## 1) Что было изучено

Проверены:

- реальная структура файлов в `data_processing/dataset` (размеры/кол-во/типы),
- внутренние структуры `.npy`, `.jsonl`, `.lmdb`,
- скрипты генерации и преобразования данных,
- конфиги и код, которые эти данные потребляют.

Ключевые источники:

- `data_processing/processing.py`
- `data_processing/utils/calculate_features.py`
- `data_processing/utils/aggregation.py`
- `data_processing/utils/dawidskene.py`
- `data_processing/dataset/lmdb_convert.py`
- `data_processing/dataset/lmdb_convert_only_spectr.py`
- `data_processing/dataset/hug_dataset/make_raw.py`
- `data_processing/dataset/iemocap/make_raw.py`
- `data_processing/README.md`
- `my_experiments/lmdb_utils.py`

---

## 2) Общая карта хранилища

По факту в папке:

- **~204G** данных,
- **300960 `.npy`** файлов (фичи),
- **27 `.jsonl`** файлов (разные слои манифестов),
- **12 `.lmdb`** + **12 `.lmdb-lock`**.

Крупные сегменты:

- `features/` — ~31G (`.npy`)
- `processed_dataset_090/aggregated_dataset/` — ~83G (jsonl + lmdb)
- `crowd_train/` — ~26G (raw wav + разметка)
- `crowd_test/` — ~2.5G
- `iemocap/` — ~2.5G
- `hug_dataset/` — ~1.3G
- `tars/` — ~58G (`crowd.tar`, `podcast.tar`, `features.tar`)

---

## 3) Формат `.npy` (акустические признаки)

### Где лежат

- Основной массив: `data_processing/dataset/features/*.npy`
- Количество: **300960**

### Как создаются

Источник: `data_processing/utils/calculate_features.py`

- WAV загружается через `librosa.load(..., sr=16000)`
- Строится mel-спектрограмма:
  - `n_mels=64`
  - окно ~20ms (`win_length = 0.02 * sr`)
  - шаг ~10ms (`hop_length = 0.01 * sr`)
- Перевод в dB: `librosa.power_to_db(...)`
- Сохранение: `np.save(..., mel_spec[None])`

### Фактическая структура

- формат файла: NumPy `.npy` (magic version **1.0**),
- `dtype`: **float32**,
- `fortran_order`: `False`,
- shape: **`(1, 64, T)`**, где `T` переменная длина по времени,
- диапазон значений (по выборке): примерно **[-80.0, ~0.0] dB**.

### Как используются

- В `processed_dataset_090/train/*.jsonl` и `.../test/*.jsonl` поле `tensor` хранит путь к `.npy`.
- Загрузка обычно через `np.load(path)` (см. `experiments/core/dataset.py`, `my_experiments/...`).

---

## 4) Формат `.jsonl` (манифесты)

В каталоге фактически 3 основных схемы JSONL.

> Важно: часть файлов содержит `NaN` как значение. Это читается Python `json.loads`, но это не строгий RFC-JSON.

---

### 4.1 Raw-разметка (`raw_*.jsonl`)

Поля:

- `hash_id`
- `audio_path`
- `duration`
- `annotator_emo`
- `golden_emo`
- `annotator_id`
- `speaker_text`
- `speaker_emo`
- `source_id`

Примеры файлов и объёмы:

- `crowd_train/raw_crowd_train.jsonl` — **898136** строк
- `crowd_test/raw_crowd_test.jsonl` — **78205**
- `podcast_train/raw_podcast_train.jsonl` — **645813**
- `podcast_test/raw_podcast_test.jsonl` — **83447**
- `iemocap/raw_train.jsonl` — **8631**
- `hug_dataset/data/raw_train-...jsonl` — **916**
- `hug_dataset/data/raw_test-...jsonl` — **224**

Как появляются:

- crowd/podcast — исходная разметка из сырого датасета,
- iemocap/hug_dataset — генерируются скриптами:
  - `dataset/iemocap/make_raw.py`
  - `dataset/hug_dataset/make_raw.py`
  из parquet в `raw_*.jsonl` + `wavs/`.

---

### 4.2 Aggregated JSONL (`processed_dataset_090/aggregated_dataset/*.jsonl`)

Поля:

- `hash_id`
- `audio_path`
- `duration`
- `emotion`
- `golden_emo`
- `speaker_text`
- `speaker_emo`
- `source_id`

Ключевые файлы:

- `crowd_train.jsonl` — **148547**
- `crowd_test.jsonl` — **13859**
- `podcast_train.jsonl` — **79966**
- `podcast_test.jsonl` — **10662**
- `combine_balanced_train.jsonl` — **89943**
- `combine_balanced_test.jsonl` — **9164**
- `combine_balanced_train_small.jsonl` — **26983**
- `combine_balanced_test_small.jsonl` — **2749**

Как появляются:

1. `processing.py` вызывает `aggregate_data(...)` из `utils/aggregation.py`.
2. `utils/dawidskene.py` агрегирует аннотации (`CrowdKitDawidSkene`) и пишет вероятности в `meta.tsv`.
3. В выборку попадают элементы, прошедшие threshold (обычно `0.9` => `processed_dataset_090`).

Доп. балансировка:

- `my_experiments/data_analise/build_balanced_aggregated_jsonl.py`
  формирует `combine_balanced_*` и `*_small`.

---

### 4.3 Экспериментальные манифесты (`processed_dataset_090/train/*.jsonl`, `.../test/*.jsonl`)

Поля:

- `id`
- `tensor` (относительный путь к `.npy`)
- `wav_length`
- `label` (int: angry=0, sad=1, neutral=2, positive=3)
- `emotion` (str)

Ключевые файлы:

- `train/train.jsonl` — **226866**
- `test/test.jsonl` — **24326**
- `train/crowd_train.jsonl` — **147093**
- `train/podcast_train.jsonl` — **79773**
- `train/crowd_{small,medium,large}.jsonl` — **7354 / 14709 / 73546**
- `train/podcast_{small,medium,large}.jsonl` — **3988 / 7977 / 39886**
- `test/crowd_test.jsonl` — **13691**
- `test/podcast_test.jsonl` — **10635**

Как появляются:

- `utils/aggregation.py -> make_exp_data(...)`, где `tensor` строится как
  `../../features/<hash_id>.npy`.

---

## 5) Формат `.lmdb`

### Физическое представление

- используются **file-backed LMDB** файлы:
  - `<name>.lmdb`
  - `<name>.lmdb-lock`
- это подтверждается логикой `lmdb.open(..., subdir=False)` при наличии суффикса (`.lmdb`).

### Ключи/метаданные

- ключи записей: строки индексов `"0"`, `"1"`, ... (bytes),
- спецключ: `b"__len__"` — длина датасета,
- значения: `pickle`-сериализованный `dict`.

### Реально обнаруженные схемы payload

1. **Only waveform/text** (hug raw lmdb):
   - `id`, `text`, `waveform`, `waveform_sr`, `y`

2. **Only spectrogram** (`*_only_spectr.lmdb`):
   - `id`, `x`, `y`

3. **Multimodal (spectrogram + waveform + text)** (`combine_balanced*.lmdb`, `dusha_resd*.lmdb`):
   - `id`, `x`, `y`, `waveform`, `waveform_sr`, `text`

Где:

- `x` — `np.ndarray(float32)` shape `(1, 64, T)`
- `waveform` — `np.ndarray(float32)` shape `(N,)`, SR обычно 16000
- `y` — int-класс

### Размеры и длины (факт)

- `combine_balanced_train_only_spectr.lmdb` — `meta_len=89943`
- `combine_balanced_test_only_spectr.lmdb` — `meta_len=9164`
- `combine_balanced_train.lmdb` — `meta_len=68203`
- `combine_balanced_test.lmdb` — `meta_len=6392`
- `combine_balanced_train_small_only_spectr.lmdb` — `meta_len=26983`
- `combine_balanced_train_small.lmdb` — `meta_len=20474`
- `combine_balanced_test_small_only_spectr.lmdb` — `meta_len=2749`
- `combine_balanced_test_small.lmdb` — `meta_len=1863`
- `dusha_resd_train.lmdb` — `meta_len=69119`
- `dusha_resd_test.lmdb` — `meta_len=6616`
- `hug_dataset/data/raw_train-00000-of-00001.lmdb` — `meta_len=916`
- `hug_dataset/data/raw_test-00000-of-00001.lmdb` — `meta_len=224`

### Критичная деталь про несоответствие JSONL↔LMDB

`lmdb_convert.py` имеет фильтр `_is_podcast_row(...)` и **пропускает podcast-строки**.

Из-за этого:

- `combine_balanced_train.jsonl` (89943) -> `combine_balanced_train.lmdb` (68203)
- `combine_balanced_test.jsonl` (9164) -> `combine_balanced_test.lmdb` (6392)
- аналогично для `*_small`.

Для `*_only_spectr.lmdb` такого фильтра нет, поэтому длины совпадают с JSONL.

---

## 6) Как данные создаются: полный пайплайн

### 6.1 Сырой этап

- исходники в tar: `tars/crowd.tar`, `tars/podcast.tar`, `tars/features.tar`,
- для внешних датасетов:
  - `iemocap/data/*.parquet` -> `iemocap/make_raw.py`
  - `hug_dataset/data/*.parquet` -> `hug_dataset/make_raw.py`
  создают WAV и `raw_*.jsonl`.

### 6.2 Фичи и агрегация

- `processing.py`:
  1. считает mel-фичи в `features/*.npy`,
  2. агрегирует разметку Dawid-Skene (`meta.tsv`),
  3. строит JSONL для train/test + aggregated_dataset в `processed_dataset_090`.

### 6.3 Балансировка

- `build_balanced_aggregated_jsonl.py` создаёт:
  - `combine_balanced_train/test`
  - `combine_balanced_train/test_small`

### 6.4 Конвертация в LMDB

- `dataset/lmdb_convert_only_spectr.py`:
  JSONL + NPY -> LMDB (`x,y,id`)
- `dataset/lmdb_convert.py`:
  JSONL + NPY + WAV + text -> LMDB (`x,y,id,waveform,waveform_sr,text`)
  и фильтрует podcast-строки.

---

## 7) Что именно потребляет код обучения

- Классические эксперименты (`experiments/*`) работают с JSONL манифестами и `tensor` -> `.npy`.
- Новые мультимодальные эксперименты (`my_experiments/*`) читают LMDB через `my_experiments/lmdb_utils.py`.
- Текущие train/test пути в `my_experiments/train_data.config` и `test_data.config` указывают на:
  - `processed_dataset_090/aggregated_dataset/dusha_resd_train.lmdb`
  - `processed_dataset_090/aggregated_dataset/dusha_resd_test.lmdb`

---

## 8) Практические рекомендации по использованию

1. Если нужны только акустические признаки: использовать `.jsonl` с полем `tensor` + `.npy`.
2. Если нужен быстрый random access и меньше IO-штрафа: использовать `*_only_spectr.lmdb`.
3. Для мультимодали (audio waveform + text + mel): использовать `combine_balanced*.lmdb` или `dusha_resd*.lmdb`.
4. При построении LMDB внимательно проверять фильтр podcast в `lmdb_convert.py`, иначе длины выборки будут неожиданно меньше исходного JSONL.

---

## 9) Неочевидные и важные наблюдения

- В `processed_dataset_090/aggregated_dataset/make_manifest.py` сейчас пустой файл.
- В raw/aggregated JSONL встречаются `NaN` значения (например `golden_emo`, иногда `speaker_text`).
- `features/*.npy` для `hug_dataset` raw hash_id в текущем состоянии не обнаружены (прямых совпадений по имени нет), значит для смешивания с общими фичами нужна отдельная синхронизация идентификаторов/фичей.
