# Отчет по обработке данных в `dusha/data_processing`

## 1) Что было изучено

Разбор сделан по всем скриптам в папке `dusha/data_processing` и `dusha/data_processing/utils`:

- `dusha/data_processing/processing.py`
- `dusha/data_processing/utils/aggregation.py`
- `dusha/data_processing/utils/calculate_features.py`
- `dusha/data_processing/utils/dawidskene.py`
- `dusha/data_processing/utils/datacls.py`
- `dusha/data_processing/README.md`
- `dusha/data_processing/requirements.txt`

Пустые файлы `__init__.py` логики не содержат.

## 2) Точка входа и общий пайплайн

Главный скрипт: `dusha/data_processing/processing.py` (команда Click `processing`).

Запуск (по README):

```bash
python processing.py -dataset_path DATASET_PATH
```

Параметры:

- `--dataset_path` (`-dataset_path`) - путь к корню датасета (обязательный).
- `--use_tsv` (`-tsv`) - читать и писать манифесты в TSV вместо JSONL.
- `--recalculate_features` (`-rf`) - пересчитать все признаки заново.
- `--threshold` (`-threshold`) - порог уверенности для Dawid-Skene, по умолчанию `0.9`.

Валидация: `threshold` должен быть в диапазоне `[0, 1]`, иначе выбрасывается `AttributeError`.

Общий процесс:

1. Создается папка результата `processed_dataset_0XX`, где `XX = int(threshold * 100)`.
2. Создаются подпапки результата: `train`, `test`, `aggregated_dataset`.
3. Гарантируется наличие папки `features` в корне датасета.
4. Для каждого набора (`crowd_train`, `crowd_test`, `podcast_train`, `podcast_test`):
   - читается разметка (`raw_*.jsonl` или `raw_*.tsv`),
   - выделяются `hash_id`/имена wav,
   - считаются (или догоняются) акустические признаки `.npy`.
5. Выполняется агрегация эмоций Dawid-Skene и формирование выходных манифестов.

## 3) Ожидаемая структура входных данных

Скрипт ожидает в `dataset_path` как минимум:

- `crowd_train/raw_crowd_train.jsonl` или `.tsv`
- `crowd_test/raw_crowd_test.jsonl` или `.tsv`
- `podcast_train/raw_podcast_train.jsonl` или `.tsv`
- `podcast_test/raw_podcast_test.jsonl` или `.tsv`
- `crowd_train/wavs/*.wav`
- `crowd_test/wavs/*.wav`
- `podcast_train/wavs/*.wav`
- `podcast_test/wavs/*.wav`

Формат строк разметки (`MarkupDataclass` из `utils/datacls.py`):

- `hash_id: str`
- `audio_path: str`
- `duration: str`
- `annotator_emo: str`
- `golden_emo: str`
- `speaker_text: str`
- `speaker_emo: str`
- `source_id: str`
- `annotator_id: str`

Примечание: в `MarkupDataclass` поля `audio_path` и `annotator_emo` объявлены повторно в коде; на практическую схему входа это обычно не влияет, но это технический дефект структуры dataclass.

## 4) Чтение сырой разметки

Функция: `read_data_markup()` в `utils/aggregation.py`.

- Если `--use_tsv`:
  - читается `raw_*.tsv`,
  - первая строка используется как заголовок,
  - поля маппятся в `MarkupDataclass`.
- Если без `--use_tsv`:
  - читается `raw_*.jsonl`,
  - каждая строка - JSON объект для `MarkupDataclass`.

Итог: список объектов разметки (по одному объекту на аннотацию).

## 5) Вычисление акустических признаков

Файлы: `utils/calculate_features.py` (`load_features()`, `create_features()`).

### 5.1 Что именно считается

Используется mel-spectrogram через `librosa`:

- `sample_rate = 16000`
- `hop_length_coef = 0.01` -> `hop_length = 160` сэмплов (10 ms)
- `win_length_coef = 0.02` -> `n_fft = 320` сэмплов (20 ms)
- `n_mels = 64`

Далее:

1. `librosa.load(..., sr=16000)`
2. `librosa.feature.melspectrogram(...)`
3. `librosa.power_to_db(spec, ref=np.max)`
4. сохранение в `features/<hash_id>.npy` как `mel_spec[None]`

Фактическая форма тензора: `(1, 64, T)`, где `T` зависит от длительности записи.

### 5.2 Режимы пересчета

- `--recalculate_features`:
  - пересчитываются все `.wav`, найденные в `wavs/`.
- без `--recalculate_features`:
  - ищутся уже готовые `.npy` в `features/`,
  - считаются только отсутствующие признаки.

### 5.3 Проверки и поведение

- Если каких-то wav не хватает относительно `hash_id` из разметки, печатается число пропусков.
- Если аудиосигнал пустой (`len(data) == 0`), выбрасывается `AttributeError`.

## 6) Агрегация эмоций Dawid-Skene

Файлы: `utils/aggregation.py` + `utils/dawidskene.py`.

### 6.1 Какие данные агрегируются

Все четыре части (`podcast_test`, `podcast_train`, `crowd_train`, `crowd_test`) объединяются в один список аннотаций.

Для Dawid-Skene берутся тройки:

- `task = hash_id`
- `worker = annotator_id`
- `label = annotator_emo`

### 6.2 Алгоритм

- Используется `crowdkit.aggregation.DawidSkene`.
- `n_iter = 100` (по умолчанию в `get_dawidskene_pred`).
- Вычисляются вероятности классов по каждому `task`.
- Все вероятности сохраняются в `meta.tsv` (`to_csv(sep="\t")`).

### 6.3 Порог уверенности

Для каждого `task`:

- берется максимальная вероятность класса,
- если `max_proba >= threshold`, задача попадает в результат,
- иначе задача отбрасывается (в финальные манифесты не идет).

## 7) Постобработка и формирование выходных датасетов

### 7.1 Формирование `aggregated_dataset/*`

Функция: `filter_data()` -> `agg_data_to_file()`.

Для каждой части датасета:

- берется только первая запись для каждого `hash_id` (дедуп через `used_wavs`),
- запись включается только если по `hash_id` есть агрегированный класс,
- формируется строка `AggDataclass`:
  - `hash_id`
  - `audio_path` (переписывается как `dataset/audio_path`)
  - `duration`
  - `emotion` (агрегированная)
  - `golden_emo`
  - `speaker_text`
  - `speaker_emo`
  - `source_id`

Сохранение:

- JSONL: объект на строку,
- TSV: табличный формат с заголовком `HEADER`.

### 7.2 Формирование экспериментальных манифестов (`train/*`, `test/*`)

Функция: `make_exp_data()` -> `exp_data_to_file()`.

Запись включается в экспериментальный манифест, если:

- `golden_emo` пустой или не строка,
- и агрегированная эмоция `!= "other"`.

Поля `DataForExp`:

- `id` = `hash_id`
- `tensor` = `../../features/<hash_id>.npy`
- `wav_length` = `duration`
- `label` = числовой класс
- `emotion` = строковая эмоция

Маппинг меток (`Emotion` enum):

- `angry -> 0`
- `sad -> 1`
- `neutral -> 2`
- `positive -> 3`

Файлы на выходе:

- по частям:
  - `processed_dataset_0XX/train/crowd_train.(jsonl|tsv)`
  - `processed_dataset_0XX/train/podcast_train.(jsonl|tsv)`
  - `processed_dataset_0XX/test/crowd_test.(jsonl|tsv)`
  - `processed_dataset_0XX/test/podcast_test.(jsonl|tsv)`
- объединенные:
  - `processed_dataset_0XX/train/train.(jsonl|tsv)` = `podcast_train + crowd_train`
  - `processed_dataset_0XX/test/test.(jsonl|tsv)` = `podcast_test + crowd_test`

## 8) Форматы сохранения

### JSONL

- Одна JSON-запись на строку.
- `ensure_ascii=False` (кириллица/Unicode сохраняются как есть).

### TSV

- Первая строка - заголовок.
- Поля разделены `\t`.

## 9) Ключевые характеристики и ограничения обработки

- Аудиопризнаки: log-mel (`64` мел-канала, окно `20 ms`, шаг `10 ms`, `16 kHz`).
- Результат признаков: `.npy` на каждый `hash_id`.
- Агрегация разметки: Dawid-Skene по аннотаторам.
- Фильтрация по уверенности: только задачи выше порога `threshold`.
- Исключение из exp-манифестов: `emotion == "other"` и записи с заполненным `golden_emo`.
- При `threshold` ниже/выше валидного диапазона скрипт аварийно завершится.

## 10) Зависимости, влияющие на обработку

`dusha/data_processing/requirements.txt`:

- `pandas==1.3.5`
- `crowd-kit==1.0.0`
- `click==8.0.4`
- `tqdm==4.62.3`
- `numpy==1.21.5`
- `librosa==0.8.1`

## 11) Практические замечания (риски)

- В `MarkupDataclass` есть дубли полей (`audio_path`, `annotator_emo`) - это стоит починить для прозрачности схемы.
- В ряде мест используется `AttributeError` для валидации/ошибок данных, что затрудняет диагностику (лучше `ValueError`/кастомные исключения).
- Путь к аудио в `aggregated_dataset` формируется как `dataset/audio_path`; это важно учитывать при переносе артефактов.
- Если `features/` уже содержит `.npy`, без `-rf` пересчет не выполняется, даже если параметры признаков в коде были изменены.

---

Если нужен, могу отдельно подготовить вторую версию отчета в виде табличной спецификации (field-by-field) для всех входных/выходных файлов.
