# Обработка данных Dusha (`ruintona/data_processing/`)

Пайплайн первичной обработки сырого датасета Dusha: извлечение акустических признаков, агрегация разметки (Dawid-Skene) и формирование манифестов для экспериментов.

> **Лицензия и атрибуция.** Этот код и сам датасет Dusha адаптированы из проекта
> [Salute Developers — Golos](https://github.com/salute-developers/golos) и
> предоставляются по лицензии Dusha/Golos (attribution + share-alike). Текст
> лицензии — [`license/`](../../license/) (RU/EN), описание и атрибуция — [`DUSHA.md`](../../DUSHA.md).

```
raw (crowd.tar / podcast.tar) → processing.py → processed_dataset_0XX/
                                                     ├── features/*.npy
                                                     ├── aggregated_dataset/*.jsonl
                                                     ├── train/*.jsonl
                                                     └── test/*.jsonl
                                                        → make_data_scripts/ → LMDB → ruintona/my_experiments/
```

## Установка зависимостей

```bash
poetry install --with data-processing
# либо вручную из ruintona/data_processing/requirements.txt:
# pandas==1.3.5, crowd-kit==1.0.0, click==8.0.4, tqdm==4.62.3, numpy==1.21.5, librosa==0.8.1
```

## Запуск

```bash
poetry run python ruintona/data_processing/processing.py -dataset_path DATASET_PATH
```

### CLI-параметры `processing.py`

| Флаг | Описание | По умолчанию |
|---|---|---|
| `-dataset_path` (`--dataset_path`) | путь к корню датасета (обязательный) | — |
| `-tsv` (`--use_tsv`) | читать/писать манифесты в TSV вместо JSONL | False |
| `-rf` (`--recalculate_features`) | пересчитать все признаки заново | False |
| `-threshold` (`--threshold`) | порог уверенности для Dawid-Skene | 0.9 |

Валидация: `threshold` должен лежать в `[0, 1]`, иначе — `AttributeError`.

## Ожидаемая структура входа

```
DATASET_PATH/
├── crowd_train/raw_crowd_train.jsonl (или .tsv) + wavs/*.wav
├── crowd_test/raw_crowd_test.jsonl   + wavs/*.wav
├── podcast_train/raw_podcast_train.jsonl + wavs/*.wav
└── podcast_test/raw_podcast_test.jsonl + wavs/*.wav
```

Строка разметки (`MarkupDataclass`):

`hash_id`, `audio_path`, `duration`, `annotator_emo`, `golden_emo`, `speaker_text`, `speaker_emo`, `source_id`, `annotator_id`.

## Шаги обработки

1. Создаётся папка результата `processed_dataset_0XX` (где `XX = int(threshold * 100)`), подпапки `train/`, `test/`, `aggregated_dataset/`.
2. Для каждого набора (`crowd_train`, `crowd_test`, `podcast_train`, `podcast_test`) читается сырая разметка (`read_data_markup()`), извлекаются `hash_id`/имена wav.
3. Считаются акустические признаки (`utils/calculate_features.py`):
   - log-mel спектрограмма через `librosa`: `sr=16000`, окно `n_fft=320` (20 мс), шаг `hop=160` (10 мс), `n_mels=64`, `power_to_db(ref=np.max)`;
   - результат — `features/<hash_id>.npy`, тензор формы `(1, 64, T)`.
   - без `-rf` считаются только отсутствующие `.npy` (кэш не пересчитывается при изменении параметров).
4. Агрегация эмоций **Dawid-Skene** (`utils/dawidskene.py`, `crowdkit`, `n_iter=100`): тройки `task=hash_id`, `worker=annotator_id`, `label=annotator_emo`; вероятности классов сохраняются в `meta.tsv`.
5. Фильтрация по порогу: запись попадает в результат только если `max(proba) >= threshold`.

## Выходные манифесты

### `aggregated_dataset/*` (`filter_data()` → `agg_data_to_file()`)

По одной записи на `hash_id` (дедуп через `used_wavs`) с агрегированной эмоцией. Поля: `hash_id`, `audio_path` (переписывается как `dataset/audio_path`), `duration`, `emotion`, `golden_emo`, `speaker_text`, `speaker_emo`, `source_id`.

### `train/*`, `test/*` (`make_exp_data()` → `exp_data_to_file()`)

- по частям: `train/crowd_train`, `train/podcast_train`, `test/crowd_test`, `test/podcast_test`;
- объединённые: `train/train` = `podcast_train + crowd_train`, `test/test` = `podcast_test + crowd_test`.

Запись включается, если `golden_emo` пустой/не строка **и** агрегированная эмоция `!= "other"`. Поля: `id` (=`hash_id`), `tensor` (путь к `../../features/<hash_id>.npy`), `wav_length`, `label`, `emotion`.

Маппинг меток (совпадает с `EMO2LABEL` в `my_experiments`):

```python
angry -> 0, sad -> 1, neutral -> 2, positive -> 3
```

Форматы: JSONL (одна JSON-запись на строку, `ensure_ascii=False`) или TSV (заголовок + `\t`).

## Известные ограничения и риски

- В `utils/datacls.py` поля `audio_path` и `annotator_emo` объявлены повторно (технический дефект dataclass).
- Валидация/ошибки данных реализованы через `AttributeError` — затрудняет диагностику.
- Путь к аудио в `aggregated_dataset` формируется как `dataset/audio_path` — важно при переносе артефактов.
- Если `features/` уже содержит `.npy`, без `-rf` пересчёт не выполняется, даже если параметры признаков в коде изменились.

## Дальнейшие шаги

Сборка сбалансированных датасетов и конвертация в LMDB — [`make_data_scripts/README.md`](./dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/README.md). Описание корпусов (включая совмещённый корпус `dusha_resd` = Dusha + RESD) и полные правила их построения из `aggregated_dataset` — [`CORPUS.md`](../../CORPUS.md). Использование в экспериментах — [`my_experiments/README.md`](../my_experiments/README.md).


