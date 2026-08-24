# Датасет Dusha

> Примечание: эта страница — описание датасета Dusha с точки зрения этого репозитория: что это за данные, где скачивать и как они здесь используются. Авторитетная информация — на официальной странице проекта [Salute Developers — Golos](https://github.com/salute-developers/golos/tree/master/dusha) (см. раздел [Официальные источники](#официальные-источники)).

Dusha — **бимодальный корпус** русской речи, предназначенный для задач распознавания эмоций (Speech Emotion Recognition, SER). Датасет состоит примерно из **300 000 аудиозаписей** (~350 часов) с транскриптами и эмоциональными метками и на момент публикации являлся крупнейшей открытой бимодальной коллекцией для SER. Эмоции соответствуют четырём базовым классам, типичным для диалога с виртуальным ассистентом: **позитив (счастье), грусть, злость и нейтральное состояние**.

> **Примечание о производных корпусах.** Эксперименты в этом репозитории в основном проводятся на совмещённом корпусе, построенном из Dusha и датасета [RESD](RESD.md) (Aniemore, Hugging Face). Описание корпусов и правила их сборки — в [CORPUS.md](CORPUS.md).

> Сводка источников всех сторонних компонентов проекта (модели + датасеты) — [`SOURCES.md`](./SOURCES.md).

## Официальные источники

| Источник | Ссылка |
|---|---|
| GitHub-репозиторий проекта (Salute Developers / Сбер) | <https://github.com/salute-developers/golos> |
| Раздел Dusha в репозитории (README датасета) | <https://github.com/salute-developers/golos/tree/master/dusha> |
| Статья «Large Raw Emotional Dataset with Aggregation Mechanism» (arXiv) | <https://arxiv.org/abs/2212.12266> |
| DOI статьи | <https://doi.org/10.48550/arXiv.2212.12266> |
| Статья Golos «Golos: Russian dataset for speech research» | <https://arxiv.org/abs/2106.10161> |
| Лицензия датасета Dusha/Golos (EN) | [`license/en_us.pdf`](./license/en_us.pdf) · <https://github.com/salute-developers/golos/blob/master/license/en_us.pdf> |
| Лицензия датасета Dusha/Golos (RU) | [`license/ru.pdf`](./license/ru.pdf) · <https://github.com/salute-developers/golos/blob/master/license/ru.pdf> |

## Структура датасета

| Домен | Кол-во файлов | Длительность (ч.) | Уникальных спикеров |
|---|---|---|---|
| Crowd (acted) | 201 850 | 255.7 | 2068 |
| Podcast (real-life) | 102 113 | 90.9 | 6240 |
| Итого | 303 963 | 346.6 | 8308 |

Датасет включает два подмножества:

- **Crowd** — «сыгранная» речь (acted), с более сбалансированным распределением классов; подходит для предобучения моделей.
- **Podcast** — «естественная» речь (real-life) из подкастов, с несбалансированным распределением; используется для тонкой настройки и валидации.

## Разметка

- Аннотация выполнена на краудсорсинговой платформе; каждая запись размечена несколькими аннотаторами.
- Итоговые метки агрегируются механизмом **Dawid-Skene** (учёт компетентности аннотаторов).
- В этом репозитории используется порог уверенности **0.9** → результат обработки лежит в `processed_dataset_090` (см. [`ruintona/data_processing/README.md`](ruintona/data_processing/README.md)).

## Скачивание

> **Важно про аудио подкастов.** В связи с лицензионными ограничениями официальный дистрибутив не содержит аудиофайлов подкастов — вместо них предоставляются предвычисленные признаки и ссылки на оригинальные подкасты с таймингами (см. [Issue #1](https://github.com/salute-developers/golos/issues/1); там же есть стороннее зеркало `podcast_wavs.tar.gz`, 8.03 GB, md5 `31283c7747c30685eddb451690a4cc73`).

| Архив | Размер | Ссылка (официальный CDN) |
|---|---|---|
| `crowd.tar` | 28 GB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/crowd.tar> |
| `podcast.tar` | 360 MB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/podcast.tar> |
| `features.tar` | 30 GB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/features.tar> |
| `paper_setups.tgz` | 16 MB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/paper_setups.tgz> |

## Использование в этом репозитории

Пайплайн обработки сырых данных: mel-спектрограммы (`features/*.npy`), агрегация разметки Dawid-Skene, агрегированные манифесты `crowd_{train,test}.jsonl` / `podcast_{train,test}.jsonl` → сбалансированные наборы → базы LMDB. Описание цепочки сборки и корпусов — в [CORPUS.md](CORPUS.md).

## Атрибуция в этом репозитории

Этот репозиторий содержит:

- **Адаптированный материал** из оригинального проекта [Salute Developers — Golos/Dusha](https://github.com/salute-developers/golos):
  - пайплайн `ruintona/data_processing/` (обработка сырых данных, извлечение признаков, агрегация меток Dawid-Skene),
  - структуру датасета и производные артефакты датасета (признаки, манифесты, базы LMDB).
- **Оригинальную работу** автора этого репозитория: всё остальное в `ruintona/my_experiments/`, `ruintona/DEMO/`, `ruintona/configs/` — лицензировано отдельно (см. [LICENSE](./LICENSE)).

Датасет и адаптированный код предоставляются по лицензии Dusha/Golos (см. [английскую версию](./license/en_us.pdf) / [русскую версию](./license/ru.pdf)): **с указанием авторства и оговоренными условиями** (Share-Alike). При распространении самого датасета или производных артефактов данных сохраняйте указанную выше атрибуцию и текст лицензии.

## Цитирование

```bibtex
@misc{kondratenko2022large,
  title = {Large Raw Emotional Dataset with Aggregation Mechanism},
  author = {Vladimir Kondratenko and Artem Sokolov and Nikolay Karpov
            and Oleg Kutuzov and Nikita Savushkin and Fyodor Minkin},
  year = {2022},
  eprint = {2212.12266},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  url = {https://arxiv.org/abs/2212.12266}
}
```
