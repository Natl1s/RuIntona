# Датасет RESD (Aniemore / Russian Emotional Speech Dialogues)

> Примечание: эта страница — краткое описание датасета со ссылкой на первоисточник. Авторитетная информация — на странице Hugging Face: <https://huggingface.co/datasets/Aniemore/resd_annotated> (DOI [10.57967/hf/1272](https://doi.org/10.57967/hf/1272)).

RESD — **Russian Emotional Speech Dialogues** — русский датасет для распознавания эмоций в речи, опубликованный командой [Aniemore](https://huggingface.co/Aniemore). В этом репозитории используется версия `resd_annotated` как дополнительный источник для расширения корпуса [Dusha](DUSHA.md) (см. [CORPUS.md](CORPUS.md)).

## Описание

- Записан в студии **20 актёрами озвучивания**.
- Без сценария: каждый актёр в паре получал приватно одну эмоцию для передачи, а диалог импровизировался. Слова спонтанны, а эмоция намеренна — в этом суть датасета и одновременно его ограничение (сыгранная эмоция — не спонтанная).
- 7 классов эмоций: `anger`, `happiness`, `sadness`, `neutral`, `disgust`, `fear`, `enthusiasm`.
- Язык: русский.

## Разбиение

| Сплит | Записей | Часов | Средняя длина |
|---|---|---|---|
| `train` | 1116 | 1.88 | 6.1 с |
| `test` | 280 | 0.46 | 5.9 с |

## Поля

| Колонка | Значение |
|---|---|
| `name` | Идентификатор клипа (например, `32_happiness_enthusiasm_h_120`) |
| `path` | Исходный путь к файлу |
| `speech` | Аудио |
| `text` | Транскрипт реплики |
| `emotion` | Метка эмоции записи |

> **Важно про частоту дискретизации.** Она неоднородна: в `train` 565 клипов — 16000 Гц и 551 — 44100 Гц. Всегда приводите аудио к единой частоте (например, 16 кГц) перед подачей в экстрактор признаков. Пайплайн в этом репозитории ресемплирует всё в 16 кГц.

## Использование в этом репозитории

- Оставляются только записи, чья эмоция отображается в 4 целевых класса: `anger → angry`, `happiness → positive`, `sadness → sad`, `neutral → neutral`. Остальные классы RESD (`disgust`, `fear`, `enthusiasm`) исключаются.
- Отобранные записи объединяются с корпусом Dusha в корпус `dusha_resd`: `dusha_resd_train.lmdb` / `dusha_resd_test.lmdb`. См. [CORPUS.md](CORPUS.md) и `dusha/data_processing/dataset/hug_dataset/make_raw.py`.

## Использование

```python
from datasets import load_dataset

ds = load_dataset("Aniemore/resd_annotated")
print(ds["train"][0]["emotion"])
```

## Цитирование

```bibtex
@misc{Aniemore,
  author = {Артем Аментес, Илья Лубенец, Никита Давидчук},
  title = {Открытая библиотека искусственного интеллекта для анализа и выявления эмоциональных оттенков речи человека},
  year = {2022},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.com/aniemore/Aniemore}},
  email = {hello@socialcode.ru}
}
```

## Лицензия

MIT.
