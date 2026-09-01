# Демо (`ruintona/DEMO/`)

Интерактивная демонстрация инференса обученных моделей на одном примере: **аудио** (mel-спектрограмма → CNN-BiLSTM) + **текст** (RuBERT) → soft-voting (**late-fusion**).

> Источники и лицензии сторонних предобученных моделей и датасетов — [`SOURCES.md`](../../SOURCES.md).

## Содержимое

| Файл | Назначение |
|---|---|
| `demo.ipynb` | Ноутбук-демо: мультимодальный инференс, сравнение модальностей, визуализация |
| `data/001ce26c07c20eaa0d666b824c6c6924.wav` | Пример аудиозаписи |
| `data/example1.json` | Пример транскрипта к аудио |
| `results/demo_results.png` | Визуализация вероятностей классов по модальностям |

> **Лицензия данных.** Сэмпл в `data/*.wav` и транскрипт `example1.json` — пример из датасета Dusha и распространяются по лицензии Dusha/Golos (attribution + share-alike), см. [`DUSHA.md`](../../DUSHA.md). Код демо — MIT (`LICENSE`).

## Запуск

```bash
poetry run jupyter notebook ruintona/DEMO/demo.ipynb
```

Первая загрузка моделей занимает ~1–2 минуты (RuBERT ~714 MB), далее модели кэшируются. Если чекпоинтов нет в `ruintona/my_experiments/checkpoints/`, они **автоматически скачиваются с Hugging Face** (коллекция **RuIntona SER**, кэш — `checkpoints/hf/`); отключить скачивание можно флагом `--no-download`.

## Из командной строки

Тот же инференс доступен из `ruintona/my_experiments/inference.py` и для **ваших собственных** аудио/текста. Отсутствующие веса скачиваются с Hugging Face автоматически.

```bash
# Мультимодально (аудио + текст, late-fusion)
poetry run python ruintona/my_experiments/inference.py --model late-fusion \
    --audio ruintona/DEMO/data/001ce26c07c20eaa0d666b824c6c6924.wav \
    --text "шестьдесят тысяч тенге сколько будет стоить"

# Только аудио (укажите путь к своему .wav/.mp3)
poetry run python ruintona/my_experiments/inference.py --model audio --audio /path/to/your.wav

# Только текст
poetry run python ruintona/my_experiments/inference.py --model text --text "я очень рад сегодня"

# Без скачивания весов (только локальные чекпоинты)
poetry run python ruintona/my_experiments/inference.py --model audio --audio /path/to/your.wav --no-download
```

Справка по CLI и реестр моделей — `--help` и [`my_experiments/README.md`](../my_experiments/README.md).
