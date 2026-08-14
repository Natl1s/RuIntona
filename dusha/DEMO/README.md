# Демо (`dusha/DEMO/`)

Интерактивная демонстрация инференса обученных моделей на одном примере: **аудио** (mel-спектрограмма → CNN-BiLSTM) + **текст** (RuBERT) → soft-voting (**late-fusion**).

## Содержимое

| Файл | Назначение |
|---|---|
| `demo.ipynb` | Ноутбук-демо: мультимодальный инференс, сравнение модальностей, визуализация |
| `data/001ce26c07c20eaa0d666b824c6c6924.wav` | Пример аудиозаписи |
| `data/example1.json` | Пример транскрипта к аудио |
| `results/demo_results.png` | Визуализация вероятностей классов по модальностям |

## Запуск

```bash
poetry run jupyter notebook dusha/DEMO/demo.ipynb
```

Первая загрузка моделей занимает ~1–2 минуты (RuBERT ~714 MB), далее модели кэшируются. Требуются обученные чекпоинты: CNN-BiLSTM (`checkpoints/audio/`) и RuBERT (`checkpoints/text/`).

## Из командной строки

Тот же инференс доступен из CLI — см. `dusha/my_experiments/inference.py` и [`my_experiments/README.md`](../my_experiments/README.md).
