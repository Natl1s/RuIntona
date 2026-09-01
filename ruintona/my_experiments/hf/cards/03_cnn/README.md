---
language:
  - ru
license: cc-by-sa-4.0
tags:
  - speech-emotion-recognition
  - audio-classification
  - russian
  - ser
pipeline_tag: audio-classification
library_name: custom
---

# CNN — классификатор эмоций русской речи (аудио, базовая модель)

Четырёхклассовый классификатор эмоций по аудио:
`angry`, `sad`, `neutral`, `positive` (маппинг `angry=0, sad=1, neutral=2, positive=3`).

- Архитектура: `EmotionCNN` — CNN (каналы 16/32/64) → классификатор (dropout 0.2).
- Обучен на корпусе `combine_balanced_train` (Dusha, mel-спектрограммы).
- Вход: mel-спектрограмма (1, 64, T) из пайплайна `ruintona/data_processing/`.

## Метрики

Метрики получены оценкой обученного чекпоинта на `dusha_resd_test` (6616 записей)
в `model_analise/audio_models_analise.ipynb`.

| Split | Corpus | Accuracy | F1-macro |
|---|---|---|---|
| test | dusha_resd (перекрёстная оценка) | 0.571 | 0.564 |

## Как использовать

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "Natlis/cnn-emotion-classification-ru",
    "CNN_combine_balanced_train_model.pt",
)

# Чекпоинт — единый формат dict (utils/model_io.py проекта):
ckpt = torch.load(path, map_location="cpu")
print(ckpt["model_class"], ckpt["model_params"])
```

Вход — готовые mel-спектрограммы (1, 64, T) из пайплайна проекта.
Архитектура и загрузка — `audio_models/CNN/CNN.py`.

## Обучение

Полные гиперпараметры — в `hyperparams.json` и `configs/audio/cnn.json`
(epochs 5, batch 16, lr 1e-3, weight_decay 1e-5, seed 42).

## Ограничения

- Базовая модель (baseline); признаки вычисляются на этапе сборки корпуса.
- Эмоции размечены краудсорсингом (Dawid-Skene, порог 0.9).

## Лицензия и атрибуция

**Веса лицензированы CC BY-SA 4.0** (модель обучена с нуля на корпусе Dusha).
Полные условия — `LICENSE`, атрибуция и данные обучения — `NOTICE`.

## Цитирование

Dusha — Kondratenko et al., arXiv:2212.12266. Подробности — `DUSHA.md` кодового репозитория.