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

# CNN-BiLSTM — классификатор эмоций русской речи (аудио)

Четырёхклассовый классификатор эмоций по аудио:
`angry`, `sad`, `neutral`, `positive` (маппинг `angry=0, sad=1, neutral=2, positive=3`).

- Архитектура: `EmotionCNNBiLSTM` — CNN (каналы 16/32/64) → BiLSTM
  (hidden 128, 2 слоя, dropout 0.2, bidirectional) → классификатор (dropout 0.3).
- Обучен на корпусе `combine_balanced_train` (Dusha, mel-спектрограммы).
- Вход: mel-спектрограмма (1, 64, T) из пайплайна `ruintona/data_processing/`.

## Метрики

Метрики получены оценкой обученного чекпоинта на `dusha_resd_test` (6616 записей)
в `model_analise/multimodal_models_analise.ipynb`.

| Split | Corpus | Accuracy | F1-macro |
|---|---|---|---|
| test | dusha_resd (перекрёстная оценка) | 0.740 | 0.732 |

## Как использовать

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "Natlis/cnn-bilstm-emotion-classification-ru",
    "CNN_BiLSTM_combine_balanced_train_model.pt",
)

# Чекпоинт — единый формат dict (utils/model_io.py проекта):
ckpt = torch.load(path, map_location="cpu")
print(ckpt["model_class"], ckpt["model_params"])
```

Вход — готовые mel-спектрограммы (1, 64, T), которые строит пайплайн проекта
(`ruintona/data_processing/`, ресемпл аудио в 16 кГц моно). Архитектура и загрузка —
`audio_models/CNN/CNN_BiLSTM.py`.

## Обучение

Полные гиперпараметры — в `hyperparams.json` и `configs/audio/cnn_bilstm.json`
(epochs 30, batch 32, lr 1e-3, weight_decay 1e-5, seed 42).

## Ограничения

- Признаки вычисляются на этапе сборки корпуса; для новых записей нужен пайплайн проекта.
- Эмоции размечены краудсорсингом (Dawid-Skene, порог 0.9); переносимость на спонтанную речь
  вне дистрибутива не гарантируется.

## Лицензия и атрибуция

**Веса лицензированы CC BY-SA 4.0** (модель обучена с нуля на корпусе Dusha).
Полные условия — `LICENSE`, атрибуция и данные обучения — `NOTICE`.

## Цитирование

Dusha — Kondratenko et al., arXiv:2212.12266. Подробности — `DUSHA.md` кодового репозитория.