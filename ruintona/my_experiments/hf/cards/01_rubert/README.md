---
language:
  - ru
license: apache-2.0
base_model: DeepPavlov/rubert-base-cased
tags:
  - speech-emotion-recognition
  - text-classification
  - russian
  - ser
pipeline_tag: text-classification
library_name: custom
---

# RuBERT — классификатор эмоций русской речи (текст)

Четырёхклассовый классификатор эмоционального тона реплики по транскрипту:
`angry`, `sad`, `neutral`, `positive` (маппинг `angry=0, sad=1, neutral=2, positive=3`).

- Архитектура: `DeepPavlov/rubert-base-cased` → [CLS] → Linear → GELU → Dropout → Linear.
- Обучение в два этапа: 1 эпоха — весь BERT заморожен, обучается только head;
  далее — разморожены слои `encoder.layer.6–11`.
- Обучен на корпусе `dusha_resd_train` (69 119 реплик) = Dusha + RESD.
- Вход: транскрипт реплики, токенизация `AutoTokenizer` (папка токенизатора в этом репозитории).

## Метрики

Оценка на `dusha_resd_test` (6616 реплик), усреднение по 5 сидам.

| Split | Corpus | Accuracy | F1-macro |
|---|---|---|---|
| test | dusha_resd | 0.586 | 0.601 |

## Как использовать

```python
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

ckpt_path = hf_hub_download(
    "Natlis/rubert-emotion-classification-ru",
    "RuBERT_dusha_resd_train_model.pt",
)
tok_dir = snapshot_download(
    "Natlis/rubert-emotion-classification-ru",
    allow_patterns=["RuBERT_dusha_resd_train_tokenizer/*"],
)
tokenizer = AutoTokenizer.from_pretrained(tok_dir)

# Чекпоинт — единый формат dict (см. utils/model_io.py репозитория проекта):
ckpt = torch.load(ckpt_path, map_location="cpu")
print(ckpt["model_class"], ckpt["model_params"])
```

Для воспроизведения архитектуры и предсказаний используйте код проекта
(https://github.com/Natl1s/RuIntona), `text_models/transformers/RuBERT.py`:
модель `EmotionClassifier` воссоздаётся из `ckpt["model_params"]`.

## Обучение

Полные гиперпараметры — в `hyperparams.json` и `configs/text/rubert.json` проекта
(seed 42, lr 2e-5, warmup 0.1, batch 16, grad-accum 8, max_len 128, label_smoothing 0.05).

## Ограничения

- Модель обучена на размеченных в краудсорсинге и сыгранных эмоциях; переносимость
  на спонтанную речь вне дистрибутива не гарантируется.
- Качество зависит от качества ASR-транскрипта реплики.

## Лицензия и атрибуция

**Веса лицензированы Apache-2.0** (бэкбон `DeepPavlov/rubert-base-cased`, Apache-2.0).
Полный текст — `LICENSE`, обязательства по атрибуции и данные обучения — `NOTICE`.

## Цитирование

Dusha — Kondratenko et al., arXiv:2212.12266. RESD — Aniemore, DOI 10.57967/hf/1272.
Подробности и bibtex — в `DUSHA.md` / `RESD.md` кодового репозитория.