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

# SVM (RBF) — классификатор эмоций русской речи (аудио, базовая модель)

Четырёхклассовый классификатор эмоций по audio-функционалам:
`angry`, `sad`, `neutral`, `positive` (маппинг `angry=0, sad=1, neutral=2, positive=3`).

- Признаки: статистические характеристики аудио (питч, энергия и др.).
- Модель: `sklearn.svm.SVC` (kernel `rbf`, C=1.0, gamma `scale`).
- Обучен на корпусе `dusha_resd_train` (69 136 записей) = Dusha + RESD.

## Метрики

Оценка на `dusha_resd_test` (6616 записей) — из training-отчёта эксперимента.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| angry | 0.521 | 0.441 | 0.478 | 1378 |
| sad | 0.581 | 0.656 | 0.616 | 2213 |
| neutral | 0.377 | 0.385 | 0.381 | 1730 |
| positive | 0.546 | 0.499 | 0.521 | 1295 |
| **Accuracy** | | | **0.510** | 6616 |
| **F1-macro** | | | **0.499** | |

## Как использовать

```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    "Natlis/svm-emotion-classification-ru",
    "svm_dusha_resd_train_model.pkl",
)
scaler_path = hf_hub_download(
    "Natlis/svm-emotion-classification-ru",
    "svm_dusha_resd_train_scaler.pkl",
)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 1) извлечь те же признаки из аудио (те же, что в пайплайне проекта),
# 2) стандартизировать:  x = scaler.transform(feats)
pred = model.predict(x)
```

## Обучение

Полные гиперпараметры — в `hyperparams.json` и `configs/audio/svm.json`
(kernel `rbf`, C=1.0, gamma `scale`, seed 42).

## Ограничения

- Базовая модель (baseline); признаки вычисляются на этапе сборки корпуса.
- Эмоции размечены краудсорсингом; переносимость на спонтанную речь вне дистрибутива не гарантируется.

## Лицензия и атрибуция

**Веса лицензированы CC BY-SA 4.0** (модель обучена на корпусе dusha_resd).
Полные условия — `LICENSE`, атрибуция и данные обучения — `NOTICE`.

## Цитирование

Dusha — Kondratenko et al., arXiv:2212.12266. RESD — Aniemore, DOI 10.57967/hf/1272.
Подробности — `DUSHA.md` / `RESD.md` кодового репозитория.