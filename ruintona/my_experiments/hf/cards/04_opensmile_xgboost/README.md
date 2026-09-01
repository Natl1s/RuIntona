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

# openSMILE + XGBoost — классификатор эмоций русской речи (аудио, базовые признаки)

Четырёхклассовый классификатор эмоций по audio-функционалам:
`angry`, `sad`, `neutral`, `positive` (маппинг `angry=0, sad=1, neutral=2, positive=3`).

- Признаки: **OpenSMILE eGeMAPSv02 Functionals** (88 функционалов).
- Модель: XGBoost (n_estimators 500, learning_rate 0.05, max_depth 6, seed 42).
- Обучен на корпусе `dusha_resd_train` (69 253 записи) = Dusha + RESD.

## Метрики

Оценка на `dusha_resd_test` (6616 записей) — из training-отчёта эксперимента.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| angry | 0.612 | 0.579 | 0.595 | 1378 |
| sad | 0.684 | 0.709 | 0.697 | 2213 |
| neutral | 0.502 | 0.514 | 0.508 | 1730 |
| positive | 0.569 | 0.547 | 0.558 | 1295 |
| **Accuracy** | | | **0.600** | 6616 |
| **F1-macro** | | | **0.590** | |

Также: ROC-AUC (ovo, macro) 0.838, MCC 0.456, log-loss 0.933.

## Как использовать

```python
import joblib
import opensmile
import numpy as np
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    "Natlis/opensmile-xgboost-emotion-classification-ru",
    "openSmile_XGBoost_dusha_resd_train_model.pkl",
)
scaler_path = hf_hub_download(
    "Natlis/opensmile-xgboost-emotion-classification-ru",
    "openSmile_XGBoost_dusha_resd_train_scaler.pkl",
)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
feats = smile.process_file("audio.wav")            # pd.Series, 88 функционалов
proba = model.predict_proba(
    scaler.transform(np.asarray(feats).reshape(1, -1))
)
```

Стандартизация признаков (`scaler`) обязательна перед подачей в модель.
Порядок признаков — стандартный порядок eGeMAPSv02 Functionals.

## Обучение

Полные гиперпараметры — в `hyperparams.json` и `configs/audio/opensmile_xgboost.json`.
Источник признаков: eGeMAPS-кэш `combine_balanced` + извлечение по записям RESD.

## Ограничения

- Модель чувствительна к особенностям экстрактора OpenSMILE (версия, версии дел).
- Эмоции размечены краудсорсингом; переносимость на спонтанную речь вне дистрибутива не гарантируется.

## Лицензия и атрибуция

**Веса лицензированы CC BY-SA 4.0** (модель обучена на корпусе dusha_resd).
Полные условия — `LICENSE`, атрибуция и данные обучения — `NOTICE`.

## Цитирование

Dusha — Kondratenko et al., arXiv:2212.12266. RESD — Aniemore, DOI 10.57967/hf/1272.
Подробности — `DUSHA.md` / `RESD.md` кодового репозитория.