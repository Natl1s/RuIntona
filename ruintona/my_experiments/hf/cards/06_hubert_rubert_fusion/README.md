---
language:
  - ru
license: cc-by-sa-4.0
base_model:
  - xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned
  - DeepPavlov/rubert-base-cased
tags:
  - speech-emotion-recognition
  - audio-classification
  - text-classification
  - russian
  - ser
library_name: custom
---

# Late fusion: HuBERT (audio) + RuBERT (text — наш fine-tune)

Late-fusion-классификатор эмоций русской речи: прогноз = взвешенная сумма вероятностей
двух независимых модальностей. **Отдельных обученных весов здесь нет** — артефакт
этого репозитория — константа слияния `α` и отчёт с метриками
(`late_fusion_hubert_rubert_weights_*.json`).

Эмоции: `angry`, `sad`, `neutral`, `positive` (маппинг `angry=0, sad=1, neutral=2, positive=3`).

## Формула слияния

```
probs_fused = α · probs_audio + (1 − α) · probs_text,   α = 0.5
```

- **audio**: `xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned`
  (основа — `facebook/hubert-large-ls960-ft`, Apache-2.0), прогноз по аудио.
- **text**: наш дообученный RuBERT — `Natlis/rubert-emotion-classification-ru`,
  прогноз по транскрипту.

## Метрики

| Split | Corpus | Accuracy | F1-macro |
|---|---|---|---|
| val | 20% от train | 0.880 | 0.887 |
| test | combine_balanced_test (в отчёте JSON) | 0.834 | 0.843 |
| test | dusha_resd_test (сводка README проекта) | 0.822 | 0.830 |

## Как использовать

Взять прогнозы обоих классификаторов и усреднить с весами `α=0.5`:

```python
probs_audio = hubert_model.predict_proba(audio_segment)      # (1, 4)
probs_text  = rubert_model.predict_proba(transcript)         # (1, 4)
probs = 0.5 * probs_audio + 0.5 * probs_text
emo = ["angry", "sad", "neutral", "positive"][int(probs.argmax())]
```

Полная воспроизводимая реализация — `multimodal/late_fusion/Late_Fusion_HuBERT_RuBERT.py`
в кодовом репозитории.

## Лицензия и атрибуция

**Отчёт и константа слияния — CC BY-SA 4.0** (получены на корпусе Dusha).
Базовые модели сохраняют собственные лицензии: HuBERT-SER — Apache-2.0,
наш RuBERT fine-tune — Apache-2.0. Полные условия — `LICENSE`, атрибуция — `NOTICE`.

## Цитирование

Dusha — Kondratenko et al., arXiv:2212.12266. RESD — Aniemore, DOI 10.57967/hf/1272.
HuBERT-SER — xbgoose на Hugging Face. Подробности — кодовый репозиторий.