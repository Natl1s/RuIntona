# Мультимодальные модели (`my_experiments/multimodal/`)

Совместная классификация эмоций по аудио и тексту. Лучший результат по проекту: **late fusion HuBERT + RuBERT (α = 0.5) → Test Acc 0.822, F1-macro 0.830 на `dusha_resd_test`** (вес α=0.5 подобран на `combine_balanced`).

## Модели

| Скрипт | Подход | Компоненты | Выход |
|---|---|---|---|
| `late_fusion/Late_Fusion_CNN_BiLSTM_RuBERT.py` | **Late fusion** — взвешенная сумма вероятностей | CNN-BiLSTM (audio) + RuBERT (text) | перебор веса α на валидации |
| `late_fusion/Late_Fusion_Baseline_SVM_TF_IDF.py` | Late fusion (базлайн) | SVM (audio) + TF-IDF LogReg (text) | перебор веса α |
| `early_fusion/Early_Fusion_Baseline.py` | **Early fusion** — конкатенация признаков + LSTM | CNN-BiLSTM (audio) + RuBERT (text) → projection → LSTM → Linear | единая сеть |
| `co-attention/Co_Attention_Baseline.py` | Co-attention | Wav2Vec2 XLS-R 300M (audio) + RuBERT (text) | cross-attention |

## Быстрый запуск

```bash
# Late fusion CNN-BiLSTM + RuBERT (нужны чекпоинты CNN-BiLSTM и RuBERT)
poetry run python dusha/my_experiments/multimodal/late_fusion/Late_Fusion_CNN_BiLSTM_RuBERT.py --mode auto \
    --audio-model-path checkpoints/audio/CNN_BiLSTM_combine_balanced_train_model.pt \
    --text-model-path checkpoints/text/RuBERT_dusha_resd_train_model.pt

# Late fusion базлайн (SVM + TF-IDF LogReg)
poetry run python dusha/my_experiments/multimodal/late_fusion/Late_Fusion_Baseline_SVM_TF_IDF.py --mode auto

# Early fusion базлайн
poetry run python dusha/my_experiments/multimodal/early_fusion/Early_Fusion_Baseline.py --mode train

# Co-attention (занимает много GPU-памяти, batch-size 4)
poetry run python dusha/my_experiments/multimodal/co-attention/Co_Attention_Baseline.py --mode train
```

Результаты сохраняются в `results/*.json` каждой подпапки.

## Ключевые флаги

- Общие: `--mode {train,load,auto,smoke}`, `--audio-model-path`, `--text-model-path`, `--results-dir`, `--batch-size`, `--val-size`, `--seed`, `--device`.
- Late fusion (вес α): `--alpha-step` (шаг перебора, по умолч. 0.05); `Late_Fusion_CNN_BiLSTM_RuBERT.py` — ещё `--audio-model-path`, `--text-model-path`.
- `Late_Fusion_Baseline_SVM_TF_IDF.py`: `--audio-scaler-path`, `--text-vectorizer-path`.
- `Early_Fusion_Baseline.py`: `--epochs`, `--lr`, `--weight-decay`, `--projection-dim`, `--dropout`, `--max-len`, `--audio-lstm-hidden-size`, `--audio-lstm-layers`, `--audio-unidirectional`, `--grad-clip-norm`.
- `Co_Attention_Baseline.py`: `--audio-pretrained-name`, `--audio-warm-start-path`, `--text-max-len`, `--audio-max-length`.

## Результаты

| Модель | Датасет | Test Acc | F1-macro |
|---|---|---|---|
| Early fusion (CNN-BiLSTM + RuBERT) | dusha_resd | 0.795 | 0.795 |
| Late fusion CNN-BiLSTM + RuBERT (α=0.5) | dusha_resd | 0.790 | 0.786 |
| **Late fusion HuBERT + RuBERT (α=0.5)** | **dusha_resd** | **0.822** | **0.830** |
| Late fusion базлайн (SVM + TF-IDF LogReg, α=0.35) | dusha_resd | 0.621 | 0.629 |

Все 4 мультимодальные модели оценены на `dusha_resd_test` (6616 сэмплов) в ноутбуке `multimodal_models_analise.ipynb`, поэтому напрямую сравнимы. Аудио/текст-бэкбоны CNN-BiLSTM и HuBERT+RuBERT (CNN-BiLSTM, HuBERT, RuBERT) и веса α=0.5 обучены/подобраны на `combine_balanced`. HuBERT — предобученная foundation-модель (описание и источник — [`model_analise/README.md`](../model_analise/README.md)). Состав корпусов и правила их сборки — [`CORPUS.md`](../../../CORPUS.md). Полные отчёты — в `results/` (val/test, confusion matrices, history по эпохам).
