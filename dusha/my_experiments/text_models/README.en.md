# Text models (`my_experiments/text_models/`)

Emotion classification from text (transcripts). Data is read from LMDB (text keys: `speaker_text` / `text` / `transcript` / `utterance`).

## Models

| Model | Features | Architecture | README |
|---|---|---|---|
| `TF-IDF_LogReg.py` | TF-IDF (1–2 grams) | Logistic Regression | [baseline/README.md](./baseline/README.md) |
| `Embeddings_LogReg.py` | FastText `cc.ru.300.bin` (mean vector) | Logistic Regression | [baseline/README.md](./baseline/README.md), [EMBEDDINGS_SETUP.md](./baseline/EMBEDDINGS_SETUP.md) |
| `BiLSTM/BiLSTM.py` | FastText embeddings (frozen matrix) | Embedding → BiLSTM → Linear | — |
| `transformers/RuBERT.py` | DeepPavlov `rubert-base-cased` tokens | RuBERT + classifier | — |
| `baseline/example_usage.py` | — | example of baseline usage | — |

## Quick start

```bash
# TF-IDF + Logistic Regression
poetry run python dusha/my_experiments/text_models/baseline/TF-IDF_LogReg.py --mode train

# FastText + Logistic Regression
poetry run python dusha/my_experiments/text_models/baseline/Embeddings_LogReg.py --mode train

# FastText + BiLSTM
poetry run python dusha/my_experiments/text_models/BiLSTM/BiLSTM.py --mode train

# RuBERT
poetry run python dusha/my_experiments/text_models/transformers/RuBERT.py --mode train --config text/rubert.json
```

## Key flags

- Common: `--mode {train,load,auto,smoke}`, `--config`, `--train-data-path`, `--test-data-path`, `--no-save`.
- `BiLSTM.py`: `--epochs`, `--batch-size`, `--max-len`, `--hidden-size`, `--num-layers`, `--max-vocab-size`, `--min-freq`.
- `RuBERT.py`: `--backbone-name`, `--epochs`, `--stage1-epochs`, `--batch-size`, `--max-len`, `--lr`, `--loss-name {ce,focal}`, `--label-smoothing`, `--fp16`.

## Pretrained FastText

`Embeddings_LogReg.py` and `BiLSTM.py` require the `cc.ru.300.bin` model (2.3 GB). It is downloaded automatically via `utils/pretrained.load_fasttext_model()` into `checkpoints/pretrained/fasttext/` or manually — see [baseline/EMBEDDINGS_SETUP.md](./baseline/EMBEDDINGS_SETUP.md).

## Artifacts

- sklearn: `{Model}_{dataset}_model.pkl` + `{...}_vectorizer.pkl` / `{...}_scaler.pkl`.
- PyTorch: `{Model}_{dataset}_model.pt` (+ timestamped backups, vocab, meta).
- Save folder — `checkpoints/text/`.

## Results

Text model metrics are recorded in the [`model_analise/text_models_analise.ipynb`](../model_analise/text_models_analise.ipynb) notebook. All models are evaluated on `dusha_resd_test` (6616 samples):

| Model | Train | Test | Test Acc | F1-macro |
|---|---|---|---|---|
| TF-IDF + LogReg | `combine_balanced` | `dusha_resd` | 0.540 | 0.556 |
| Embeddings (FastText) + LogReg | `combine_balanced` | `dusha_resd` | 0.531 | 0.541 |
| BiLSTM | `combine_balanced` | `dusha_resd` | 0.560 | 0.580 |
| **RuBERT** | **`dusha_resd`** | **`dusha_resd`** | **0.586** | **0.601** |

Corpus composition — [`CORPUS.md`](../../../CORPUS.md). Audio/text baselines of other modalities and multimodal results — see [`audio_models/README.md`](../audio_models/README.md) and [`multimodal/README.md`](../multimodal/README.md).

The analyzed TF-IDF/Embeddings/BiLSTM checkpoints were trained on `combine_balanced`, RuBERT on `dusha_resd` (see the notebook). By default the text model scripts (including RuBERT, checkpoint `RuBERT_dusha_resd_train_model.pt`) are trained on the combined `dusha_resd` corpus (Dusha + RESD).
