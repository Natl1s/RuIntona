# Model analysis (`my_experiments/model_analise/`)

Jupyter notebooks with model result analysis + evaluation results of pretrained (foundation) models.

## Notebooks

| Notebook | What it analyzes |
|---|---|
| `audio_models_analise.ipynb` | Metrics, confusion matrices, error analysis of audio models |
| `random_forest_hyperparameter_tuning.ipynb` | Overfitting diagnostics and fighting it: OOB curve, `max_depth`/`min_samples_leaf`/`ccp_alpha`, RandomizedSearchCV, baseline vs tuned comparison |
| `text_models_analise.ipynb` | Metrics and errors of text models |
| `multimodal_models_analise.ipynb` | Analysis of late/early fusion, impact of the α weight, etc. |
| `pretrained_models_analise.ipynb` | Description and metrics of the pretrained (foundation) models on `dusha_resd_test`: HuBERT-large, Whisper-large-v3, WavLM-BERT fusion |

## Text model metrics

Text models are evaluated on `dusha_resd_test` (6616 samples) in the `text_models_analise.ipynb` notebook:

| Model | Train | Test Acc | F1-macro |
|---|---|---|---|
| TF-IDF + LogReg | `combine_balanced` | 0.540 | 0.556 |
| Embeddings (FastText) + LogReg | `combine_balanced` | 0.531 | 0.541 |
| BiLSTM | `combine_balanced` | 0.560 | 0.580 |
| **RuBERT** | **`dusha_resd`** | **0.586** | **0.601** |

## Evaluation of pretrained models

Evaluation scripts and checkpoints (`checkpoints/pretrained/`) are not part of the repository (`.gitignore`), but the aggregated evaluation results (`*_eval_*.json`) are committed. The evaluation was run on `dusha_resd_test` (6616 samples, audio modality; see corpus composition in [`CORPUS.md`](../../../CORPUS.md)). Detailed description of each model and its metrics — in the `pretrained_models_analise.ipynb` notebook:

| Model | Source | Test Acc | F1-macro |
|---|---|---|---|
| Whisper-large-v3 | [`firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) | 0.435 | 0.345 |
| WavLM-BERT fusion | [`Aniemore/wavlm-bert-fusion-s-emotion-russian-resd`](https://huggingface.co/Aniemore/wavlm-bert-fusion-s-emotion-russian-resd) | 0.552 | 0.503 |
| **HuBERT-large (Dusha-finetuned)** | [`xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned`](https://huggingface.co/xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned) | **0.805** | **0.815** |

HuBERT embeddings are further used in late fusion (see [`multimodal/README.md`](../multimodal/README.md)).

## Run

```bash
poetry run jupyter notebook ruintona/my_experiments/model_analise/
```

Notebooks read metrics from the `results/` files of the model groups and `checkpoints/pretrained/*_eval_*.json`.
