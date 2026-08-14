# Demo (`ruintona/DEMO/`)

Interactive demo of inference of trained models on a single example: **audio** (mel-spectrogram → CNN-BiLSTM) + **text** (RuBERT) → soft-voting (**late-fusion**).

## Contents

| File | Purpose |
|---|---|
| `demo.ipynb` | Demo notebook: multimodal inference, modality comparison, visualization |
| `data/001ce26c07c20eaa0d666b824c6c6924.wav` | Sample audio recording |
| `data/example1.json` | Sample transcript for the audio |
| `results/demo_results.png` | Visualization of class probabilities per modality |

## Run

```bash
poetry run jupyter notebook ruintona/DEMO/demo.ipynb
```

First model load takes ~1–2 minutes (RuBERT ~714 MB), then models are cached. Trained checkpoints are required: CNN-BiLSTM (`checkpoints/audio/`) and RuBERT (`checkpoints/text/`).

## From the CLI

The same inference is available from the CLI — see `ruintona/my_experiments/inference.py` and [`my_experiments/README.md`](../my_experiments/README.md).
