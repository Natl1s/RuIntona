# Data analysis (`my_experiments/data_analise/`)

Jupyter notebooks with exploratory data analysis (EDA) of the Dusha dataset and corpus research.

| Notebook | What it analyzes |
|---|---|
| `corpus_statistics.ipynb` | Statistics of key LMDB corpora (`combine_balanced`, `combine_balanced_small`, `dusha_resd`) and **corpus justification**: emotion distribution, corpus overlap by ID, quality and fairness of train/test splits (χ² on emotions, KS on durations) |
| `dusha_resd_analyse.ipynb` | Research on the `dusha_resd` corpus (Dusha + RESD): composition, sources, emotion distribution, overlap with Dusha |
| `dusha_resd_leak_audio_check.ipynb` | Leak check between train/test in the `dusha_resd` corpus by audio identifiers |
| `text_analise.ipynb` | Text features: transcript length, emotion-specific vocabulary, typical words |
| `audio_analise.ipynb` | Audio features: durations, spectral characteristics, waveform/spectrogram visualization |

## Run

```bash
poetry run jupyter notebook ruintona/my_experiments/data_analise/
```

Notebooks read data from JSONL/CSV manifests (see `data_processing/dataset/processed_dataset_090/aggregated_dataset/`) and/or LMDB corpora (see `CORPUS.md`). Data paths are set at the top of each notebook and should be adjusted to the local dataset location (see `my_experiments/data.json`).

## Output files

The current notebooks in this directory are interactive and **do not write files** on execution — results are shown in cells.

The directory contains `data_statistics.csv` and `data_statistics.json` — a dataset summary table (rows, hours, durations, text share, speakers, sources, emotion distribution) and a full report (datasets, LMDB lengths, train/test comparison). These are **stale artifacts** from a previous notebook version (formerly `main_statistic.ipynb`): the current `corpus_statistics.ipynb` does not update them. If an up-to-date dump is needed, extend the notebook or delete the files.

## Dependencies

`pandas`, `matplotlib`, `seaborn`, `scipy`, `librosa` (for `audio_analise.ipynb`), `lmdb` (for reading `.lmdb` corpora in `corpus_statistics.ipynb`). Installed via `poetry install`.
