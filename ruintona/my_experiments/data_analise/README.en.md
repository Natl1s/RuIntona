# Data analysis (`my_experiments/data_analise/`)

Jupyter notebooks with exploratory data analysis (EDA) of the Dusha dataset.

| Notebook | What it analyzes |
|---|---|
| `main_statistic.ipynb` | Statistics of all datasets and **corpus justification**: sources (crowd/podcast), emotion distribution, annotation quality (annotators → Dawid-Skene), reproduction of the build pipeline (balancing, 30% small, RESD, LMDB), quality and fairness of train/test splits |
| `text_analise.ipynb` | Text features: transcript length, emotion-specific vocabulary, typical words |
| `audio_analise.ipynb` | Audio features: durations, spectral characteristics, waveform/spectrogram visualization |

## Run

```bash
poetry run jupyter notebook ruintona/my_experiments/data_analise/
```

Notebooks read data from JSONL/CSV manifests (see `data_processing/dataset/processed_dataset_090/aggregated_dataset/`). Data paths are set at the top of each notebook and should be adjusted to the local dataset location.

`main_statistic.ipynb` additionally reads LMDB lengths (`__len__`) and a sample of records to verify formats (requires the `lmdb` module, part of the project).

## Output files

Running `main_statistic.ipynb` saves next to the notebook:

- `data_statistics.csv` — dataset summary table (rows, hours, durations, text share, speakers);
- `data_statistics.json` — full report (datasets, LMDB lengths, train/test comparison).

## Dependencies

`pandas`, `matplotlib`, `seaborn`, `librosa` (for `audio_analise.ipynb`). Installed via `poetry install`.
