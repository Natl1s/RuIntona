# Dusha data processing (`ruintona/data_processing/`)

Pipeline for the raw Dusha dataset: acoustic feature extraction, label aggregation (Dawid-Skene) and manifest building for experiments.

> Sources and licenses of third-party pretrained models and datasets — [`SOURCES.en.md`](../../SOURCES.en.md).

> **License and attribution.** This code and the Dusha dataset itself are adapted
> from the [Salute Developers — Golos](https://github.com/salute-developers/golos)
> project and are provided under the Dusha/Golos license (attribution + share-alike).
> License text — [`license/`](../../license/) (EN/RU); description and attribution — [`DUSHA.md`](../../DUSHA.md).

```
raw (crowd.tar / podcast.tar) → processing.py → processed_dataset_0XX/
                                                     ├── features/*.npy
                                                     ├── aggregated_dataset/*.jsonl
                                                     ├── train/*.jsonl
                                                     └── test/*.jsonl
                                                        → make_data_scripts/ → LMDB → ruintona/my_experiments/
```

## Install dependencies

```bash
poetry install --with data-processing
# or manually from ruintona/data_processing/requirements.txt:
# pandas==1.3.5, crowd-kit==1.0.0, click==8.0.4, tqdm==4.62.3, numpy==1.21.5, librosa==0.8.1
```

## Run

```bash
poetry run python ruintona/data_processing/processing.py -dataset_path DATASET_PATH
```

### `processing.py` CLI flags

| Flag | Description | Default |
|---|---|---|
| `-dataset_path` (`--dataset_path`) | path to the dataset root (required) | — |
| `-tsv` (`--use_tsv`) | read/write manifests in TSV instead of JSONL | False |
| `-rf` (`--recalculate_features`) | recompute all features from scratch | False |
| `-threshold` (`--threshold`) | Dawid-Skene confidence threshold | 0.9 |

Validation: `threshold` must be in `[0, 1]`, otherwise an `AttributeError` is raised.

## Expected input layout

```
DATASET_PATH/
├── crowd_train/raw_crowd_train.jsonl (or .tsv) + wavs/*.wav
├── crowd_test/raw_crowd_test.jsonl   + wavs/*.wav
├── podcast_train/raw_podcast_train.jsonl + wavs/*.wav
└── podcast_test/raw_podcast_test.jsonl + wavs/*.wav
```

Markup record (`MarkupDataclass`):

`hash_id`, `audio_path`, `duration`, `annotator_emo`, `golden_emo`, `speaker_text`, `speaker_emo`, `source_id`, `annotator_id`.

## Processing steps

1. The result folder `processed_dataset_0XX` is created (where `XX = int(threshold * 100)`), with `train/`, `test/`, `aggregated_dataset/` subfolders.
2. For each set (`crowd_train`, `crowd_test`, `podcast_train`, `podcast_test`) the raw markup is read (`read_data_markup()`), `hash_id`/wav names are extracted.
3. Acoustic features are computed (`utils/calculate_features.py`):
   - log-mel spectrogram via `librosa`: `sr=16000`, window `n_fft=320` (20 ms), hop `160` (10 ms), `n_mels=64`, `power_to_db(ref=np.max)`;
   - result — `features/<hash_id>.npy`, a tensor of shape `(1, 64, T)`.
   - without `-rf` only missing `.npy` files are computed (the cache is not recomputed if the feature parameters change).
4. **Dawid-Skene** emotion aggregation (`utils/dawidskene.py`, `crowdkit`, `n_iter=100`): triples `task=hash_id`, `worker=annotator_id`, `label=annotator_emo`; class probabilities are saved into `meta.tsv`.
5. Confidence filtering: a record is kept only if `max(proba) >= threshold`.

## Output manifests

### `aggregated_dataset/*` (`filter_data()` → `agg_data_to_file()`)

One record per `hash_id` (deduplicated via `used_wavs`) with the aggregated emotion. Fields: `hash_id`, `audio_path` (rewritten as `dataset/audio_path`), `duration`, `emotion`, `golden_emo`, `speaker_text`, `speaker_emo`, `source_id`.

### `train/*`, `test/*` (`make_exp_data()` → `exp_data_to_file()`)

- per part: `train/crowd_train`, `train/podcast_train`, `test/crowd_test`, `test/podcast_test`;
- combined: `train/train` = `podcast_train + crowd_train`, `test/test` = `podcast_test + crowd_test`.

A record is included if `golden_emo` is empty/not a string **and** the aggregated emotion `!= "other"`. Fields: `id` (=`hash_id`), `tensor` (path to `../../features/<hash_id>.npy`), `wav_length`, `label`, `emotion`.

Label mapping (matches `EMO2LABEL` in `my_experiments`):

```python
angry -> 0, sad -> 1, neutral -> 2, positive -> 3
```

Formats: JSONL (one JSON record per line, `ensure_ascii=False`) or TSV (header + `\t`).

## Known limitations and risks

- In `utils/datacls.py` the fields `audio_path` and `annotator_emo` are declared twice (dataclass defect).
- Data validation/errors are implemented via `AttributeError` — makes diagnostics harder.
- The audio path in `aggregated_dataset` is built as `dataset/audio_path` — keep in mind when moving artifacts.
- If `features/` already contains `.npy`, without `-rf` recomputation is skipped even if the feature parameters in the code changed.

## Next steps

Building balanced datasets and converting to LMDB — [`make_data_scripts/README.md`](./dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/README.md). The corpora (including the combined `dusha_resd` = Dusha + RESD corpus) and the full rules for building them from `aggregated_dataset` — [`CORPUS.md`](../../CORPUS.md). Using them in experiments — [`my_experiments/README.md`](../my_experiments/README.md).


