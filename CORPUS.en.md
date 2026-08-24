# Data corpora (CORPUS)

This repository trains and evaluates models on corpora built from two source datasets:

- **Dusha** (Sber / Salute Developers) — see [DUSHA.md](DUSHA.md)
- **RESD** (Aniemore, Hugging Face) — see [RESD.md](RESD.md)

All artifacts live under `ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/`.

> Sources and licenses of third-party pretrained models and datasets — [`SOURCES.en.md`](./SOURCES.en.md).

## License

The corpora inherit the licenses of their source datasets:

- **`combine_balanced`** / **`combine_balanced_small`** (Dusha only): Dusha/Golos license (attribution + share-alike) — text in [`license/`](./license/), description in [`DUSHA.md`](DUSHA.md).
- **`dusha_resd`** (Dusha + RESD): Dusha/Golos for the Dusha part and MIT for the RESD part (see [`RESD.md`](RESD.md)).

When redistributing the corpora or derived artifacts, keep the attribution and license texts of both parts.

## Corpora

| Corpus | Composition | Train | Test | Used by |
|---|---|---|---|---|
| `combine_balanced` | Dusha only (crowd + podcast), 4 emotions, balanced | 68203 | 6392 | CNN/CNN-BiLSTM, HuBERT+RuBERT late fusion, foundation-model evaluation |
| `combine_balanced_small` | 30% of `combine_balanced` | 20474 | 1863 | fast runs, wav2vec2 warm-start checkpoints |
| `dusha_resd` | `combine_balanced` + RESD (916 / 224 rows) | 69119 | 6616 | **most models**: RuBERT, Logistic Regression, Random Forest (default + tuned), SVM, openSMILE+XGBoost, early/late-fusion baselines |

Sizes are the number of **LMDB records** (after conversion). They may differ from the JSONL line counts — see [Known caveats](#known-caveats).

## Building rules

The build chain from raw data to LMDBs:

```
raw (crowd.tar / podcast.tar)
  → data_processing/processing.py            (mel features, Dawid-Skene, threshold 0.9)
  → aggregated_dataset/crowd_*.jsonl, podcast_*.jsonl
  → make_data_scripts/build_balanced_aggregated_jsonl.py
  → combine_balanced_{train,test}.jsonl (+ *_small.jsonl)
  → make_data_scripts/lmdb_convert.py        (JSONL + WAV → LMDB)
  → hug_dataset/add_missing_spectrograms.py  (compute x for records that lack it)

RESD parquet (Aniemore/resd_annotated)
  → hug_dataset/make_raw.py                  (parquet → raw_*.jsonl + wavs, emotion from name)
  → merged with combine_balanced_*.jsonl     → dusha_resd_{train,test}.jsonl
  → make_data_scripts/lmdb_convert.py        → dusha_resd_{train,test}.lmdb
  → hug_dataset/add_missing_spectrograms.py  (RESD rows have no x after conversion)
```

### Step 0. Aggregated manifests (Dusha only)

`ruintona/data_processing/processing.py` extracts mel-spectrograms (`features/*.npy`), aggregates annotations with **Dawid-Skene** (crowdkit, threshold 0.9 → `processed_dataset_090`) and writes aggregated manifests `crowd_{train,test}.jsonl` / `podcast_{train,test}.jsonl` in `aggregated_dataset/` (fields: `hash_id`, `audio_path`, `duration`, `emotion`, `golden_emo`, `speaker_text`, `speaker_emo`, `source_id`).

### Step 1. Balancing → `combine_balanced`

`make_data_scripts/build_balanced_aggregated_jsonl.py`:

1. Train source: `crowd_train.jsonl + podcast_train.jsonl`; test source: `crowd_test.jsonl + podcast_test.jsonl`.
2. Only target emotions are kept: `angry`, `sad`, `neutral`, `positive`.
3. Full sets: `neutral <= 2 * min(count of non-neutral classes)`; non-neutral classes are taken in full.
4. Small sets: size = 30% (`--small-ratio`) of the full set, class proportions preserved as close as possible (`_scaled_targets_with_same_ratio`).
5. `--seed 42` (default) makes sampling reproducible.

Outputs: `combine_balanced_train.jsonl`, `combine_balanced_test.jsonl`, `combine_balanced_train_small.jsonl`, `combine_balanced_test_small.jsonl`.

### Step 2. RESD → raw manifests

`hug_dataset/make_raw.py` converts the parquet files of `Aniemore/resd_annotated` (train 1116 / test 280 rows) into `raw_*.jsonl` + `wavs/*.wav`:

- emotion is read from the clip `name` (e.g. `32_happiness_enthusiasm_h_120`);
- mapping `happiness → positive`, `anger → angry`, `sadness → sad`, `neutral → neutral`;
- rows whose first two name tokens are not in the target set (`disgust`, `fear`, `enthusiasm`, …) are skipped → train 916 / test 224 rows.

### Step 3. Merge → `dusha_resd`

`dusha_resd_train.jsonl = combine_balanced_train.jsonl + raw_train_*.jsonl (RESD)`; likewise for test. (No dedicated merge script — concatenate the manifests.)

### Step 4. Convert to LMDB

`make_data_scripts/lmdb_convert.py` (JSONL + WAV → LMDB). Record format: key `<index>` (bytes), value pickled dict:

| Key | Type | Description |
|---|---|---|
| `y` | `int` | emotion label (0–3) |
| `id` | `str` | sample id |
| `waveform` | `np.ndarray(float32)` | mono signal, resampled to 16 kHz |
| `waveform_sr` | `int` | sample rate (`16000`) |
| `text` | `str` | transcript |

Plus a metadata key `b"__len__"` with the record count.

> **Podcast filter.** `lmdb_convert.py` skips rows whose audio path contains `podcast` (they are excluded from the LMDBs). As a result the LMDB length is smaller than the JSONL line count.

### Step 5. Compute spectrograms

`hug_dataset/add_missing_spectrograms.py` computes the mel-spectrogram `x` (shape `(1, 64, T)`, hop=160, n_fft=320, n_mels=64, `power_to_db(ref=np.max)`) from `waveform` for any record that lacks `x`. This is required for `dusha_resd` because `lmdb_convert.py` does not write `x` for the RESD rows.

### Alternative: spectrograms only

`make_data_scripts/lmdb_convert_only_spectr.py` (JSONL + NPY → LMDB with just `x`, `y`, `id`) — faster and smaller, no WAV needed.

## Which models run on which corpus

| Model | Corpus |
|---|---|
| SVM (RBF) | `dusha_resd` (retrained 08.08, test acc 0.510) |
| openSMILE+XGBoost | `dusha_resd` (retrained 08.08, test acc 0.600) |
| Logistic Regression | `dusha_resd` (retrained from `combine_balanced` on 06.08, test acc 0.474) |
| Random Forest | `dusha_resd` (retrained from `combine_balanced` on 06.08, test acc 0.471) |
| CNN / CNN-BiLSTM | `combine_balanced` |
| Wav2Vec2 XLS-R 300M + Self-Attention | `combine_balanced_small` (warm-start) |
| RuBERT | `dusha_resd` |
| Random Forest (tuned) | `dusha_resd` |
| Early fusion (CNN-BiLSTM + RuBERT) | `dusha_resd` |
| Late fusion baseline (SVM + TF-IDF LogReg) | `dusha_resd` |
| Late fusion HuBERT + RuBERT (α=0.5) | `combine_balanced` |
| Foundation models (Whisper, WavLM-BERT, HuBERT) | `combine_balanced` |

Detailed description and sources of the foundation models (Whisper, WavLM-BERT, HuBERT) — [`model_analise/README.md`](./ruintona/my_experiments/model_analise/README.md).

## Known caveats

- **JSONL ↔ LMDB mismatch.** Because of the podcast filter in `lmdb_convert.py`, LMDB lengths are smaller than the JSONL line counts. E.g. `combine_balanced_train.jsonl` has 89943 lines but the LMDB has 68203 records (see [`make_data_scripts/README.md`](ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/README.md), which documents the podcast filter and corpus sizes).
- **On-disk JSONL may be stale/truncated.** Rebuild the manifests before re-converting; check line counts against the LMDB `__len__`.
- **RESD sample rate is not uniform** (16 kHz / 44.1 kHz) — `lmdb_convert.py` resamples everything to 16 kHz.
- **`make_manifest.py` is an empty stub** and is not used.

## References

- Build scripts: [`make_data_scripts/README.md`](ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/README.md)
- HuggingFace datasets in `ruintona/data_processing/dataset/hug_dataset/`: `make_raw.py`, `dataset_stats.py`, `add_missing_spectrograms.py`
- Raw processing pipeline: [`data_processing/README.md`](ruintona/data_processing/README.md)
