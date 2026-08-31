# Dataset build scripts (`make_data_scripts/`)

Scripts to prepare datasets from the processed manifests of `data_processing`. All commands below can be run from the repository root (or use paths matching your local layout).

> Sources and licenses of third-party pretrained models and datasets — [`SOURCES.en.md`](../../../../../../SOURCES.en.md).

The scripts live in:
`ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/`

The full description of the corpora and the overall build pipeline (including the combined Dusha + RESD corpus) — see [`CORPUS.en.md`](../../../../../../CORPUS.en.md) (repository root).

## Overview

```
raw (crowd.tar / podcast.tar)
  → data_processing/processing.py            (mel features, Dawid-Skene, threshold 0.9)
  → aggregated_dataset/crowd_*.jsonl, podcast_*.jsonl
  → build_balanced_aggregated_jsonl.py
  → combine_balanced_{train,test}.jsonl (+ *_small.jsonl)
  → lmdb_convert.py                          (JSONL + WAV → LMDB)
  → hug_dataset/add_missing_spectrograms.py  (compute x for records that lack it)

RESD parquet (Aniemore/resd_annotated)
  → hug_dataset/make_raw.py                  (parquet → raw_*.jsonl + wavs)
  → merged with combine_balanced_*.jsonl     → dusha_resd_{train,test}.jsonl
  → lmdb_convert.py                          → dusha_resd_{train,test}.lmdb
  → hug_dataset/add_missing_spectrograms.py  (RESD rows have no x after conversion)
```

## `build_balanced_aggregated_jsonl.py`

Creates 4 balanced JSONL datasets inside `aggregated_dataset`:

- `combine_balanced_train.jsonl`
- `combine_balanced_test.jsonl`
- `combine_balanced_train_small.jsonl`
- `combine_balanced_test_small.jsonl`

Rules:

1. Train source: `crowd_train.jsonl + podcast_train.jsonl`
2. Test source: `crowd_test.jsonl + podcast_test.jsonl`
3. Only target emotions are used: `angry`, `sad`, `neutral`, `positive`
4. For full sets: `neutral <= 2 * min(non-neutral class count)`; non-neutral classes are taken in full
5. For small sets: size is 30% (configurable, `--small-ratio`) of the full set, class ratio preserved as close as possible
6. `--seed` (default 42) makes sampling reproducible

```bash
python ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/build_balanced_aggregated_jsonl.py

# with custom options
python ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/build_balanced_aggregated_jsonl.py \
  --aggregated-dir ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset \
  --small-ratio 0.3 \
  --seed 42
```

These JSONL files are used to build the LMDBs and run the experiments (`ruintona/my_experiments/`).

## `lmdb_convert.py`

Converter **JSONL + WAV → LMDB** (full records: waveform, text, label).

```bash
python ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/lmdb_convert.py \
  --manifest ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train.jsonl \
  --data-root ruintona/data_processing/dataset/processed_dataset_090 \
  --output ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train.lmdb
```

Supports the manifest fields `audio_path` / `wav_path` / `wav` (audio), `hash_id` / `id` (id) and labels from `label` / `emotion` / `annotator_emo` / `speaker_emo` / `target` / `class` / `emo`. LMDB record format:

- key: `<index>` (bytes)
- value: pickle:
  - `y`: label (int, angry=0, sad=1, neutral=2, positive=3)
  - `id`: sample_id
  - `waveform`: raw mono signal (`np.float32`, resampled to 16 kHz)
  - `waveform_sr`: sample rate (`16000`)
  - `text`: transcript (`speaker_text` / `text` / `transcript` / `utterance`)
- metadata key `b"__len__"`: record count

> **Podcast filter.** The script skips rows whose audio path contains `podcast` (checked in `audio_path`/`wav_path`/`wav`/`tensor`/`feature_path`/`id`/`hash_id`). Therefore the LMDB length is **smaller** than the JSONL line count (e.g. `combine_balanced_train.jsonl` → `combine_balanced_train.lmdb`).
>
> **About spectrograms.** `lmdb_convert.py` does **not** write the mel-spectrogram `x`. For records without `x` compute it from `waveform` with `ruintona/data_processing/dataset/hug_dataset/add_missing_spectrograms.py` (required for `dusha_resd`). The existing `combine_balanced*` LMDBs on disk do contain `x` — they were built by an older converter version.

Flags: `--manifest`, `--output`, `--data-root` (default: manifest directory), `--commit-interval` (records per transaction, default 1024).

## `lmdb_convert_only_spectr.py`

Converter **JSONL + NPY → LMDB (spectrograms only)** — no waveform or text, just `x`, `y`, `id`. Faster and more compact than `lmdb_convert.py`; no audio files needed.

```bash
python ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/lmdb_convert_only_spectr.py \
  --manifest ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train.jsonl \
  --data-root ruintona/data_processing/dataset/processed_dataset_090 \
  --output ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/combine_balanced_train_spectr.lmdb
```

Features are read from `tensor` / `feature_path` or `features/<hash_id>.npy`. Flags: same as `lmdb_convert.py`.

## Building the combined corpus `dusha_resd` (Dusha + RESD)

Most models (`my_experiments/`) are trained on the `dusha_resd` corpus — a merge of `combine_balanced` and the RESD dataset (Aniemore, Hugging Face). Build sequence:

```bash
# 1. RESD parquet → raw jsonl + wavs (train 916 / test 224 rows, emotion from name)
python ruintona/data_processing/dataset/hug_dataset/make_raw.py \
  --input-dir ruintona/data_processing/dataset/hug_dataset/data \
  --wavs-dir ruintona/data_processing/dataset/hug_dataset/wavs

# 2. Merge (concatenate) raw_train + combine_balanced_train → dusha_resd_train.jsonl;
#    same for test. (No dedicated merge script.)

# 3. Convert to LMDB
python ruintona/data_processing/dataset/processed_dataset_090/aggregated_dataset/make_data_scripts/lmdb_convert.py \
  --manifest .../dusha_resd_train.jsonl --data-root .../processed_dataset_090 \
  --output .../aggregated_dataset/dusha_resd_train.lmdb
# (same for dusha_resd_test.lmdb)

# 4. Compute mel-spectrograms x for records lacking them (needed for RESD rows)
python ruintona/data_processing/dataset/hug_dataset/add_missing_spectrograms.py \
  --lmdb .../aggregated_dataset/dusha_resd_train.lmdb \
  --lmdb .../aggregated_dataset/dusha_resd_test.lmdb
```

Final sizes: `dusha_resd_train.lmdb` — 69119 records (68203 combine_balanced + 916 RESD), `dusha_resd_test.lmdb` — 6616 (6392 + 224).

## Scripts in `ruintona/data_processing/dataset/hug_dataset/`

| Script | Purpose |
|---|---|
| `make_raw.py` | Convert parquet (RESD / other HF datasets) to `raw_*.jsonl` + `wavs/*.wav`; emotion is read from the clip name, mapping `happiness→positive`, `anger→angry`, `sadness→sad`, `neutral→neutral` |
| `dataset_stats.py` | Statistics over raw JSONL (emotion distribution, durations, text length, duplicates) |
| `add_missing_spectrograms.py` | Compute mel-spectrogram `x` from `waveform` for records without `x` (by default works with `dusha_resd_train/test.lmdb`) |

## See also

- [`CORPUS.en.md`](../../../../../../CORPUS.en.md) — corpus composition and full building rules
- [`ruintona/data_processing/README.en.md`](../../../../README.en.md) — raw processing pipeline (Dawid-Skene, features)
- [`ruintona/data_processing/dataset/hug_dataset/`](../../../hug_dataset/) — scripts for external HF datasets
