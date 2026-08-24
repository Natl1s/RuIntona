# Dusha dataset

> Note: this page describes the Dusha dataset from the perspective of this repository: what the data is, where to download it, and how it is used here. Authoritative information is on the official project page of [Salute Developers — Golos](https://github.com/salute-developers/golos/tree/master/dusha) (see [Official sources](#official-sources)).

Dusha is a **bi-modal corpus** of Russian speech designed for Speech Emotion Recognition (SER) tasks. The dataset consists of about **300 000 audio recordings** (~350 hours) with transcripts and emotional labels and, at the time of publication, was the largest open bi-modal collection for SER. Emotions correspond to four basic classes typical for a dialog with a virtual assistant: **Happiness (Positive), Sadness, Anger and Neutral** emotion.

> **Note on derived corpora.** The experiments in this repository mostly run on a combined corpus built from Dusha and the [RESD](RESD.md) dataset (Aniemore, Hugging Face). See [CORPUS.md](CORPUS.md) for a description of the corpora and the rules for building them.

> Consolidated sources of all third-party components (models + datasets) — [`SOURCES.en.md`](./SOURCES.en.md).

## Official sources

| Source | Link |
|---|---|
| Project GitHub repository (Salute Developers / Sber) | <https://github.com/salute-developers/golos> |
| Dusha section in the repository (dataset README) | <https://github.com/salute-developers/golos/tree/master/dusha> |
| Paper "Large Raw Emotional Dataset with Aggregation Mechanism" (arXiv) | <https://arxiv.org/abs/2212.12266> |
| Paper DOI | <https://doi.org/10.48550/arXiv.2212.12266> |
| Golos paper "Golos: Russian dataset for speech research" | <https://arxiv.org/abs/2106.10161> |
| Dusha/Golos dataset license (EN) | [`license/en_us.pdf`](./license/en_us.pdf) · <https://github.com/salute-developers/golos/blob/master/license/en_us.pdf> |
| Dusha/Golos dataset license (RU) | [`license/ru.pdf`](./license/ru.pdf) · <https://github.com/salute-developers/golos/blob/master/license/ru.pdf> |

## Dataset structure

| Domain | Number of Files | Duration (Hr.) | Unique Speakers |
|---|---|---|---|
| Crowd (acted) | 201 850 | 255.7 | 2068 |
| Podcast (real-life) | 102 113 | 90.9 | 6240 |
| Total | 303 963 | 346.6 | 8308 |

The dataset consists of two subsets:

- **Crowd** — acted speech with a more balanced class distribution; suitable for model pre-training.
- **Podcast** — real-life speech from podcasts, with an unbalanced distribution; intended for fine-tuning and validation.

## Annotation

- The data is annotated on a crowdsourcing platform; each recording is labeled by several annotators.
- Final labels are aggregated with the **Dawid-Skene** mechanism (accounts for annotator competence).
- This repository uses the confidence threshold **0.9** → processing output is stored in `processed_dataset_090` (see [`ruintona/data_processing/README.md`](ruintona/data_processing/README.md)).

## Downloads

> **Podcast audio caveat.** Due to license restrictions, the official distribution does not include podcast audio files. Instead, precalculated features and links to the original podcasts with timings are provided (see [Issue #1](https://github.com/salute-developers/golos/issues/1); it also contains a community mirror `podcast_wavs.tar.gz`, 8.03 GB, md5 `31283c7747c30685eddb451690a4cc73`).

| Archive | Size | Link (official CDN) |
|---|---|---|
| `crowd.tar` | 28 GB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/crowd.tar> |
| `podcast.tar` | 360 MB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/podcast.tar> |
| `features.tar` | 30 GB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/features.tar> |
| `paper_setups.tgz` | 16 MB | <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/paper_setups.tgz> |

## Usage in this repository

Raw data processing pipeline: mel-spectrograms (`features/*.npy`), Dawid-Skene label aggregation, aggregated manifests `crowd_{train,test}.jsonl` / `podcast_{train,test}.jsonl` → balanced sets → LMDB databases. See [CORPUS.md](CORPUS.md) for the full build chain and corpora description.

## Attribution in this repository

This repository contains:

- **Adapted Material** from the original [Salute Developers — Golos/Dusha](https://github.com/salute-developers/golos) project:
  - the `ruintona/data_processing/` pipeline (raw data processing, feature extraction, Dawid-Skene label aggregation),
  - the dataset structure and any derived dataset artifacts (features, manifests, LMDB databases).
- **Original work** by the author of this repository: everything else under `ruintona/my_experiments/`, `ruintona/DEMO/`, `ruintona/configs/` — licensed separately (see [LICENSE](./LICENSE)).

The dataset and the adapted code are provided under the Dusha/Golos license (see [English Version](./license/en_us.pdf) / [Russian Version](./license/ru.pdf)): **with attribution and conditions reserved** (Share-Alike). When distributing the dataset itself or derived data artifacts, keep the attribution above and the license text.

## Citation

```bibtex
@misc{kondratenko2022large,
  title = {Large Raw Emotional Dataset with Aggregation Mechanism},
  author = {Vladimir Kondratenko and Artem Sokolov and Nikolay Karpov
            and Oleg Kutuzov and Nikita Savushkin and Fyodor Minkin},
  year = {2022},
  eprint = {2212.12266},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS},
  url = {https://arxiv.org/abs/2212.12266}
}
```
