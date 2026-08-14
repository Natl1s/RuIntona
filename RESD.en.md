# RESD dataset (Aniemore / Russian Emotional Speech Dialogues)

> Note: this page is a mirror/summary of the upstream dataset card. The authoritative source is the Hugging Face page: <https://huggingface.co/datasets/Aniemore/resd_annotated> (DOI [10.57967/hf/1272](https://doi.org/10.57967/hf/1272)).

RESD — **Russian Emotional Speech Dialogues** — is a Russian speech emotion recognition dataset published by the [Aniemore](https://huggingface.co/Aniemore) team. In this repository the `resd_annotated` version is used as a secondary source to augment the [Dusha](DUSHA.md) corpus (see [CORPUS.md](CORPUS.md)).

## Description

- Recorded in a studio by **20 voice actors**.
- No script: each actor in a pair was privately given an emotion to convey, and the dialogue was improvised. The words are spontaneous while the emotion is deliberate — this is the point of the dataset and also its limitation (acted emotion is not spontaneous emotion).
- 7 emotion classes: `anger`, `happiness`, `sadness`, `neutral`, `disgust`, `fear`, `enthusiasm`.
- Language: Russian.

## Splits

| Split | Rows | Hours | Mean clip |
|---|---|---|---|
| `train` | 1116 | 1.88 | 6.1 s |
| `test` | 280 | 0.46 | 5.9 s |

## Fields

| Column | Meaning |
|---|---|
| `name` | Clip identifier (e.g. `32_happiness_enthusiasm_h_120`) |
| `path` | Original file path |
| `speech` | Audio |
| `text` | Transcript of the utterance |
| `emotion` | Emotion label of the recording |

> **Sample rate caveat.** The sample rate is not uniform: in `train`, 565 clips are 16000 Hz and 551 clips are 44100 Hz. Always resample to a single rate (e.g. 16 kHz) before feeding a feature extractor. The pipeline in this repository resamples everything to 16 kHz.

## Use in this repository

- Only rows whose emotion maps to the four target classes are kept: `anger → angry`, `happiness → positive`, `sadness → sad`, `neutral → neutral`. Other RESD classes (`disgust`, `fear`, `enthusiasm`) are excluded.
- The selected rows are merged into the Dusha corpus to form the `dusha_resd` corpus: `dusha_resd_train.lmdb` / `dusha_resd_test.lmdb`. See [CORPUS.md](CORPUS.md) and `ruintona/data_processing/dataset/hug_dataset/make_raw.py`.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Aniemore/resd_annotated")
print(ds["train"][0]["emotion"])
```

## Citation

```bibtex
@misc{Aniemore,
  author = {Артем Аментес, Илья Лубенец, Никита Давидчук},
  title = {Открытая библиотека искусственного интеллекта для анализа и выявления эмоциональных оттенков речи человека},
  year = {2022},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.com/aniemore/Aniemore}},
  email = {hello@socialcode.ru}
}
```

## License

MIT.
