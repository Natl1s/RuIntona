# Research: Speech Emotion Recognition in Russian (SER)

> This document describes **why** this research is being conducted, **which data** were chosen and why,
> which **ready-made solutions** were evaluated, as well as the **hypothesis, experiments and results**.
> Technical details of the models and corpora — in [README.en.md](README.en.md),
> [CORPUS.en.md](../../CORPUS.en.md), [DUSHA.en.md](../../DUSHA.en.md),
> [RESD.en.md](../../RESD.en.md) and the [`model_analise/`](model_analise/) notebooks.
>
> Sources and licenses of third-party pretrained models and datasets — [`SOURCES.en.md`](../../SOURCES.en.md).

## 1. Motivation and problem statement

Speech Emotion Recognition (SER) is a key technology for human–machine interfaces: voice assistants,
support centers, and call analysis. For Russian the task is complicated by a **shortage of labeled
emotion speech corpora**:

- open Russian-language corpora with emotion labels number only a few;
- most of them are of **small size**;
- almost all were recorded **by actors in a studio** — "ideal" sound quality and **acted**, not
  spontaneous emotions.

## 2. Existing Russian emotion corpora

| Corpus | Size | Recording | Emotions | Limitations |
|---|---|---|---|---|
| **RESD** (Aniemore, 2022) | ~1396 clips / ~2.3 h | 20 voice actors, studio, improvised dialogues without a script | 7 (`anger`, `happiness`, `sadness`, `neutral`, `disgust`, `fear`, `enthusiasm`) | small; actors; studio quality; deliberate, not spontaneous emotion |
| **RAMAS** (Neurodata Lab, 2018) | ~7 h | 10 semi-professional actors (5M/5F), acted dyadic interactions, studio | 4 | small; actors; studio; more multimodal (video+physiology) than speech |
| **Dusha** (Sber / Salute Developers, 2022) | **~304k recordings / ~347 h, 8308 speakers** | **crowdsourcing** (acted crowd + real-life podcasts) | 4 (`angry`, `sad`, `neutral`, `positive`) | — |

Conclusion: except for Dusha, the available corpora are **small** (hours, not hundreds of hours) and
**acted by actors in a studio**, which does not reflect real, spontaneous user speech.

## 3. Why Dusha is the main corpus

1. **Size and bi-modality.** The largest open bi-modal corpus for SER: ~304k audio recordings
   (~347 h) with transcripts and emotion labels. It allows training audio-, text- and multimodal models.
2. **Crowdsourced labeling without professional actors.** The recordings were labeled on a
   crowdsourcing platform by several annotators; the final labels are aggregated by the
   **Dawid-Skene** mechanism (accounts for annotator competence; this repository uses a confidence
   threshold of 0.9).
3. **Natural speech.** The `podcast` subset (real-life speech from podcasts) brings the task closer to
   a real assistant-dialog scenario, unlike "ideal" studio speech.
4. **Few models trained on this dataset.** A Hugging Face Hub search (**as of 10.08.2026**) finds only
   ~10 open-source models fine-tuned on Dusha, and all of them are **audio-only**:

   | Author | Model |
   |---|---|
   | `xbgoose` | `hubert-large` / `hubert-base` / `wavlm-base` / `wavlm-large` (speech-emotion-recognition-russian-**dusha**-finetuned) |
   | `growingpenguin` | `dusha_emotion-hubert-base` |
   | `nixiieee` | `whisper-small` / `whisper-small-2` / `whisper-large-v3` / `gigaam-rnnt` (emotion-classifier-**dusha**) |
   | `KELONMYOSA` | `wav2vec2-xls-r-300m-emotion-ru` |

   The list may change over time. The absence of multimodal and text models on Dusha is an
   additional argument for our own experiments.

## 4. Evaluation of ready-made solutions (pretrained, zero-shot)

Open Hugging Face models were run **without fine-tuning** on our test corpus
`dusha_resd_test` (6616 samples). For each model we describe **what it was trained on**, what metrics
are **claimed in the authors' documentation** and what is **actually obtained** on our sample:

| Model | Trained on | Claimed in documentation | Actual on `dusha_resd_test` (Acc / F1-macro) |
|---|---|---|---|
| **Whisper-large-v3** (SER, `firdhokk`) | English acted datasets RAVDESS, SAVEE, TESS, URDU | acc 0.92, F1 0.92 | **0.435 / 0.345** — collapse |
| **WavLM-BERT fusion** (`Aniemore`) | Russian **RESD** (7 classes) | RESD test macro-F1 0.794; on Dusha podcast — UA 0.35 / WA 0.07 / F1 0.092 (the authors honestly note degradation on spontaneous speech) | **0.552 / 0.503** |
| **HuBERT-large** (`xbgoose`) | **Dusha** (half of the train split, 2 epochs) | acc 0.86, balanced 0.76, macro-F1 0.81 | **0.805 / 0.815** |

**Conclusion.** The claimed metrics **do not reproduce** on Dusha: Whisper, trained on English acted
data, barely works on Russian spontaneous speech; WavLM-BERT fusion degrades on real speech (which is
honestly acknowledged by its authors). The only high-quality model is HuBERT, **fine-tuned on Dusha
itself**. Hence:

- ready-made solutions trained on other data do not work on our corpus;
- it is necessary to **train models on the Dusha corpus** (also confirmed by the HuBERT result).

## 5. Hypothesis and experiments

**Hypothesis.** A multimodal model (**audio + text**) will improve emotion recognition over unimodal
ones: audio carries prosodic information (intonation, tempo, timbre), text carries lexical and
semantic information; the errors of the modalities partially do not overlap and compensate each other.

### Protocol

- **Data.** The `combine_balanced` (Dusha only) and `dusha_resd` (Dusha + RESD, mapped to 4 classes)
  corpora. Composition and building rules — [CORPUS.en.md](../../CORPUS.en.md).
- **Test.** `dusha_resd_test` — a single fixed split, 6616 samples, 4 classes
  (`angry` · `sad` · `neutral` · `positive`).
- **Metrics.** Accuracy and F1-macro (additionally in notebooks — UAR, MCC, ROC-AUC).
- **Reproducibility.** Scripts fix the seed (default 42); the α weights for late fusion are tuned on
  validation.

### Unimodal models

**Audio** (features — mel-spectrograms from LMDB; for wav2vec2 — raw signal):

| Model | Type |
|---|---|
| Logistic Regression | classic baseline |
| Random Forest (default and tuned) | classic baseline |
| SVM (RBF) | classic baseline |
| openSMILE + XGBoost / LightGBM | classic baseline on openSMILE features |
| CNN | deep model on mel/MFCC |
| CNN-BiLSTM | deep model on mel |
| Wav2Vec2 XLS-R 300M + Self-Attention | pretrained + head |

**Text** (transcripts from LMDB):

| Model | Features |
|---|---|
| TF-IDF + LogReg | TF-IDF (1–2 grams) |
| FastText (embeddings) + LogReg | average `cc.ru.300.bin` vector |
| BiLSTM | FastText matrix (frozen) + BiLSTM |
| RuBERT | `DeepPavlov/rubert-base-cased` tokens |

### Multimodal approaches

| Approach | Components |
|---|---|
| Late fusion (soft-voting, α sweep) | 3 variants: SVM+TF-IDF, CNN-BiLSTM+RuBERT, **HuBERT+RuBERT** |
| Early fusion | concatenation of CNN-BiLSTM + RuBERT features → LSTM → Linear |
| Co-attention | Wav2Vec2 XLS-R 300M (audio) + RuBERT (text), cross-attention |

## 6. Results

### Best audio model

**CNN-BiLSTM** (trained on `combine_balanced`, evaluated on `dusha_resd_test`):

| Model | Test Acc | F1-macro |
|---|---|---|
| **CNN-BiLSTM** | **0.740** | **0.732** |

> Notes.
> - Best classic baseline — **openSMILE + XGBoost** (Test Acc 0.600, F1-macro 0.590); the other
>   sklearn baselines are noticeably lower (LogReg 0.474/0.465, RF 0.471/0.458, RF tuned 0.485/0.465, SVM 0.510/0.499).
> - In the multimodal model the pretrained **HuBERT-large** is used as the audio backbone (fine-tuned
>   on Dusha, Test Acc 0.805 / F1-macro 0.815) — it proved stronger than our own CNN-BiLSTM.

### Best text model

**RuBERT** (trained on `dusha_resd`, evaluated on `dusha_resd_test`):

| Model | Test Acc | F1-macro |
|---|---|---|
| **RuBERT** | **0.586** | **0.601** |

> Other text models: TF-IDF+LogReg 0.540/0.556, FastText+LogReg 0.531/0.541, BiLSTM 0.560/0.580.

### Best multimodal model

**Late fusion HuBERT + RuBERT (α = 0.5)**:

| Model | Test Acc | F1-macro |
|---|---|---|
| **Late fusion HuBERT + RuBERT (α=0.5)** | **0.822** | **0.830** |

Comparison with other multimodal variants (`dusha_resd_test`):

| Model | Test Acc | F1-macro |
|---|---|---|
| Late fusion baseline SVM + TF-IDF LogReg (α=0.35) | 0.621 | 0.629 |
| Late fusion CNN-BiLSTM + RuBERT (α=0.5) | 0.790 | 0.786 |
| Early fusion CNN-BiLSTM + RuBERT | 0.795 | 0.795 |
| **Late fusion HuBERT + RuBERT (α=0.5)** | **0.822** | **0.830** |

**Is the best multimodal model a combination of the best unimodal ones?**

Partially, but not fully:

- the text branch — the **best text model** (RuBERT, 0.586/0.601);
- the audio branch — **not** the best own audio model (CNN-BiLSTM), but the pretrained **HuBERT-large**
  (0.805/0.815). Fusion specifically with HuBERT gives the maximum gain:
  - CNN-BiLSTM + RuBERT → 0.790/0.786;
  - **HuBERT + RuBERT → 0.822/0.830** (+0.032/+0.044 compared with fusion with CNN-BiLSTM).

I.e. multimodality provides a gain over any unimodal model, but the best result is the combination of
the best text encoder with the **strongest** audio encoder, not with the "best own" audio model.

## 7. Statistical significance of the multimodal combination

Verified on `dusha_resd_test.lmdb` (n = 6616, paired predictions) in
[`multimodal_models_analise.ipynb`](model_analise/multimodal_models_analise.ipynb):
**McNemar test** on discordant pairs (with continuity correction, H0: the models' error rates are equal)
and **paired bootstrap** (B = 1000, seed = 42) for the metric difference
`fusion − modality` (95% percentile CI, two-sided p-value). For each model the fusion is compared
with each unimodal branch on the same test examples.

Notation: `n01` — fusion correct / branch not correct, `n10` — branch correct / fusion not correct;
`*** p<0.001`, `** p<0.01`, `* p<0.05`, `n.s.` otherwise.

### 7.1 McNemar test (paired predictions)

| Model | Comparison | n01 | n10 | χ² | p-value |
|---|---|---|---|---|---|
| Late fusion Baseline (SVM + TF-IDF, α=0.35) | Fusion vs Audio | 1259 | 491 | 336.2 | <0.001 `***` |
| | Fusion vs Text | 1124 | 589 | 166.5 | <0.001 `***` |
| Late fusion CNN-BiLSTM + RuBERT (α=0.5) | Fusion vs Audio | 647 | 292 | 133.5 | <0.001 `***` |
| | Fusion vs Text | 1695 | 342 | 897.4 | <0.001 `***` |
| Late fusion HuBERT + RuBERT (α=0.5) | Fusion vs Audio | 334 | 224 | 21.3 | <0.001 `***` |
| | Fusion vs Text | 1891 | 329 | 1097.6 | <0.001 `***` |
| Early fusion CNN-BiLSTM + RuBERT | Fusion vs Audio | 793 | 406 | 124.3 | <0.001 `***` |
| | Fusion vs Text | 1740 | 355 | 914.3 | <0.001 `***` |

### 7.2 Paired bootstrap: Accuracy / F1-macro difference (95% CI)

| Model | Comparison | Δ Accuracy | 95% CI | Δ F1-macro | 95% CI |
|---|---|---|---|---|---|
| Late fusion Baseline | Fusion vs Audio | +0.116 | [0.104, 0.129] | +0.135 | [0.123, 0.148] |
| | Fusion vs Text | +0.081 | [0.070, 0.093] | +0.073 | [0.062, 0.084] |
| Late fusion CNN-BiLSTM + RuBERT | Fusion vs Audio | +0.054 | [0.044, 0.062] | +0.059 | [0.049, 0.069] |
| | Fusion vs Text | +0.204 | [0.193, 0.217] | +0.185 | [0.174, 0.197] |
| Late fusion HuBERT + RuBERT | Fusion vs Audio | +0.017 | [0.010, 0.024] | +0.015 | [0.008, 0.022] |
| | Fusion vs Text | +0.236 | [0.224, 0.249] | +0.229 | [0.218, 0.242] |
| Early fusion CNN-BiLSTM + RuBERT | Fusion vs Audio | +0.058 | [0.049, 0.068] | +0.068 | [0.058, 0.080] |
| | Fusion vs Text | +0.209 | [0.197, 0.222] | +0.194 | [0.183, 0.206] |

### Conclusion

For **all** multimodal models the combination of modalities is statistically significantly better than
both unimodal branches: McNemar p < 0.001 in all comparisons, and the bootstrap CIs of the Accuracy and
F1-macro difference never cross zero (all gains `***`; Balanced Accuracy gives the same picture).
Fusion improves the text branch the most (RuBERT, Δ Accuracy up to +0.24) and notably the audio branch
(Δ Accuracy up to +0.12). The smallest but still significant gain over audio is for **HuBERT + RuBERT**
(+0.017 accuracy, 95% CI [0.010, 0.024]): the strong audio branch (HuBERT-large, 0.805) leaves less
"headroom" for fusion, yet even here the combination of modalities is significantly better.

## 8. References

- Corpora and their building: [CORPUS.en.md](../../CORPUS.en.md)
- Datasets: [DUSHA.en.md](../../DUSHA.en.md), [RESD.en.md](../../RESD.en.md)
- Audio models: [`audio_models/README.en.md`](audio_models/README.en.md)
- Text models: [`text_models/README.en.md`](text_models/README.en.md)
- Multimodal models: [`multimodal/README.en.md`](multimodal/README.en.md)
- Result and pretrained-model analysis: [`model_analise/README.en.md`](model_analise/README.en.md)
- Pretrained models: `xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned`,
  `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`,
  `Aniemore/wavlm-bert-fusion-s-emotion-russian-resd` — sources and licenses: [`SOURCES.en.md`](../../SOURCES.en.md)
