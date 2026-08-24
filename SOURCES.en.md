# Third-Party Pretrained Models and Data Sources

This project uses third-party pretrained models and datasets. Model weights and
data are not vendored in the repository — they are downloaded at runtime
(`ruintona/my_experiments/utils/pretrained.py`) or loaded from the Hugging Face Hub
by the experiment scripts via `from_pretrained(...)`.

The legal licensing summary for all parts of the project — [`NOTICE`](./NOTICE).

## Models

| Component | Source | License |
|---|---|---|
| RuBERT (`rubert-base-cased`, DeepPavlov) | <https://huggingface.co/DeepPavlov/rubert-base-cased> | Apache-2.0 |
| Wav2Vec2 XLS-R 300M (Facebook/Meta) | <https://huggingface.co/facebook/wav2vec2-xls-r-300m> | Apache-2.0 |
| HuBERT-large SER fine-tuned on Dusha (xbgoose) | <https://huggingface.co/xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned> | Apache-2.0 |
| Whisper large-v3 SER fine-tune (firdhokk; base `openai/whisper-large-v3` is MIT) | <https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3> | Apache-2.0 |
| WavLM-BERT fusion SER on RESD (Aniemore) | <https://huggingface.co/Aniemore/wavlm-bert-fusion-s-emotion-russian-resd> | MIT |
| FastText `cc.ru.300.bin` (word vectors) | <https://fasttext.cc/docs/en/crawl-vectors.html> | CC BY-SA 3.0 |

## Datasets

| Dataset | Source | License |
|---|---|---|
| Dusha (Salute Developers / Sber) | <https://github.com/salute-developers/golos/tree/master/dusha> · paper <https://arxiv.org/abs/2212.12266> · archives <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/> | Dusha/Golos (attribution + share-alike) |
| RESD (`Aniemore/resd_annotated`, Hugging Face) | <https://huggingface.co/datasets/Aniemore/resd_annotated> · DOI <https://doi.org/10.57967/hf/1272> | MIT |

Details on the datasets and their usage — [`DUSHA.en.md`](./DUSHA.en.md) and
[`RESD.en.md`](./RESD.en.md); corpus building rules — [`CORPUS.en.md`](./CORPUS.en.md).
