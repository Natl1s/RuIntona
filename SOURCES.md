# Сторонние предобученные модели и данные: источники

В проекте используются сторонние предобученные модели и датасеты. Веса моделей и данные
не хранятся в репозитории — они скачиваются при запуске
(`ruintona/my_experiments/utils/pretrained.py`) либо загружаются с Hugging Face Hub
экспериментальными скриптами через `from_pretrained(...)`.

Юридическая сводка лицензий по всем частям проекта — [`NOTICE`](./NOTICE).

## Модели

| Компонент | Источник | Лицензия |
|---|---|---|
| RuBERT (`rubert-base-cased`, DeepPavlov) | <https://huggingface.co/DeepPavlov/rubert-base-cased> | Apache-2.0 |
| Wav2Vec2 XLS-R 300M (Facebook/Meta) | <https://huggingface.co/facebook/wav2vec2-xls-r-300m> | Apache-2.0 |
| HuBERT-large SER, дообучен на Dusha (xbgoose) | <https://huggingface.co/xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned> | Apache-2.0 |
| Whisper large-v3 SER, дообучен (firdhokk; база `openai/whisper-large-v3` — MIT) | <https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3> | Apache-2.0 |
| WavLM-BERT fusion SER на RESD (Aniemore) | <https://huggingface.co/Aniemore/wavlm-bert-fusion-s-emotion-russian-resd> | MIT |
| FastText `cc.ru.300.bin` (word vectors) | <https://fasttext.cc/docs/en/crawl-vectors.html> | CC BY-SA 3.0 |

## Датасеты

| Датасет | Источник | Лицензия |
|---|---|---|
| Dusha (Salute Developers / Сбер) | <https://github.com/salute-developers/golos/tree/master/dusha> · статья <https://arxiv.org/abs/2212.12266> · архивы <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/> | Dusha/Golos (attribution + share-alike) |
| RESD (`Aniemore/resd_annotated`, Hugging Face) | <https://huggingface.co/datasets/Aniemore/resd_annotated> · DOI <https://doi.org/10.57967/hf/1272> | MIT |

Подробнее о датасетах и их использовании в проекте — [`DUSHA.md`](./DUSHA.md) и
[`RESD.md`](./RESD.md); правила сборки корпусов — [`CORPUS.md`](./CORPUS.md).
