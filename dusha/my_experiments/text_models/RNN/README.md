# RNN Text Models

`BiLSTM_Text.py` - BiLSTM-классификатор 4 эмоций (`angry`, `sad`, `neutral`, `positive`) по тексту из LMDB.

## Что делает скрипт

- грузит пути train/test из конфигов `dusha/my_experiments/train_data.config` и `dusha/my_experiments/test_data.config`
- читает только текст и метку из LMDB-record payload
- строит словарь по train-текстам
- обучает `Embedding -> BiLSTM -> Linear`
- показывает метрики и матрицу ошибок
- поддерживает режимы `train/load/auto`
- сохраняет веса, словарь, meta и текстовый отчет в `models_params/`

## Быстрый запуск

```bash
poetry run python dusha/my_experiments/text_models/RNN/BiLSTM_Text.py --mode train
```

```bash
poetry run python dusha/my_experiments/text_models/RNN/BiLSTM_Text.py --mode auto --device auto
```

## Полезные параметры

- `--epochs` (default: `12`)
- `--batch-size` (default: `64`)
- `--embed-dim` (default: `256`)
- `--hidden-size` (default: `128`)
- `--lstm-layers` (default: `2`)
- `--max-vocab-size` (default: `40000`)
- `--min-freq` (default: `2`)
- `--max-len` (default: `256`)
- `--device` (`cuda`, `cpu`, `auto`)

## Артефакты

Для датасета `combine_balanced_train` будут созданы:

- `BiLSTM_Text_combine_balanced_train_model.pt`
- `BiLSTM_Text_combine_balanced_train_model_<timestamp>.pt`
- `BiLSTM_Text_combine_balanced_train_vocab.json`
- `BiLSTM_Text_combine_balanced_train_meta.json`
- `BiLSTM_Text_combine_balanced_train_training_report.txt`
