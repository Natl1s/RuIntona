# Data analysis scripts

## build_balanced_aggregated_jsonl.py

Creates 4 balanced JSONL datasets inside `aggregated_dataset`:

- `combine_balanced_train.jsonl`
- `combine_balanced_test.jsonl`
- `combine_balanced_train_small.jsonl`
- `combine_balanced_test_small.jsonl`

Rules:

1. Source for train: `crowd_train.jsonl + podcast_train.jsonl`
2. Source for test: `crowd_test.jsonl + podcast_test.jsonl`
3. Uses only target emotions: `angry`, `sad`, `neutral`, `positive`
4. For full sets: `neutral <= 2 * min(non-neutral class count)`
5. For small sets: size is 30% (configurable) of full set, class ratio is preserved as close as possible.

Run:

```bash
python /home/natlis/PycharmProjects/dusha_new/dusha/my_experiments/data_analise/build_balanced_aggregated_jsonl.py
```

With custom options:

```bash
python /home/natlis/PycharmProjects/dusha_new/dusha/my_experiments/data_analise/build_balanced_aggregated_jsonl.py \
  --aggregated-dir /home/natlis/PycharmProjects/dusha_new/dusha/dataset/processed_dataset_090/aggregated_dataset \
  --small-ratio 0.3 \
  --seed 42
```
