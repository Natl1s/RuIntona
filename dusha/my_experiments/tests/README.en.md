# Tests (`my_experiments/tests/`)

Smoke tests for the models: each script is run in `--mode smoke` on tiny synthetic LMDBs. Run in CI (`.github/workflows/ci.yml`: `ruff` + `pytest`).

## Structure

| File | Purpose |
|---|---|
| `conftest.py` | Fixtures: tiny audio/text/multimodal datasets |
| `smoke_helpers.py` | Synthetic LMDB generators and helper functions |
| `test_smoke_audio.py` | Smoke tests for audio models |
| `test_smoke_text.py` | Smoke tests for text models |

`test_smoke_audio.py` coverage: `logreg`, `svm`, `random-forest`, `cnn`, `cnn-bilstm`.
`test_smoke_text.py` coverage: `tfidf-logreg`, `embeddings-logreg`, `bilstm`, `bilstm-text`, `rubert`.

## Run

```bash
poetry run pytest dusha/my_experiments/tests/ -v

# local lint (as in CI)
poetry run ruff check dusha/my_experiments/tests/
```

Note: the tests do not check model quality; they only ensure the pipeline (train/inference) runs end-to-end without errors.
