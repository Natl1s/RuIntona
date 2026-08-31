# Tests (`my_experiments/tests/`)

Smoke tests for the models: each script is run in `--mode smoke` on tiny synthetic LMDBs. Run in CI (`.github/workflows/ci.yml`: `ruff` + `pytest`).

> Sources and licenses of third-party pretrained models and datasets — [`SOURCES.en.md`](../../../SOURCES.en.md).

## Structure

| File | Purpose |
|---|---|
| `conftest.py` | Fixtures: tiny audio/text/multimodal datasets |
| `smoke_helpers.py` | Synthetic LMDB generators and helper functions |
| `test_smoke_audio.py` | Smoke tests for audio models |
| `test_smoke_text.py` | Smoke tests for text models |
| `test_smoke_multimodal.py` | Smoke tests for multimodal models (early fusion) |
| `test_model_io.py` | PyTorch checkpoint load tests (numpy globals / `weights_only` fallback) |
| `test_pretrained.py` | Tests for `utils/pretrained.py` path resolution (FastText) |

`test_smoke_audio.py` coverage: `logreg`, `svm`, `random-forest`, `cnn`, `cnn-bilstm`.
`test_smoke_text.py` coverage: `tfidf-logreg`, `embeddings-logreg`, `bilstm`, `rubert`.
`test_smoke_multimodal.py` coverage: `early-fusion` (requires `torch` + `transformers`).

## Run

```bash
poetry run pytest ruintona/my_experiments/tests/ -v

# local lint (as in CI)
poetry run ruff check ruintona/my_experiments/tests/
```

Note: the tests do not check model quality; they only ensure the pipeline (train/inference) runs end-to-end without errors.
