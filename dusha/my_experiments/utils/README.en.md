# Utilities (`my_experiments/utils/`)

Shared modules used by all experiments: configuration, LMDB access, metrics, model save/load, model registry, pretrained models, PyTorch/sklearn helpers, CLI.

| Module | Purpose | Key functions |
|---|---|---|
| `config_utils.py` | Central configuration: `data.json`, constants, checkpoint paths, JSON config loading | `EMO2LABEL`, `TARGET_NAMES`, `DATASET_PATH`, `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `CHECKPOINTS_DIR`, `resolve_model_path()`, `checkpoints_dir_for()`, `load_experiment_config()`, `apply_config_to_args()`, `add_config_arg()`, `add_data_path_args()`, `resolve_data_paths()`, `find_pretrained_model()` |
| `lmdb_utils.py` | LMDB reading: safe deserialization, loading features/texts/audio | `open_lmdb_readonly()`, `safe_pickle_loads()`, `iter_lmdb_payloads()`, `load_feature_vectors_from_lmdb()`, `load_texts_from_lmdb()`, `load_audio_features_from_lmdb()`, `to_fixed_vector()` |
| `metrics.py` | Classification metrics computation and display (incl. notebooks) | `compute_classification_metrics()`, `weighted_accuracy()`, `print_eval_block()` |
| `model_io.py` | Save/load sklearn (joblib) and PyTorch models, CSV experiment log | `save_checkpoint()`, `load_checkpoint()`, `save_sklearn_model()`, `save_pytorch_model()`, `load_torch_with_weights()`, `log_experiment_to_csv()` |
| `model_registry.py` | Registry of trained models in `checkpoints/registry.json` | `ModelRegistry.register()`, `.get()`, `.find()`, `.latest()`, `.remove()`, `.list_all()` |
| `pretrained.py` | Paths to pretrained models (FastText, RuBERT, Wav2Vec2) | `get_fasttext_path()`, `load_fasttext_model()`, `get_rubert_path()`, `get_wav2vec_path()`, `resolve_pretrained()`, `list_pretrained()` |
| `sklearn_utils.py` | Unified sklearn classifier evaluation | `evaluate_sklearn_classifier()` |
| `text_utils.py` | Text utilities: extraction, preprocessing, FastText | `extract_text()`, `preprocess_text()`, `load_texts_from_manifest()`, `load_fasttext_model()` |
| `torch_utils.py` | PyTorch utilities: reproducibility, device, loaders, evaluation | `set_seed()`, `resolve_device()`, `build_loader()`, `EarlyStopping`, `evaluate_split()`, `ensure_transformers_compat()` |
| `cli_utils.py` | Common CLI for experiments | `add_mode_args()`, `dispatch_mode()` |

## Key conventions

- **Emotion labels**: `EMO2LABEL = {'angry': 0, 'sad': 1, 'neutral': 2, 'positive': 3}`, `TARGET_NAMES = ['angry', 'sad', 'neutral', 'positive']`.
- **Data paths** are set in `data.json` and resolved via `config_utils`.
- **Checkpoints**: name convention `{Model}_{dataset}_model.{pt|pkl}`; folders `checkpoints/{text,audio,multimodal}/` are detected automatically from the script path.
- **Modes**: `--mode train|load|auto|smoke` — shared by all models (`cli_utils.dispatch_mode`).
