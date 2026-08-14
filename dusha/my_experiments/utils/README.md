# Утилиты (`my_experiments/utils/`)

Общие модули, используемые всеми экспериментами: конфигурация, работа с LMDB, метрики, сохранение/загрузка моделей, реестр моделей, предобученные модели, PyTorch/sklearn-хелперы, CLI.

| Модуль | Назначение | Ключевые функции |
|---|---|---|
| `config_utils.py` | Центральная конфигурация: `data.json`, константы, пути чекпоинтов, загрузка JSON-конфигов | `EMO2LABEL`, `TARGET_NAMES`, `DATASET_PATH`, `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `CHECKPOINTS_DIR`, `resolve_model_path()`, `checkpoints_dir_for()`, `load_experiment_config()`, `apply_config_to_args()`, `add_config_arg()`, `add_data_path_args()`, `resolve_data_paths()`, `find_pretrained_model()` |
| `lmdb_utils.py` | Чтение LMDB: безопасная десериализация, загрузка признаков/текстов/аудио | `open_lmdb_readonly()`, `safe_pickle_loads()`, `iter_lmdb_payloads()`, `load_feature_vectors_from_lmdb()`, `load_texts_from_lmdb()`, `load_audio_features_from_lmdb()`, `to_fixed_vector()` |
| `metrics.py` | Расчёт и вывод метрик классификации (в т.ч. в ноутбуках) | `compute_classification_metrics()`, `weighted_accuracy()`, `print_eval_block()` |
| `model_io.py` | Сохранение/загрузка sklearn (joblib) и PyTorch-моделей, CSV-лог экспериментов | `save_checkpoint()`, `load_checkpoint()`, `save_sklearn_model()`, `save_pytorch_model()`, `load_torch_with_weights()`, `log_experiment_to_csv()` |
| `model_registry.py` | Реестр обученных моделей в `checkpoints/registry.json` | `ModelRegistry.register()`, `.get()`, `.find()`, `.latest()`, `.remove()`, `.list_all()` |
| `pretrained.py` | Пути к предобученным моделям (FastText, RuBERT, Wav2Vec2) | `get_fasttext_path()`, `load_fasttext_model()`, `get_rubert_path()`, `get_wav2vec_path()`, `resolve_pretrained()`, `list_pretrained()` |
| `sklearn_utils.py` | Единая оценка sklearn-классификаторов | `evaluate_sklearn_classifier()` |
| `text_utils.py` | Текстовые утилиты: извлечение текста, предобработка, FastText | `extract_text()`, `preprocess_text()`, `load_texts_from_manifest()`, `load_fasttext_model()` |
| `torch_utils.py` | PyTorch-утилиты: воспроизводимость, устройство, даталоадеры, оценка | `set_seed()`, `resolve_device()`, `build_loader()`, `EarlyStopping`, `evaluate_split()`, `ensure_transformers_compat()` |
| `cli_utils.py` | Единый CLI для экспериментов | `add_mode_args()`, `dispatch_mode()` |

## Ключевые конвенции

- **Метки эмоций**: `EMO2LABEL = {'angry': 0, 'sad': 1, 'neutral': 2, 'positive': 3}`, `TARGET_NAMES = ['angry', 'sad', 'neutral', 'positive']`.
- **Пути к данным** задаются в `data.json` и резолвятся через `config_utils`.
- **Чекпоинты**: конвенция имён `{Model}_{dataset}_model.{pt|pkl}`; папки `checkpoints/{text,audio,multimodal}/` определяются автоматически по пути скрипта.
- **Режимы**: `--mode train|load|auto|smoke` — общий для всех моделей (`cli_utils.dispatch_mode`).
