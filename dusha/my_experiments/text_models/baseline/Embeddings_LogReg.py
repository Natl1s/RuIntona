"""
FastText Embeddings + Logistic Regression для классификации эмоций по тексту.

Требования:
    pip install gensim
    Предобученная модель FastText для русского языка: https://fasttext.cc/docs/en/crawl-vectors.html
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import sys
_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, add_data_path_args, resolve_data_paths
from my_experiments.utils.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists, save_metrics_report
from my_experiments.utils.pretrained import get_fasttext_path
from my_experiments.utils.text_utils import preprocess_text, load_fasttext_model
from my_experiments.utils.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem
DEFAULT_EMBEDDINGS_PATH = get_fasttext_path()


def load_texts_from_manifest(manifest_path):
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_text)


def text_to_vector(text, fasttext_model):
    words = text.split()
    if not words:
        return np.zeros(fasttext_model.wv.vector_size, dtype=np.float32)
    word_vectors = []
    for word in words:
        try:
            word_vectors.append(fasttext_model.wv[word])
        except KeyError:
            continue
    if not word_vectors:
        return np.zeros(fasttext_model.wv.vector_size, dtype=np.float32)
    return np.mean(word_vectors, axis=0).astype(np.float32)


def texts_to_vectors(texts, fasttext_model, verbose=True):
    if verbose:
        print(f"Преобразование {len(texts)} текстов в векторы...")
        try:
            from tqdm import tqdm
            texts_iter = tqdm(texts, desc="Векторизация")
        except ImportError:
            texts_iter = texts
    else:
        texts_iter = texts
    vectors = [text_to_vector(text, fasttext_model) for text in texts_iter]
    vectors_matrix = np.vstack(vectors)
    if verbose:
        print(f"✓ Создана матрица векторов: {vectors_matrix.shape}")
    return vectors_matrix


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test, dataset_name, model_name=MODEL_NAME):
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Количество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"Размер матрицы коэффициентов: {model.coef_.shape}")
    print(f"Количество итераций: {model.n_iter_}")

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ ПРИЗНАКОВ")
    print(f"{'=' * 60}")
    print(f"Размерность embedding вектора: {X_train.shape[1]}")
    print(f"Количество обучающих примеров: {X_train.shape[0]}")
    print(f"Количество тестовых примеров: {X_test.shape[0]}")

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    train_pred = model.predict(X_train_scaled)
    train_report_text = classification_report(y_train, train_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0)
    print(train_report_text)

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    test_pred = model.predict(X_test_scaled)
    test_report_text = classification_report(y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0)
    print(test_report_text)

    print("\nМатрица ошибок:")
    test_cm = confusion_matrix(y_test, test_pred)
    print(test_cm)

    train_accuracy = float(accuracy_score(y_train, train_pred))
    test_accuracy = float(accuracy_score(y_test, test_pred))
    train_report_dict = classification_report(y_train, train_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0, output_dict=True)
    test_report_dict = classification_report(y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0, output_dict=True)

    metrics_lines = [
        "train_classification_report:", train_report_text.strip(), "",
        "test_classification_report:", test_report_text.strip(), "",
        "test_confusion_matrix:", json.dumps(test_cm.tolist(), ensure_ascii=False, indent=2), "",
        f"train_accuracy: {train_accuracy:.6f}", f"test_accuracy: {test_accuracy:.6f}",
    ]
    save_metrics_report(metrics_lines, dataset_name, models_dir=MODELS_DIR, model_name=model_name)

    return {
        "train_accuracy": train_accuracy, "train_classification_report_text": train_report_text,
        "train_classification_report": train_report_dict,
        "test_accuracy": test_accuracy, "test_classification_report_text": test_report_text,
        "test_classification_report": test_report_dict, "test_confusion_matrix": test_cm.tolist(),
    }


DEFAULTS = {
    "logreg": {
        "solver": "lbfgs", "max_iter": 1000,
        "random_state": 42, "class_weight": "balanced",
    },
}


def train_embeddings_logreg(embeddings_path=None, save=True, config=None, train_path=None, test_path=None):
    cfg_logreg = {**DEFAULTS["logreg"], **(config or {}).get("logreg", {})}
    cfg = {**(config or {}), "logreg": cfg_logreg}

    if embeddings_path is None:
        embeddings_path = cfg.get("embeddings_path") or DEFAULT_EMBEDDINGS_PATH

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА FASTTEXT EMBEDDINGS")
    print(f"{'=' * 60}")
    fasttext_model = load_fasttext_model(embeddings_path)

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"\n📊 Датасет: {dataset_name}\n")

    print(f"{'=' * 60}")
    print("ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ")
    print(f"{'=' * 60}")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")
    print(f"Пример текста после предобработки: '{X_train_texts[0]}'")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print(f"{'=' * 60}")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    print(f"\n{'=' * 60}")
    print("ПРЕОБРАЗОВАНИЕ ТЕКСТОВ В ВЕКТОРЫ")
    print(f"{'=' * 60}")
    X_train = texts_to_vectors(X_train_texts, fasttext_model, verbose=True)
    X_test = texts_to_vectors(X_test_texts, fasttext_model, verbose=True)

    print(f"\n{'=' * 60}")
    print("НОРМАЛИЗАЦИЯ ПРИЗНАКОВ")
    print(f"{'=' * 60}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Признаки нормализованы (StandardScaler)")

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ МОДЕЛИ LOGISTIC REGRESSION")
    print(f"{'=' * 60}")

    model = LogisticRegression(**cfg_logreg)
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    metrics = evaluate_model(model, scaler, X_train, y_train, X_test, y_test, dataset_name)

    if save:
        training_params = {
            "embeddings_path": str(embeddings_path),
            "logreg": {
                "solver": model.solver, "max_iter": model.max_iter,
                "random_state": model.random_state, "class_weight": model.class_weight,
            },
            "scaler": {"type": "StandardScaler"},
            "train_manifest": str(train_manifest), "test_manifest": str(test_manifest),
        }
        save_sklearn_model(
            model, scaler, dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            training_params=training_params, test_metrics=metrics,
        )

    return model, scaler, dataset_name


def load_and_evaluate(embeddings_path=None, train_path=None, test_path=None):
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    if embeddings_path is None:
        embeddings_path = DEFAULT_EMBEDDINGS_PATH

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА FASTTEXT EMBEDDINGS")
    print(f"{'=' * 60}")
    fasttext_model = load_fasttext_model(embeddings_path)

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    model, scaler = load_sklearn_model(dataset_name, models_dir=MODELS_DIR, model_name=MODEL_NAME)

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ")
    print(f"{'=' * 60}")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print(f"{'=' * 60}")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    print(f"\n{'=' * 60}")
    print("ПРЕОБРАЗОВАНИЕ ТЕКСТОВ В ВЕКТОРЫ")
    print(f"{'=' * 60}")
    X_train = texts_to_vectors(X_train_texts, fasttext_model, verbose=False)
    X_test = texts_to_vectors(X_test_texts, fasttext_model, verbose=False)

    evaluate_model(model, scaler, X_train, y_train, X_test, y_test, dataset_name)
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение или загрузка модели FastText Embeddings + LogReg')
    parser.add_argument('--mode', type=str, choices=['train', 'load', 'auto', 'smoke'], default='auto')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--embeddings-path', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Путь к JSON-конфигу (относительно configs/ или абсолютный)')
    add_data_path_args(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)

    dataset_name = get_dataset_name(train_path)

    if args.mode == "smoke":
        print("💨 Режим: Smoke-тест\n")
        train_embeddings_logreg(embeddings_path=args.embeddings_path, save=False, config=experiment_config, train_path=train_path, test_path=test_path)
    elif args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        train_embeddings_logreg(embeddings_path=args.embeddings_path, save=not args.no_save, config=experiment_config, train_path=train_path, test_path=test_path)
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        load_and_evaluate(embeddings_path=args.embeddings_path, train_path=train_path, test_path=test_path)
    else:
        if sklearn_model_exists(dataset_name, models_dir=MODELS_DIR, model_name=MODEL_NAME):
            print("📂 Режим: AUTO — найдена существующая модель, загружаем...\n")
            load_and_evaluate(embeddings_path=args.embeddings_path, train_path=train_path, test_path=test_path)
        else:
            print("🎯 Режим: AUTO — модель не найдена, начинаем обучение...\n")
            train_embeddings_logreg(embeddings_path=args.embeddings_path, save=not args.no_save, config=experiment_config, train_path=train_path, test_path=test_path)
