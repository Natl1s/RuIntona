"""
TF-IDF + Logistic Regression для классификации эмоций по тексту.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
_PROJECT_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if _parent.name == "my_experiments":
        _PROJECT_ROOT = _parent.parent
        break
if _PROJECT_ROOT and str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from my_experiments.config_utils import TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config
from my_experiments.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists, save_metrics_report
from my_experiments.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb
from my_experiments.text_utils import preprocess_text

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem


def load_texts_from_manifest(manifest_path):
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_text)


def evaluate_model(model, vectorizer, X_train_texts, y_train, X_test_texts, y_test, dataset_name, model_name=MODEL_NAME):
    X_train_tfidf = vectorizer.transform(X_train_texts)
    X_test_tfidf = vectorizer.transform(X_test_texts)

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Количество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"Размер матрицы коэффициентов: {model.coef_.shape}")
    print(f"Количество итераций: {model.n_iter_}")

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ TF-IDF ВЕКТОРИЗАТОРА")
    print(f"{'=' * 60}")
    print(f"Размер словаря: {len(vectorizer.vocabulary_)}")
    print(f"Диапазон n-грамм: {vectorizer.ngram_range}")
    print(f"Максимальное количество признаков: {vectorizer.max_features}")
    print(f"Минимальная частота документов (min_df): {vectorizer.min_df}")
    print(f"Максимальная частота документов (max_df): {vectorizer.max_df}")

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    train_pred = model.predict(X_train_tfidf)
    print(classification_report(y_train, train_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0))

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    test_pred = model.predict(X_test_tfidf)
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
    "tfidf": {
        "ngram_range": [1, 2], "max_features": 10000,
        "min_df": 2, "max_df": 0.8, "sublinear_tf": True,
    },
    "logreg": {
        "solver": "lbfgs", "max_iter": 1000,
        "random_state": 42, "class_weight": "balanced",
    },
}


def train_tfidf_logreg(save=True, config=None):
    cfg = {**DEFAULTS, **(config or {})}
    tfidf_params = cfg["tfidf"]
    logreg_params = cfg["logreg"]
    # ngram_range хранится как список в JSON, преобразуем в tuple
    if isinstance(tfidf_params.get("ngram_range"), list):
        tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    print("Загрузка обучающих данных...")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")
    print(f"Пример текста после предобработки: '{X_train_texts[0]}'")

    print("\nЗагрузка тестовых данных...")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    print(f"\n{'=' * 60}")
    print("ПОСТРОЕНИЕ TF-IDF ПРИЗНАКОВ")
    print(f"{'=' * 60}")

    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train_texts)
    print(f"✓ Размер матрицы TF-IDF признаков (train): {X_train_tfidf.shape}")
    print(f"  - Количество документов: {X_train_tfidf.shape[0]}")
    print(f"  - Количество признаков (размер словаря): {X_train_tfidf.shape[1]}")

    X_test_tfidf = vectorizer.transform(X_test_texts)
    print(f"✓ Размер матрицы TF-IDF признаков (test): {X_test_tfidf.shape}")

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ МОДЕЛИ LOGISTIC REGRESSION")
    print(f"{'=' * 60}")

    model = LogisticRegression(**logreg_params)
    model.fit(X_train_tfidf, y_train)
    print("✓ Обучение завершено!")

    metrics = evaluate_model(model, vectorizer, X_train_texts, y_train, X_test_texts, y_test, dataset_name)

    if save:
        training_params = {
            "tfidf_vectorizer": {
                "ngram_range": vectorizer.ngram_range, "max_features": vectorizer.max_features,
                "min_df": vectorizer.min_df, "max_df": vectorizer.max_df, "sublinear_tf": vectorizer.sublinear_tf,
            },
            "logreg": {
                "solver": model.solver, "max_iter": model.max_iter,
                "random_state": model.random_state, "class_weight": model.class_weight,
            },
            "train_manifest": str(train_manifest), "test_manifest": str(test_manifest),
        }
        save_sklearn_model(
            model, vectorizer, dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            artifact_name="vectorizer", training_params=training_params, test_metrics=metrics,
        )

    return model, vectorizer, dataset_name


def load_and_evaluate():
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    model, vectorizer = load_sklearn_model(dataset_name, models_dir=MODELS_DIR, model_name=MODEL_NAME, artifact_name="vectorizer")

    print("\nЗагрузка обучающих данных...")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")

    print("\nЗагрузка тестовых данных...")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    evaluate_model(model, vectorizer, X_train_texts, y_train, X_test_texts, y_test, dataset_name)
    return model, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение или загрузка модели TF-IDF + LogReg для классификации эмоций по тексту')
    parser.add_argument('--mode', type=str, choices=['train', 'load', 'auto'], default='auto')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--config', type=str, default=None, help='Путь к JSON-конфигу (относительно configs/ или абсолютный)')
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.config)

    dataset_name = get_dataset_name(TRAIN_DATA_PATH)
    _exists = lambda dn: sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME, artifact_name="vectorizer")

    if args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        train_tfidf_logreg(save=not args.no_save, config=experiment_config)
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        load_and_evaluate()
    else:
        if _exists(dataset_name):
            print("📂 Режим: AUTO — найдена существующая модель, загружаем...\n")
            load_and_evaluate()
        else:
            print("🎯 Режим: AUTO — модель не найдена, начинаем обучение...\n")
            train_tfidf_logreg(save=not args.no_save, config=experiment_config)
