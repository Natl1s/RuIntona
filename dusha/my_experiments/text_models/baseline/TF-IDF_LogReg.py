"""
TF-IDF + Logistic Regression для классификации эмоций по тексту.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dusha.my_experiments.utils.config_utils import get_dataset_name, models_dir_for, load_experiment_config, add_data_path_args, resolve_data_paths
from dusha.my_experiments.utils.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists
from dusha.my_experiments.utils.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb
from dusha.my_experiments.utils.text_utils import preprocess_text
from dusha.my_experiments.utils.sklearn_utils import evaluate_sklearn_classifier

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem


def load_texts_from_manifest(manifest_path):
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_text)


def print_tfidf_vec_params(model, vectorizer):
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


def train_tfidf_logreg(save=True, config=None, train_path=None, test_path=None):
    cfg = {**DEFAULTS, **(config or {})}
    tfidf_params = cfg["tfidf"]
    logreg_params = cfg["logreg"]
    if isinstance(tfidf_params.get("ngram_range"), list):
        tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])

    train_manifest = train_path
    test_manifest = test_path
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

    metrics = evaluate_sklearn_classifier(
        model, X_train_texts, y_train, X_test_texts, y_test,
        transform_fn=vectorizer.transform,
        print_params_fn=lambda m: print_tfidf_vec_params(m, vectorizer),
    )

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


def load_and_evaluate(train_path=None, test_path=None):
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    model, vectorizer = load_sklearn_model(dataset_name, models_dir=MODELS_DIR, model_name=MODEL_NAME, artifact_name="vectorizer")

    print("\nЗагрузка обучающих данных...")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")

    print("\nЗагрузка тестовых данных...")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    evaluate_sklearn_classifier(
        model, X_train_texts, y_train, X_test_texts, y_test,
        transform_fn=vectorizer.transform,
        print_params_fn=lambda m: print_tfidf_vec_params(m, vectorizer),
    )
    return model, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение или загрузка модели TF-IDF + LogReg для классификации эмоций по тексту')
    parser.add_argument('--mode', type=str, choices=['train', 'load', 'auto', 'smoke'], default='auto')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--config', type=str, default=None, help='Путь к JSON-конфигу (относительно configs/ или абсолютный)')
    add_data_path_args(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)

    dataset_name = get_dataset_name(train_path)
    _exists = lambda dn: sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME, artifact_name="vectorizer")

    if args.mode == "smoke":
        print("💨 Режим: Smoke-тест\n")
        train_tfidf_logreg(save=False, config=experiment_config, train_path=train_path, test_path=test_path)
    elif args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        train_tfidf_logreg(save=not args.no_save, config=experiment_config, train_path=train_path, test_path=test_path)
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        load_and_evaluate(train_path=train_path, test_path=test_path)
    else:
        if _exists(dataset_name):
            print("📂 Режим: AUTO — найдена существующая модель, загружаем...\n")
            load_and_evaluate(train_path=train_path, test_path=test_path)
        else:
            print("🎯 Режим: AUTO — модель не найдена, начинаем обучение...\n")
            train_tfidf_logreg(save=not args.no_save, config=experiment_config, train_path=train_path, test_path=test_path)
