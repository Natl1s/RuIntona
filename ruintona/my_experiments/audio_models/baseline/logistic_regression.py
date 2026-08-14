import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, add_data_path_args, resolve_data_paths
from ruintona.my_experiments.utils.cli_utils import add_mode_args, dispatch_mode
from ruintona.my_experiments.utils.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists
from ruintona.my_experiments.utils.lmdb_utils import load_audio_features_from_lmdb
from ruintona.my_experiments.utils.sklearn_utils import evaluate_sklearn_classifier

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem


def print_logreg_params(model) -> None:
    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Количество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"Размер матрицы коэффициентов: {model.coef_.shape}")
    print(f"Количество итераций: {model.n_iter_}")


DEFAULTS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": None,
}


def train_logistic_regression(save=True, config=None, train_path=None, test_path=None):
    cfg = {**DEFAULTS, **(config or {})}
    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    print("Загрузка обучающих данных...")
    X_train, y_train = load_audio_features_from_lmdb(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_audio_features_from_lmdb(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    print("Нормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print(f"{'=' * 60}")
    model = LogisticRegression(
        solver=cfg["solver"],
        max_iter=cfg["max_iter"],
        random_state=cfg["random_state"],
        class_weight=cfg["class_weight"],
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    metrics = evaluate_sklearn_classifier(
        model, X_train, y_train, X_test, y_test,
        transform_fn=scaler.transform,
        print_params_fn=print_logreg_params,
    )

    if save:
        training_params = {
            "solver": model.solver,
            "max_iter": model.max_iter,
            "random_state": model.random_state,
            "class_weight": model.class_weight,
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
        }
        save_sklearn_model(
            model, scaler, dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            training_params=training_params, test_metrics=metrics,
        )

    return model, scaler, dataset_name


def load_and_evaluate(train_path=None, test_path=None):
    """Загрузить существующую модель и оценить её."""
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    model, scaler = load_sklearn_model(
        dataset_name, models_dir=MODELS_DIR, model_name=MODEL_NAME,
    )

    print("\nЗагрузка обучающих данных...")
    X_train, y_train = load_audio_features_from_lmdb(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_audio_features_from_lmdb(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    evaluate_sklearn_classifier(
        model, X_train, y_train, X_test, y_test,
        transform_fn=scaler.transform,
        print_params_fn=print_logreg_params,
    )
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение или загрузка логистической регрессии для классификации эмоций"
    )
    add_mode_args(parser)
    add_data_path_args(parser)
    parser.add_argument("--config", type=str, default=None, help="Путь к JSON-конфигу (относительно configs/ или абсолютный)")
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)

    dataset_name = get_dataset_name(train_path)
    dispatch_mode(
        args,
        train_fn=lambda save: train_logistic_regression(save=save, config=experiment_config, train_path=train_path, test_path=test_path),
        load_fn=lambda: load_and_evaluate(train_path=train_path, test_path=test_path),
        model_exists_fn=lambda dn: sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME),
        dataset_name=dataset_name,
    )
