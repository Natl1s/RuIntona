import argparse
import numpy as np
from pathlib import Path
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
from my_experiments.utils.cli_utils import add_mode_args, dispatch_mode
from my_experiments.utils.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists
from my_experiments.utils.lmdb_utils import load_feature_vectors_from_lmdb

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem


def _to_fixed_vector(feat: np.ndarray) -> np.ndarray:
    """Преобразует feature-тензор в вектор фиксированной длины."""
    arr = np.asarray(feat)
    if arr.ndim == 0:
        raise ValueError("Пустой/скалярный feature-тензор")
    if arr.ndim == 1:
        return arr.astype(np.float32)
    mean_part = arr.mean(axis=-1).reshape(-1)
    std_part = arr.std(axis=-1).reshape(-1)
    return np.concatenate([mean_part, std_part]).astype(np.float32)


def load_features_from_lmdb(lmdb_path):
    return load_feature_vectors_from_lmdb(
        lmdb_path=Path(lmdb_path),
        vectorize_fn=_to_fixed_vector,
        label_kind="emotion",
    )


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Оценка модели на обучающей и тестовой выборках."""
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
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    train_pred = model.predict(X_train_scaled)
    train_report_text = classification_report(
        y_train, train_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0,
    )
    print(train_report_text)

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    test_pred = model.predict(X_test_scaled)
    test_report_text = classification_report(
        y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0,
    )
    print(test_report_text)

    print("\nМатрица ошибок:")
    test_cm = confusion_matrix(y_test, test_pred)
    print(test_cm)

    test_report_dict = classification_report(
        y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES,
        zero_division=0, output_dict=True,
    )

    return {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_classification_report_text": test_report_text,
        "test_classification_report": test_report_dict,
        "test_confusion_matrix": test_cm.tolist(),
    }


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
    X_train, y_train = load_features_from_lmdb(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_features_from_lmdb(test_manifest)
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

    metrics = evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

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
    X_train, y_train = load_features_from_lmdb(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_features_from_lmdb(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    evaluate_model(model, scaler, X_train, y_train, X_test, y_test)
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
