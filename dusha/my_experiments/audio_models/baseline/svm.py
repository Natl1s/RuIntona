import argparse
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
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

from my_experiments.config_utils import TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, apply_config_to_args, add_config_arg
from my_experiments.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists
from my_experiments.lmdb_utils import load_feature_vectors_from_lmdb

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


def print_svm_parameters(model):
    """Вывод критичных параметров SVM модели."""
    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ SVM МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Ядро (kernel): {model.kernel}")
    print(f"Параметр C (регуляризация): {model.C}")
    if model.kernel in ["rbf", "poly", "sigmoid"]:
        print(f"Параметр gamma: {model.gamma}")
    if model.kernel == "poly":
        print(f"Степень полинома (degree): {model.degree}")
    print(f"\nКоличество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"\nОбщее количество опорных векторов: {model.n_support_.sum()}")
    print(f"Опорные векторы по классам: {dict(zip(model.classes_, model.n_support_))}")
    print(f"Процент опорных векторов: {model.n_support_.sum() / len(model.support_) * 100:.2f}%")
    print(f"\nФорма матрицы опорных векторов: {model.support_vectors_.shape}")
    print(f"Форма матрицы dual_coef: {model.dual_coef_.shape}")
    print(f"\nСтатистика dual coefficients:")
    print(f"  Min: {model.dual_coef_.min():.6f}")
    print(f"  Max: {model.dual_coef_.max():.6f}")
    print(f"  Mean: {model.dual_coef_.mean():.6f}")
    print(f"  Std: {model.dual_coef_.std():.6f}")


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Оценка модели на обучающей и тестовой выборках."""
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print_svm_parameters(model)

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


def train_svm(save=True, kernel="rbf", C=1.0, gamma="scale"):
    """Обучение SVM классификатора."""
    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
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

    print("\nНормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ SVM МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Параметры: kernel={kernel}, C={C}, gamma={gamma}")

    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42, verbose=True)
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    metrics = evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

    if save:
        training_params = {
            "kernel": model.kernel, "C": model.C, "gamma": model.gamma,
            "degree": model.degree, "coef0": model.coef0,
            "class_weight": model.class_weight, "probability": model.probability,
            "random_state": model.random_state,
            "train_manifest": str(train_manifest), "test_manifest": str(test_manifest),
        }
        save_sklearn_model(
            model, scaler, dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            training_params=training_params, test_metrics=metrics,
        )

    return model, scaler, dataset_name


def load_and_evaluate():
    """Загрузить существующую модель и оценить её."""
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
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
        description="Обучение или загрузка SVM для классификации эмоций"
    )
    parser.add_argument("--mode", type=str, choices=["train", "load", "auto"], default="auto")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--kernel", type=str, choices=["linear", "rbf", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    add_config_arg(parser)
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config)

    try:
        gamma_value = float(args.gamma)
    except ValueError:
        gamma_value = args.gamma

    dataset_name = get_dataset_name(TRAIN_DATA_PATH)
    _exists = lambda dn: sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME)

    if args.mode == "train":
        print("🎯 Режим: Обучение новой модели\n")
        train_svm(save=not args.no_save, kernel=args.kernel, C=args.C, gamma=gamma_value)
    elif args.mode == "load":
        print("📂 Режим: Загрузка существующей модели\n")
        load_and_evaluate()
    else:
        if _exists(dataset_name):
            print("📂 Режим: AUTO — найдена существующая модель, загружаем...\n")
            load_and_evaluate()
        else:
            print("🎯 Режим: AUTO — модель не найдена, начинаем обучение...\n")
            train_svm(save=not args.no_save, kernel=args.kernel, C=args.C, gamma=gamma_value)
