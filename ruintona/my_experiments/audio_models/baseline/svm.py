import argparse
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from ruintona.my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, apply_config_to_args, add_config_arg, add_data_path_args, resolve_data_paths
from ruintona.my_experiments.utils.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists
from ruintona.my_experiments.utils.lmdb_utils import load_audio_features_from_lmdb
from ruintona.my_experiments.utils.sklearn_utils import evaluate_sklearn_classifier

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem


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


def train_svm(save=True, kernel="rbf", C=1.0, gamma="scale", train_path=None, test_path=None):
    """Обучение SVM классификатора."""
    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"Датасет: {dataset_name}\n")

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

    print("\nНормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ SVM МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Параметры: kernel={kernel}, C={C}, gamma={gamma}")

    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=False, random_state=42, verbose=True)
    model.fit(X_train_scaled, y_train)
    print("Обучение завершено!")

    metrics = evaluate_sklearn_classifier(
        model, X_train, y_train, X_test, y_test,
        transform_fn=scaler.transform,
        print_params_fn=print_svm_parameters,
    )

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


def load_and_evaluate(train_path=None, test_path=None):
    """Загрузить существующую модель и оценить её."""
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"Датасет: {dataset_name}\n")

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
        print_params_fn=print_svm_parameters,
    )
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение или загрузка SVM для классификации эмоций"
    )
    parser.add_argument("--mode", type=str, choices=["train", "load", "auto", "smoke"], default="auto")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--kernel", type=str, choices=["linear", "rbf", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    add_data_path_args(parser)
    add_config_arg(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config, parser)

    try:
        gamma_value = float(args.gamma)
    except ValueError:
        gamma_value = args.gamma

    dataset_name = get_dataset_name(train_path)
    _exists = lambda dn: sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME)

    if args.mode == "smoke":
        print("Режим: Smoke-тест\n")
        train_svm(save=False, kernel=args.kernel, C=args.C, gamma=gamma_value, train_path=train_path, test_path=test_path)
    elif args.mode == "train":
        print("Режим: Обучение новой модели\n")
        train_svm(save=not args.no_save, kernel=args.kernel, C=args.C, gamma=gamma_value, train_path=train_path, test_path=test_path)
    elif args.mode == "load":
        print("Режим: Загрузка существующей модели\n")
        load_and_evaluate(train_path=train_path, test_path=test_path)
    else:
        if _exists(dataset_name):
            print("Режим: AUTO — найдена существующая модель, загружаем...\n")
            load_and_evaluate(train_path=train_path, test_path=test_path)
        else:
            print("Режим: AUTO — модель не найдена, начинаем обучение...\n")
            train_svm(save=not args.no_save, kernel=args.kernel, C=args.C, gamma=gamma_value, train_path=train_path, test_path=test_path)
