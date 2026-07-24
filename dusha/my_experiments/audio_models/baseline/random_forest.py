import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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

from my_experiments.utils.config_utils import TARGET_NAMES, get_dataset_name, models_dir_for, load_experiment_config, apply_config_to_args, add_config_arg, add_data_path_args, resolve_data_paths
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


def print_random_forest_parameters(model, feature_dim):
    """Вывод критичных параметров Random Forest модели."""
    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ RANDOM FOREST МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Количество деревьев (n_estimators): {model.n_estimators}")
    print(f"Максимальная глубина (max_depth): {model.max_depth if model.max_depth else 'Неограничена'}")
    print(f"Минимум samples для split (min_samples_split): {model.min_samples_split}")
    print(f"Минимум samples в листе (min_samples_leaf): {model.min_samples_leaf}")
    print(f"Максимум признаков для split (max_features): {model.max_features}")
    print(f"Критерий разделения: {model.criterion}")
    print(f"\nКоличество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"\nСтатистика по деревьям:")
    tree_depths = [tree.get_depth() for tree in model.estimators_]
    tree_leaves = [tree.get_n_leaves() for tree in model.estimators_]
    print(f"  Глубина деревьев — Min: {min(tree_depths)}, Max: {max(tree_depths)}, Mean: {np.mean(tree_depths):.2f}")
    print(f"  Листьев в деревьях — Min: {min(tree_leaves)}, Max: {max(tree_leaves)}, Mean: {np.mean(tree_leaves):.2f}")
    print(f"\nВажность признаков (Feature Importance):")
    feature_importances = model.feature_importances_
    top_indices = np.argsort(feature_importances)[-10:][::-1]
    print(f"  Топ-10 признаков:")
    for i, idx in enumerate(top_indices, 1):
        print(f"    {i}. Признак {idx}: {feature_importances[idx]:.6f}")
    print(f"\nСтатистика важности признаков:")
    print(f"  Min: {feature_importances.min():.6f}, Max: {feature_importances.max():.6f}")
    print(f"  Mean: {feature_importances.mean():.6f}, Std: {feature_importances.std():.6f}")
    print(f"  Признаков с нулевой важностью: {np.sum(feature_importances == 0)}/{len(feature_importances)}")
    if hasattr(model, "oob_score_"):
        print(f"\nOut-of-Bag Score: {model.oob_score_:.4f}")


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Оценка модели на обучающей и тестовой выборках."""
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print_random_forest_parameters(model, X_train.shape[1])

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    train_pred = model.predict(X_train_scaled)
    print(classification_report(y_train, train_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0))

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    test_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0))

    print("\nМатрица ошибок:")
    test_cm = confusion_matrix(y_test, test_pred)
    print(test_cm)

    test_report_dict = classification_report(
        y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0, output_dict=True,
    )
    return {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_classification_report_text": classification_report(y_test, test_pred, labels=TARGET_NAMES, target_names=TARGET_NAMES, zero_division=0),
        "test_classification_report": test_report_dict,
        "test_confusion_matrix": test_cm.tolist(),
    }


def train_random_forest(save=True, n_estimators=100, max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        max_features="sqrt", oob_score=False,
                        train_path=None, test_path=None):
    """Обучение Random Forest классификатора."""
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

    print("\nНормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ RANDOM FOREST МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Параметры: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
          f"max_features={max_features}, oob_score={oob_score}")

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, oob_score=oob_score,
        random_state=42, n_jobs=-1, verbose=1,
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    metrics = evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

    if save:
        training_params = {
            "n_estimators": model.n_estimators, "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split, "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features, "criterion": model.criterion,
            "oob_score": model.oob_score, "random_state": model.random_state,
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
    print(f"📊 Датасет: {dataset_name}\n")

    model, scaler = load_sklearn_model(dataset_name, models_dir=MODELS_DIR, model_name=MODEL_NAME)

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
        description="Обучение или загрузка Random Forest для классификации эмоций"
    )
    parser.add_argument("--mode", type=str, choices=["train", "load", "auto", "smoke"], default="auto")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--oob-score", action="store_true")
    add_data_path_args(parser)
    add_config_arg(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config)

    if args.max_features not in ["sqrt", "log2", "None"]:
        try:
            max_features_value = int(args.max_features)
        except ValueError:
            try:
                max_features_value = float(args.max_features)
            except ValueError:
                max_features_value = args.max_features
    else:
        max_features_value = None if args.max_features == "None" else args.max_features

    dataset_name = get_dataset_name(train_path)
    _exists = lambda dn: sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME)

    def _train(save=True):
        return train_random_forest(
            save=save, n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            max_features=max_features_value, oob_score=args.oob_score,
            train_path=train_path, test_path=test_path,
        )

    if args.mode == "smoke":
        print("💨 Режим: Smoke-тест\n")
        train_random_forest(
            save=False, n_estimators=10, max_depth=3,
            min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", oob_score=False,
            train_path=train_path, test_path=test_path,
        )
    elif args.mode == "train":
        print("🎯 Режим: Обучение новой модели\n")
        _train(save=not args.no_save)
    elif args.mode == "load":
        print("📂 Режим: Загрузка существующей модели\n")
        load_and_evaluate(train_path=train_path, test_path=test_path)
    else:
        if _exists(dataset_name):
            print("📂 Режим: AUTO — найдена существующая модель, загружаем...\n")
            load_and_evaluate(train_path=train_path, test_path=test_path)
        else:
            print("🎯 Режим: AUTO — модель не найдена, начинаем обучение...\n")
            _train(save=not args.no_save)
