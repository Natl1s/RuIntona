import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from dusha.my_experiments.utils.config_utils import get_dataset_name, models_dir_for, load_experiment_config, apply_config_to_args, add_config_arg, add_data_path_args, resolve_data_paths
from dusha.my_experiments.utils.model_io import save_sklearn_model, load_sklearn_model, sklearn_model_exists
from dusha.my_experiments.utils.lmdb_utils import load_audio_features_from_lmdb
from dusha.my_experiments.utils.sklearn_utils import evaluate_sklearn_classifier

MODELS_DIR = models_dir_for(__file__)
MODEL_NAME = Path(__file__).stem

DEFAULT_SEARCH_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
}


def print_random_forest_params(model):
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
    print(f"Cost-complexity pruning (ccp_alpha): {model.ccp_alpha}")
    print(f"class_weight: {model.class_weight}")
    print(f"\nКоличество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print("\nСтатистика по деревьям:")
    tree_depths = [tree.get_depth() for tree in model.estimators_]
    tree_leaves = [tree.get_n_leaves() for tree in model.estimators_]
    print(f"  Глубина деревьев — Min: {min(tree_depths)}, Max: {max(tree_depths)}, Mean: {np.mean(tree_depths):.2f}")
    print(f"  Листьев в деревьях — Min: {min(tree_leaves)}, Max: {max(tree_leaves)}, Mean: {np.mean(tree_leaves):.2f}")
    print("\nВажность признаков (Feature Importance):")
    feature_importances = model.feature_importances_
    top_indices = np.argsort(feature_importances)[-10:][::-1]
    print("  Топ-10 признаков:")
    for i, idx in enumerate(top_indices, 1):
        print(f"    {i}. Признак {idx}: {feature_importances[idx]:.6f}")
    print("\nСтатистика важности признаков:")
    print(f"  Min: {feature_importances.min():.6f}, Max: {feature_importances.max():.6f}")
    print(f"  Mean: {feature_importances.mean():.6f}, Std: {feature_importances.std():.6f}")
    print(f"  Признаков с нулевой важностью: {np.sum(feature_importances == 0)}/{len(feature_importances)}")
    if hasattr(model, "oob_score_"):
        print(f"\nOut-of-Bag Score: {model.oob_score_:.4f}")


def _stratified_subsample(X, y, max_samples, random_state=42):
    """Стратифицированная подвыборка для ускорения подбора гиперпараметров."""
    if max_samples and max_samples < len(y):
        X, _, y, _ = train_test_split(
            X, y, train_size=int(max_samples), stratify=y, random_state=random_state,
        )
    return X, y


def train_random_forest(save=True, n_estimators=100, max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        max_features="sqrt", oob_score=False,
                        ccp_alpha=0.0, class_weight=None,
                        train_path=None, test_path=None):
    """Обучение Random Forest классификатора."""
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

    print("\nНормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ RANDOM FOREST МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Параметры: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
          f"max_features={max_features}, oob_score={oob_score}, "
          f"ccp_alpha={ccp_alpha}, class_weight={class_weight}")

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, oob_score=oob_score,
        ccp_alpha=ccp_alpha, class_weight=class_weight,
        random_state=42, n_jobs=-1, verbose=1,
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    metrics = evaluate_sklearn_classifier(
        model, X_train, y_train, X_test, y_test,
        transform_fn=scaler.transform,
        print_params_fn=print_random_forest_params,
    )

    if save:
        training_params = {
            "n_estimators": model.n_estimators, "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split, "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features, "criterion": model.criterion,
            "oob_score": model.oob_score, "random_state": model.random_state,
            "ccp_alpha": model.ccp_alpha, "class_weight": str(model.class_weight),
            "train_manifest": str(train_manifest), "test_manifest": str(test_manifest),
        }
        save_sklearn_model(
            model, scaler, dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            training_params=training_params, test_metrics=metrics,
        )

    return model, scaler, dataset_name


def _save_search_results(search, dataset_name, grid):
    """Сохраняет результаты RandomizedSearchCV в JSON."""
    results = {
        "dataset_name": dataset_name,
        "scoring": search.scoring,
        "cv_folds": search.cv,
        "n_iter": search.n_iter,
        "search_grid": {k: list(v) for k, v in grid.items()},
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "top_results": [
            {
                "params": dict(p),
                "mean_cv_score": float(score),
                "std_cv_score": float(std),
            }
            for score, std, p in zip(
                search.cv_results_["mean_test_score"][:10],
                search.cv_results_["std_test_score"][:10],
                search.cv_results_["params"][:10],
            )
        ],
    }
    results_path = MODELS_DIR / f"{MODEL_NAME}_{dataset_name}_search_results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n✓ Результаты поиска сохранены: {results_path.absolute()}")


def tune_random_forest(save=True, search_iterations=20, cv_folds=3,
                       max_samples=None, tune_grid=None,
                       train_path=None, test_path=None):
    """Подбор гиперпараметров RandomForest: RandomizedSearchCV + итоговое обучение.

    Поиск идёт на подвыборке (для скорости), лучшие параметры затем
    используются для обучения на полном train и оценки на test
    (test не участвует в подборе — исключается риск подгонки под тест).
    """
    from sklearn.ensemble import RandomForestClassifier

    train_manifest = train_path
    test_manifest = test_path
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    print("Загрузка обучающих данных...")
    X_train_full, y_train_full = load_audio_features_from_lmdb(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train_full)}")
    print(f"Размер обучающей выборки: {X_train_full.shape}")

    X_search, y_search = _stratified_subsample(X_train_full, y_train_full, max_samples)
    if len(X_search) < len(X_train_full):
        print(f"Используем подвыборку для поиска: {len(y_search)} примеров "
              f"(стратифицировано по классам)")
    print(f"Распределение классов в подвыборке: {np.unique(y_search, return_counts=True)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_audio_features_from_lmdb(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    print("\nНормализация features (на поисковой подвыборке)...")
    scaler_search = StandardScaler()
    X_search_scaled = scaler_search.fit_transform(X_search)

    grid = tune_grid if tune_grid is not None else DEFAULT_SEARCH_GRID

    print(f"\n{'=' * 60}")
    print("ПОДБОР ГИПЕРПАРАМЕТРОВ (RandomizedSearchCV)")
    print(f"{'=' * 60}")
    print(f"Поиск: n_iter={search_iterations}, cv={cv_folds}, scoring='f1_macro'")
    print(f"Сетка: {json.dumps({k: list(v) for k, v in grid.items()}, default=str)}")

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
    search = RandomizedSearchCV(
        base_model, grid,
        n_iter=search_iterations, cv=cv_folds,
        scoring="f1_macro", n_jobs=-1, random_state=42, verbose=1,
    )
    search.fit(X_search_scaled, y_search)
    print("✓ Поиск завершён!")

    print(f"\nЛучшие параметры: {search.best_params_}")
    print(f"Лучший CV f1-macro: {search.best_score_:.4f}")
    print("\nТоп-10 комбинаций:")
    for score, std, params in zip(
        search.cv_results_["mean_test_score"][:10],
        search.cv_results_["std_test_score"][:10],
        search.cv_results_["params"][:10],
    ):
        print(f"  f1-macro={score:.4f} ± {std:.4f}  {params}")

    if save:
        _save_search_results(search, dataset_name, grid)

    print(f"\n{'=' * 60}")
    print("ИТОГОВОЕ ОБУЧЕНИЕ НА ПОЛНЫХ ДАННЫХ")
    print(f"{'=' * 60}")
    print(f"Параметры: {search.best_params_}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)

    best_params = dict(search.best_params_)
    model = RandomForestClassifier(
        **best_params, random_state=42, n_jobs=-1, oob_score=True,
    )
    model.fit(X_train_scaled, y_train_full)
    print("✓ Обучение завершено!")

    metrics = evaluate_sklearn_classifier(
        model, X_train_full, y_train_full, X_test, y_test,
        transform_fn=scaler.transform,
        print_params_fn=print_random_forest_params,
    )

    if save:
        training_params = {
            **best_params,
            "criterion": model.criterion,
            "oob_score": model.oob_score, "random_state": model.random_state,
            "search_n_iter": search_iterations, "search_cv_folds": cv_folds,
            "search_best_cv_f1_macro": float(search.best_score_),
            "search_max_samples": int(max_samples) if max_samples else None,
            "train_manifest": str(train_manifest), "test_manifest": str(test_manifest),
        }
        save_sklearn_model(
            model, scaler, dataset_name,
            models_dir=MODELS_DIR, model_name=MODEL_NAME,
            training_params=training_params, test_metrics=metrics,
        )

    return model, scaler, dataset_name, search


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
    X_train, y_train = load_audio_features_from_lmdb(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_audio_features_from_lmdb(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")

    evaluate_sklearn_classifier(
        model, X_train, y_train, X_test, y_test,
        transform_fn=scaler.transform,
        print_params_fn=print_random_forest_params,
    )
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение или загрузка Random Forest для классификации эмоций"
    )
    parser.add_argument("--mode", type=str, choices=["train", "load", "auto", "smoke", "tune"], default="auto")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--oob-score", action="store_true")
    parser.add_argument("--ccp-alpha", type=float, default=0.0)
    parser.add_argument("--class-weight", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--search-iterations", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=3)
    add_data_path_args(parser)
    add_config_arg(parser)
    args = parser.parse_args()

    train_path, test_path = resolve_data_paths(args)
    experiment_config = load_experiment_config(args.config)
    if experiment_config:
        args = apply_config_to_args(args, experiment_config, parser)

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

    def _exists(dn):
        return sklearn_model_exists(dn, models_dir=MODELS_DIR, model_name=MODEL_NAME)

    def _train(save=True):
        return train_random_forest(
            save=save, n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
            max_features=max_features_value, oob_score=args.oob_score,
            ccp_alpha=args.ccp_alpha, class_weight=args.class_weight,
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
    elif args.mode == "tune":
        print("🔍 Режим: Подбор гиперпараметров\n")
        tune_random_forest(
            save=not args.no_save,
            search_iterations=args.search_iterations, cv_folds=args.cv_folds,
            max_samples=args.max_samples,
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
