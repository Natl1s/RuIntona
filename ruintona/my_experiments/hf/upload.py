"""Публикация обученных весов проекта на Hugging Face Hub.

Для каждой модели:
  1) создаёт (при необходимости) репозиторий на HF;
  2) собирает staging-папку в /tmp из карточек (hf/cards/<model>) + весов
     (checkpoints/...);
  3) заливает папку через HfApi.upload_folder (HTTP-протокол, git-lfs не нужен).

Использование:
    poetry run python ruintona/my_experiments/hf/upload.py                  # все репо
    poetry run python ruintona/my_experiments/hf/upload.py --repo rubert    # один
    poetry run python ruintona/my_experiments/hf/upload.py --dry-run        # без заливки

Аутентификация: используется залогиненный токен huggingface_hub
(poetry run huggingface-cli login) или переменная окружения HF_TOKEN.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

HF_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HF_DIR.parents[2]
CARDS_DIR = HF_DIR / "cards"
STAGING_ROOT = Path("/tmp/hf-upload")

REPOS = [
    {
        "repo_id": "Natlis/rubert-emotion-classification-ru",
        "cards": "01_rubert",
        "weights": [
            "ruintona/my_experiments/checkpoints/text/RuBERT_dusha_resd_train_model.pt",
        ],
        "weight_dirs": [
            "ruintona/my_experiments/checkpoints/text/RuBERT_dusha_resd_train_tokenizer",
        ],
        "commit": "Add fine-tuned RuBERT checkpoint + tokenizer + model card",
    },
    {
        "repo_id": "Natlis/cnn-bilstm-emotion-classification-ru",
        "cards": "02_cnn_bilstm",
        "weights": [
            "ruintona/my_experiments/checkpoints/audio/CNN_BiLSTM_combine_balanced_train_model.pt",
        ],
        "weight_dirs": [],
        "commit": "Add trained CNN-BiLSTM checkpoint + model card",
    },
    {
        "repo_id": "Natlis/cnn-emotion-classification-ru",
        "cards": "03_cnn",
        "weights": [
            "ruintona/my_experiments/checkpoints/audio/CNN_combine_balanced_train_model.pt",
        ],
        "weight_dirs": [],
        "commit": "Add trained CNN checkpoint + model card",
    },
    {
        "repo_id": "Natlis/opensmile-xgboost-emotion-classification-ru",
        "cards": "04_opensmile_xgboost",
        "weights": [
            "ruintona/my_experiments/checkpoints/audio/openSmile_XGBoost_dusha_resd_train_model.pkl",
            "ruintona/my_experiments/checkpoints/audio/openSmile_XGBoost_dusha_resd_train_scaler.pkl",
        ],
        "weight_dirs": [],
        "commit": "Add trained openSMILE+XGBoost checkpoint + model card",
    },
    {
        "repo_id": "Natlis/svm-emotion-classification-ru",
        "cards": "05_svm",
        "weights": [
            "ruintona/my_experiments/checkpoints/audio/svm_dusha_resd_train_model.pkl",
            "ruintona/my_experiments/checkpoints/audio/svm_dusha_resd_train_scaler.pkl",
        ],
        "weight_dirs": [],
        "commit": "Add trained SVM checkpoint + model card",
    },
    {
        "repo_id": "Natlis/hubert-rubert-late-fusion-emotion-classification-ru",
        "cards": "06_hubert_rubert_fusion",
        "weights": [],
        "weight_dirs": [],
        "commit": "Add late-fusion report (alpha) + model card",
    },
]


def _check_weights(spec: dict) -> None:
    missing = [
        rel for rel in (*spec["weights"], *spec["weight_dirs"])
        if not (PROJECT_ROOT / rel).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Отсутствуют файлы весов для публикации:\n  "
            + "\n  ".join(missing)
        )


def _build_staging(spec: dict) -> Path:
    repo_id = spec["repo_id"]
    staging = STAGING_ROOT / repo_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    cards_dir = CARDS_DIR / spec["cards"]
    if not cards_dir.exists():
        raise FileNotFoundError(f"Карточки не найдены: {cards_dir}")

    for src in cards_dir.iterdir():
        _copy(src, staging)

    for rel in spec["weights"]:
        _copy(PROJECT_ROOT / rel, staging)
    for rel in spec["weight_dirs"]:
        _copy_dir(PROJECT_ROOT / rel, staging)
    return staging


def _copy(src: Path, dst_dir: Path) -> None:
    dst = dst_dir / src.name
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"  stage  {src.name}")


def _copy_dir(src: Path, dst_dir: Path) -> None:
    shutil.copytree(src, dst_dir / src.name)
    print(f"  stage  {src.name}/ ({sum(1 for _ in src.rglob('*'))} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=None,
        help="repo_id или имя карточки (например 01_rubert) для частичной заливки",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только показать, что будет залито (без обращений к HF)",
    )
    args = parser.parse_args()

    repos = REPOS
    if args.repo:
        repos = [r for r in REPOS if args.repo in (r["repo_id"], r["cards"])]
        if not repos:
            print(f"Репозиторий не найден: {args.repo}")
            sys.exit(1)

    api = HfApi()
    for spec in repos:
        repo_id = spec["repo_id"]
        print(f"\n=== {repo_id} ===")
        _check_weights(spec)

        if args.dry_run:
            print("  [dry-run] staging:")
            _build_staging(spec)
            continue

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        staging = _build_staging(spec)
        print("  upload ...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staging),
            commit_message=spec["commit"],
        )
        print(f"  OK  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()