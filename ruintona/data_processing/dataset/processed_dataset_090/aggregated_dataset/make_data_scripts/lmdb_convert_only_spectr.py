import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from tqdm import tqdm

EMO2LABEL = {"angry": 0, "sad": 1, "neutral": 2, "positive": 3}
META_LEN_KEY = b"__len__"
LABEL_FIELD_PRIORITY = ("label", "emotion", "annotator_emo", "speaker_emo", "target", "class", "emo")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    feature_path: Path
    label: int


def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Не удалось распарсить label: {value}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value.isdigit():
            return int(value)
        if value in EMO2LABEL:
            return EMO2LABEL[value]
    raise ValueError(f"Не удалось распарсить label: {value}")


def _is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def _extract_label_value(record: dict[str, Any]) -> Any:
    for key in LABEL_FIELD_PRIORITY:
        value = record.get(key)
        if _is_nan_like(value):
            continue
        return value
    raise ValueError(
        f"Не найден label. Ожидаются поля: {', '.join(LABEL_FIELD_PRIORITY)}; "
        f"доступные ключи: {', '.join(sorted(record.keys()))}"
    )


def resolve_feature_path(record: dict[str, Any], manifest_path: Path, data_root: Path) -> Path:
    raw_path = record.get("tensor") or record.get("feature_path")
    if raw_path is None:
        hash_id = record.get("hash_id") or record.get("id")
        if hash_id is None:
            raise ValueError("Нет пути к фичам: ожидается tensor/feature_path/hash_id/id")
        raw_path = f"features/{hash_id}.npy"

    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = [
        (manifest_path.parent / path).resolve(),
        (data_root / path).resolve(),
        (data_root / "features" / path.name).resolve(),
        (data_root.parent / path).resolve(),
        (data_root.parent / "features" / path.name).resolve(),
        (data_root.parent.parent / path).resolve(),
        (data_root.parent.parent / "features" / path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _should_use_subdir(path: Path) -> bool:
    return path.suffix == ""


def _open_lmdb(path: Path, map_size: int) -> lmdb.Environment:
    subdir = _should_use_subdir(path)
    if subdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(path),
        map_size=map_size,
        subdir=subdir,
        readonly=False,
        lock=True,
        meminit=False,
        map_async=True,
        max_dbs=1,
    )


def _write_batch(env: lmdb.Environment, batch: list[tuple[bytes, bytes]]) -> None:
    while True:
        txn = env.begin(write=True)
        try:
            for key, value in batch:
                txn.put(key=key, value=value)
            txn.commit()
            return
        except lmdb.MapFullError:
            txn.abort()
            prev_size = env.info()["map_size"]
            new_size = prev_size * 2
            env.set_mapsize(new_size)
            print(f"LMDB map_size увеличен: {prev_size // (1024**2)}MB -> {new_size // (1024**2)}MB")


def read_manifest(manifest_path: Path, data_root: Path) -> tuple[list[Sample], int]:
    samples: list[Sample] = []
    total_feature_bytes = 0

    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                label = parse_label(_extract_label_value(row))
            except ValueError as error:
                raise ValueError(f"{error} (строка {line_num})") from error
            feature_path = resolve_feature_path(row, manifest_path=manifest_path, data_root=data_root)
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Файл фичей не найден (строка {line_num}): {feature_path}"
                )
            sample_id = str(row.get("id", row.get("hash_id", len(samples))))
            total_feature_bytes += feature_path.stat().st_size
            samples.append(Sample(sample_id=sample_id, feature_path=feature_path, label=label))

    return samples, total_feature_bytes


def estimate_map_size(total_feature_bytes: int, num_samples: int) -> int:
    overhead = num_samples * 2048 + 32 * 1024 * 1024
    estimated = int(total_feature_bytes * 1.8) + overhead
    min_size = 512 * 1024 * 1024
    return max(estimated, min_size)


def convert_to_lmdb(
    manifest_path: Path,
    data_root: Path,
    lmdb_path: Path,
    commit_interval: int = 1024,
) -> None:
    print("Читаем manifest...")
    samples, total_feature_bytes = read_manifest(manifest_path=manifest_path, data_root=data_root)
    if not samples:
        raise ValueError("Manifest пустой: нечего конвертировать")

    print(f"Всего записей: {len(samples)}")
    map_size = estimate_map_size(total_feature_bytes=total_feature_bytes, num_samples=len(samples))
    print(f"Создаём LMDB (начальный map_size={map_size // (1024**2)}MB)")

    env = _open_lmdb(path=lmdb_path, map_size=map_size)
    batch: list[tuple[bytes, bytes]] = []
    try:
        for index, sample in enumerate(tqdm(samples, desc="Конвертация"), start=0):
            array = np.load(sample.feature_path)
            payload = {
                "x": np.asarray(array, dtype=np.float32),
                "y": sample.label,
                "id": sample.sample_id,
            }
            serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            batch.append((str(index).encode("utf-8"), serialized))

            if len(batch) >= commit_interval:
                _write_batch(env=env, batch=batch)
                batch.clear()

        batch.append((META_LEN_KEY, str(len(samples)).encode("utf-8")))
        _write_batch(env=env, batch=batch)
        env.sync()
    finally:
        env.close()

    print(f"Готово: {lmdb_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Конвертер JSONL + NPY в LMDB (только спектры)")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Путь до manifest в формате JSONL",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Путь до выходного LMDB (файл .lmdb или директория)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Корневой путь данных. По умолчанию: директория manifest-файла",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=1024,
        help="Сколько записей писать за одну транзакцию",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else manifest_path.parent.resolve()
    output_path = args.output.resolve()

    if args.commit_interval <= 0:
        raise ValueError("--commit-interval должен быть > 0")

    convert_to_lmdb(
        manifest_path=manifest_path,
        data_root=data_root,
        lmdb_path=output_path,
        commit_interval=args.commit_interval,
    )


if __name__ == "__main__":
    main()
