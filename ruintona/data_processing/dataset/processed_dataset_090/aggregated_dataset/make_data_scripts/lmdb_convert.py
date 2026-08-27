import argparse
import json
import math
import pickle
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from tqdm import tqdm

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

EMO2LABEL = {"angry": 0, "sad": 1, "neutral": 2, "positive": 3}
META_LEN_KEY = b"__len__"
TARGET_SAMPLE_RATE = 16000
LABEL_FIELD_PRIORITY = ("label", "emotion", "annotator_emo", "speaker_emo", "target", "class", "emo")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    audio_path: Path
    text: str
    duration_sec: float | None
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
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def _normalize_text(record: dict[str, Any]) -> str:
    for key in ("speaker_text", "text", "transcript", "utterance"):
        value = record.get(key)
        if _is_nan_like(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


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


def _is_podcast_row(record: dict[str, Any]) -> bool:
    keys_to_check = ("audio_path", "wav_path", "wav", "tensor", "feature_path", "id", "hash_id")
    for key in keys_to_check:
        value = record.get(key)
        if value is None:
            continue
        if "podcast" in str(value).lower():
            return True
    return False


def resolve_audio_path(record: dict[str, Any], manifest_path: Path, data_root: Path) -> Path:
    raw_path = record.get("audio_path") or record.get("wav_path") or record.get("wav")
    sample_id = record.get("hash_id") or record.get("id")
    search_roots = (data_root, data_root.parent, data_root.parent.parent)

    if raw_path is not None:
        path = Path(str(raw_path))
        if path.is_absolute():
            return path

        candidates = [
            (manifest_path.parent / path).resolve(),
        ]
        candidates.extend((root / path).resolve() for root in search_roots)
        if sample_id is not None:
            sample_name = f"{sample_id}.wav"
            for root in search_roots:
                candidates.extend(
                    [
                        (root / "wavs" / sample_name).resolve(),
                        (root / "crowd_train" / "wavs" / sample_name).resolve(),
                        (root / "crowd_test" / "wavs" / sample_name).resolve(),
                        (root / "podcast_train" / "wavs" / sample_name).resolve(),
                        (root / "podcast_test" / "wavs" / sample_name).resolve(),
                    ]
                )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    if sample_id is None:
        raise ValueError("Нет пути к аудио: ожидается audio_path/wav_path/wav или hash_id/id")

    default_candidates: list[Path] = []
    for root in search_roots:
        default_candidates.extend(
            [
                (root / "crowd_train" / "wavs" / f"{sample_id}.wav").resolve(),
                (root / "crowd_test" / "wavs" / f"{sample_id}.wav").resolve(),
                (root / "podcast_train" / "wavs" / f"{sample_id}.wav").resolve(),
                (root / "podcast_test" / "wavs" / f"{sample_id}.wav").resolve(),
                (root / "wavs" / f"{sample_id}.wav").resolve(),
            ]
        )
    for candidate in default_candidates:
        if candidate.exists():
            return candidate
    return default_candidates[0]


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


def _resample_linear(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(waveform, dtype=np.float32)
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Некорректная частота дискретизации: {source_sr} -> {target_sr}")
    if waveform.size == 0:
        return np.asarray(waveform, dtype=np.float32)

    duration = waveform.shape[0] / float(source_sr)
    target_len = max(1, int(round(duration * target_sr)))
    src_x = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    resampled = np.interp(dst_x, src_x, waveform)
    return np.asarray(resampled, dtype=np.float32)


def _load_wav_builtin(audio_path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_sr = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        signed = np.where(signed & sign_bit, signed - (1 << 24), signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Неподдерживаемая sample width={sample_width} в {audio_path}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    data = _resample_linear(np.asarray(data, dtype=np.float32), source_sr, target_sr)
    return data, target_sr


def load_waveform_16khz(audio_path: Path) -> tuple[np.ndarray, int]:
    if librosa is not None:
        waveform, sample_rate = librosa.load(
            str(audio_path),
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )
        return np.asarray(waveform, dtype=np.float32), int(sample_rate)

    if audio_path.suffix.lower() != ".wav":
        raise RuntimeError(
            "librosa не установлен и файл не .wav. "
            "Установите librosa или используйте WAV."
        )
    return _load_wav_builtin(audio_path=audio_path, target_sr=TARGET_SAMPLE_RATE)


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
    total_audio_bytes_estimate = 0
    skipped_podcast_rows = 0

    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if _is_podcast_row(row):
                skipped_podcast_rows += 1
                continue
            try:
                label = parse_label(_extract_label_value(row))
            except ValueError as error:
                raise ValueError(f"{error} (строка {line_num})") from error
            audio_path = resolve_audio_path(row, manifest_path=manifest_path, data_root=data_root)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Аудио файл не найден (строка {line_num}): {audio_path}"
                )
            sample_id = str(row.get("id", row.get("hash_id", len(samples))))
            duration_raw = row.get("duration", row.get("wav_length"))
            duration_sec = None
            if not _is_nan_like(duration_raw):
                duration_sec = float(duration_raw)

            if duration_sec is not None and duration_sec > 0:
                total_audio_bytes_estimate += int(duration_sec * TARGET_SAMPLE_RATE * 4)
            else:
                total_audio_bytes_estimate += int(audio_path.stat().st_size * 2.0)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    text=_normalize_text(row),
                    duration_sec=duration_sec,
                    label=label,
                )
            )

    if skipped_podcast_rows:
        print(f"Пропущено строк podcast: {skipped_podcast_rows}")

    return samples, total_audio_bytes_estimate


def estimate_map_size(total_payload_bytes: int, num_samples: int) -> int:
    overhead = num_samples * 2048 + 32 * 1024 * 1024
    estimated = int(total_payload_bytes * 1.6) + overhead
    min_size = 512 * 1024 * 1024
    return max(estimated, min_size)


def convert_to_lmdb(
    manifest_path: Path,
    data_root: Path,
    lmdb_path: Path,
    commit_interval: int = 1024,
) -> None:
    print("Читаем manifest...")
    samples, total_payload_bytes = read_manifest(manifest_path=manifest_path, data_root=data_root)
    if not samples:
        raise ValueError("Manifest пустой: нечего конвертировать")

    print(f"Всего записей: {len(samples)}")
    map_size = estimate_map_size(total_payload_bytes=total_payload_bytes, num_samples=len(samples))
    print(f"Создаём LMDB (начальный map_size={map_size // (1024**2)}MB)")

    env = _open_lmdb(path=lmdb_path, map_size=map_size)
    batch: list[tuple[bytes, bytes]] = []
    try:
        for index, sample in enumerate(tqdm(samples, desc="Конвертация"), start=0):
            waveform, sample_rate = load_waveform_16khz(sample.audio_path)
            payload = {
                "y": sample.label,
                "id": sample.sample_id,
                "waveform": np.asarray(waveform, dtype=np.float32),
                "waveform_sr": int(sample_rate),
                "text": sample.text,
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
    parser = argparse.ArgumentParser(description="Конвертер JSONL + WAV в LMDB")
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
