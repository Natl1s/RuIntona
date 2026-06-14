import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

DEFAULT_AUDIO_COLUMNS = ("audio", "speech")
DEFAULT_TEXT_COLUMNS = ("text",)
TARGET_NAME_EMOTIONS = {"happiness", "anger", "neutral", "sadness"}
NAME_TO_OUTPUT_EMOTION = {
    "happiness": "positive",
    "anger": "angry",
    "neutral": "neutral",
    "sadness": "sad",
}


def _sanitize_hash_id(raw_value: str, fallback: str) -> str:
    value = str(raw_value).strip()
    if not value:
        value = fallback
    value = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE).strip("_")
    return value or fallback


def _extract_audio(row: pd.Series, parquet_path: Path, audio_columns: tuple[str, ...]) -> tuple[np.ndarray, int]:
    for column in audio_columns:
        if column not in row.index:
            continue
        value = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue

        if isinstance(value, dict):
            sampling_rate = value.get("sampling_rate")

            if value.get("array") is not None and sampling_rate is not None:
                audio = np.asarray(value["array"], dtype=np.float32)
                return audio, int(sampling_rate)

            if value.get("bytes"):
                audio, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
                return np.asarray(audio, dtype=np.float32), int(sr)

            if value.get("path"):
                audio_path = Path(value["path"])
                if not audio_path.is_absolute():
                    audio_path = parquet_path.parent / audio_path
                audio, sr = sf.read(audio_path, dtype="float32")
                return np.asarray(audio, dtype=np.float32), int(sr)

        if isinstance(value, (str, Path)):
            audio_path = Path(value)
            if not audio_path.is_absolute():
                audio_path = parquet_path.parent / audio_path
            audio, sr = sf.read(audio_path, dtype="float32")
            return np.asarray(audio, dtype=np.float32), int(sr)

    raise ValueError(f"Audio data not found in columns {audio_columns}")


def _pick_emotion_from_name(source_name: str) -> str | None:
    name_parts = source_name.split("_")
    if len(name_parts) < 3:
        return None
    first_candidate = name_parts[1].strip().lower()
    second_candidate = name_parts[2].strip().lower()
    if first_candidate in TARGET_NAME_EMOTIONS:
        return NAME_TO_OUTPUT_EMOTION[first_candidate]
    if second_candidate in TARGET_NAME_EMOTIONS:
        return NAME_TO_OUTPUT_EMOTION[second_candidate]
    return None


def _make_raw_row(hash_id: str, duration: float, emotion: str, speaker_text: str | None) -> dict[str, Any]:
    return {
        "hash_id": hash_id,
        "audio_path": f"wavs/{hash_id}.wav",
        "duration": duration,
        "annotator_emo": emotion,
        "golden_emo": float("nan"),
        "annotator_id": "ext_annotator_1",
        "speaker_text": speaker_text if speaker_text is not None else "",
        "speaker_emo": emotion,
        "source_id": float("nan"),
    }


def convert_parquet_to_raw_jsonl(
    parquet_path: Path,
    wavs_dir: Path,
    output_jsonl: Path,
    audio_columns: tuple[str, ...],
    text_columns: tuple[str, ...],
) -> dict[str, int]:
    df = pd.read_parquet(parquet_path)

    wavs_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    used_ids: set[str] = set()
    raw_rows: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "saved": 0,
        "skipped_name_emotion": 0,
    }

    for idx, row in df.iterrows():
        stats["total"] += 1
        source_name = row.get("name", row.get("id", row.get("path", f"sample_{idx}")))
        source_name = Path(str(source_name)).stem
        emotion = _pick_emotion_from_name(source_name)
        if emotion is None:
            stats["skipped_name_emotion"] += 1
            continue

        hash_id = _sanitize_hash_id(source_name, fallback=f"sample_{idx}")
        suffix = 1
        while hash_id in used_ids:
            hash_id = f"{_sanitize_hash_id(source_name, fallback=f'sample_{idx}')}_{suffix}"
            suffix += 1
        used_ids.add(hash_id)

        try:
            audio, sample_rate = _extract_audio(row=row, parquet_path=parquet_path, audio_columns=audio_columns)
        except Exception as exc:
            raise ValueError(f"{parquet_path.name}: failed to read audio for row {idx}") from exc

        wav_path = wavs_dir / f"{hash_id}.wav"
        sf.write(wav_path, audio, sample_rate, format="WAV")
        duration = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0

        speaker_text = ""
        for column in text_columns:
            if column in row.index:
                value = row[column]
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    speaker_text = str(value)
                break

        raw_rows.append(_make_raw_row(hash_id=hash_id, duration=duration, emotion=emotion, speaker_text=speaker_text))
        stats["saved"] += 1

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    return stats


def _resolve_output_jsonl(parquet_path: Path, output_data_dir: Path) -> Path:
    return output_data_dir / f"raw_{parquet_path.stem}.jsonl"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert parquet to raw_*.jsonl and WAV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "data",
        help="Directory with parquet files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.parquet",
        help="Glob for parquet files in input-dir.",
    )
    parser.add_argument(
        "--wavs-dir",
        type=Path,
        default=base_dir / "wavs",
        help="Output directory for wav files.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=base_dir / "data",
        help="Output directory for raw_*.jsonl files.",
    )
    parser.add_argument(
        "--audio-columns",
        type=str,
        default=",".join(DEFAULT_AUDIO_COLUMNS),
        help="Comma-separated candidate columns with audio payload.",
    )
    parser.add_argument(
        "--text-columns",
        type=str,
        default=",".join(DEFAULT_TEXT_COLUMNS),
        help="Comma-separated candidate columns with transcription text.",
    )
    args = parser.parse_args()

    parquet_files = sorted(args.input_dir.glob(args.glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir} with glob {args.glob!r}")

    audio_columns = tuple(col.strip() for col in args.audio_columns.split(",") if col.strip())
    text_columns = tuple(col.strip() for col in args.text_columns.split(",") if col.strip())

    for parquet_path in parquet_files:
        out_jsonl = _resolve_output_jsonl(parquet_path, args.output_data_dir)
        stats = convert_parquet_to_raw_jsonl(
            parquet_path=parquet_path,
            wavs_dir=args.wavs_dir,
            output_jsonl=out_jsonl,
            audio_columns=audio_columns,
            text_columns=text_columns,
        )
        print(
            f"[{parquet_path.name}] saved={stats['saved']} total={stats['total']} "
            f"skipped_name_emotion={stats['skipped_name_emotion']} -> {out_jsonl}"
        )


if __name__ == "__main__":
    main()
