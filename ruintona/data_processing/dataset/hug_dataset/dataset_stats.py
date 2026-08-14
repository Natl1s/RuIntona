import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from ruintona.my_experiments.utils.config_utils import DATASET_PATH


def _resolve_dataset_path() -> Path:
    if DATASET_PATH is None:
        raise FileNotFoundError(
            "Путь к датасету не задан (data.json отсутствует). "
            "Укажите --train и --test явно."
        )
    return DATASET_PATH


def default_train_path() -> Path:
    return _resolve_dataset_path() / "hug_dataset" / "data" / "raw_train-00000-of-00001-1f5fe73d1293189c.jsonl"


def default_test_path() -> Path:
    return _resolve_dataset_path() / "hug_dataset" / "data" / "raw_test-00000-of-00001-a2b788d59856c4ae.jsonl"


DEFAULT_LABEL_PRIORITY = ("annotator_emo", "emotion", "label", "speaker_emo")
KNOWN_EMOTIONS = ("angry", "sad", "neutral", "positive")


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def normalize_emotion(value: Any) -> str | None:
    if is_nan_like(value):
        return None
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate:
            return candidate
    if isinstance(value, (int,)):
        return str(value)
    return str(value).strip().lower() or None


def choose_label(record: dict[str, Any], label_priority: tuple[str, ...]) -> tuple[str | None, str | None]:
    for col in label_priority:
        if col in record:
            value = normalize_emotion(record.get(col))
            if value is not None:
                return value, col
    return None, None


def extract_duration(record: dict[str, Any]) -> float | None:
    for key in ("duration", "wav_length", "audio_length", "length"):
        if key not in record:
            continue
        value = record[key]
        if is_nan_like(value):
            return None
        try:
            dur = float(value)
            if math.isfinite(dur) and dur > 0:
                return dur
        except (TypeError, ValueError):
            return None
    return None


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    alpha = pos - lo
    return sorted_values[lo] * (1 - alpha) + sorted_values[hi] * alpha


def summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean(sorted_values),
        "median": median(sorted_values),
        "p90": quantile(sorted_values, 0.90),
        "p95": quantile(sorted_values, 0.95),
    }


def pct(part: int, total: int) -> float:
    return 0.0 if total == 0 else part * 100.0 / total


def format_float(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{ndigits}f}"


def compute_dataset_stats(path: Path, label_priority: tuple[str, ...]) -> dict[str, Any]:
    label_counts = Counter()
    duration_by_label: dict[str, list[float]] = defaultdict(list)
    unknown_emotion_values = Counter()
    label_source_counts = Counter()
    speaker_vs_target = Counter()

    durations: list[float] = []
    text_char_lengths: list[float] = []
    text_word_lengths: list[float] = []

    hash_ids = []
    audio_paths = []

    missing_duration = 0
    missing_text = 0
    missing_audio_path = 0
    parse_errors = 0
    total_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            label, label_source = choose_label(row, label_priority)
            if label_source is not None:
                label_source_counts[label_source] += 1

            if label is None:
                unknown_emotion_values["<missing>"] += 1
            else:
                if label not in KNOWN_EMOTIONS:
                    unknown_emotion_values[label] += 1
                label_counts[label] += 1

            speaker_emo = normalize_emotion(row.get("speaker_emo"))
            if label is not None and speaker_emo is not None:
                speaker_vs_target[(speaker_emo, label)] += 1

            duration = extract_duration(row)
            if duration is None:
                missing_duration += 1
            else:
                durations.append(duration)
                if label is not None:
                    duration_by_label[label].append(duration)

            text = row.get("speaker_text")
            if is_nan_like(text):
                missing_text += 1
            else:
                text = str(text).strip()
                if not text:
                    missing_text += 1
                else:
                    text_char_lengths.append(float(len(text)))
                    text_word_lengths.append(float(len(text.split())))

            audio_path = row.get("audio_path")
            if is_nan_like(audio_path):
                missing_audio_path += 1
            else:
                audio_paths.append(str(audio_path))

            hash_id = row.get("hash_id")
            if not is_nan_like(hash_id):
                hash_ids.append(str(hash_id))

    duplicate_hash_ids = len(hash_ids) - len(set(hash_ids))
    duplicate_audio_paths = len(audio_paths) - len(set(audio_paths))

    duration_stats = summary_stats(durations)
    text_char_stats = summary_stats(text_char_lengths)
    text_word_stats = summary_stats(text_word_lengths)

    per_label_duration_stats = {
        label: summary_stats(values) for label, values in sorted(duration_by_label.items())
    }

    return {
        "path": str(path),
        "rows_total": total_rows,
        "parse_errors": parse_errors,
        "label_counts": dict(label_counts),
        "label_source_counts": dict(label_source_counts),
        "unknown_or_unmapped_labels": dict(unknown_emotion_values),
        "duration_stats": duration_stats,
        "text_char_len_stats": text_char_stats,
        "text_word_len_stats": text_word_stats,
        "missing_duration": missing_duration,
        "missing_text": missing_text,
        "missing_audio_path": missing_audio_path,
        "duplicate_hash_ids": duplicate_hash_ids,
        "duplicate_audio_paths": duplicate_audio_paths,
        "per_label_duration_stats": per_label_duration_stats,
        "speaker_vs_target_top_pairs": dict(speaker_vs_target.most_common(12)),
    }


def print_dataset_report(stats: dict[str, Any]) -> None:
    total = stats["rows_total"]
    print(f"\n=== {stats['path']} ===")
    print(f"rows_total: {total}")
    print(f"parse_errors: {stats['parse_errors']}")
    print("label_source_counts:", stats["label_source_counts"])

    print("\nlabel_distribution:")
    for label, count in sorted(stats["label_counts"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:>8}: {count:>7} ({pct(count, total):6.2f}%)")

    if stats["unknown_or_unmapped_labels"]:
        print("unknown_or_unmapped_labels:", stats["unknown_or_unmapped_labels"])

    duration = stats["duration_stats"]
    print("\nduration_sec:")
    print(
        "  count={count} min={min} mean={mean} median={median} p90={p90} p95={p95} max={max}".format(
            count=duration["count"],
            min=format_float(duration["min"]),
            mean=format_float(duration["mean"]),
            median=format_float(duration["median"]),
            p90=format_float(duration["p90"]),
            p95=format_float(duration["p95"]),
            max=format_float(duration["max"]),
        )
    )

    text_char = stats["text_char_len_stats"]
    text_word = stats["text_word_len_stats"]
    print(
        "text_chars: count={count} mean={mean} median={median} p90={p90}".format(
            count=text_char["count"],
            mean=format_float(text_char["mean"]),
            median=format_float(text_char["median"]),
            p90=format_float(text_char["p90"]),
        )
    )
    print(
        "text_words: count={count} mean={mean} median={median} p90={p90}".format(
            count=text_word["count"],
            mean=format_float(text_word["mean"]),
            median=format_float(text_word["median"]),
            p90=format_float(text_word["p90"]),
        )
    )

    print(
        "missing: duration={d} ({dp:.2f}%), text={t} ({tp:.2f}%), audio_path={a} ({ap:.2f}%)".format(
            d=stats["missing_duration"],
            dp=pct(stats["missing_duration"], total),
            t=stats["missing_text"],
            tp=pct(stats["missing_text"], total),
            a=stats["missing_audio_path"],
            ap=pct(stats["missing_audio_path"], total),
        )
    )

    print(
        f"duplicates: hash_id={stats['duplicate_hash_ids']}, "
        f"audio_path={stats['duplicate_audio_paths']}"
    )

    print("\nper_label_duration_sec:")
    for label, s in stats["per_label_duration_stats"].items():
        print(
            f"  {label:>8}: n={s['count']:>6} "
            f"mean={format_float(s['mean'])} median={format_float(s['median'])} "
            f"p95={format_float(s['p95'])}"
        )

    if stats["speaker_vs_target_top_pairs"]:
        print("\nspeaker_emo vs target (top pairs):")
        for pair, count in stats["speaker_vs_target_top_pairs"].items():
            print(f"  {pair}: {count}")


def print_split_comparison(train_stats: dict[str, Any], test_stats: dict[str, Any]) -> None:
    print("\n=== train vs test comparison ===")
    print(f"rows: train={train_stats['rows_total']}, test={test_stats['rows_total']}")

    labels = sorted(set(train_stats["label_counts"]) | set(test_stats["label_counts"]))
    print("label_share_difference_pp (train - test):")
    for label in labels:
        train_share = pct(train_stats["label_counts"].get(label, 0), train_stats["rows_total"])
        test_share = pct(test_stats["label_counts"].get(label, 0), test_stats["rows_total"])
        print(f"  {label:>8}: {train_share - test_share:+7.3f} pp")

    train_dur = train_stats["duration_stats"]["mean"]
    test_dur = test_stats["duration_stats"]["mean"]
    if train_dur is not None and test_dur is not None:
        print(f"mean_duration_sec: train={train_dur:.3f}, test={test_dur:.3f}, delta={train_dur - test_dur:+.3f}")

    train_words = train_stats["text_word_len_stats"]["mean"]
    test_words = test_stats["text_word_len_stats"]["mean"]
    if train_words is not None and test_words is not None:
        print(f"mean_text_words: train={train_words:.3f}, test={test_words:.3f}, delta={train_words - test_words:+.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Основные статистики train/test JSONL для задачи классификации эмоций."
    )
    parser.add_argument("--train", type=Path, default=None, help="Путь к train JSONL")
    parser.add_argument("--test", type=Path, default=None, help="Путь к test JSONL")
    parser.add_argument(
        "--label-priority",
        nargs="+",
        default=list(DEFAULT_LABEL_PRIORITY),
        help="Порядок колонок для выбора целевой метки.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Опционально: путь для сохранения отчёта в JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train is None:
        args.train = default_train_path()
    if args.test is None:
        args.test = default_test_path()
    if not args.train.exists():
        raise FileNotFoundError(f"Train файл не найден: {args.train}")
    if not args.test.exists():
        raise FileNotFoundError(f"Test файл не найден: {args.test}")

    label_priority = tuple(args.label_priority)

    train_stats = compute_dataset_stats(args.train, label_priority=label_priority)
    test_stats = compute_dataset_stats(args.test, label_priority=label_priority)

    print_dataset_report(train_stats)
    print_dataset_report(test_stats)
    print_split_comparison(train_stats, test_stats)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"train": train_stats, "test": test_stats}
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report saved: {args.json_out}")


if __name__ == "__main__":
    main()
