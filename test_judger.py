import argparse
import csv
import importlib
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read captions from a CSV file and label them with a Judger."
    )
    parser.add_argument(
        "--judger",
        required=True,
        help=(
            "Judger class to use. Supported forms: 'QwenGuardJudger' "
            "or 'module_name:ClassName'."
        ),
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Source CSV file path.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU device id, for example '0'. If omitted, use the default runtime device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name passed to the Judger constructor.",
    )
    return parser.parse_args()


def resolve_judger_class(judger_spec: str):
    if ":" in judger_spec:
        module_name, class_name = judger_spec.split(":", 1)
    else:
        module_name, class_name = "qwen_guard_judger", judger_spec

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_judger(args: argparse.Namespace):
    if args.gpu is not None and args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    judger_class = resolve_judger_class(args.judger)
    kwargs: Dict[str, Any] = {}
    if args.model_name:
        kwargs["model_name"] = args.model_name
    return judger_class(**kwargs)


def iter_caption_rows(input_csv: Path) -> Iterable[Tuple[str, str]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "video_id" not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise ValueError("CSV must contain 'video_id' and 'caption' columns.")

        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not caption:
                continue
            yield video_id, caption


def count_rows(input_csv: Path) -> Tuple[int, int]:
    total_rows = 0
    non_empty_rows = 0
    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "video_id" not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise ValueError("CSV must contain 'video_id' and 'caption' columns.")

        for row in reader:
            total_rows += 1
            caption = (row.get("caption") or "").strip()
            if caption:
                non_empty_rows += 1
    return total_rows, non_empty_rows


def batched(items: Sequence[Tuple[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_negative_types(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def print_progress(processed: int, total: int, elapsed: float) -> None:
    if total <= 0:
        return

    bar_width = 30
    ratio = min(processed / total, 1.0)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    avg = elapsed / processed if processed else 0.0
    print(
        f"\rProgress [{bar}] {processed}/{total} "
        f"({ratio * 100:6.2f}%) elapsed={format_seconds(elapsed)} avg={format_seconds(avg)}",
        end="",
        flush=True,
    )


def write_results(
    output_csv: Path,
    rows: List[Dict[str, Any]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["video_id", "text", "is_negative", "negative_types", "safety_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    judger = make_judger(args)
    total_rows, non_empty_rows = count_rows(input_csv)
    source_rows = list(iter_caption_rows(input_csv))
    skipped_rows = total_rows - non_empty_rows

    output_rows: List[Dict[str, Any]] = []
    negative_count = 0
    processed_count = 0
    inference_elapsed = 0.0

    if non_empty_rows:
        print_progress(0, non_empty_rows, 0.0)

    for batch in batched(source_rows, args.batch_size):
        texts = [caption for _, caption in batch]
        batch_start = time.perf_counter()
        results = judger.batch_judge(texts)
        inference_elapsed += time.perf_counter() - batch_start

        for (video_id, caption), result in zip(batch, results):
            if result.get("is_negative"):
                negative_count += 1
            output_rows.append(
                {
                    "video_id": video_id,
                    "text": caption,
                    "is_negative": result.get("is_negative"),
                    "negative_types": normalize_negative_types(result.get("negative_types")),
                    "safety_label": result.get("safety_label"),
                }
            )
        processed_count += len(batch)
        print_progress(processed_count, non_empty_rows, inference_elapsed)

    write_results(output_csv, output_rows)

    if non_empty_rows:
        print()

    average_time = inference_elapsed / processed_count if processed_count else 0.0
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Total rows: {total_rows}")
    print(f"Processed rows: {processed_count}")
    print(f"Skipped empty captions: {skipped_rows}")
    print(f"Negative rows: {negative_count}")
    print(f"Non-negative rows: {processed_count - negative_count}")
    print(f"Total inference time: {format_seconds(inference_elapsed)}")
    print(f"Average inference time per text: {format_seconds(average_time)}")


if __name__ == "__main__":
    main()
