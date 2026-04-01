from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from api_neg_label import NegLabelClient
from utils import read_unique_titles
from workflow_neg_label import WorkflowConfig, WorkflowNegLabelClient


def run_api_labeling(
    input_path: str,
    column_name: Optional[str],
    max_items: Optional[int],
    batch_size: int,
    output_xlsx: str,
    base_url: str,
    model: str,
) -> None:
    """
    使用 utils 读取表格/目录中的标题，再通过 NegLabelClient 调用 API 打标并写入 xlsx。
    """
    titles = read_unique_titles(
        path=input_path,
        column_name=column_name,
        max_items=max_items,
    )
    if not titles:
        print("[WARN] 未读取到任何标题，跳过打标。")
        return

    client = NegLabelClient(base_url=base_url, model=model)
    client.label_titles(
        titles=titles,
        batch_size=batch_size,
        output_xlsx=output_xlsx,
    )


def _derive_workflow_output_paths(output_path: str) -> tuple[str, str, str]:
    output = Path(output_path)
    suffix = output.suffix or ".xlsx"
    if suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
        suffix = ".xlsx"

    stem = output.stem
    parent = output.parent
    initial_output = parent / f"{stem}_initial{suffix}"
    final_non_negative_output = parent / f"{stem}_non_negative{suffix}"
    final_negative_output = parent / f"{stem}_negative{suffix}"
    return (
        str(initial_output),
        str(final_non_negative_output),
        str(final_negative_output),
    )


def run_workflow_labeling(
    input_path: str,
    column_name: Optional[str],
    max_items: Optional[int],
    output_xlsx: str,
    base_url: str,
    screening_model: str,
    negative_review_model: Optional[str],
    non_negative_review_model: Optional[str],
    screening_batch_size: int,
    negative_review_batch_size: int,
    negative_review_labels: Optional[list[str]],
    non_negative_review_batch_size: int,
    non_negative_review_ratio: float,
    non_negative_review_seed: int,
) -> None:
    """
    使用 workflow 方式进行初筛和复核，并输出 3 份 xlsx 结果文件。
    """
    titles = read_unique_titles(
        path=input_path,
        column_name=column_name,
        max_items=max_items,
    )
    if not titles:
        print("[WARN] 未读取到任何标题，跳过打标。")
        return

    initial_output_xlsx, final_non_negative_output_xlsx, final_negative_output_xlsx = (
        _derive_workflow_output_paths(output_xlsx)
    )
    config = WorkflowConfig(
        screening_batch_size=screening_batch_size,
        negative_review_batch_size=negative_review_batch_size,
        negative_review_labels=negative_review_labels,
        non_negative_review_batch_size=non_negative_review_batch_size,
        non_negative_review_ratio=non_negative_review_ratio,
        non_negative_review_seed=non_negative_review_seed,
    )
    client = WorkflowNegLabelClient(
        base_url=base_url,
        screening_model=screening_model,
        negative_review_model=negative_review_model,
        non_negative_review_model=non_negative_review_model,
        config=config,
    )
    client.label_titles(
        titles=titles,
        initial_output_xlsx=initial_output_xlsx,
        final_non_negative_output_xlsx=final_non_negative_output_xlsx,
        final_negative_output_xlsx=final_negative_output_xlsx,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="视频负面内容打标入口；根据 --method 选择打标方式。",
    )
    p.add_argument(
        "--method",
        required=True,
        choices=["api", "workflow"],
        help='打标方式：`api` 或 `workflow`。',
    )
    p.add_argument(
        "--input",
        "-i",
        help="数据路径：单个表格文件（.xlsx / .xls / .csv）或含子目录的数据根目录。",
    )
    p.add_argument(
        "--column",
        "-c",
        default=None,
        help="列名；文件模式默认 caption，目录模式默认 视频标题；可显式指定。",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="最多读取多少条去重后的标题；不传表示不限制。",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="`api` 方法每批送给模型的标题条数；`workflow` 下作为初筛批大小。",
    )
    p.add_argument(
        "--output",
        "-o",
        default="neg_label_results.xlsx",
        help="输出 xlsx 路径；`api` 直接写该文件，`workflow` 会派生出 initial/non_negative/negative 三个文件。",
    )
    p.add_argument(
        "--base-url",
        default="https://api.siliconflow.cn/v1",
        help="OpenAI 兼容 API 的 base URL。",
    )
    p.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V3.2",
        help="主模型名称；`api` 直接使用，`workflow` 默认作为初筛模型。",
    )
    p.add_argument(
        "--negative-review-model",
        default=None,
        help="`workflow` 负样本复核模型；不传则跟随 --model。",
    )
    p.add_argument(
        "--non-negative-review-model",
        default=None,
        help="`workflow` 非负样本抽检复核模型；不传则跟随 --model。",
    )
    p.add_argument(
        "--negative-review-batch-size",
        type=int,
        default=5,
        help="`workflow` 负样本复核批大小。",
    )
    p.add_argument(
        "--negative-review-labels",
        nargs="*",
        default=None,
        help="`workflow` 需要进行复核的负面标签列表；不传表示全部复核。",
    )
    p.add_argument(
        "--non-negative-review-batch-size",
        type=int,
        default=5,
        help="`workflow` 非负样本抽检复核批大小。",
    )
    p.add_argument(
        "--non-negative-review-ratio",
        type=float,
        default=0.2,
        help="`workflow` 初筛非负样本抽检比例。",
    )
    p.add_argument(
        "--non-negative-review-seed",
        type=int,
        default=42,
        help="`workflow` 初筛非负样本抽检随机种子。",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input:
        parser.error("必须提供 --input（数据文件或目录路径）。")

    if args.method == "api":
        run_api_labeling(
            input_path=args.input,
            column_name=args.column,
            max_items=args.max_items,
            batch_size=args.batch_size,
            output_xlsx=args.output,
            base_url=args.base_url,
            model=args.model,
        )
        return 0

    if args.method == "workflow":
        run_workflow_labeling(
            input_path=args.input,
            column_name=args.column,
            max_items=args.max_items,
            output_xlsx=args.output,
            base_url=args.base_url,
            screening_model=args.model,
            negative_review_model=args.negative_review_model,
            non_negative_review_model=args.non_negative_review_model,
            screening_batch_size=args.batch_size,
            negative_review_batch_size=args.negative_review_batch_size,
            negative_review_labels=args.negative_review_labels,
            non_negative_review_batch_size=args.non_negative_review_batch_size,
            non_negative_review_ratio=args.non_negative_review_ratio,
            non_negative_review_seed=args.non_negative_review_seed,
        )
        return 0

    print(f"[ERROR] 不支持的打标方法: {args.method}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
