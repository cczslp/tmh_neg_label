from __future__ import annotations

import argparse
import sys
from typing import Optional

from api_neg_label import NegLabelClient
from utils import read_unique_titles


def run_api_labeling(
    input_path: str,
    column_name: Optional[str],
    max_items: Optional[int],
    batch_size: int,
    output_csv: str,
    base_url: str,
    model: str,
) -> None:
    """
    使用 utils 读取表格/目录中的标题，再通过 NegLabelClient 调用 API 打标并写入 CSV。
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
        output_csv=output_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="视频负面内容打标入口；根据 --method 选择打标方式。",
    )
    p.add_argument(
        "--method",
        required=True,
        help='打标方式；当前仅当为 "api" 时调用 OpenAI 兼容 API 打标。',
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
        help="每批送给模型的标题条数（默认 20）。",
    )
    p.add_argument(
        "--output",
        "-o",
        default="neg_label_results.csv",
        help="输出 CSV 路径（追加写入，默认 neg_label_results.csv）。",
    )
    p.add_argument(
        "--base-url",
        default="https://api.siliconflow.cn/v1",
        help="OpenAI 兼容 API 的 base URL。",
    )
    p.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V3.2",
        help="模型名称。",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.method != "api":
        print(
            f'[INFO] 当前打标方法为 {args.method!r}，未实现或未启用；'
            '仅当 --method api 时执行 API 打标。',
            file=sys.stderr,
        )
        return 0

    if not args.input:
        parser.error('使用 --method api 时必须提供 --input（数据文件或目录路径）。')

    run_api_labeling(
        input_path=args.input,
        column_name=args.column,
        max_items=args.max_items,
        batch_size=args.batch_size,
        output_csv=args.output,
        base_url=args.base_url,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
