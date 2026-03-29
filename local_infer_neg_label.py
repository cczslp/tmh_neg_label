"""
本地 Transformers 多卡并行负面打标；分类标准、提示词、解析与输出格式与 api_neg_label 一致。
"""
from __future__ import annotations

import csv
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from api_neg_label import (
    LabelResult,
    NegLabelClient,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def _build_user_message(titles: List[str]) -> str:
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    return USER_PROMPT_TEMPLATE.format(n=len(titles), titles=numbered)


@dataclass
class _WorkerPayload:
    gpu_id: int
    tasks: List[Tuple[int, List[str]]]
    model_path: str
    max_new_tokens: int


def _infer_one_batch(
    model,
    tokenizer,
    device: torch.device,
    titles: List[str],
    max_new_tokens: int,
) -> str:
    user_text = _build_user_message(titles)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, in_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _gpu_worker_run(payload: _WorkerPayload) -> List[Tuple[int, List[LabelResult]]]:
    """子进程：仅可见一张卡，按 tasks 顺序推理，返回 (全局 batch 下标, 该批 LabelResult 列表)。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload.gpu_id)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(payload.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        payload.model_path,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    out: List[Tuple[int, List[LabelResult]]] = []
    for batch_idx, batch_titles in payload.tasks:
        raw = _infer_one_batch(
            model, tokenizer, device, batch_titles, payload.max_new_tokens
        )
        parsed = NegLabelClient._parse_response(batch_titles, raw)
        out.append((batch_idx, parsed))
    return out


class LocalNegLabelClient:
    """使用 Transformers 在本地多 GPU 上并行对标题打标（结果顺序与输入批次顺序一致）。"""

    def __init__(
        self,
        model_path: str,
        device_ids: Sequence[int] | None = None,
        max_new_tokens: int = 2048,
        mp_context: str = "spawn",
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.mp_context = mp_context
        if device_ids is None:
            n = torch.cuda.device_count()
            if n == 0:
                raise RuntimeError("未检测到 CUDA 设备，无法进行本地 GPU 推理。")
            self.device_ids = list(range(n))
        else:
            self.device_ids = list(device_ids)

    def _split_batches_across_gpus(
        self, batches: List[List[str]]
    ) -> List[_WorkerPayload]:
        n_gpu = len(self.device_ids)
        per_gpu: List[List[Tuple[int, List[str]]]] = [[] for _ in range(n_gpu)]
        for bi, batch in enumerate(batches):
            per_gpu[bi % n_gpu].append((bi, batch))
        payloads: List[_WorkerPayload] = []
        for slot, gpu_id in enumerate(self.device_ids):
            tasks = per_gpu[slot]
            if tasks:
                payloads.append(
                    _WorkerPayload(
                        gpu_id=gpu_id,
                        tasks=tasks,
                        model_path=self.model_path,
                        max_new_tokens=self.max_new_tokens,
                    )
                )
        return payloads

    def _run_parallel(self, batches: List[List[str]]) -> List[List[LabelResult]]:
        if not batches:
            return []
        payloads = self._split_batches_across_gpus(batches)
        if len(payloads) == 1:
            chunk = _gpu_worker_run(payloads[0])
            chunk.sort(key=lambda x: x[0])
            return [results for _, results in chunk]

        ctx = get_context(self.mp_context)
        with ctx.Pool(processes=len(payloads)) as pool:
            chunks = pool.map(_gpu_worker_run, payloads)

        flat: List[Tuple[int, List[LabelResult]]] = []
        for ch in chunks:
            flat.extend(ch)
        flat.sort(key=lambda x: x[0])
        return [results for _, results in flat]

    def label_titles(
        self,
        titles: List[str],
        batch_size: int = 20,
        output_csv: str = "neg_label_results.csv",
    ) -> List[LabelResult]:
        if not titles:
            return []

        batches = [
            titles[i : i + batch_size] for i in range(0, len(titles), batch_size)
        ]

        output_path = Path(output_csv)
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        all_results: List[LabelResult] = []

        pbar_ctx = (
            tqdm(
                total=len(titles),
                desc="打标",
                unit="条",
                smoothing=0.05,
            )
            if tqdm is not None
            else nullcontext()
        )

        with output_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "title",
                "is_negative",
                "negative_labels",
                "negative_labels_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            with pbar_ctx as pbar:
                batch_results_list = self._run_parallel(batches)
                for batch, results in zip(batches, batch_results_list):
                    if pbar is None or tqdm is None:
                        print(
                            f"[INFO] 写入批次（{len(batch)} 条），累计进度 "
                            f"{len(all_results) + len(batch)}/{len(titles)}"
                        )
                    for r in results:
                        writer.writerow(r.to_dict())
                    f.flush()
                    all_results.extend(results)
                    if pbar is not None:
                        pbar.update(len(batch))

        print(f"[INFO] 打标完成，结果已追加至 {output_path.resolve()}")
        return all_results

    def label_file(
        self,
        input_file: str,
        batch_size: int = 20,
        output_csv: str = "neg_label_results.csv",
        encoding: str = "utf-8",
    ) -> List[LabelResult]:
        with open(input_file, "r", encoding=encoding) as f:
            titles = [line.strip() for line in f if line.strip()]
        print(f"[INFO] 从 {input_file} 读取到 {len(titles)} 条标题")
        return self.label_titles(titles, batch_size=batch_size, output_csv=output_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="本地 Transformers 多卡并行负面打标（与 api_neg_label 标准一致）。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/huangbeining/qwen2.5-7b-it",
        help="本地模型目录（HuggingFace 格式）。",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="逗号分隔的物理 GPU 编号，如 0,1,2,3；默认使用当前可见的全部 GPU。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="每批送给模型的标题条数。",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="neg_label_results_local.csv",
        help="输出 CSV 路径（追加）。",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="可选：每行一个标题的文本文件；不传则运行内置示例。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="生成最大新 token 数。",
    )
    args = parser.parse_args()

    device_ids = None
    if args.gpus:
        device_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    client = LocalNegLabelClient(
        model_path=args.model,
        device_ids=device_ids,
        max_new_tokens=args.max_new_tokens,
    )

    if args.input:
        t0 = time.time()
        client.label_file(
            input_file=args.input,
            batch_size=args.batch_size,
            output_csv=args.output,
        )
        print(f"[INFO] 总耗时: {time.time() - t0:.2f}s")
    else:
        sample_titles = [
            "超性感小姐姐教你这样穿，男生都把持不住了",
            "今天去公园散步，天气真好！",
            "河南人真的很坏，骗子最多",
            "股市暴跌，明天必崩，赶紧跑！",
            "宝宝第一次走路，超可爱",
        ]
        t0 = time.time()
        client.label_titles(
            titles=sample_titles,
            batch_size=2,
            output_csv=args.output,
        )
        print(f"[INFO] 总耗时: {time.time() - t0:.2f}s")
