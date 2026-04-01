import json
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import openpyxl

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

NEGATIVE_LABELS = [
    "色情擦边",
    "地域歧视",
    "群体歧视",
    "焦虑抑郁",
    "网络戾气",
    "虚假谣言",
    "奢靡拜金",
    "不良婚育观",
    "饭圈娱乐",
    "经济低迷",
]

SYSTEM_PROMPT = """判断短视频标题是否属于以下负面主题，并输出JSON数组。

负面标签（标签名:核心特征）：
色情擦边:通过暴露、暗示性动作、低俗挑逗文案、暧昧暗示、低俗性诱惑等方式，打色情 “擦边球”，引发低俗联想
地域歧视:以地域、省份、城市、籍贯、方言、风土习俗等为依据，对特定地域人群进行贬低、嘲讽、刻板印象、恶意攻击、排斥对立的言论及内容
群体歧视:针对性别、职业、学历、外貌、残障、疾病、贫富、性取向、民族、宗教等非地域身份群体，进行贬低、侮辱、排斥、区别对待、恶意攻击的内容
焦虑抑郁:渲染焦虑、抑郁、绝望等极端负面情绪，传播消极厌世、自我否定、无意义感，或暗示、描述、教唆自残、自杀、自伤行为
网络戾气:极端偏激、恶意攻击、辱骂嘲讽、煽动对立、人肉搜索、造谣网暴等内容，充斥负面戾气，对他人进行人格侮辱、精神伤害、隐私泄露、恶意抹黑，破坏网络文明环境
虚假谣言:故意捏造、歪曲事实，容易误导公众、引发恐慌、破坏秩序、侵害他人或组织合法权益
奢靡拜金:炫富攀比，鼓吹金钱至上，挥霍浪费
不良婚育观:煽动性别对立，不婚不育，宣扬出轨，抛弃家庭责任，极端婚恋观，重男轻女等
饭圈娱乐:拉踩引战，非理性应援，偶像崇拜，抬高明星
经济低迷:夸大本国失业裁员，企业倒闭，国内经济下行，股市下跌等现状，制造焦虑与恐慌

对于地域歧视等负面主题，请深入理解文本，不要把中性描述或报道当成负面内容。

输出为JSON数组，每项字段：index(0起)、is_negative(bool)、negative_labels(list)。
只输出JSON，不加任何额外文字。示例：
[{{"index":0,"is_negative":false,"negative_labels":[]}},{{"index":1,"is_negative":true,"negative_labels":["色情擦边"]}}]"""

USER_PROMPT_TEMPLATE = "判断以下{n}条标题：\n{titles}"


@dataclass
class LabelResult:
    title: str
    is_negative: bool
    negative_labels: List[str]
    raw_output: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "is_negative": self.is_negative,
            "negative_labels": "|".join(self.negative_labels),
            "negative_labels_count": len(self.negative_labels),
        }


class NegLabelClient:
    """通过 OpenAI 兼容 API 对短视频标题进行负面主题打标。"""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "环境变量 OPENAI_API_KEY 未设置，请先执行: export OPENAI_API_KEY=your_key"
            )
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = self._normalize_base_url(base_url)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self._system_prompt = SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Normalize OpenAI-compatible base URL to the API root."""
        normalized = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized

    def _build_user_message(self, titles: List[str]) -> str:
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
        return USER_PROMPT_TEMPLATE.format(n=len(titles), titles=numbered)

    def _call_api(self, titles: List[str]) -> str:
        """Call the LLM and return raw text output."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": self._build_user_message(titles)},
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == self.max_retries:
                    raise
                print(f"[WARN] API 调用失败（第 {attempt} 次）: {exc}，{self.retry_delay}s 后重试…")
                time.sleep(self.retry_delay)
        return ""

    @staticmethod
    def _parse_response(titles: List[str], raw: str) -> List[LabelResult]:
        """Parse JSON array from model output into LabelResult list."""
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract a JSON array with regex
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                items = json.loads(m.group())
            else:
                # If parsing totally fails, mark all as error
                return [
                    LabelResult(
                        title=t,
                        is_negative=False,
                        negative_labels=[],
                        raw_output=raw,
                    )
                    for t in titles
                ]

        results: List[LabelResult] = []
        idx_map = {item.get("index", i): item for i, item in enumerate(items)}
        for i, title in enumerate(titles):
            item = idx_map.get(i, {})
            labels = [
                lb for lb in item.get("negative_labels", []) if lb in NEGATIVE_LABELS
            ]
            results.append(
                LabelResult(
                    title=title,
                    is_negative=bool(item.get("is_negative", False)),
                    negative_labels=labels,
                    raw_output=raw,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_titles(
        self,
        titles: List[str],
        batch_size: int = 20,
        output_xlsx: str = "neg_label_results.xlsx",
    ) -> List[LabelResult]:
        """
        对标题列表进行负面主题打标，结果以追加模式写入 Excel 文件。

        Args:
            titles:      视频标题字符串列表。
            batch_size:  每次发送给大模型的标题数量。
            output_xlsx: 输出 xlsx 文件路径（追加模式）。

        Returns:
            所有标题的 LabelResult 列表。
        """
        if not titles:
            return []

        output_path = Path(output_xlsx)
        fieldnames = ["title", "is_negative", "negative_labels", "negative_labels_count"]

        if output_path.exists():
            wb = openpyxl.load_workbook(output_path)
            ws = wb.active
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(fieldnames)

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
        with pbar_ctx as pbar:
            for batch_start in range(0, len(titles), batch_size):
                batch = titles[batch_start: batch_start + batch_size]
                if pbar is None or tqdm is None:
                    print(
                        f"[INFO] 处理第 {batch_start + 1}–{batch_start + len(batch)} 条标题"
                        f"（共 {len(titles)} 条）…"
                    )
                try:
                    raw = self._call_api(batch)
                    results = self._parse_response(batch, raw)
                except Exception as exc:
                    msg = f"[ERROR] 批次处理失败: {exc}"
                    if tqdm is not None and pbar is not None:
                        tqdm.write(msg)
                    else:
                        print(msg)
                    results = [
                        LabelResult(
                            title=t,
                            is_negative=False,
                            negative_labels=[],
                        )
                        for t in batch
                    ]

                for r in results:
                    d = r.to_dict()
                    ws.append([d[k] for k in fieldnames])
                wb.save(output_path)
                all_results.extend(results)
                if pbar is not None:
                    pbar.update(len(batch))

        print(f"[INFO] 打标完成，结果已追加至 {output_path.resolve()}")
        return all_results

    def label_file(
        self,
        input_file: str,
        batch_size: int = 20,
        output_xlsx: str = "neg_label_results.xlsx",
        encoding: str = "utf-8",
    ) -> List[LabelResult]:
        """
        从文本文件中读取标题（每行一个），然后调用 label_titles 进行打标。

        Args:
            input_file:  输入文本文件路径。
            batch_size:  每批大小。
            output_xlsx: 输出 xlsx 文件路径（追加模式）。
            encoding:    输入文件编码。

        Returns:
            所有标题的 LabelResult 列表。
        """
        with open(input_file, "r", encoding=encoding) as f:
            titles = [line.strip() for line in f if line.strip()]
        print(f"[INFO] 从 {input_file} 读取到 {len(titles)} 条标题")
        return self.label_titles(titles, batch_size=batch_size, output_xlsx=output_xlsx)


if __name__ == "__main__":
    client = NegLabelClient(
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V3.2",
    )
    # client = NegLabelClient(
    #     base_url="https://api.openai.com/v1",
    #     model="gpt-5.4-2026-03-05",
    # )

    sample_titles = [
        "超性感小姐姐教你这样穿，男生都把持不住了",
        "今天去公园散步，天气真好！",
        "河南人真的很坏，骗子最多",
        "股市暴跌，明天必崩，赶紧跑！",
        "宝宝第一次走路，超可爱",
        "生二胎的三个条件：好的经济条件，体贴上进的老公，靠谱的公婆会帮忙带孩子，如果一样都没占，谁催都别生，不然会崩溃的#产后",
        "都是出来玩的 叫我一声老公怎么了？",
        "感觉生完孩子都有一些轻微抑郁，有时候情绪低落，孩子有一点不听话就想吼他，看着满屋子乱糟糟的玩具还有吃东西掉的渣渣，心里烦的很#快手创作者服务中心# #感谢快手我要上热门# #推广小助手# #支持快手传播正能量# ##情感语录 #"
    ]
    
    t0 = time.time()
    results = client.label_titles(
        titles=sample_titles,
        batch_size=10,
        output_xlsx="neg_label_results.xlsx",
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
    for r in results:
        print(r)
