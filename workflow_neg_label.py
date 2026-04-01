import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import openpyxl

from openai import OpenAI

from api_neg_label import NEGATIVE_LABELS

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


SCREENING_SYSTEM_PROMPT = """判断短视频标题是否属于以下负面主题，并输出JSON数组。

负面标签（标签名:核心特征）：
色情擦边:通过挑逗文案、暧昧暗示、暗示性动作、露骨衣着和身材描述等方式，打色情“擦边球”，引发色情联想
地域歧视:以地域、省份、城市、籍贯、方言、风土习俗等为依据，对特定地域人群进行贬低、嘲讽、刻板印象、恶意攻击、排斥对立
群体歧视:针对性别、职业、学历、外貌、残障、疾病、贫富、性取向、民族、宗教等非地域身份群体，进行贬低、侮辱、排斥、区别对待、恶意攻击的内容
焦虑抑郁:渲染焦虑、抑郁、绝望等极端负面情绪，传播消极厌世、自我否定、无意义感，或暗示、描述、教唆自残、自杀、自伤行为
网络戾气:极端偏激、恶意攻击、辱骂嘲讽、煽动对立、人肉搜索、造谣网暴等内容，充斥负面戾气，对他人进行人格侮辱、精神伤害、隐私泄露、恶意抹黑，破坏网络文明环境
虚假谣言:故意捏造、歪曲事实，容易误导公众、引发恐慌、破坏秩序、侵害他人或组织合法权益
奢靡拜金:炫富攀比，鼓吹金钱至上，挥霍浪费
不良婚育观:煽动性别对立，不婚不育，宣扬出轨，抛弃家庭责任，极端婚恋观，重男轻女等
饭圈娱乐:拉踩引战，非理性应援，偶像崇拜，抬高明星
经济低迷:夸大本国失业裁员，企业倒闭，国内经济下行，股市下跌等现状，制造焦虑与恐慌

对于地域歧视等负面主题，请深入理解文本，不要把中性描述、引用他人观点或正常新闻概述误判为负面内容。

输出为JSON数组，每项字段：
- index: 标题序号（从0开始）
- is_negative: 是否负面（bool）
- negative_labels: 命中的负面标签列表

只输出JSON，不加任何额外文字。示例：
[{"index":0,"is_negative":false,"negative_labels":[]},{"index":1,"is_negative":true,"negative_labels":["色情擦边"]}]"""

NEGATIVE_LABEL_REVIEW_PROMPTS: Dict[str, str] = {
    "色情擦边": "你在复核标题是否真的属于“色情擦边”。只有明确存在性暗示、低俗挑逗、色情擦边式吸引点击时才保留该标签；普通穿搭、美妆、情感表达、正常颜值夸赞不算。",
    "地域歧视": "你在复核标题是否真的属于“地域歧视”。只有针对特定地域人群实施贬低、羞辱、刻板印象攻击、排斥对立时才保留；普通地域信息、旅游、美食、新闻讨论不算。",
    "群体歧视": "你在复核标题是否真的属于“群体歧视”。只有针对某类非地域身份群体进行贬低、侮辱、排斥、恶意攻击时才保留；一般观点表达、个体吐槽、非群体性批评不算。",
    "焦虑抑郁": "你在复核标题是否真的属于“焦虑抑郁”。只有明显渲染绝望、厌世、自我否定，或描述、暗示、教唆自残自杀时才保留；普通情绪低落、生活压力表达、心理健康科普不算。",
    "网络戾气": "你在复核标题是否真的属于“网络戾气”。只有明显存在极端辱骂、恶意攻击、煽动对立、网暴导向时才保留；一般争议、普通批评、轻微吐槽不算。",
    "虚假谣言": "你在复核标题是否真的属于“虚假谣言”。只有明显捏造、歪曲事实并可能误导公众时才保留；无法从标题单独确认真伪时要谨慎，不要轻易判定。",
    "奢靡拜金": "你在复核标题是否真的属于“奢靡拜金”。只有明确炫富攀比、鼓吹金钱至上、挥霍浪费时才保留；普通消费分享、理财、产品展示不算。",
    "不良婚育观": "你在复核标题是否真的属于“不良婚育观”。只有明显煽动性别对立、鼓吹不负责任婚育观、宣扬出轨或极端婚恋立场时才保留；普通婚恋经验分享、育儿吐槽不算。",
    "饭圈娱乐": "你在复核标题是否真的属于“饭圈娱乐”。只有明显拉踩引战、非理性应援、偶像崇拜失范、刻意抬高明星时才保留；普通娱乐资讯和粉丝表达不算。",
    "经济低迷": "你在复核标题是否真的属于“经济低迷”。只有明显夸大失业、倒闭、下行、恐慌并制造悲观情绪时才保留；客观经济新闻、就业讨论、个人求职经历不算。",
}

NEGATIVE_REVIEW_SYSTEM_PROMPT_TEMPLATE = """你正在复核一批已被初筛命中标签“{label}”的短视频标题。

{label_review_instruction}

请认真反思初筛结论是否站得住脚，可以逐条审慎判断，但最终只输出JSON数组。每项字段：
- index: 标题序号（从0开始）
- keep_label: 是否保留该标签（bool）
- reason: 不超过40字的简短理由

只输出JSON，不加任何额外文字。"""

NEGATIVE_REVIEW_USER_PROMPT_TEMPLATE = """请复核以下{n}条标题与标签“{label}”是否匹配：
{titles}"""

NON_NEGATIVE_REVIEW_SYSTEM_PROMPT = """你正在复核一批初筛为“非负面”的短视频标题，目标是找出被漏判的负面样本。

请重新审阅每条标题，必要时可进行简短推理，但最终只输出JSON数组。仅在标题本身已经足以支持判断时，才标记为负面。

可选负面标签：
{labels}

输出字段：
- index: 标题序号（从0开始）
- is_negative: 是否应改判为负面（bool）
- negative_labels: 命中的负面标签列表
- reason: 不超过40字的简短理由

只输出JSON，不加任何额外文字。"""

NON_NEGATIVE_REVIEW_USER_PROMPT_TEMPLATE = """以下{n}条标题在初筛中被判定为非负面，请复核是否存在漏判：
{titles}"""


@dataclass
class LabelResult:
    item_index: int
    title: str
    is_negative: bool
    negative_labels: List[str]
    raw_output: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "item_index": self.item_index,
            "title": self.title,
            "is_negative": self.is_negative,
            "negative_labels": "|".join(self.negative_labels),
            "negative_labels_count": len(self.negative_labels),
        }


@dataclass
class WorkflowResult:
    item_index: int
    title: str
    initial_is_negative: bool
    initial_negative_labels: List[str]
    final_is_negative: bool
    final_negative_labels: List[str]
    review_type: str
    review_status: str
    review_notes: str = ""
    raw_output: str = field(default="", repr=False)

    def to_initial_row(self) -> dict:
        return {
            "item_index": self.item_index,
            "title": self.title,
            "is_negative": self.initial_is_negative,
            "negative_labels": "|".join(self.initial_negative_labels),
            "negative_labels_count": len(self.initial_negative_labels),
        }

    def to_final_row(self) -> dict:
        return {
            "item_index": self.item_index,
            "title": self.title,
            "initial_is_negative": self.initial_is_negative,
            "initial_negative_labels": "|".join(self.initial_negative_labels),
            "final_is_negative": self.final_is_negative,
            "final_negative_labels": "|".join(self.final_negative_labels),
            "final_negative_labels_count": len(self.final_negative_labels),
            "review_type": self.review_type,
            "review_status": self.review_status,
            "review_notes": self.review_notes,
        }

    def to_compact_final_row(self) -> dict:
        return {
            "title": self.title,
            "is_negative": self.final_is_negative,
            "negative_labels": "|".join(self.final_negative_labels),
            "negative_labels_count": len(self.final_negative_labels),
        }


@dataclass
class WorkflowConfig:
    screening_batch_size: int = 20
    negative_review_batch_size: int = 8
    negative_review_labels: List[str] | None = None
    non_negative_review_batch_size: int = 20
    non_negative_review_ratio: float = 0.1
    non_negative_review_seed: int = 42


class WorkflowNegLabelClient:
    """基于初筛和复核工作流的短视频标题负面打标客户端。"""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        screening_model: str = "gpt-4o-mini",
        negative_review_model: str | None = None,
        non_negative_review_model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        config: WorkflowConfig | None = None,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "环境变量 OPENAI_API_KEY 未设置，请先执行: export OPENAI_API_KEY=your_key"
            )

        self.base_url = self._normalize_base_url(base_url)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.screening_model = screening_model
        self.negative_review_model = negative_review_model or screening_model
        self.non_negative_review_model = (
            non_negative_review_model or screening_model
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.config = config or WorkflowConfig()

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if normalized.endswith(suffix):
                return normalized[: -len(suffix)]
        return normalized

    @staticmethod
    def _build_numbered_titles(titles: Sequence[str]) -> str:
        return "\n".join(f"{i}. {title}" for i, title in enumerate(titles))

    def _call_api(self, model: str, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == self.max_retries:
                    raise
                print(
                    f"[WARN] API 调用失败（第 {attempt} 次）: {exc}，"
                    f"{self.retry_delay}s 后重试…"
                )
                time.sleep(self.retry_delay)
        return ""

    @staticmethod
    def _extract_json(raw: str):
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def _screen_titles(
        self,
        titles: Sequence[str],
        start_index: int = 0,
    ) -> List[LabelResult]:
        raw = self._call_api(
            model=self.screening_model,
            system_prompt=SCREENING_SYSTEM_PROMPT,
            user_prompt=f"判断以下{len(titles)}条标题：\n{self._build_numbered_titles(titles)}",
        )
        try:
            items = self._extract_json(raw)
        except Exception:
            return [
                LabelResult(
                    item_index=start_index + i,
                    title=title,
                    is_negative=False,
                    negative_labels=[],
                    raw_output=raw,
                )
                for i, title in enumerate(titles)
            ]

        idx_map = {item.get("index", i): item for i, item in enumerate(items)}
        results: List[LabelResult] = []
        for i, title in enumerate(titles):
            item = idx_map.get(i, {})
            labels = [
                label
                for label in item.get("negative_labels", [])
                if label in NEGATIVE_LABELS
            ]
            results.append(
                LabelResult(
                    item_index=start_index + i,
                    title=title,
                    is_negative=bool(item.get("is_negative", False)) and bool(labels),
                    negative_labels=labels,
                    raw_output=raw,
                )
            )
        return results

    def _review_negative_label(
        self,
        label: str,
        titles: Sequence[str],
    ) -> List[bool]:
        if not titles:
            return []

        system_prompt = NEGATIVE_REVIEW_SYSTEM_PROMPT_TEMPLATE.format(
            label=label,
            label_review_instruction=NEGATIVE_LABEL_REVIEW_PROMPTS[label],
        )
        user_prompt = NEGATIVE_REVIEW_USER_PROMPT_TEMPLATE.format(
            label=label,
            n=len(titles),
            titles=self._build_numbered_titles(titles),
        )
        raw = self._call_api(
            model=self.negative_review_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        try:
            items = self._extract_json(raw)
        except Exception:
            return [True] * len(titles)

        idx_map = {item.get("index", i): item for i, item in enumerate(items)}
        return [bool(idx_map.get(i, {}).get("keep_label", True)) for i in range(len(titles))]

    def _review_negative_results(
        self,
        initial_results: Sequence[LabelResult],
    ) -> Dict[int, List[str]]:
        reviewable_labels = (
            set(self.config.negative_review_labels)
            if self.config.negative_review_labels is not None
            else set(NEGATIVE_LABELS)
        )
        confirmed_labels_by_index: Dict[int, List[str]] = {}

        for result in initial_results:
            if not result.is_negative:
                continue
            confirmed_labels_by_index[result.item_index] = [
                label
                for label in result.negative_labels
                if label not in reviewable_labels
            ]

        for label in NEGATIVE_LABELS:
            if label not in reviewable_labels:
                continue

            item_group = [
                (result.item_index, result.title)
                for result in initial_results
                if result.is_negative and label in result.negative_labels
            ]
            if not item_group:
                continue

            for start in range(0, len(item_group), self.config.negative_review_batch_size):
                batch = item_group[start: start + self.config.negative_review_batch_size]
                keep_flags = self._review_negative_label(
                    label=label,
                    titles=[title for _, title in batch],
                )
                for (item_index, _), keep in zip(batch, keep_flags):
                    if keep:
                        confirmed_labels_by_index[item_index].append(label)

        return confirmed_labels_by_index

    def _sample_non_negative_titles(
        self,
        initial_results: Sequence[LabelResult],
    ) -> List[tuple[int, str]]:
        candidates = [
            (result.item_index, result.title)
            for result in initial_results
            if not result.is_negative
        ]
        if not candidates:
            return []

        ratio = min(max(self.config.non_negative_review_ratio, 0.0), 1.0)
        if ratio <= 0:
            return []

        sample_size = min(len(candidates), max(1, math.ceil(len(candidates) * ratio)))
        rng = random.Random(self.config.non_negative_review_seed)
        sampled = rng.sample(candidates, sample_size)
        sampled.sort(key=lambda item: item[0])
        return sampled

    def _review_non_negative_batch(self, titles: Sequence[str]) -> List[LabelResult]:
        if not titles:
            return []

        system_prompt = NON_NEGATIVE_REVIEW_SYSTEM_PROMPT.format(
            labels="、".join(NEGATIVE_LABELS)
        )
        user_prompt = NON_NEGATIVE_REVIEW_USER_PROMPT_TEMPLATE.format(
            n=len(titles),
            titles=self._build_numbered_titles(titles),
        )
        raw = self._call_api(
            model=self.non_negative_review_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        try:
            items = self._extract_json(raw)
        except Exception:
            return [
                LabelResult(
                    item_index=i,
                    title=title,
                    is_negative=False,
                    negative_labels=[],
                    raw_output=raw,
                )
                for i, title in enumerate(titles)
            ]

        idx_map = {item.get("index", i): item for i, item in enumerate(items)}
        results: List[LabelResult] = []
        for i, title in enumerate(titles):
            item = idx_map.get(i, {})
            labels = [
                label
                for label in item.get("negative_labels", [])
                if label in NEGATIVE_LABELS
            ]
            results.append(
                LabelResult(
                    item_index=i,
                    title=title,
                    is_negative=bool(item.get("is_negative", False)) and bool(labels),
                    negative_labels=labels,
                    raw_output=raw,
                )
            )
        return results

    def _review_non_negative_results(
        self,
        initial_results: Sequence[LabelResult],
    ) -> Dict[int, LabelResult]:
        sampled_items = self._sample_non_negative_titles(initial_results)
        reviewed: Dict[int, LabelResult] = {}

        for start in range(0, len(sampled_items), self.config.non_negative_review_batch_size):
            batch = sampled_items[start: start + self.config.non_negative_review_batch_size]
            results = self._review_non_negative_batch([title for _, title in batch])
            for (item_index, title), result in zip(batch, results):
                reviewed[item_index] = LabelResult(
                    item_index=item_index,
                    title=title,
                    is_negative=result.is_negative,
                    negative_labels=result.negative_labels,
                    raw_output=result.raw_output,
                )
        return reviewed

    @staticmethod
    def _save_xlsx(
        rows: List[dict],
        output_path: str | Path,
        headers: Sequence[str] | None = None,
    ) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        workbook = openpyxl.Workbook()
        worksheet = workbook.active

        if rows or headers:
            headers = list(headers or rows[0].keys())
            worksheet.append(headers)
            for row in rows:
                worksheet.append([row.get(header, "") for header in headers])
        workbook.save(output)

    def label_titles(
        self,
        titles: List[str],
        initial_output_xlsx: str = "output/workflow_initial_results.xlsx",
        final_non_negative_output_xlsx: str = "output/workflow_final_non_negative.xlsx",
        final_negative_output_xlsx: str = "output/workflow_final_negative.xlsx",
    ) -> List[WorkflowResult]:
        if not titles:
            self._save_xlsx(
                [],
                initial_output_xlsx,
                headers=["item_index", "title", "is_negative", "negative_labels", "negative_labels_count"],
            )
            self._save_xlsx(
                [],
                final_non_negative_output_xlsx,
                headers=["title", "is_negative", "negative_labels", "negative_labels_count"],
            )
            self._save_xlsx(
                [],
                final_negative_output_xlsx,
                headers=["title", "is_negative", "negative_labels", "negative_labels_count"],
            )
            return []

        initial_results: List[LabelResult] = []
        pbar_ctx = (
            tqdm(total=len(titles), desc="初筛", unit="条", smoothing=0.05)
            if tqdm is not None
            else nullcontext()
        )
        with pbar_ctx as pbar:
            for start in range(0, len(titles), self.config.screening_batch_size):
                batch = titles[start: start + self.config.screening_batch_size]
                results = self._screen_titles(batch, start_index=start)
                initial_results.extend(results)
                if pbar is not None:
                    pbar.update(len(batch))

        self._save_xlsx(
            [result.to_dict() for result in initial_results],
            initial_output_xlsx,
        )

        negative_review_map = self._review_negative_results(initial_results)
        non_negative_review_map = self._review_non_negative_results(initial_results)

        final_results: List[WorkflowResult] = []
        for result in initial_results:
            if result.is_negative:
                final_labels = negative_review_map.get(result.item_index, [])
                final_results.append(
                    WorkflowResult(
                        item_index=result.item_index,
                        title=result.title,
                        initial_is_negative=True,
                        initial_negative_labels=result.negative_labels,
                        final_is_negative=bool(final_labels),
                        final_negative_labels=final_labels,
                        review_type="negative_recheck",
                        review_status="confirmed_negative" if final_labels else "reversed_to_non_negative",
                        review_notes="按标签逐类复核",
                    )
                )
                continue

            reviewed_result = non_negative_review_map.get(result.item_index)
            if reviewed_result is None:
                final_results.append(
                    WorkflowResult(
                        item_index=result.item_index,
                        title=result.title,
                        initial_is_negative=False,
                        initial_negative_labels=[],
                        final_is_negative=False,
                        final_negative_labels=[],
                        review_type="non_negative_sampling",
                        review_status="not_sampled",
                        review_notes="未进入抽检复核样本",
                    )
                )
                continue

            final_results.append(
                WorkflowResult(
                    item_index=result.item_index,
                    title=result.title,
                    initial_is_negative=False,
                    initial_negative_labels=[],
                    final_is_negative=reviewed_result.is_negative,
                    final_negative_labels=reviewed_result.negative_labels,
                    review_type="non_negative_sampling",
                    review_status=(
                        "corrected_to_negative"
                        if reviewed_result.is_negative
                        else "confirmed_non_negative"
                    ),
                    review_notes="非负样本抽检复核",
                )
            )

        final_non_negative_rows = [
            result.to_compact_final_row()
            for result in final_results
            if (
                not result.initial_is_negative
                and result.review_status == "confirmed_non_negative"
            )
        ]
        final_negative_rows = [
            result.to_compact_final_row()
            for result in final_results
            if result.final_is_negative
        ]
        self._save_xlsx(final_non_negative_rows, final_non_negative_output_xlsx)
        self._save_xlsx(final_negative_rows, final_negative_output_xlsx)

        print(f"[INFO] 初筛结果已写入 {Path(initial_output_xlsx).resolve()}")
        print(f"[INFO] 最终非负结果已写入 {Path(final_non_negative_output_xlsx).resolve()}")
        print(f"[INFO] 最终负面结果已写入 {Path(final_negative_output_xlsx).resolve()}")
        return final_results

    def label_file(
        self,
        input_file: str,
        initial_output_xlsx: str = "output/workflow_initial_results.xlsx",
        final_non_negative_output_xlsx: str = "output/workflow_final_non_negative.xlsx",
        final_negative_output_xlsx: str = "output/workflow_final_negative.xlsx",
        encoding: str = "utf-8",
    ) -> List[WorkflowResult]:
        with open(input_file, "r", encoding=encoding) as file:
            titles = [line.strip() for line in file if line.strip()]
        print(f"[INFO] 从 {input_file} 读取到 {len(titles)} 条标题")
        return self.label_titles(
            titles=titles,
            initial_output_xlsx=initial_output_xlsx,
            final_non_negative_output_xlsx=final_non_negative_output_xlsx,
            final_negative_output_xlsx=final_negative_output_xlsx,
        )


if __name__ == "__main__":
    client = WorkflowNegLabelClient(
        base_url="https://api.siliconflow.cn/v1",
        screening_model="deepseek-ai/DeepSeek-V3.2",
        negative_review_model="deepseek-ai/DeepSeek-V3.2",
        non_negative_review_model="deepseek-ai/DeepSeek-V3.2",
        config=WorkflowConfig(
            screening_batch_size=10,
            negative_review_batch_size=4,
            negative_review_labels=["色情擦边", "焦虑抑郁", "网络戾气"],
            non_negative_review_batch_size=8,
            non_negative_review_ratio=0.2,
            non_negative_review_seed=42,
        ),
    )

    sample_titles = [
        "超性感小姐姐教你这样穿，男生都把持不住了",
        "今天去公园散步，天气真好！",
        "河南人真的很坏，骗子最多",
        "股市暴跌，明天必崩，赶紧跑！",
        "宝宝第一次走路，超可爱",
        "生二胎的三个条件：好的经济条件，体贴上进的老公，靠谱的公婆会帮忙带孩子，如果一样都没占，谁催都别生，不然会崩溃的#产后",
        "都是出来玩的 叫我一声老公怎么了？",
        "感觉生完孩子都有一些轻微抑郁，有时候情绪低落，孩子有一点不听话就想吼他",
    ]

    start_time = time.time()
    results = client.label_titles(titles=sample_titles)
    print(f"Time taken: {time.time() - start_time} seconds")
    for result in results:
        print(result)
