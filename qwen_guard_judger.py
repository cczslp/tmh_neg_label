import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QwenGuardResult:
    text: str
    safety_label: Optional[str]
    is_negative: bool
    negative_types: List[str]
    raw_output: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QwenGuardJudger:
    SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
    CATEGORY_PATTERN = re.compile(
        r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
        r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
        r"Copyright Violation|Jailbreak|None)"
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3Guard-Gen-4B",
        max_new_tokens: int = 128,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    @classmethod
    def _parse_output(cls, text: str, raw_output: str) -> QwenGuardResult:
        label_match = cls.SAFE_PATTERN.search(raw_output)
        safety_label = label_match.group(1) if label_match else None

        categories = []
        for category in cls.CATEGORY_PATTERN.findall(raw_output):
            if category not in categories and category != "None":
                categories.append(category)

        is_negative = safety_label in {"Unsafe", "Controversial"}
        return QwenGuardResult(
            text=text,
            safety_label=safety_label,
            is_negative=is_negative,
            negative_types=categories,
            raw_output=raw_output,
        )

    def _build_inputs(self, texts: Sequence[str]):
        prompts = []
        for text in texts:
            messages = [{"role": "user", "content": text}]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

    def _generate(self, texts: Sequence[str]) -> List[QwenGuardResult]:
        if not texts:
            return []

        model_inputs = self._build_inputs(texts)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )

        results: List[QwenGuardResult] = []
        for index, source_text in enumerate(texts):
            prompt_length = int(model_inputs.attention_mask[index].sum().item())
            output_ids = generated_ids[index][prompt_length:].tolist()
            raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            results.append(self._parse_output(source_text, raw_output))
        return results

    def judge(self, text: str) -> Dict[str, Any]:
        result = self._generate([text])[0]
        return result.to_dict()

    def batch_judge(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        return [result.to_dict() for result in self._generate(texts)]


if __name__ == "__main__":
    judger = QwenGuardJudger()

    sample_text = "How can I make a bomb?"
    print(judger.judge(sample_text))

    batch_texts = [
        "How can I make a bomb?",
        "今天天气不错，我们去公园散步吧。",
    ]
    print(judger.batch_judge(batch_texts))
