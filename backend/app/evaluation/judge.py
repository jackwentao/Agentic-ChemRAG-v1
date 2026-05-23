from __future__ import annotations

import json
from typing import Any, Protocol

from backend.app.config import Settings


class JudgeClient(Protocol):
    def create(self, *, model: str, messages: list[dict[str, str]], response_format: dict[str, str]) -> Any:
        ...


class LlmJudge:
    def __init__(self, settings: Settings, client: JudgeClient | None = None) -> None:
        self.model = settings.deepseek_model
        if client is None:
            if not settings.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when --llm-judge on.")
            from openai import OpenAI

            openai_client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.deepseek_timeout_seconds,
            )
            client = openai_client.chat.completions
        self.client = client

    def score(
        self,
        question: str,
        gold_hints: dict[str, Any],
        answer: str,
        citations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are judging a chemistry/materials-science RAG answer. "
                    "Use only the provided question, gold hints, answer, and citations. "
                    "Do not use external knowledge. Return strict JSON with integer scores "
                    "faithfulness, completeness, relevance from 1 to 5 and a short rationale."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "gold_hints": gold_hints,
                        "answer": answer,
                        "citations": citations,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.client.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return parse_judge_payload(content)


def parse_judge_payload(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc}") from exc
    scores: dict[str, Any] = {}
    for key in ["faithfulness", "completeness", "relevance"]:
        value = payload.get(key)
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"Judge score {key} must be an integer from 1 to 5.")
        scores[key] = value
    rationale = payload.get("rationale")
    scores["rationale"] = rationale if isinstance(rationale, str) else ""
    return scores
