from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Any, Protocol

from backend.app.domain.documents import Evidence
from backend.app.evaluation.cases import GoldenCase
from backend.app.providers.embeddings import EmbeddingProvider


BASIC_RAGAS_METRICS = ("faithfulness", "answer_relevancy")
RETRIEVAL_RAGAS_METRICS = (*BASIC_RAGAS_METRICS, "context_precision")
REFERENCE_RAGAS_METRICS = ("context_recall", "answer_correctness")
ALL_RAGAS_METRICS = (*RETRIEVAL_RAGAS_METRICS, *REFERENCE_RAGAS_METRICS)


@dataclass(frozen=True)
class RagasEvaluationConfig:
    metric_set: str = "basic"
    sample_limit: int | None = None
    max_context_chars: int = 4000
    fail_on_error: bool = False


class RagasRunner(Protocol):
    def evaluate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        ...


def build_ragas_row(
    case: GoldenCase,
    answer: str,
    evidence: list[Evidence],
    *,
    max_context_chars: int = 4000,
) -> dict[str, Any]:
    contexts = [_trim_context(item.text, max_context_chars) for item in evidence if item.text.strip()]
    reference = case.reference_answer
    row = {
        "case_id": case.case_id,
        "user_input": case.question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": reference,
        "reference_contexts": case.reference_contexts,
        "reference_hint": _reference_from_case_hints(case),
        "ragas_tags": case.ragas_tags,
        # 保留旧版 Ragas 常见字段名，便于跨版本兼容。
        "question": case.question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": reference,
    }
    return row


def merge_ragas_report(report: dict[str, Any], ragas_report: dict[str, Any] | None) -> dict[str, Any]:
    if ragas_report is None:
        report["ragas"] = {"enabled": False}
        return report
    report["ragas"] = {
        "enabled": bool(ragas_report.get("enabled", False)),
        "summary": ragas_report.get("summary", {}),
        "metrics": ragas_report.get("metrics", []),
        "skipped_metrics": ragas_report.get("skipped_metrics", []),
        "errors": ragas_report.get("errors", []),
    }
    by_case_id = {item["case_id"]: item for item in ragas_report.get("results", []) if "case_id" in item}
    for result in report.get("results", []):
        case_id = result.get("case", {}).get("case_id")
        result["ragas"] = by_case_id.get(case_id, {"status": "not_evaluated"})
    return report


class RagasEvaluator:
    def __init__(
        self,
        *,
        settings: Any,
        embedding_provider: EmbeddingProvider,
        config: RagasEvaluationConfig,
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.config = config

    def evaluate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        limited_rows = rows[: self.config.sample_limit] if self.config.sample_limit else rows
        try:
            return run_ragas_evaluation(
                limited_rows,
                settings=self.settings,
                embedding_provider=self.embedding_provider,
                config=self.config,
            )
        except Exception as exc:
            if self.config.fail_on_error:
                raise
            return {
                "enabled": True,
                "summary": {},
                "metrics": [],
                "skipped_metrics": [],
                "results": [],
                "errors": [str(exc)],
            }


def run_ragas_evaluation(
    rows: list[dict[str, Any]],
    *,
    settings: Any,
    embedding_provider: EmbeddingProvider,
    config: RagasEvaluationConfig,
) -> dict[str, Any]:
    if not rows:
        return {"enabled": True, "summary": {}, "metrics": [], "skipped_metrics": [], "results": [], "errors": []}

    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError("Ragas 评测需要安装可选依赖：python -m pip install -e \".[eval]\"") from exc

    metric_names, skipped_metrics = _metric_names_for_rows(config.metric_set, rows)
    metrics = _load_ragas_metrics(metric_names)
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=_build_ragas_llm(settings),
        embeddings=_build_ragas_embeddings(embedding_provider),
        raise_exceptions=config.fail_on_error,
        column_map={
            "user_input": "user_input",
            "response": "response",
            "retrieved_contexts": "retrieved_contexts",
            "reference": "reference",
        },
    )
    result_rows = _rows_from_ragas_result(result)
    return {
        "enabled": True,
        "summary": _summarize_ragas_rows(result_rows, metric_names),
        "metrics": list(metric_names),
        "skipped_metrics": skipped_metrics,
        "results": [
            {
                "case_id": str(item.get("case_id", rows[index].get("case_id", ""))),
                "metrics": {
                    name: item[name]
                    for name in metric_names
                    if name in item and isinstance(item[name], (int, float)) and not math.isnan(float(item[name]))
                },
            }
            for index, item in enumerate(result_rows)
        ],
        "errors": [],
    }


def _metric_names_for_rows(metric_set: str, rows: list[dict[str, Any]]) -> tuple[tuple[str, ...], list[str]]:
    requested = {
        "basic": BASIC_RAGAS_METRICS,
        "retrieval": RETRIEVAL_RAGAS_METRICS,
        "all": ALL_RAGAS_METRICS,
    }.get(metric_set, BASIC_RAGAS_METRICS)
    has_reference = all(str(row.get("reference") or "").strip() for row in rows)
    metrics = []
    skipped = []
    for name in requested:
        if name in REFERENCE_RAGAS_METRICS and not has_reference:
            skipped.append(name)
            continue
        metrics.append(name)
    return tuple(metrics), skipped


def _load_ragas_metrics(metric_names: tuple[str, ...]) -> list[Any]:
    from ragas import metrics as ragas_metrics

    loaded = []
    for name in metric_names:
        if not hasattr(ragas_metrics, name):
            raise RuntimeError(f"当前 Ragas 版本不支持指标：{name}")
        loaded.append(getattr(ragas_metrics, name))
    return loaded


def _build_ragas_llm(settings: Any) -> Any:
    if not getattr(settings, "deepseek_api_key", None):
        raise RuntimeError("启用 Ragas 需要配置 DEEPSEEK_API_KEY。")
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError("Ragas 使用 DeepSeek 评测需要安装 langchain-openai 与 ragas。") from exc

    chat_model = ChatOpenAI(
        model=getattr(settings, "deepseek_model"),
        api_key=getattr(settings, "deepseek_api_key"),
        base_url=getattr(settings, "deepseek_base_url"),
        timeout=getattr(settings, "deepseek_timeout_seconds"),
        temperature=0,
    )
    return LangchainLLMWrapper(chat_model)


def _build_ragas_embeddings(embedding_provider: EmbeddingProvider) -> Any:
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise RuntimeError("Ragas embedding wrapper 不可用，请确认 ragas 已安装。") from exc
    return LangchainEmbeddingsWrapper(_EmbeddingProviderWrapper(embedding_provider))


class _EmbeddingProviderWrapper:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.provider.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


def _rows_from_ragas_result(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "to_pandas"):
        return result.to_pandas().to_dict("records")
    if hasattr(result, "scores"):
        return list(result.scores)
    if isinstance(result, dict):
        values = [value for value in result.values() if isinstance(value, list)]
        if values:
            row_count = len(values[0])
            return [{key: value[index] for key, value in result.items() if isinstance(value, list)} for index in range(row_count)]
        return [result]
    raise RuntimeError("无法解析 Ragas evaluate 返回结果。")


def _summarize_ragas_rows(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name in metric_names:
        values = [
            float(item[name])
            for item in rows
            if name in item and isinstance(item[name], (int, float)) and not math.isnan(float(item[name]))
        ]
        if values:
            summary[name] = sum(values) / len(values)
    return summary


def _reference_from_case_hints(case: GoldenCase) -> str | None:
    hints = [*case.expected_answer_terms, *case.expected_aspects]
    if not hints:
        return None
    unique_hints = list(dict.fromkeys(term for term in hints if term.strip()))
    return "Expected answer should cover: " + "; ".join(unique_hints)


def _trim_context(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()
