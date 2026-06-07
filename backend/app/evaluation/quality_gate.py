from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


DEFAULT_MIN_SUMMARY_METRICS = {
    "route_hit": 0.95,
    "target_hit": 0.95,
    "citation_page_hit": 0.90,
    "source_url_valid": 1.0,
    "retrieval_behavior_hit": 0.95,
    "answer_term_coverage": 0.80,
    "final_answer_accepted": 0.95,
}
DEFAULT_MAX_SUMMARY_METRICS = {
    "unsupported_claim_risk": 0.0,
    "max_rounds_exhausted": 0.0,
}
DEFAULT_REQUIRED_CASE_METRICS = (
    "route_hit",
    "target_hit",
    "source_url_valid",
    "retrieval_behavior_hit",
    "final_answer_accepted",
)
DEFAULT_FORBIDDEN_TRUE_CASE_METRICS = (
    "unsupported_claim_risk",
    "max_rounds_exhausted",
)


@dataclass(frozen=True)
class RetrievalQualityGateConfig:
    min_case_count: int = 1
    min_summary_metrics: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIN_SUMMARY_METRICS))
    max_summary_metrics: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MAX_SUMMARY_METRICS))
    required_case_metrics: tuple[str, ...] = DEFAULT_REQUIRED_CASE_METRICS
    forbidden_true_case_metrics: tuple[str, ...] = DEFAULT_FORBIDDEN_TRUE_CASE_METRICS
    min_case_answer_term_coverage: float = 0.50
    required_diagnostic_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalQualityGateResult:
    passed: bool
    failures: list[str]


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval_quality_gate(
    report: dict[str, Any],
    config: RetrievalQualityGateConfig | None = None,
) -> RetrievalQualityGateResult:
    config = config or RetrievalQualityGateConfig()
    failures: list[str] = []
    summary = _mapping(report.get("summary"))
    metrics = _mapping(summary.get("metrics"))
    results = _list(report.get("results"))

    case_count = int(summary.get("case_count", len(results)) or 0)
    if case_count < config.min_case_count:
        failures.append(f"案例数量不足：实际 {case_count}，要求至少 {config.min_case_count}。")

    for metric, minimum in config.min_summary_metrics.items():
        value = _numeric_metric(metrics, metric)
        if value is None:
            failures.append(f"缺少总体指标 `{metric}`。")
        elif value < minimum:
            failures.append(f"总体指标 `{metric}` 过低：实际 {_fmt(value)}，要求 >= {_fmt(minimum)}。")

    for metric, maximum in config.max_summary_metrics.items():
        value = _numeric_metric(metrics, metric)
        if value is None:
            failures.append(f"缺少总体指标 `{metric}`。")
        elif value > maximum:
            failures.append(f"总体指标 `{metric}` 过高：实际 {_fmt(value)}，要求 <= {_fmt(maximum)}。")

    for result in results:
        case_id = str(_mapping(result.get("case")).get("case_id", "<unknown>"))
        case_metrics = _mapping(result.get("metrics"))
        for metric in config.required_case_metrics:
            if case_metrics.get(metric) is not True:
                failures.append(f"案例 `{case_id}` 未通过逐项指标 `{metric}`。")
        for metric in config.forbidden_true_case_metrics:
            if case_metrics.get(metric) is True:
                failures.append(f"案例 `{case_id}` 触发禁止指标 `{metric}`。")
        answer_coverage = _numeric_metric(case_metrics, "answer_term_coverage")
        if answer_coverage is not None and answer_coverage < config.min_case_answer_term_coverage:
            failures.append(
                f"案例 `{case_id}` 答案术语覆盖过低：实际 {_fmt(answer_coverage)}，"
                f"要求 >= {_fmt(config.min_case_answer_term_coverage)}。"
            )
        for key in config.required_diagnostic_keys:
            if _get_nested(result, key) is None:
                failures.append(f"案例 `{case_id}` 缺少诊断字段 `{key}`。")

    return RetrievalQualityGateResult(passed=not failures, failures=failures)


def format_quality_gate_result(result: RetrievalQualityGateResult) -> str:
    if result.passed:
        return "检索质量门禁通过。"
    lines = ["检索质量门禁未通过："]
    lines.extend(f"- {failure}" for failure in result.failures)
    return "\n".join(lines)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _numeric_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
