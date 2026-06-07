import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.evaluation.quality_gate import (  # noqa: E402
    RetrievalQualityGateConfig,
    evaluate_retrieval_quality_gate,
    format_quality_gate_result,
    load_report,
)


def main() -> None:
    args = _parse_args()
    config = RetrievalQualityGateConfig(
        min_case_count=args.min_case_count,
        min_summary_metrics=_metric_thresholds(args.min_metric),
        max_summary_metrics=_metric_thresholds(args.max_metric),
        required_case_metrics=tuple(args.require_case_metric),
        forbidden_true_case_metrics=tuple(args.forbid_case_metric),
        min_case_answer_term_coverage=args.min_case_answer_term_coverage,
        required_diagnostic_keys=tuple(args.require_diagnostic_key),
    )
    result = evaluate_retrieval_quality_gate(load_report(args.report), config)
    print(format_quality_gate_result(result))
    if not result.passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 RAG 离线评测报告是否达到检索交付门禁。")
    parser.add_argument("--report", type=Path, default=Path("eval/reports/latest.json"))
    parser.add_argument("--min-case-count", type=int, default=1)
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[
            "route_hit=0.95",
            "target_hit=0.95",
            "citation_page_hit=0.90",
            "source_url_valid=1.0",
            "retrieval_behavior_hit=0.95",
            "answer_term_coverage=0.80",
            "final_answer_accepted=0.95",
        ],
        help="总体指标下限，格式 name=value；可重复传入。",
    )
    parser.add_argument(
        "--max-metric",
        action="append",
        default=["unsupported_claim_risk=0.0", "max_rounds_exhausted=0.0"],
        help="总体指标上限，格式 name=value；可重复传入。",
    )
    parser.add_argument(
        "--require-case-metric",
        action="append",
        default=["route_hit", "target_hit", "source_url_valid", "retrieval_behavior_hit", "final_answer_accepted"],
        help="每个案例必须为 true 的指标；可重复传入。",
    )
    parser.add_argument(
        "--forbid-case-metric",
        action="append",
        default=["unsupported_claim_risk", "max_rounds_exhausted"],
        help="每个案例不得为 true 的指标；可重复传入。",
    )
    parser.add_argument("--min-case-answer-term-coverage", type=float, default=0.50)
    parser.add_argument(
        "--require-diagnostic-key",
        action="append",
        default=[],
        help="每个案例必须存在的点号路径，例如 diagnostics.recall.lexical_candidate_count。",
    )
    return parser.parse_args()


def _metric_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"指标阈值格式错误：`{raw}`，应为 name=value。")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"指标阈值名称为空：`{raw}`。")
        try:
            thresholds[name] = float(value)
        except ValueError as exc:
            raise SystemExit(f"指标阈值不是数字：`{raw}`。") from exc
    return thresholds


if __name__ == "__main__":
    main()
