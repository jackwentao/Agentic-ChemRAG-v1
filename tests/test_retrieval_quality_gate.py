from backend.app.evaluation.quality_gate import (
    RetrievalQualityGateConfig,
    evaluate_retrieval_quality_gate,
    format_quality_gate_result,
)


def _report(metric_overrides=None, case_metric_overrides=None):
    metrics = {
        "route_hit": 1.0,
        "target_hit": 1.0,
        "citation_page_hit": 1.0,
        "source_url_valid": 1.0,
        "retrieval_behavior_hit": 1.0,
        "answer_term_coverage": 1.0,
        "final_answer_accepted": 1.0,
        "unsupported_claim_risk": 0.0,
        "max_rounds_exhausted": 0.0,
    }
    metrics.update(metric_overrides or {})
    case_metrics = {
        "route_hit": True,
        "target_hit": True,
        "source_url_valid": True,
        "retrieval_behavior_hit": True,
        "final_answer_accepted": True,
        "unsupported_claim_risk": False,
        "max_rounds_exhausted": False,
        "answer_term_coverage": 1.0,
    }
    case_metrics.update(case_metric_overrides or {})
    return {
        "summary": {"case_count": 1, "metrics": metrics},
        "results": [
            {
                "case": {"case_id": "c1"},
                "metrics": case_metrics,
                "diagnostics": {"recall": {"lexical_candidate_count": 2}},
            }
        ],
    }


def test_retrieval_quality_gate_passes_clean_report():
    result = evaluate_retrieval_quality_gate(
        _report(),
        RetrievalQualityGateConfig(required_diagnostic_keys=("diagnostics.recall.lexical_candidate_count",)),
    )

    assert result.passed is True
    assert format_quality_gate_result(result) == "检索质量门禁通过。"


def test_retrieval_quality_gate_fails_low_summary_metric():
    result = evaluate_retrieval_quality_gate(_report(metric_overrides={"target_hit": 0.5}))

    assert result.passed is False
    assert any("总体指标 `target_hit` 过低" in failure for failure in result.failures)


def test_retrieval_quality_gate_fails_case_metric_and_answer_coverage():
    result = evaluate_retrieval_quality_gate(
        _report(case_metric_overrides={"final_answer_accepted": False, "answer_term_coverage": 0.25})
    )

    assert result.passed is False
    assert any("`final_answer_accepted`" in failure for failure in result.failures)
    assert any("答案术语覆盖过低" in failure for failure in result.failures)


def test_retrieval_quality_gate_requires_diagnostic_key():
    result = evaluate_retrieval_quality_gate(
        _report(),
        RetrievalQualityGateConfig(required_diagnostic_keys=("diagnostics.recall.figure_sidecar_candidate_count",)),
    )

    assert result.passed is False
    assert any("缺少诊断字段" in failure for failure in result.failures)
