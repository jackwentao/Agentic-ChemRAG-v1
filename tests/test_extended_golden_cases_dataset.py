from pathlib import Path

from backend.app.evaluation.cases import load_golden_cases


def test_extended_golden_cases_keeps_original_cases_and_adds_more():
    original_path = Path("eval/golden_cases.jsonl")
    extended_path = Path("eval/golden_cases_extended.jsonl")

    original_cases = load_golden_cases(original_path)
    extended_cases = load_golden_cases(extended_path)

    original_ids = {case.case_id for case in original_cases}
    extended_ids = {case.case_id for case in extended_cases}

    assert len(original_cases) == 20
    assert len(extended_cases) > len(original_cases)
    assert original_ids.issubset(extended_ids)
    assert len(extended_ids) == len(extended_cases)


def test_extended_golden_cases_cover_architecture_risks_and_followups():
    cases = load_golden_cases(Path("eval/golden_cases_extended.jsonl"))

    task_categories = {case.task_category for case in cases}
    risk_levels = {case.risk_level for case in cases}
    behaviors = {case.expected_retrieval_behavior for case in cases}

    assert {"simple_fact", "figure_or_summary", "multi_condition", "comparison", "context_followup", "high_risk", "insufficient_evidence"} <= task_categories
    assert {"low", "medium", "high"} <= risk_levels
    assert {"single_round_expected", "retry_if_missing", "insufficient_should_stop"} <= behaviors


def test_extended_golden_cases_include_realistic_cross_doc_and_high_risk_examples():
    cases = {case.case_id: case for case in load_golden_cases(Path("eval/golden_cases_extended.jsonl"))}

    high_risk = cases["refusal-high-risk-001"]
    followup = cases["context-zno-followup-001"]
    cross_doc = cases["mixed-cross-zno-pl-001"]

    assert high_risk.answer_type == "refusal"
    assert high_risk.risk_level == "high"
    assert "实验" in high_risk.required_disclosures

    assert followup.task_category == "context_followup"
    assert followup.scope == "current_document"
    assert followup.max_acceptable_rounds == 2

    assert cross_doc.scope == "all_documents"
    assert cross_doc.target_type == "mixed"
    assert cross_doc.expected_route == "figure_index"
