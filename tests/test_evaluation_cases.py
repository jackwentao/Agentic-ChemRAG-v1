import pytest

from backend.app.evaluation.cases import parse_golden_case


def valid_payload():
    return {
        "case_id": "c1",
        "question": "What is shown?",
        "scope": "current_document",
        "document_ids": ["doc1"],
        "expected_route": "figure_index",
        "target_type": "figure",
        "expected_documents": ["doc1"],
        "expected_pages": [2],
        "expected_figures": [{"page": 2, "caption_terms": ["PL"]}],
        "expected_answer_terms": ["PL"],
        "answer_type": "figure_query",
        "task_category": "figure_or_summary",
        "risk_level": "low",
        "expected_aspects": ["PL"],
        "required_disclosures": [],
        "expected_retrieval_behavior": "retry_if_missing",
        "max_acceptable_rounds": 2,
        "notes": None,
    }


def test_parse_golden_case_accepts_required_schema():
    case = parse_golden_case(valid_payload())

    assert case.case_id == "c1"
    assert case.expected_route == "figure_index"
    assert case.target_type == "figure"
    assert case.expected_figures == [{"page": 2, "caption_terms": ["PL"]}]
    assert case.task_category == "figure_or_summary"
    assert case.expected_retrieval_behavior == "retry_if_missing"
    assert case.max_acceptable_rounds == 2
    assert case.reference_answer is None
    assert case.reference_contexts == []
    assert case.ragas_tags == []


def test_parse_golden_case_accepts_optional_ragas_reference_fields():
    payload = valid_payload()
    payload["reference_answer"] = "The answer should mention PL."
    payload["reference_contexts"] = ["Fig. 1 PL spectrum."]
    payload["ragas_tags"] = ["figure", "chemistry"]

    case = parse_golden_case(payload)

    assert case.reference_answer == "The answer should mention PL."
    assert case.reference_contexts == ["Fig. 1 PL spectrum."]
    assert case.ragas_tags == ["figure", "chemistry"]


def test_parse_golden_case_requires_expected_route_and_target_type_enums():
    payload = valid_payload()
    payload["expected_route"] = "wrong"

    with pytest.raises(ValueError, match="expected_route"):
        parse_golden_case(payload)

    payload = valid_payload()
    payload["target_type"] = "wrong"

    with pytest.raises(ValueError, match="target_type"):
        parse_golden_case(payload)


def test_parse_golden_case_rejects_invalid_pages_and_figure_shapes():
    payload = valid_payload()
    payload["expected_pages"] = [0]

    with pytest.raises(ValueError, match="expected_pages"):
        parse_golden_case(payload)

    payload = valid_payload()
    payload["expected_figures"] = [{"page": 2, "caption_terms": "PL"}]

    with pytest.raises(ValueError, match="caption_terms"):
        parse_golden_case(payload)


def test_parse_golden_case_reports_missing_fields():
    payload = valid_payload()
    del payload["expected_route"]

    with pytest.raises(ValueError, match="missing required fields"):
        parse_golden_case(payload)


def test_parse_golden_case_supplies_backward_compatible_evaluation_defaults():
    payload = valid_payload()
    for field in [
        "task_category",
        "risk_level",
        "expected_aspects",
        "required_disclosures",
        "expected_retrieval_behavior",
        "max_acceptable_rounds",
    ]:
        del payload[field]

    case = parse_golden_case(payload)

    assert case.task_category == "figure_or_summary"
    assert case.risk_level == "low"
    assert case.expected_retrieval_behavior == "single_round_expected"
    assert case.max_acceptable_rounds == 1


def test_parse_golden_case_rejects_invalid_cost_baseline_fields():
    payload = valid_payload()
    payload["risk_level"] = "critical"

    with pytest.raises(ValueError, match="risk_level"):
        parse_golden_case(payload)

    payload = valid_payload()
    payload["max_acceptable_rounds"] = 0

    with pytest.raises(ValueError, match="max_acceptable_rounds"):
        parse_golden_case(payload)
