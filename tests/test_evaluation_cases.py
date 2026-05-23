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
        "notes": None,
    }


def test_parse_golden_case_accepts_required_schema():
    case = parse_golden_case(valid_payload())

    assert case.case_id == "c1"
    assert case.expected_route == "figure_index"
    assert case.target_type == "figure"
    assert case.expected_figures == [{"page": 2, "caption_terms": ["PL"]}]


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
