import pytest

from backend.app.domain.documents import Evidence
from backend.app.evaluation.cases import GoldenCase
from backend.app.evaluation.ragas_adapter import (
    RagasEvaluationConfig,
    _EmbeddingProviderWrapper,
    _metric_names_for_rows,
    build_ragas_row,
    merge_ragas_report,
)


def make_case(reference_answer=None):
    return GoldenCase(
        case_id="c1",
        question="What does the ZnO paper report?",
        scope="current_document",
        document_ids=["doc1"],
        expected_route="chunk_search",
        target_type="page",
        expected_pages=[2],
        expected_answer_terms=["ZnO", "PL"],
        answer_type="fact_qa",
        reference_answer=reference_answer,
        reference_contexts=["Reference context"],
        ragas_tags=["chemistry"],
    )


def test_build_ragas_row_maps_project_fields_to_new_and_legacy_columns():
    case = make_case(reference_answer="The paper reports ZnO PL behavior.")
    evidence = [Evidence("c1", "doc1", "ZnO photoluminescence is reported.", 2, 2, "Results", 0.9)]

    row = build_ragas_row(case, "It reports ZnO PL.", evidence)

    assert row["case_id"] == "c1"
    assert row["user_input"] == "What does the ZnO paper report?"
    assert row["response"] == "It reports ZnO PL."
    assert row["retrieved_contexts"] == ["ZnO photoluminescence is reported."]
    assert row["reference"] == "The paper reports ZnO PL behavior."
    assert row["reference_contexts"] == ["Reference context"]
    assert row["ragas_tags"] == ["chemistry"]
    assert row["question"] == row["user_input"]
    assert row["answer"] == row["response"]
    assert row["contexts"] == row["retrieved_contexts"]
    assert row["ground_truth"] == row["reference"]


def test_build_ragas_row_trims_contexts_without_using_hints_as_reference():
    case = make_case()
    evidence = [Evidence("c1", "doc1", "A" * 20, 1, 1, "Intro", 0.5)]

    row = build_ragas_row(case, "Answer", evidence, max_context_chars=5)

    assert row["retrieved_contexts"] == ["AAAAA"]
    assert row["reference"] is None
    assert "ZnO" in row["reference_hint"]


def test_reference_metrics_are_skipped_without_explicit_reference_answer():
    rows = [build_ragas_row(make_case(), "Answer", [])]

    metrics, skipped = _metric_names_for_rows("all", rows)

    assert "faithfulness" in metrics
    assert "answer_correctness" not in metrics
    assert "context_recall" not in metrics
    assert skipped == ["context_recall", "answer_correctness"]


def test_reference_metrics_are_enabled_when_reference_answer_exists():
    rows = [build_ragas_row(make_case(reference_answer="Reference"), "Answer", [])]

    metrics, skipped = _metric_names_for_rows("all", rows)

    assert "context_recall" in metrics
    assert "answer_correctness" in metrics
    assert skipped == []


def test_merge_ragas_report_adds_summary_and_per_case_scores():
    report = {"results": [{"case": {"case_id": "c1"}}, {"case": {"case_id": "c2"}}]}
    ragas_report = {
        "enabled": True,
        "summary": {"faithfulness": 0.7},
        "metrics": ["faithfulness"],
        "skipped_metrics": [],
        "errors": [],
        "results": [{"case_id": "c1", "metrics": {"faithfulness": 0.7}}],
    }

    merged = merge_ragas_report(report, ragas_report)

    assert merged["ragas"]["summary"] == {"faithfulness": 0.7}
    assert merged["results"][0]["ragas"]["metrics"] == {"faithfulness": 0.7}
    assert merged["results"][1]["ragas"] == {"status": "not_evaluated"}


def test_ragas_config_defaults_are_safe_for_regular_evaluation():
    config = RagasEvaluationConfig()

    assert config.metric_set == "basic"
    assert config.sample_limit is None
    assert config.max_context_chars == 4000
    assert config.fail_on_error is False


@pytest.mark.asyncio
async def test_embedding_wrapper_supports_async_langchain_methods():
    class Provider:
        def embed_documents(self, texts):
            return [[float(len(text))] for text in texts]

        def embed_query(self, text):
            return [float(len(text))]

    wrapper = _EmbeddingProviderWrapper(Provider())

    assert await wrapper.aembed_documents(["ZnO", "PL"]) == [[3.0], [2.0]]
    assert await wrapper.aembed_query("ZnO") == [3.0]
