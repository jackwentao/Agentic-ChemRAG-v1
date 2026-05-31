from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.evaluation.cases import GoldenCase
from backend.app.evaluation.metrics import (
    evaluate_case_metrics,
    ranking_metrics,
    selected_figure_hit,
    source_urls_valid,
    summarize_latencies,
)


def test_ranking_metrics_calculates_recall_precision_mrr_and_ndcg():
    metrics = ranking_metrics([False, True, True], expected_count=2, top_k=3)

    assert metrics["recall@k"] == 1.0
    assert metrics["precision@k"] == 2 / 3
    assert metrics["mrr@k"] == 0.5
    assert 0 < metrics["ndcg@k"] < 1


def test_case_metrics_selects_page_target_metrics():
    case = GoldenCase(
        case_id="c1",
        question="What is reported?",
        scope="current_document",
        document_ids=["doc1"],
        expected_route="chunk_search",
        target_type="page",
        expected_documents=["doc1"],
        expected_pages=[3],
        expected_figures=[],
        expected_answer_terms=["ZnO"],
        answer_type="fact_qa",
        notes=None,
    )
    evidence = [
        Evidence("c1", "doc1", "ZnO improves emission.", 2, 2, "Results", 0.9),
        Evidence("c2", "doc1", "ZnO emission peak is described.", 3, 3, "Results", 0.8),
    ]

    metrics = evaluate_case_metrics(
        case,
        actual_route="chunk_search",
        evidence=evidence,
        answer="ZnO emission is described.",
        citations=[{"page_start": 3, "page_end": 3, "source_url": "/api/documents/doc1/pdf#page=3"}],
        selected_images=[],
        top_k=2,
    )

    assert metrics["route_hit"] is True
    assert metrics["page_recall@k"] == 1.0
    assert metrics["page_mrr@k"] == 0.5
    assert "document_recall@k" not in metrics
    assert metrics["citation_page_hit"] is True
    assert metrics["answer_term_coverage"] == 1.0


def test_case_metrics_handles_mixed_target_type():
    case = GoldenCase(
        case_id="c1",
        question="Which figure shows PL?",
        scope="all_documents",
        document_ids=[],
        expected_route="figure_index",
        target_type="mixed",
        expected_documents=["doc1"],
        expected_pages=[4],
        expected_figures=[{"page": 4, "caption_terms": ["PL", "spectrum"]}],
        expected_answer_terms=["PL"],
        answer_type="figure_query",
        notes=None,
    )
    evidence = [
        Evidence(
            "figure:img1",
            "doc1",
            "Fig. 2 PL spectrum.",
            4,
            4,
            "Figure Index",
            0.9,
            images=[EvidenceImage("img1", page=4, caption="Fig. 2 PL spectrum.")],
        )
    ]

    metrics = evaluate_case_metrics(
        case,
        actual_route="figure_index",
        evidence=evidence,
        answer="PL spectrum is shown.",
        citations=[{"page_start": 4, "page_end": 4, "source_url": "/api/documents/doc1/pdf#page=4"}],
        selected_images=[
            {
                "image_id": "img1",
                "page": 4,
                "caption": "Fig. 2 PL spectrum.",
                "source_url": "/api/documents/doc1/pdf#page=4",
            }
        ],
        top_k=5,
    )

    assert metrics["document_recall@k"] == 1.0
    assert metrics["page_recall@k"] == 1.0
    assert metrics["figure_recall@k"] == 1.0
    assert metrics["target_hit"] is True
    assert metrics["figure_hit_mode"] == "page_caption"


def test_figure_hit_modes_are_exact_page_caption_and_none():
    assert selected_figure_hit([{"image_id": "img1", "page": 7}], ["img1"]) == (True, "exact_id")
    assert selected_figure_hit(
        [{"image_id": "other", "page": 7, "caption": "Fig. 1 PL spectrum"}],
        [{"page": 7, "caption_terms": ["PL", "spectrum"]}],
    ) == (True, "page_caption")
    assert selected_figure_hit(
        [{"image_id": "other", "page": 8, "caption": "Fig. 1 PL spectrum"}],
        [{"page": 7, "caption_terms": ["PL", "spectrum"]}],
    ) == (False, "none")
    assert selected_figure_hit(
        [{"image_id": "other", "page": 2, "caption": "Fig. 1. The packing principles between NLO properties and iCOFs structures."}],
        [{"page": 2, "caption_terms": ["nonlinear"]}],
    ) == (True, "page_caption")


def test_source_url_validation_rejects_missing_or_non_pdf_page_urls():
    assert source_urls_valid(
        [{"source_url": "/api/documents/doc1/pdf#page=2"}],
        [{"source_url": "/api/documents/doc1/pdf#page=3"}],
    )
    assert not source_urls_valid([{"source_url": "/api/images/img1"}], [])
    assert not source_urls_valid([], [{"image_id": "img1"}])


def test_none_target_focuses_on_refusal():
    case = GoldenCase(
        case_id="none",
        question="Unrelated?",
        scope="current_document",
        document_ids=["doc1"],
        expected_route="chunk_search",
        target_type="none",
        expected_documents=[],
        expected_pages=[],
        expected_figures=[],
        expected_answer_terms=[],
        answer_type="refusal",
        notes=None,
    )

    metrics = evaluate_case_metrics(case, "chunk_search", [], "I could not find enough evidence.", [], [], top_k=5)

    assert metrics["target_hit"] is True
    assert metrics["refusal_correct"] is True


def test_conservative_insufficient_answer_is_not_counted_as_unsupported_claim():
    case = GoldenCase(
        case_id="insufficient",
        question="Summarize unavailable details.",
        scope="current_document",
        document_ids=["doc1"],
        expected_route="chunk_search",
        target_type="page",
        expected_pages=[2],
        answer_type="fact_qa",
    )

    metrics = evaluate_case_metrics(
        case,
        "chunk_search",
        [],
        "检索到的证据不足以可靠回答该问题。",
        [],
        [],
        top_k=5,
        limitations=["资料缺口：未找到直接证据"],
        retrieval_round=2,
        stop_reason="no_new_evidence",
    )

    assert metrics["unsupported_claim_risk"] is False


def test_adaptive_retrieval_metrics_report_retry_gain_disclosure_and_cost():
    case = GoldenCase(
        case_id="retry",
        question="Is it stable at high temperature?",
        scope="current_document",
        document_ids=["doc1"],
        expected_route="chunk_search",
        target_type="page",
        expected_pages=[2],
        expected_answer_terms=["stable"],
        answer_type="fact_qa",
        task_category="multi_condition",
        risk_level="medium",
        expected_aspects=["high temperature", "stable"],
        required_disclosures=["测试条件"],
        expected_retrieval_behavior="retry_if_missing",
        max_acceptable_rounds=3,
    )

    metrics = evaluate_case_metrics(
        case,
        "chunk_search",
        [Evidence("c1", "doc1", "Stable at high temperature.", 2, 2, "Results", 0.9)],
        "Stable at high temperature; 测试条件尚有限。",
        [{"page_start": 2, "page_end": 2, "source_url": "/api/documents/doc1/pdf#page=2"}],
        [],
        top_k=5,
        limitations=["资料缺口：测试条件"],
        retrieval_round=2,
        stop_reason="sufficient_evidence",
        assessment_rounds=[{"round_index": 1}, {"round_index": 2}],
        llm_call_counts={"evidence_assessment": 2, "answer_generation": 1, "image_selection": 1},
    )

    assert metrics["retry_triggered"] is True
    assert metrics["observed_retry_success"] is True
    assert metrics["wasted_retry"] is False
    assert metrics["retrieval_behavior_hit"] is True
    assert metrics["limitation_disclosure_hit"] is True
    assert metrics["aspect_coverage_rate"] == 1.0
    assert metrics["assessment_round_count"] == 2
    assert metrics["image_selection_llm_calls"] == 1
    assert metrics["total_tracked_llm_calls"] == 4


def test_latency_summary_reports_p50_p95_and_max():
    summary = summarize_latencies(
        [
            {"latency_total_ms": 10, "latency_retrieval_ms": 3, "latency_answer_ms": 7},
            {"latency_total_ms": 20, "latency_retrieval_ms": 5, "latency_assessment_ms": 2, "latency_answer_ms": 15},
            {"latency_total_ms": 30, "latency_retrieval_ms": 7, "latency_assessment_ms": 4, "latency_answer_ms": 23},
        ]
    )

    assert summary["latency_total_ms"]["p50"] == 20
    assert summary["latency_total_ms"]["p95"] == 30
    assert summary["latency_total_ms"]["max"] == 30
    assert "latency_rerank_ms" in summary
    assert summary["latency_assessment_ms"]["max"] == 4
