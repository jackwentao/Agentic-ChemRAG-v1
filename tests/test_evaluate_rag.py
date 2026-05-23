from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.evaluation.cases import GoldenCase
from scripts.evaluate_rag import evaluate_cases, render_markdown_report


class FakeGraph:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, question, document_ids=None):
        self.calls.append((question, document_ids))
        evidence = [
            Evidence(
                chunk_id="figure:img1",
                document_id="doc1",
                text="Fig. 2 PL spectrum of ZnO.",
                page_start=4,
                page_end=4,
                section_title="Figure Index",
                score=0.9,
                images=[EvidenceImage("img1", page=4, caption="Fig. 2 PL spectrum of ZnO.")],
            )
        ]
        return {
            "route": "figure_index",
            "verified_evidence": evidence,
            "answer": "Fig. 2 shows the PL spectrum of ZnO.",
            "selected_images": {"figure:img1": [evidence[0].images[0]]},
            "retrieval_latency_ms": 12.0,
            "rerank_latency_ms": 2.5,
            "rerank_enabled": True,
            "answer_latency_ms": 3.0,
        }


class FailingJudge:
    def score(self, **kwargs):
        raise RuntimeError("judge unavailable")


def make_case():
    return GoldenCase(
        case_id="c1",
        question="What figures are shown?",
        scope="current_document",
        document_ids=["doc1"],
        expected_route="figure_index",
        target_type="figure",
        expected_documents=["doc1"],
        expected_pages=[4],
        expected_figures=[{"image_id": "img1", "page": 4, "caption_terms": ["PL", "spectrum"]}],
        expected_answer_terms=["PL", "ZnO"],
        answer_type="figure_query",
        notes=None,
    )


def test_evaluate_cases_records_route_metrics_outputs_and_latencies():
    graph = FakeGraph()

    report = evaluate_cases(
        [make_case()],
        graph=graph,
        top_k=5,
        warmup_info={
            "warmup_latency_ms": 7.0,
            "router_cache_status": "disk_hit",
            "router_cache_enabled": True,
            "router_cache_path": "cache.json",
            "router_cache_error": None,
            "reranker_warmup_latency_ms": 4.0,
        },
    )

    result = report["results"][0]
    assert graph.calls == [("What figures are shown?", ["doc1"])]
    assert report["warmup"]["warmup_latency_ms"] == 7.0
    assert report["warmup"]["router_cache_status"] == "disk_hit"
    assert result["actual_route"] == "figure_index"
    assert result["metrics"]["route_hit"] is True
    assert result["metrics"]["figure_hit_mode"] == "exact_id"
    assert result["citations"][0]["source_url"] == "/api/documents/doc1/pdf#page=4"
    assert result["selected_images"][0]["source_url"] == "/api/documents/doc1/pdf#page=4"
    assert result["latency_retrieval_ms"] == 12.0
    assert result["latency_rerank_ms"] == 2.5
    assert result["rerank_enabled"] is True
    assert "by_expected_route" in report["summary"]


def test_evaluate_cases_scope_override_all_documents_passes_no_document_filter():
    graph = FakeGraph()

    evaluate_cases([make_case()], graph=graph, top_k=5, scope_override="all_documents")

    assert graph.calls[0][1] is None


def test_evaluate_cases_records_judge_errors_without_failing():
    report = evaluate_cases([make_case()], graph=FakeGraph(), top_k=5, judge=FailingJudge())

    assert report["results"][0]["judge_error"] == "judge unavailable"


def test_render_markdown_report_includes_expected_sections():
    report = evaluate_cases([make_case()], graph=FakeGraph(), top_k=5)

    markdown = render_markdown_report(report)

    assert "# RAG Evaluation Report" in markdown
    assert "## Overall Metrics" in markdown
    assert "## By Expected Route" in markdown
    assert "## Slowest Cases" in markdown
    assert "Router cache status" in markdown
    assert "Reranker warmup latency" in markdown
