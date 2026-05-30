from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.evaluation.cases import GoldenCase
from scripts.evaluate_rag import _warmup_evaluation_components, evaluate_cases, render_markdown_report
from backend.app.routing.query_router import RouterWarmupStatus


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
            "retrieval_round": 2,
            "stop_reason": "sufficient_evidence",
            "verified_evidence": evidence,
            "answer": "Fig. 2 shows the PL spectrum of ZnO.",
            "limitations": ["资料缺口：长期稳定性"],
            "selected_images": {"figure:img1": [evidence[0].images[0]]},
            "total_latency_ms": 18.5,
            "retrieval_latency_ms": 12.0,
            "rerank_latency_ms": 2.5,
            "rerank_enabled": True,
            "answer_latency_ms": 3.0,
            "diagnostics": {
                "rerank": {"latency_ms": 2.5, "candidate_count": 8},
                "assessment_rounds": [
                    {"round_index": 1, "latency_ms": 1.2, "sufficient": False},
                    {"round_index": 2, "latency_ms": 1.5, "sufficient": True},
                ],
                "llm_call_counts": {"evidence_assessment": 2, "answer_generation": 0},
            },
        }


class FailingJudge:
    def score(self, **kwargs):
        raise RuntimeError("judge unavailable")


class RecordingEmbedding:
    def __init__(self) -> None:
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [1.0]


class RecordingRouter:
    def __init__(self) -> None:
        self.arguments = []

    def warmup(self, warm_embedding_provider=True):
        self.arguments.append(warm_embedding_provider)
        return RouterWarmupStatus(cache_enabled=True, status="disk_hit", latency_ms=2.0)


class RecordingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def warmup(self):
        self.calls += 1


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
        task_category="figure_or_summary",
        risk_level="low",
        expected_aspects=["PL", "ZnO"],
        expected_retrieval_behavior="retry_if_missing",
        max_acceptable_rounds=2,
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
            "embedding_warmup_latency_ms": 3.0,
            "enabled": True,
        },
    )

    result = report["results"][0]
    assert graph.calls == [("What figures are shown?", ["doc1"])]
    assert report["warmup"]["warmup_latency_ms"] == 7.0
    assert report["warmup"]["router_cache_status"] == "disk_hit"
    assert report["warmup"]["embedding_warmup_latency_ms"] == 3.0
    assert result["actual_route"] == "figure_index"
    assert result["retrieval_round"] == 2
    assert result["stop_reason"] == "sufficient_evidence"
    assert result["limitations"] == ["资料缺口：长期稳定性"]
    assert result["metrics"]["route_hit"] is True
    assert result["metrics"]["figure_hit_mode"] == "exact_id"
    assert result["citations"][0]["source_url"] == "/api/documents/doc1/pdf#page=4"
    assert result["selected_images"][0]["source_url"] == "/api/documents/doc1/pdf#page=4"
    assert result["latency_retrieval_ms"] == 12.0
    assert result["latency_rerank_ms"] == 2.5
    assert result["rerank_enabled"] is True
    assert result["latency_assessment_ms"] == 2.7
    assert result["llm_call_counts"] == {"evidence_assessment": 2, "answer_generation": 0}
    assert result["metrics"]["observed_retry_success"] is True
    assert result["metrics"]["total_tracked_llm_calls"] == 2
    assert result["diagnostics"]["rerank"]["candidate_count"] == 8
    assert "by_expected_route" in report["summary"]
    assert "by_task_category" in report["summary"]
    assert report["summary"]["adaptive_retrieval_cost"]["average_retrieval_rounds"] == 2


def test_evaluate_cases_scope_override_all_documents_passes_no_document_filter():
    graph = FakeGraph()

    evaluate_cases([make_case()], graph=graph, top_k=5, scope_override="all_documents")

    assert graph.calls[0][1] is None


def test_evaluate_cases_records_judge_errors_without_failing():
    report = evaluate_cases([make_case()], graph=FakeGraph(), top_k=5, judge=FailingJudge())

    assert report["results"][0]["judge_error"] == "judge unavailable"


def test_render_markdown_report_includes_expected_sections():
    report = evaluate_cases(
        [make_case()],
        graph=FakeGraph(),
        top_k=5,
        evaluation_info={"answer_provider": "fake", "assessment_provider": "configured", "llm_judge": "off"},
    )

    markdown = render_markdown_report(report)

    assert "# RAG 离线评测报告" in markdown
    assert "证据评估提供器：`configured`" in markdown
    assert "## 总体指标" in markdown
    assert "## 按预期路由" in markdown
    assert "## 自适应检索成本" in markdown
    assert "## 停止原因" in markdown
    assert "## 按任务类别" in markdown
    assert "## 按风险等级" in markdown
    assert "## 最慢案例" in markdown
    assert "路由缓存状态" in markdown
    assert "重排器预热时延" in markdown
    assert "嵌入预热时延" in markdown


def test_evaluation_warmup_records_independent_embedding_router_and_reranker_steps():
    embedding = RecordingEmbedding()
    router = RecordingRouter()
    reranker = RecordingReranker()

    result = _warmup_evaluation_components(
        embedding_provider=embedding,
        query_router=router,
        reranker_provider=reranker,
        enabled=True,
        warm_reranker=True,
    )

    assert embedding.queries == ["evaluation embedding warmup probe"]
    assert router.arguments == [False]
    assert reranker.calls == 1
    assert result["enabled"] is True
    assert result["embedding_warmup_latency_ms"] >= 0
    assert result["warmup_latency_ms"] == 2.0
    assert result["reranker_warmup_latency_ms"] >= 0
