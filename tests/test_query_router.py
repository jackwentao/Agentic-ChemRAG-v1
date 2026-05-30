import json

from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.routing.query_router import CACHE_FILENAME, QueryRouter, RouteContext


def test_router_routes_english_cross_document_questions_to_profile():
    router = QueryRouter()

    first = router.route("Which papers discuss ZnO photoluminescence?", RouteContext(scope="all_documents"))
    second = router.route("Which papers are related to phthalocyanine?", RouteContext(scope="all_documents"))

    assert first.route == "document_profile"
    assert "cross_document_signal" in first.rationale
    assert second.route == "document_profile"


def test_router_routes_chinese_holistic_and_cross_document_questions_to_profile():
    router = QueryRouter()

    summary = router.route("这篇论文主要研究什么？", RouteContext(scope="current_document", document_ids=["doc1"]))
    cross = router.route("哪些论文是讲卟啉相关的？", RouteContext(scope="all_documents"))

    assert summary.route == "document_profile"
    assert cross.route == "document_profile"


def test_router_routes_explicit_figure_questions_to_figure_index():
    router = QueryRouter()

    assert router.route("What figures or plots does this paper contain?").route == "figure_index"
    assert router.route("Which PL spectrum figures are shown?").route == "figure_index"
    assert router.route("该文献有什么性能图？").route == "figure_index"
    assert router.route("这篇论文有哪些PL谱图？").route == "figure_index"
    assert router.route("给我看一下这篇文献里的SEM图").route == "figure_index"


def test_router_protects_fact_questions_from_weak_visual_terms():
    router = QueryRouter()

    spectra = router.route("What nonlinear spectra topic is covered?")
    photoluminescence = router.route("What photoluminescence property is discussed?")
    chinese_peak = router.route("PL谱峰位是多少？")
    chinese_figure_result = router.route("总结图3说明了什么结果")

    assert spectra.route == "chunk_search"
    assert photoluminescence.route == "chunk_search"
    assert spectra.scores["chunk_search"] > spectra.scores["figure_index"]
    assert chinese_peak.route == "chunk_search"
    assert chinese_figure_result.route == "chunk_search"
    assert "figure_local_fact_question" in chinese_figure_result.rationale


def test_router_routes_chinese_document_profile_but_not_local_facts():
    router = QueryRouter()

    summary = router.route("帮我概括一下这篇文献", RouteContext(scope="current_document", document_ids=["doc1"]))
    cross = router.route("找出和ZnO荧光相关的论文", RouteContext(scope="all_documents"))
    solvent = router.route("这篇论文中溶剂相关的结果是什么？", RouteContext(scope="current_document", document_ids=["doc1"]))

    assert summary.route == "document_profile"
    assert cross.route == "document_profile"
    assert solvent.route == "chunk_search"


def test_router_flags_unrelated_low_confidence_questions():
    decision = QueryRouter().route(
        "What does the paper say about CRISPR gene editing efficiency?",
        RouteContext(scope="current_document", document_ids=["doc1"]),
    )

    assert decision.route == "chunk_search"
    assert decision.needs_llm_router is True
    assert "likely_out_of_domain_low_confidence" in decision.rationale


def test_router_flags_multi_step_tasks():
    decision = QueryRouter().route("Find papers about ZnO and summarize their figures", RouteContext(scope="all_documents"))

    assert decision.is_multi_step is True
    assert decision.needs_llm_router is True


class StubEmbeddingProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        text_lower = text.lower()
        if "semantic profile query" in text_lower:
            return [0.0, 0.0, 1.0]
        if "which papers discuss" in text_lower or "what is this paper about" in text_lower or "summarize" in text_lower:
            return [0.0, 0.0, 1.0]
        if "figures" in text_lower or "plots" in text_lower or "performance figures" in text_lower:
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]


def test_router_semantic_score_can_lift_query_without_keywords():
    router = QueryRouter(StubEmbeddingProvider(), semantic_weight=0.70, lexical_weight=0.30)

    decision = router.route("semantic profile query", RouteContext(scope="all_documents"))

    assert decision.route == "document_profile"
    assert decision.semantic_scores["document_profile"] > decision.semantic_scores["chunk_search"]


def test_router_lexical_high_confidence_is_not_overridden_by_semantic_signal():
    router = QueryRouter(StubEmbeddingProvider(), semantic_weight=0.30, lexical_weight=0.70)

    decision = router.route("What figures or plots does this paper contain?", RouteContext(scope="current_document"))

    assert decision.route == "figure_index"
    assert decision.scores["figure_index"] > decision.scores["document_profile"]


def test_router_works_without_embedding_provider():
    decision = QueryRouter().route("What solvent is used for synthesis?")

    assert decision.route == "chunk_search"
    assert decision.semantic_scores == {"chunk_search": 0.0, "figure_index": 0.0, "document_profile": 0.0}


class CountingEmbeddingProvider(StubEmbeddingProvider):
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[float(len(text) % 7 + 1), 1.0, 0.5] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def test_router_warmup_cache_miss_writes_cache(tmp_path):
    provider = CountingEmbeddingProvider()
    router = QueryRouter(provider, cache_dir=tmp_path, provider_fingerprint="provider-a")

    status = router.warmup()

    assert status.status == "miss_written"
    assert provider.document_calls == 3
    assert provider.query_calls == 1
    assert (tmp_path / CACHE_FILENAME).exists()


def test_router_warmup_cache_hit_skips_prototype_encoding(tmp_path):
    first_provider = CountingEmbeddingProvider()
    first = QueryRouter(first_provider, cache_dir=tmp_path, provider_fingerprint="provider-a")
    first.warmup()

    second_provider = CountingEmbeddingProvider()
    second = QueryRouter(second_provider, cache_dir=tmp_path, provider_fingerprint="provider-a")
    status = second.warmup()

    assert status.status == "disk_hit"
    assert second_provider.document_calls == 0
    assert second_provider.query_calls == 1


def test_router_warmup_invalidates_changed_provider_fingerprint(tmp_path):
    first = QueryRouter(CountingEmbeddingProvider(), cache_dir=tmp_path, provider_fingerprint="provider-a")
    first.warmup()
    provider = CountingEmbeddingProvider()
    second = QueryRouter(provider, cache_dir=tmp_path, provider_fingerprint="provider-b")

    status = second.warmup()

    assert status.status == "invalid_recomputed_written"
    assert "fingerprint" in (status.error or "")
    assert provider.document_calls == 3


def test_router_warmup_recovers_from_corrupt_cache(tmp_path):
    (tmp_path / CACHE_FILENAME).write_text("not json", encoding="utf-8")
    provider = CountingEmbeddingProvider()
    router = QueryRouter(provider, cache_dir=tmp_path, provider_fingerprint="provider-a")

    status = router.warmup()

    assert status.status == "invalid_recomputed_written"
    assert status.error
    assert provider.document_calls == 3


def test_router_warmup_recovers_from_dimension_mismatch(tmp_path):
    provider = CountingEmbeddingProvider()
    router = QueryRouter(provider, cache_dir=tmp_path, provider_fingerprint="provider-a")
    router.warmup()
    cache_path = tmp_path / CACHE_FILENAME
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["dimensions"] = 99
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    second_provider = CountingEmbeddingProvider()
    second = QueryRouter(second_provider, cache_dir=tmp_path, provider_fingerprint="provider-a")
    status = second.warmup()

    assert status.status == "invalid_recomputed_written"
    assert "dimensions" in (status.error or "")
    assert second_provider.document_calls == 3


def test_router_warmup_second_call_is_memory_hit(tmp_path):
    provider = CountingEmbeddingProvider()
    router = QueryRouter(provider, cache_dir=tmp_path, provider_fingerprint="provider-a")

    router.warmup()
    status = router.warmup()

    assert status.status == "memory_hit"
    assert provider.document_calls == 3
    assert provider.query_calls == 2


def test_router_disabled_cache_uses_memory_only(tmp_path):
    provider = CountingEmbeddingProvider()
    router = QueryRouter(provider, cache_dir=tmp_path, cache_enabled=False)

    status = router.warmup()

    assert status.status == "disabled"
    assert status.cache_enabled is False
    assert not (tmp_path / CACHE_FILENAME).exists()


def test_router_warmup_can_skip_embedding_provider_query(tmp_path):
    provider = CountingEmbeddingProvider()
    router = QueryRouter(provider, cache_dir=tmp_path, provider_fingerprint="provider-a")

    router.warmup(warm_embedding_provider=False)

    assert provider.query_calls == 0
    assert provider.document_calls == 3
