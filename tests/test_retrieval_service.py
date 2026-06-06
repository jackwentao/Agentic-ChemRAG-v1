from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.rerankers import FakeRerankerProvider, RerankResult
from backend.app.services.rag import RetrievalService


class FailingReranker:
    def rerank(self, query, candidates, top_k):
        raise RuntimeError("model unavailable")


class LexicalHostileReranker:
    def rerank(self, query, candidates, top_k):
        rows = []
        for index, candidate in enumerate(candidates):
            score = 10.0 if "unrelated" in candidate.text.lower() else 0.01
            rows.append(RerankResult(candidate_id=candidate.candidate_id, score=score - index * 0.001))
        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[:top_k]


class VectorOnlyRepository:
    def __init__(self) -> None:
        self.received_vector = None
        self.document_ids = None
        self.top_k = None

    def search(self, query_embedding, top_k=5, document_ids=None):
        self.received_vector = query_embedding
        self.document_ids = document_ids
        self.top_k = top_k
        return [
            Evidence(
                chunk_id=f"c{index}",
                document_id="d1",
                text=text,
                page_start=1,
                page_end=1,
                section_title="Abstract",
                score=0.9 - index * 0.01,
            )
            for index, text in enumerate(
                [
                    "general background",
                    "ZnO photoluminescence property and emission result",
                    "methods",
                    "unrelated",
                ]
            )
        ]


class StrategyRepository:
    def __init__(self, chunk_count: int) -> None:
        self.chunk_count = chunk_count
        self.calls = []

    def count_searchable_chunks(self, document_ids=None, owner_id=None):
        self.calls.append(("count", document_ids, owner_id))
        return self.chunk_count

    def search(self, query_embedding, top_k=5, document_ids=None, owner_id=None, search_strategy="hnsw", exact_timeout_ms=None):
        self.calls.append(("search", document_ids, owner_id, search_strategy, exact_timeout_ms))
        return [
            Evidence(
                chunk_id="c1",
                document_id="d1",
                text="ZnO result",
                page_start=1,
                page_end=1,
                section_title="Results",
                score=0.9,
            )
        ]


class HybridRepository:
    def __init__(self) -> None:
        self.lexical_calls = []

    def search(self, query_embedding, top_k=5, document_ids=None, owner_id=None, search_strategy="hnsw", exact_timeout_ms=None):
        return [
            Evidence(
                chunk_id="vector-general",
                document_id="doc1",
                text="General photovoltaic background.",
                page_start=2,
                page_end=2,
                section_title="Introduction",
                score=0.72,
            )
        ]

    def search_lexical_chunks(self, query, top_k=5, document_ids=None, owner_id=None):
        self.lexical_calls.append((query, top_k, document_ids, owner_id))
        return [
            Evidence(
                chunk_id="author-hit",
                document_id="doc1",
                text="M.A. Eleruja b, Department of Physics, Obafemi Awolowo University.",
                page_start=1,
                page_end=1,
                section_title="Title Page",
                score=1.1,
                evidence_type="lexical_chunk",
                metadata={"retrieval_source": "lexical_chunk"},
            )
        ]


def test_retrieval_service_embeds_before_vector_repository_search():
    repository = VectorOnlyRepository()
    provider = FakeEmbeddingProvider(dimensions=8)

    results = RetrievalService(repository, provider, reranker_enabled=False).retrieve("ZnO photoluminescence")

    assert results
    assert isinstance(repository.received_vector, list)
    assert len(repository.received_vector) == 8


def test_retrieval_service_passes_document_filter_to_vector_repository():
    repository = VectorOnlyRepository()
    provider = FakeEmbeddingProvider(dimensions=8)

    RetrievalService(repository, provider, reranker_enabled=False).retrieve("ZnO", document_ids=["doc1"])

    assert repository.document_ids == ["doc1"]


def test_retrieve_uses_exact_then_hnsw_for_small_owner_scoped_candidate_sets():
    repository = StrategyRepository(chunk_count=3)
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
        exact_chunk_limit=10,
        exact_timeout_ms=250,
    )

    results = service.retrieve("ZnO", document_ids=["doc1"], owner_id="tenant-a")

    assert results[0].chunk_id == "c1"
    assert repository.calls[0] == ("count", ["doc1"], "tenant-a")
    assert repository.calls[1] == ("search", ["doc1"], "tenant-a", "exact_then_hnsw", 250)
    assert service.last_recall_diagnostics["search_strategy"] == "exact_then_hnsw"
    assert service.last_recall_diagnostics["candidate_chunk_count"] == 3


def test_retrieve_merges_lexical_hits_for_exact_fact_questions_without_reranker():
    repository = HybridRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
        lexical_chunk_candidates=12,
    )

    results = service.retrieve("what M.A. Eleruja do in the paper?", top_k=2, document_ids=["doc1"], owner_id="tenant-a")

    assert results[0].chunk_id == "author-hit"
    assert repository.lexical_calls == [("what M.A. Eleruja do in the paper?", 12, ["doc1"], "tenant-a")]
    assert service.last_recall_diagnostics["mode"] == "hybrid_chunk"
    assert service.last_recall_diagnostics["vector_candidate_count"] == 1
    assert service.last_recall_diagnostics["lexical_candidate_count"] == 1
    assert results[0].metadata["retrieval_sources"] == ["lexical"]


def test_retrieve_combines_duplicate_vector_and_lexical_candidates_before_rerank():
    class DuplicateRepository(HybridRepository):
        def search_lexical_chunks(self, query, top_k=5, document_ids=None, owner_id=None):
            return [
                Evidence(
                    chunk_id="vector-general",
                    document_id="doc1",
                    text="General photovoltaic background with C. Ton-That reference.",
                    page_start=2,
                    page_end=2,
                    section_title="References",
                    score=1.2,
                    evidence_type="lexical_chunk",
                    metadata={"retrieval_source": "lexical_chunk"},
                )
            ]

    service = RetrievalService(
        DuplicateRepository(),
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
    )

    results = service.retrieve("C. Ton-That reference number", top_k=5)

    assert len(results) == 1
    assert results[0].metadata["retrieval_sources"] == ["vector", "lexical"]
    assert results[0].metadata["vector_score"] == 0.72
    assert results[0].metadata["lexical_score"] == 1.2


def test_retrieve_adds_figure_sidecar_for_visual_detail_questions():
    class VisualHybridRepository(HybridRepository):
        def search_figures(self, query_embedding, top_k=8, document_ids=None, owner_id=None):
            return [
                Evidence(
                    chunk_id="figure:fig6",
                    document_id="doc1",
                    text="Fig. 6. Resonant spectra for TPP-ML/ZnO/Si(111), TPP/ZnO/Si(111), and TPP/ZnO/ITO.",
                    page_start=6,
                    page_end=6,
                    section_title="Figure Index",
                    score=0.95,
                    evidence_type="figure",
                    images=[EvidenceImage("fig6", page=6, caption="Fig. 6. Resonant spectra for TPP/ZnO systems.")],
                )
            ]

    service = RetrievalService(
        VisualHybridRepository(),
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
    )

    results = service.retrieve("What material system is studied in the resonant spectra shown?", top_k=3)

    assert any(item.chunk_id == "figure:fig6" for item in results)
    assert service.last_recall_diagnostics["figure_sidecar_candidate_count"] == 1


def test_retrieve_normalizes_known_scientific_ocr_noise_before_answering():
    class OcrNoiseRepository(HybridRepository):
        def search(self, query_embedding, top_k=5, document_ids=None, owner_id=None, search_strategy="hnsw", exact_timeout_ms=None):
            return []

        def search_figures(self, query_embedding, top_k=8, document_ids=None, owner_id=None):
            return [
                Evidence(
                    chunk_id="figure:fig4",
                    document_id="doc1",
                    text="Fig. 4. ResPES spectra for TPP-ML/ZnO/S(111).",
                    page_start=5,
                    page_end=5,
                    section_title="Figure Index",
                    score=0.95,
                    evidence_type="figure",
                    images=[
                        EvidenceImage(
                            "fig4",
                            page=5,
                            caption="Fig. 4. TPP-ML/ZnO/S(111) monolayer.",
                            nearby_text="TPP/ZnO/S(111) specimen.",
                        )
                    ],
                )
            ]

    service = RetrievalService(
        OcrNoiseRepository(),
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
    )

    results = service.retrieve("What material system is in the resonant spectra?", top_k=3)

    figure = next(item for item in results if item.chunk_id == "figure:fig4")
    assert "TPP-ML/ZnO/Si(111)" in figure.text
    assert "TPP-ML/ZnO/Si(111)" in figure.images[0].caption
    assert "TPP/ZnO/Si(111)" in figure.images[0].nearby_text
    assert figure.metadata["scientific_ocr_normalizations"] == ["zno_silicon_substrate_plane"]


def test_retrieve_normalizes_figure_sidecar_when_it_replaces_top_k_result():
    class SidecarReplacementRepository(HybridRepository):
        def search(self, query_embedding, top_k=5, document_ids=None, owner_id=None, search_strategy="hnsw", exact_timeout_ms=None):
            return [
                Evidence(
                    chunk_id="text-hit",
                    document_id="doc1",
                    text="Reference [21] discusses Ton-That work.",
                    page_start=6,
                    page_end=6,
                    section_title="References",
                    score=2.0,
                )
            ]

        def search_lexical_chunks(self, query, top_k=5, document_ids=None, owner_id=None):
            return []

        def search_figures(self, query_embedding, top_k=8, document_ids=None, owner_id=None):
            return [
                Evidence(
                    chunk_id="figure:fig4",
                    document_id="doc1",
                    text="Fig. 4. TPP-ML/ZnO/S(111) monolayer.",
                    page_start=5,
                    page_end=5,
                    section_title="Figure Index",
                    score=0.5,
                    evidence_type="figure",
                    images=[EvidenceImage("fig4", page=5, caption="Fig. 4. TPP-ML/ZnO/S(111) monolayer.")],
                )
            ]

    service = RetrievalService(
        SidecarReplacementRepository(),
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
    )

    results = service.retrieve("What reference and resonant spectra material system are shown?", top_k=1)

    assert results[0].chunk_id == "figure:fig4"
    assert "TPP-ML/ZnO/Si(111)" in results[0].text
    assert "TPP-ML/ZnO/Si(111)" in results[0].images[0].caption


def test_rerank_preserves_strong_lexical_fact_hits_in_top_k():
    class ManyCandidateHybridRepository(HybridRepository):
        def search(self, query_embedding, top_k=5, document_ids=None, owner_id=None, search_strategy="hnsw", exact_timeout_ms=None):
            return []

        def search_lexical_chunks(self, query, top_k=5, document_ids=None, owner_id=None):
            return [
                Evidence(
                    chunk_id="author-hit",
                    document_id="doc1",
                    text="M.A. Eleruja, Department of Physics, Obafemi Awolowo University.",
                    page_start=1,
                    page_end=1,
                    section_title="Title Page",
                    score=1.2,
                    evidence_type="lexical_chunk",
                    metadata={"retrieval_source": "lexical_chunk"},
                ),
                *[
                    Evidence(
                        chunk_id=f"noise-{index}",
                        document_id="doc2",
                        text=f"Unrelated photovoltaic background section {index}.",
                        page_start=2,
                        page_end=2,
                        section_title="Background",
                        score=1.1,
                        evidence_type="lexical_chunk",
                        metadata={"retrieval_source": "lexical_chunk"},
                    )
                    for index in range(4)
                ],
            ]

    service = RetrievalService(
        ManyCandidateHybridRepository(),
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=LexicalHostileReranker(),
        reranker_enabled=True,
        reranker_min_candidates=1,
        lexical_chunk_candidates=5,
    )

    results = service.retrieve("what M.A. Eleruja do in the paper?", top_k=2)

    assert any(item.chunk_id == "author-hit" for item in results)
    assert service.last_rerank_diagnostics["protected_lexical_count"] == 1


def test_retrieve_uses_hnsw_for_large_owner_scoped_candidate_sets():
    repository = StrategyRepository(chunk_count=50)
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_enabled=False,
        exact_chunk_limit=10,
        exact_timeout_ms=250,
    )

    service.retrieve("ZnO", document_ids=["doc1"], owner_id="tenant-a")

    assert repository.calls[1] == ("search", ["doc1"], "tenant-a", "hnsw", None)
    assert service.last_recall_diagnostics["search_strategy"] == "hnsw"


def test_retrieve_uses_candidate_pool_and_reranks_to_top_k():
    repository = VectorOnlyRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
        reranker_chunk_candidates=30,
        reranker_min_candidates=1,
    )

    results = service.retrieve("ZnO photoluminescence", top_k=1)

    assert repository.top_k == 30
    assert results[0].chunk_id == "c1"
    assert service.last_rerank_enabled is True
    assert service.last_rerank_latency_ms >= 0
    assert service.last_recall_diagnostics["candidate_count"] == 4
    assert service.last_recall_diagnostics["returned_count"] == 1
    assert service.last_rerank_diagnostics["candidate_count"] == 4
    assert service.last_rerank_diagnostics["inference_latency_ms"] >= 0
    assert service.last_rerank_diagnostics["skip_reason"] is None


def test_single_document_retrieve_uses_fast_path_without_rerank_candidate_expansion():
    repository = HybridRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
        reranker_chunk_candidates=30,
        reranker_min_candidates=1,
        lexical_chunk_candidates=30,
    )

    results = service.retrieve("what M.A. Eleruja do in the paper?", top_k=2, document_ids=["doc1"])

    assert [item.chunk_id for item in results] == ["author-hit", "vector-general"]
    assert repository.lexical_calls == [("what M.A. Eleruja do in the paper?", 2, ["doc1"], None)]
    assert service.last_rerank_enabled is False
    assert service.last_rerank_diagnostics["skip_reason"] == "single_document_fast_path"


class FigureRepository:
    def __init__(self) -> None:
        self.top_k = None

    def search_figures(self, query_embedding, top_k=8, document_ids=None):
        self.top_k = top_k
        return [
            Evidence(
                chunk_id="figure:title",
                document_id="doc",
                text="Article Ionic covalent organic frameworks enable laser-pulse-duration-dependent high third-order nonlinear optical responses",
                page_start=1,
                page_end=1,
                section_title="Figure Index",
                score=0.9,
                images=[
                    EvidenceImage(
                        image_id="title",
                        page=1,
                        nearby_text="Article Ionic covalent organic frameworks enable laser-pulse-duration-dependent high third-order nonlinear optical responses",
                    )
                ],
            ),
            Evidence(
                chunk_id="figure:fig1",
                document_id="doc",
                text="Fig. 1. The packing underlying principles between NLO properties and iCOFs structures.",
                page_start=2,
                page_end=2,
                section_title="Figure Index",
                score=0.4,
                images=[
                    EvidenceImage(
                        image_id="fig1",
                        page=2,
                        label="Fig. 1",
                        caption="Fig. 1. The packing underlying principles between NLO properties and iCOFs structures.",
                    )
                ],
            ),
        ]


def test_retrieve_figures_reranks_captioned_nlo_figure_above_title_page():
    repository = FigureRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
        reranker_figure_candidates=32,
        reranker_min_candidates=1,
    )

    results = service.retrieve_figures("What figures describe nonlinear optical responses?", top_k=1)

    assert repository.top_k == 32
    assert results[0].images[0].image_id == "fig1"
    assert results[0].images[0].caption


def test_document_profile_retrieval_does_not_call_reranker():
    class ProfileRepository:
        def search_document_profiles(self, query_embedding, top_k=5, document_ids=None):
            return [
                Evidence(
                    chunk_id="profile:doc",
                    document_id="doc",
                    text="Title: Profile",
                    page_start=1,
                    page_end=1,
                    section_title="Profile",
                    score=0.7,
                )
            ]

    service = RetrievalService(
        ProfileRepository(),
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
    )

    results = service.retrieve_document_profiles("What is this paper about?", top_k=1)

    assert results[0].chunk_id == "profile:doc"
    assert service.last_rerank_enabled is False
    assert service.last_recall_diagnostics["mode"] == "document_profile"
    assert service.last_rerank_diagnostics["enabled"] is True
    assert service.last_rerank_diagnostics["returned_count"] == 0
    assert service.last_rerank_diagnostics["new_candidate_count"] == 1
    assert service.last_rerank_diagnostics["skip_reason"] == "document_profile_not_reranked"


def test_retrieve_skips_rerank_when_candidates_are_below_threshold():
    repository = VectorOnlyRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
        reranker_chunk_candidates=10,
        reranker_min_candidates=10,
    )

    results = service.retrieve("ZnO photoluminescence", top_k=3)

    assert [item.chunk_id for item in results] == ["c0", "c1", "c2"]
    assert service.last_rerank_enabled is False
    assert service.last_rerank_diagnostics["enabled"] is False
    assert service.last_rerank_diagnostics["skip_reason"] == "insufficient_candidates"
    assert service.last_rerank_diagnostics["candidate_count"] == 4


def test_retrieve_falls_back_to_vector_results_when_reranker_fails():
    repository = VectorOnlyRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FailingReranker(),
        reranker_enabled=True,
        reranker_chunk_candidates=10,
        reranker_min_candidates=1,
    )

    results = service.retrieve("ZnO photoluminescence", top_k=3)

    assert [item.chunk_id for item in results] == ["c0", "c1", "c2"]
    assert service.last_rerank_enabled is False
    assert service.last_rerank_diagnostics["enabled"] is False
    assert service.last_rerank_diagnostics["skip_reason"] == "provider_error"
    assert service.last_rerank_diagnostics["returned_count"] == 3


def test_followup_retrieval_reranks_only_candidates_not_seen_in_previous_round():
    repository = VectorOnlyRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
        reranker_chunk_candidates=10,
        reranker_min_candidates=1,
    )

    results = service.retrieve(
        "ZnO photoluminescence additional evidence",
        top_k=3,
        exclude_keys={("d1", "c0"), ("d1", "c1")},
    )

    assert [item.chunk_id for item in results] == ["c2", "c3"]
    assert service.last_recall_diagnostics["candidate_count"] == 4
    assert service.last_rerank_diagnostics["incremental_only"] is True
    assert service.last_rerank_diagnostics["excluded_existing_count"] == 2
    assert service.last_rerank_diagnostics["new_candidate_count"] == 2
