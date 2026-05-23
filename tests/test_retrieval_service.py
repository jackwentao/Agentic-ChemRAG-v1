from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.rerankers import FakeRerankerProvider
from backend.app.services.rag import RetrievalService


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


def test_retrieve_uses_candidate_pool_and_reranks_to_top_k():
    repository = VectorOnlyRepository()
    service = RetrievalService(
        repository,
        FakeEmbeddingProvider(dimensions=8),
        reranker_provider=FakeRerankerProvider(),
        reranker_enabled=True,
        reranker_chunk_candidates=30,
    )

    results = service.retrieve("ZnO photoluminescence", top_k=1)

    assert repository.top_k == 30
    assert results[0].chunk_id == "c1"
    assert service.last_rerank_enabled is True
    assert service.last_rerank_latency_ms >= 0


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
