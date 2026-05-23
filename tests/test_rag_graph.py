from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.domain.documents import Chunk, Evidence, EvidenceImage
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.llm import FakeLlmProvider
from backend.app.repositories.memory import InMemoryVectorRepository
from backend.app.services.rag import RetrievalService


def test_multi_agent_rag_graph_returns_answer_and_evidence():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    repository = InMemoryVectorRepository()
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        parent_id="p1",
        text="Chemical bonding improves interfacial charge transport in composites.",
        page_start=2,
        page_end=2,
        section_title="Results and Discussion",
        section_path="Results and Discussion",
        chunk_type="child",
    )
    repository.add_chunks([chunk], embedding_provider.embed_documents([chunk.text]))

    graph = MultiAgentRagGraph(RetrievalService(repository, embedding_provider), FakeLlmProvider())
    state = graph.invoke("How does chemical bonding affect transport?")

    assert state["answer"]
    assert state["verified_evidence"]
    assert state["verified_evidence"][0].chunk_id == "c1"
    assert "route_scores" in state


class ImageCandidateRepository:
    def search(self, query_embedding, top_k=5, document_ids=None):
        return [
            Evidence(
                chunk_id="c1",
                document_id="d1",
                text="The SEM image shows a porous nanosheet morphology.",
                page_start=3,
                page_end=3,
                section_title="Results",
                score=0.9,
                image_ids=["img1"],
                images=[EvidenceImage(image_id="img1", page=3, width=640, height=480)],
            )
        ]


def test_multi_agent_rag_graph_selects_answer_images_from_candidates():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    graph = MultiAgentRagGraph(RetrievalService(ImageCandidateRepository(), embedding_provider), FakeLlmProvider())

    state = graph.invoke("What morphology is shown?")

    assert state["selected_images"]["c1"][0].image_id == "img1"
    assert state["selected_images"]["c1"][0].reason


class FigureIndexRepository:
    def search_figures(self, query_embedding, top_k=8, document_ids=None):
        return [
            Evidence(
                chunk_id="figure:img1",
                document_id="d1",
                text="Figure 1. Photoluminescence spectra.",
                page_start=5,
                page_end=5,
                section_title="Figure Index",
                score=0.91,
                image_ids=["img1"],
                images=[
                    EvidenceImage(
                        image_id="img1",
                        page=5,
                        label="Figure 1",
                        caption="Figure 1. Photoluminescence spectra.",
                    )
                ],
            )
        ]


def test_multi_agent_rag_graph_routes_figure_questions_to_figure_index():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    graph = MultiAgentRagGraph(RetrievalService(FigureIndexRepository(), embedding_provider), FakeLlmProvider())

    state = graph.invoke("该文献有什么性能图？")

    assert state["route"] == "figure_index"
    assert state["route_scores"]["figure_index"] > state["route_scores"]["chunk_search"]
    assert state["route_confidence"] > 0
    assert "Figure 1" in state["answer"]
    assert state["selected_images"]["figure:img1"][0].image_id == "img1"


class DocumentProfileRepository:
    def search_document_profiles(self, query_embedding, top_k=5, document_ids=None):
        return [
            Evidence(
                chunk_id="profile:d1",
                document_id="d1",
                text="Title: Porphyrin polymer. Entities: porphyrin. Topics: photoluminescence.",
                page_start=1,
                page_end=1,
                section_title="Porphyrin polymer",
                score=0.93,
            )
        ]


def test_multi_agent_rag_graph_routes_holistic_questions_to_document_profiles():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    graph = MultiAgentRagGraph(RetrievalService(DocumentProfileRepository(), embedding_provider), FakeLlmProvider())

    state = graph.invoke("哪些论文是讲卟啉相关的？", scope="all_documents")

    assert state["route"] == "document_profile"
    assert state["route_scores"]["document_profile"] > state["route_scores"]["chunk_search"]
    assert "cross_document_signal" in state["route_rationale"]
    assert state["verified_evidence"][0].chunk_id == "profile:d1"
