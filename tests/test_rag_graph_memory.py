from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.domain.documents import Evidence
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.llm import FakeLlmProvider
from backend.app.services.rag import RetrievalService


class TextQueryRepository:
    expects_text_query = True

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query, embedding_provider, top_k):
        self.queries.append(query)
        return [
            Evidence(
                chunk_id="c1",
                document_id="d1",
                text="Catalyst X is stable in acidic electrolyte.",
                page_start=4,
                page_end=4,
                section_title="Stability",
                score=0.9,
            )
        ]


def test_rag_graph_injects_conversation_context_into_first_query():
    repository = TextQueryRepository()
    graph = MultiAgentRagGraph(RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)), FakeLlmProvider())

    state = graph.invoke(
        "它在酸性条件下稳定吗？",
        conversation_context="最近完整对话：用户问题：这篇论文研究 Catalyst X。",
        session_id="s1",
    )

    assert state["session_id"] == "s1"
    assert "Catalyst X" in repository.queries[0]
    assert "当前问题：它在酸性条件下稳定吗？" in repository.queries[0]
    assert state["conversation_context"]
