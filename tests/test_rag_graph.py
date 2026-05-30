from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.domain.documents import Chunk, Evidence, EvidenceImage
from backend.app.domain.rag import EvidenceAssessment, RoutePlan, RouteStep
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
    assert state["original_question"] == "How does chemical bonding affect transport?"
    assert state["query_plans"][0].queries == ["How does chemical bonding affect transport?"]
    assert state["retrieval_round"] == 1
    assert state["max_retrieval_rounds"] == 1
    assert state["task_decision"].task_type == "simple_fact"
    assert state["stop_reason"] == "sufficient_evidence"
    assert state["verified_evidence"]
    assert state["verified_evidence"][0].chunk_id == "c1"
    assert state["limitations"] == []
    assert state["evidence_assessment"].sufficient is True
    assert "route_scores" in state
    assert state["diagnostics"]["route"]["latency_ms"] >= 0
    assert state["diagnostics"]["recall"]["candidate_count"] >= 1
    assert state["diagnostics"]["verifier"]["answerable"] is True
    assert state["diagnostics"]["verifier"]["verified_evidence_count"] == 1
    assert state["diagnostics"]["retrieval_rounds"][0]["round_index"] == 1
    assert state["diagnostics"]["assessment_rounds"][0]["round_index"] == 1
    assert state["diagnostics"]["llm_call_counts"] == {"evidence_assessment": 0, "answer_generation": 0}
    assert state["diagnostics"]["stop_reason"] == "sufficient_evidence"
    assert state["diagnostics"]["llm"]["answer_chars"] > 0
    assert state["total_latency_ms"] >= 0


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


class RejectingVerifierProvider(FakeLlmProvider):
    def __init__(self) -> None:
        self.answer_calls = 0
        self.assessment_calls = 0

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        self.assessment_calls += 1
        return EvidenceAssessment(
            sufficient=False,
            covered_aspects=["形貌描述"],
            missing_aspects=["导电性实验结论"],
            next_search_goal="检索导电性测量结果",
            reason="片段仅描述形貌，没有给出问题要求的实验结论。",
            used_llm=True,
        )

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        self.answer_calls += 1
        return super().answer_with_evidence(question, evidence)


def test_multi_agent_rag_graph_stops_answer_generation_when_verifier_rejects_evidence():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    provider = RejectingVerifierProvider()
    graph = MultiAgentRagGraph(RetrievalService(ImageCandidateRepository(), embedding_provider), provider)

    state = graph.invoke("What conclusion was reported for conductivity?")

    assert state["verified_evidence"] == []
    assert "证据不足以可靠回答" in state["answer"]
    assert state["limitations"] == ["资料缺口：导电性实验结论"]
    assert state["evidence_assessment"].next_search_goal == "检索导电性测量结果"
    assert "检索证据不足以回答问题" in state["warnings"][0]
    assert provider.answer_calls == 0
    assert state["diagnostics"]["verifier"]["used_llm"] is True
    assert state["diagnostics"]["verifier"]["sufficient"] is False
    assert state["diagnostics"]["verifier"]["answerable"] is False
    assert state["diagnostics"]["verifier"]["missing_aspects"] == ["导电性实验结论"]
    assert state["diagnostics"]["verifier"]["verified_evidence_count"] == 0
    assert state["diagnostics"]["llm"]["used_llm"] is False
    assert state["retrieval_round"] == 1
    assert state["max_retrieval_rounds"] == 1
    assert state["stop_reason"] == "max_rounds_reached"
    assert provider.assessment_calls == 1
    assert len(state["diagnostics"]["assessment_rounds"]) == 1
    assert state["diagnostics"]["llm_call_counts"] == {"evidence_assessment": 1, "answer_generation": 0}


def _round_evidence(chunk_id: str, text: str) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        document_id="d-round",
        text=text,
        page_start=1,
        page_end=1,
        section_title="Results",
        score=0.9,
    )


class TextQueryRoundRepository:
    expects_text_query = True

    def __init__(self, responses: list[list[Evidence]]) -> None:
        self.responses = responses
        self.queries: list[str] = []

    def search(self, query, embedding_provider, top_k):
        self.queries.append(query)
        index = min(len(self.queries) - 1, len(self.responses) - 1)
        return self.responses[index]


class CompletingAssessmentProvider(FakeLlmProvider):
    def __init__(self) -> None:
        self.assessment_evidence_ids: list[list[str]] = []

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        chunk_ids = [item.chunk_id for item in evidence]
        self.assessment_evidence_ids.append(chunk_ids)
        if "c-high-temp" in chunk_ids:
            return EvidenceAssessment(
                sufficient=True,
                covered_aspects=["常温稳定性", "高温稳定性"],
                supporting_chunk_ids=chunk_ids,
                reason="累计证据覆盖了问题要求。",
                used_llm=True,
            )
        return EvidenceAssessment(
            sufficient=False,
            covered_aspects=["常温稳定性"],
            missing_aspects=["高温稳定性"],
            next_search_goal="检索高温稳定性结果",
            reason="尚缺高温稳定性数据。",
            used_llm=True,
        )


def test_multi_agent_rag_graph_retrieves_again_for_missing_aspect_and_assesses_accumulated_evidence():
    repository = TextQueryRoundRepository(
        [
            [_round_evidence("c-room-temp", "Stable at room temperature.")],
            [_round_evidence("c-high-temp", "Stable under high-temperature aqueous conditions.")],
        ]
    )
    provider = CompletingAssessmentProvider()
    graph = MultiAgentRagGraph(RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)), provider)

    state = graph.invoke("Is catalyst X stable in water at elevated temperature?")

    assert state["stop_reason"] == "sufficient_evidence"
    assert state["retrieval_round"] == 2
    assert len(repository.queries) == 2
    assert "检索高温稳定性结果" in repository.queries[1]
    assert provider.assessment_evidence_ids == [["c-room-temp"], ["c-room-temp", "c-high-temp"]]
    assert [item.chunk_id for item in state["verified_evidence"]] == ["c-room-temp", "c-high-temp"]
    assert state["limitations"] == []
    assert len(state["diagnostics"]["retrieval_rounds"]) == 2
    assert len(state["diagnostics"]["assessment_rounds"]) == 2
    assert state["diagnostics"]["llm_call_counts"] == {"evidence_assessment": 2, "answer_generation": 0}


class NeverSufficientProvider(FakeLlmProvider):
    def __init__(self) -> None:
        self.calls = 0

    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        self.calls += 1
        return EvidenceAssessment(
            sufficient=False,
            covered_aspects=["局部信息"],
            missing_aspects=[f"缺失证据 {self.calls}"],
            next_search_goal=f"检索补充条件 {self.calls}",
            reason="证据仍不完整。",
            used_llm=True,
        )


def test_multi_agent_rag_graph_stops_at_three_retrieval_rounds_when_evidence_remains_insufficient():
    repository = TextQueryRoundRepository(
        [
            [_round_evidence("c1", "Evidence round 1.")],
            [_round_evidence("c2", "Evidence round 2.")],
            [_round_evidence("c3", "Evidence round 3.")],
            [_round_evidence("c4", "This round must not run.")],
        ]
    )
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        NeverSufficientProvider(),
    )

    state = graph.invoke("Assess catalyst durability under all requested conditions.")

    assert len(repository.queries) == 3
    assert state["retrieval_round"] == 3
    assert state["stop_reason"] == "max_rounds_reached"
    assert state["verified_evidence"] == []
    assert state["limitations"] == ["资料缺口：缺失证据 3"]
    assert state["diagnostics"]["stop_reason"] == "max_rounds_reached"


def test_multi_agent_rag_graph_stops_when_refined_search_returns_no_new_evidence():
    duplicate = _round_evidence("c-duplicate", "Only morphology is described.")
    repository = TextQueryRoundRepository([[duplicate], [duplicate]])
    provider = RejectingVerifierProvider()
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        provider,
    )

    state = graph.invoke("Compare conductivity under dry and aqueous conditions.")

    assert len(repository.queries) == 2
    assert state["stop_reason"] == "no_new_evidence"
    assert state["max_retrieval_rounds"] == 3
    assert state["diagnostics"]["retrieval_rounds"][1]["new_evidence_count"] == 0
    assert state["diagnostics"]["retrieval_rounds"][1]["rerank_incremental_only"] is True
    assert state["diagnostics"]["retrieval_rounds"][1]["rerank_excluded_existing_count"] == 1
    assert state["diagnostics"]["rerank"]["incremental_only"] is True
    assert state["diagnostics"]["rerank"]["excluded_existing_count"] == 1
    assert len(state["accumulated_evidence"]) == 1
    assert provider.assessment_calls == 1


class RepeatedRefinementProvider(FakeLlmProvider):
    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        return EvidenceAssessment(
            sufficient=False,
            missing_aspects=["循环寿命"],
            next_search_goal="检索循环寿命",
            reason="仍缺循环寿命数据。",
            used_llm=True,
        )


def test_multi_agent_rag_graph_stops_when_rule_refiner_would_repeat_a_query():
    repository = TextQueryRoundRepository(
        [
            [_round_evidence("c1", "Initial result.")],
            [_round_evidence("c2", "Different result without lifecycle data.")],
        ]
    )
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        RepeatedRefinementProvider(),
    )

    state = graph.invoke("Does catalyst X last in continuous operation?")

    assert len(repository.queries) == 2
    assert state["stop_reason"] == "no_new_query"
    assert state["retrieval_round"] == 2


class FallbackAssessmentProvider(FakeLlmProvider):
    def assess_evidence(self, question: str, evidence: list[Evidence]) -> EvidenceAssessment:
        return EvidenceAssessment(
            sufficient=True,
            covered_aspects=["候选证据"],
            missing_aspects=["证据充分性评估未完成，回答需谨慎复核"],
            supporting_chunk_ids=[item.chunk_id for item in evidence],
            reason="证据评估超时，已回退。",
            used_llm=True,
            fallback_used=True,
        )


def test_multi_agent_rag_graph_does_not_retry_after_assessment_fallback():
    repository = TextQueryRoundRepository([[_round_evidence("c1", "Candidate evidence.")]])
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        FallbackAssessmentProvider(),
    )

    state = graph.invoke("What was reported?")

    assert len(repository.queries) == 1
    assert state["stop_reason"] == "assessment_fallback"
    assert state["verified_evidence"] == []
    assert "证据不足以可靠回答" in state["answer"]
    assert state["diagnostics"]["verifier"]["sufficient"] is False
    assert state["diagnostics"]["verifier"]["answerable"] is False
    assert state["diagnostics"]["verifier"]["verified_evidence_count"] == 0
    assert state["diagnostics"]["llm"]["used_llm"] is False
    assert "证据评估已回退" in state["limitations"][1]


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
    assert state["diagnostics"]["llm"]["used_llm"] is False
    assert state["diagnostics"]["llm"]["selected_image_count"] == 1


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
    assert state["diagnostics"]["route"]["selected_route"] == "document_profile"


class PlannedProfileFigureProvider(FakeLlmProvider):
    def plan_route(self, question: str, *, conversation_context: str | None, scope: str, document_ids: list[str] | None):
        return RoutePlan(
            plan_type="multi_step",
            rationale="中文复合问题需要先概括文档，再定位PL相关图。",
            used_llm=True,
            steps=[
                RouteStep(
                    step_id="step-1",
                    route="document_profile",
                    task_type="document_summary",
                    query="概括这篇论文的主题",
                    purpose="获取论文整体主题",
                    max_rounds=1,
                ),
                RouteStep(
                    step_id="step-2",
                    route="figure_index",
                    task_type="figure_query",
                    query="查找PL相关谱图",
                    purpose="定位PL相关图题和图片",
                    max_rounds=1,
                ),
            ],
        )


class ProfileAndFigureRepository:
    def __init__(self) -> None:
        self.profile_queries = 0
        self.figure_queries = 0

    def search_document_profiles(self, query_embedding, top_k=5, document_ids=None):
        self.profile_queries += 1
        return [
            Evidence(
                chunk_id="profile:d1",
                document_id="d1",
                text="Title: ZnO photoluminescence materials. Topics: photoluminescence.",
                page_start=1,
                page_end=1,
                section_title="Document Profile",
                score=0.9,
            )
        ]

    def search_figures(self, query_embedding, top_k=8, document_ids=None):
        self.figure_queries += 1
        return [
            Evidence(
                chunk_id="figure:img1",
                document_id="d1",
                text="Figure 2. PL spectra of ZnO samples.",
                page_start=4,
                page_end=4,
                section_title="Figure Index",
                score=0.92,
                image_ids=["img1"],
                images=[EvidenceImage(image_id="img1", page=4, label="Figure 2", caption="Figure 2. PL spectra.")],
            )
        ]


def test_multi_agent_rag_graph_executes_llm_route_plan_steps():
    repository = ProfileAndFigureRepository()
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        PlannedProfileFigureProvider(),
    )

    state = graph.invoke("这篇论文主要讲什么？顺便找一下和PL相关的图。", document_ids=["d1"])

    assert repository.profile_queries == 1
    assert repository.figure_queries == 1
    assert state["route_plan"].used_llm is True
    assert len(state["route_plan"].steps) == 2
    assert [step["route"] for step in state["completed_route_steps"]] == ["document_profile", "figure_index"]
    assert [item.chunk_id for item in state["verified_evidence"]] == ["profile:d1", "figure:img1"]
    assert state["selected_images"]["figure:img1"][0].image_id == "img1"
    assert state["diagnostics"]["route_plan"]["used_llm"] is True
    assert state["diagnostics"]["route_plan"]["completed_steps"][1]["route"] == "figure_index"
