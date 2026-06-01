from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.domain.documents import Chunk, Evidence, EvidenceImage
from backend.app.domain.rag import AnswerVerification, EvidenceAssessment, ImageSelectionResult, RoutePlan, RouteStep
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.providers.llm import FakeLlmProvider
from backend.app.repositories.memory import InMemoryVectorRepository
from backend.app.services.rag import RetrievalService


class CountingAnswerProvider(FakeLlmProvider):
    def __init__(self) -> None:
        self.answer_calls = 0

    def uses_llm_for_answer(self) -> bool:
        return True

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        self.answer_calls += 1
        return super().answer_with_evidence(question, evidence)


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
    assert state["diagnostics"]["llm_call_counts"] == {
        "evidence_assessment": 0,
        "answer_generation": 0,
        "answer_verification": 0,
        "answer_revision": 0,
        "image_selection": 0,
    }
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


class LlmImageSelectionProvider(FakeLlmProvider):
    def select_images_for_answer(self, question: str, answer: str, evidence: list[Evidence], max_images: int = 3):
        image = evidence[0].images[0]
        selected = {"c1": [EvidenceImage(image_id=image.image_id, page=image.page, reason="Selected by LLM.")]}
        return ImageSelectionResult(
            images=selected,
            strategy="provider_llm",
            used_llm=True,
            candidate_count=1,
            selected_count=1,
            reason="测试 provider 模拟图片选择模型调用。",
        )


def test_multi_agent_rag_graph_counts_llm_image_selection_calls():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    graph = MultiAgentRagGraph(RetrievalService(ImageCandidateRepository(), embedding_provider), LlmImageSelectionProvider())

    state = graph.invoke("What morphology is shown?")

    assert state["selected_images"]["c1"][0].image_id == "img1"
    assert state["diagnostics"]["image_selection"]["used_llm"] is True
    assert state["diagnostics"]["llm_call_counts"]["image_selection"] == 1


def test_multi_agent_rag_graph_triggers_answer_verification_for_high_risk_question():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    repository = InMemoryVectorRepository()
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        parent_id="p1",
        text="The paper reports toxicity observations but does not provide scale-up safety evidence.",
        page_start=2,
        page_end=2,
        section_title="Safety",
        section_path="Safety",
        chunk_type="child",
    )
    repository.add_chunks([chunk], embedding_provider.embed_documents([chunk.text]))

    graph = MultiAgentRagGraph(RetrievalService(repository, embedding_provider), FakeLlmProvider())
    state = graph.invoke("Is this material safe for scale-up?")

    verification = state["diagnostics"]["answer_verification"]
    assert verification["triggered"] is True
    assert "high_risk" in verification["trigger_reasons"]
    assert verification["passed"] is True
    assert state["diagnostics"]["llm_call_counts"]["answer_verification"] == 0


class FailingAnswerVerifierProvider(FakeLlmProvider):
    def __init__(self) -> None:
        self.revision_calls = 0

    def answer_with_evidence(self, question: str, evidence: list[Evidence]) -> str:
        return "This unsupported conclusion is safe for scale-up."

    def verify_answer_with_evidence(self, question: str, answer: str, evidence: list[Evidence], **kwargs) -> AnswerVerification:
        return AnswerVerification(
            passed=False,
            triggered=True,
            trigger_reasons=kwargs["trigger_reasons"],
            unsupported_claims=["safe for scale-up"],
            revision_instruction="删除放大安全结论，并保守说明证据边界。",
            used_llm=True,
            reason="答案包含未支撑的高风险结论。",
        )

    def revise_answer_with_evidence(self, question: str, answer: str, evidence: list[Evidence], **kwargs) -> str:
        self.revision_calls += 1
        return "证据不足以支持安全放大结论；只能说明文献报告了有限观察。"


def test_multi_agent_rag_graph_revises_once_after_failed_answer_verification():
    repository = TextQueryRoundRepository([[_round_evidence("c1", "The paper reports limited safety observations.")]])
    provider = FailingAnswerVerifierProvider()
    graph = MultiAgentRagGraph(RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)), provider)

    state = graph.invoke("Is this material safe for scale-up?")

    assert provider.revision_calls == 1
    assert state["answer_revision_count"] == 1
    assert "安全放大结论" in state["answer"]
    assert state["diagnostics"]["answer_verification"]["passed"] is False
    assert state["diagnostics"]["answer_verification"]["revision_applied"] is True
    assert state["diagnostics"]["llm_call_counts"]["answer_verification"] == 1
    assert state["diagnostics"]["llm_call_counts"]["answer_revision"] == 0


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
    assert state["diagnostics"]["llm_call_counts"] == {
        "evidence_assessment": 1,
        "answer_generation": 0,
        "answer_verification": 0,
        "answer_revision": 0,
        "image_selection": 0,
    }


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
    assert state["diagnostics"]["llm_call_counts"] == {
        "evidence_assessment": 2,
        "answer_generation": 0,
        "answer_verification": 0,
        "answer_revision": 0,
        "image_selection": 0,
    }


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


class LlmRefinementAssessmentProvider(RepeatedRefinementProvider):
    def __init__(self) -> None:
        self.refinement_calls = 0

    def refine_search_queries(self, question: str, **kwargs):
        self.refinement_calls += 1
        return {
            "queries": ["catalyst continuous operation durability lifetime"],
            "search_goal": "use durability and lifetime synonyms",
            "reason": "complex task uses quality-first query expansion",
            "safety_notes": [],
        }


def test_multi_agent_rag_graph_uses_llm_refiner_without_changing_route():
    repository = TextQueryRoundRepository(
        [
            [_round_evidence("c1", "Initial result.")],
            [_round_evidence("c2", "Durability result under continuous operation.")],
        ]
    )
    provider = LlmRefinementAssessmentProvider()
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        provider,
    )

    state = graph.invoke("Does catalyst X last in continuous operation?")

    assert provider.refinement_calls >= 1
    assert len(repository.queries) == 2
    assert repository.queries[1] == "catalyst continuous operation durability lifetime"
    assert state["route"] == "chunk_search"
    assert state["diagnostics"]["retrieval_strategy"]["selected"] == "hybrid"
    assert state["diagnostics"]["query_refinement"]["used_llm"] is True


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
                evidence_type="figure",
                source_scope="page_span",
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


def test_multi_agent_rag_graph_uses_unified_answerer_for_figure_index():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    provider = CountingAnswerProvider()
    graph = MultiAgentRagGraph(RetrievalService(FigureIndexRepository(), embedding_provider), provider)

    state = graph.invoke("该文献有什么性能图？")

    assert state["route"] == "figure_index"
    assert provider.answer_calls == 1
    assert state["diagnostics"]["llm"]["used_llm"] is True
    assert state["diagnostics"]["llm_call_counts"]["answer_generation"] == 1


class DocumentProfileRepository:
    def __init__(self) -> None:
        self.document_ids = None

    def search_document_profiles(self, query_embedding, top_k=5, document_ids=None):
        self.document_ids = document_ids
        return [
            Evidence(
                chunk_id="profile:d1",
                document_id="d1",
                text="Title: Porphyrin polymer. Entities: porphyrin. Topics: photoluminescence.",
                page_start=1,
                page_end=1,
                section_title="Porphyrin polymer",
                score=0.93,
                evidence_type="document_profile",
                source_scope="document_level",
            )
        ]


def test_multi_agent_rag_graph_routes_holistic_questions_to_document_profiles():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    repository = DocumentProfileRepository()
    graph = MultiAgentRagGraph(RetrievalService(repository, embedding_provider), FakeLlmProvider())

    state = graph.invoke("哪些论文是讲卟啉相关的？", scope="all_documents")

    assert state["route"] == "document_profile"
    assert state["route_scores"]["document_profile"] > state["route_scores"]["chunk_search"]
    assert "cross_document_signal" in state["route_rationale"]
    assert state["verified_evidence"][0].chunk_id == "profile:d1"
    assert state["diagnostics"]["route"]["selected_route"] == "document_profile"
    assert state["diagnostics"]["outline_resolution"]["decision"] == "enter_outline_retrieval"


def test_multi_agent_rag_graph_resolves_pronoun_before_document_profile_retrieval():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    repository = DocumentProfileRepository()
    graph = MultiAgentRagGraph(RetrievalService(repository, embedding_provider), FakeLlmProvider())

    state = graph.invoke("它主要研究什么？", document_ids=["d1"], scope="current_document")

    assert state["route"] == "document_profile"
    assert repository.document_ids == ["d1"]
    assert state["diagnostics"]["outline_resolution"]["decision"] == "single_document_outline"
    assert state["diagnostics"]["outline_resolution"]["needs_outline_retrieval"] is False
    assert "文档 d1" in state["query_plans"][0].resolved_question


def test_multi_agent_rag_graph_uses_unified_answerer_for_document_profiles():
    embedding_provider = FakeEmbeddingProvider(dimensions=8)
    provider = CountingAnswerProvider()
    graph = MultiAgentRagGraph(RetrievalService(DocumentProfileRepository(), embedding_provider), provider)

    state = graph.invoke("哪些论文是讲卟啉相关的？", scope="all_documents")

    assert state["route"] == "document_profile"
    assert provider.answer_calls == 1
    assert state["diagnostics"]["llm"]["used_llm"] is True
    assert state["diagnostics"]["llm_call_counts"]["answer_generation"] == 1


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


class PlannedProfileOnlyProvider(FakeLlmProvider):
    def plan_route(self, question: str, *, conversation_context: str | None, scope: str, document_ids: list[str] | None):
        return RoutePlan(
            plan_type="single_step",
            rationale="先用文档画像定位候选论文。",
            used_llm=True,
            steps=[
                RouteStep(
                    step_id="step-1",
                    route="document_profile",
                    task_type="cross_document_filter",
                    query="ZnO photoluminescence",
                    purpose="定位相关论文",
                    max_rounds=1,
                ),
            ],
        )


class ProfileGuidedChunkRepository:
    def __init__(self) -> None:
        self.profile_queries = 0
        self.chunk_queries = 0
        self.chunk_document_ids = None

    def search_document_profiles(self, query_embedding, top_k=5, document_ids=None):
        self.profile_queries += 1
        return [
            Evidence(
                chunk_id="profile:d1",
                document_id="d1",
                text="Title: ZnO photoluminescence. Entities: ZnO. Topics: photoluminescence.",
                page_start=1,
                page_end=1,
                section_title="ZnO photoluminescence",
                score=0.91,
                evidence_type="document_profile",
                source_scope="document_level",
                metadata={"entities": ["ZnO"], "topics": ["photoluminescence"]},
            )
        ]

    def search(self, query_embedding, top_k=5, document_ids=None):
        self.chunk_queries += 1
        self.chunk_document_ids = document_ids
        return [
            Evidence(
                chunk_id="c1",
                document_id="d1",
                text="The Results section reports green photoluminescence emission from ZnO samples.",
                page_start=4,
                page_end=4,
                section_title="Results",
                score=0.88,
            )
        ]


def test_profile_guided_retrieval_adds_constrained_chunk_followup_for_detail_questions():
    repository = ProfileGuidedChunkRepository()
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        PlannedProfileOnlyProvider(),
    )

    state = graph.invoke("Which papers discuss ZnO photoluminescence and what evidence do they provide?", scope="all_documents")

    assert repository.profile_queries == 1
    assert repository.chunk_queries == 1
    assert repository.chunk_document_ids == ["d1"]
    assert [step["route"] for step in state["completed_route_steps"]] == ["document_profile", "chunk_search"]
    assert [item.chunk_id for item in state["verified_evidence"]] == ["profile:d1", "c1"]
    assert state["route_plan"].plan_type == "profile_guided"
    assert state["diagnostics"]["profile_guided_followup"]["route"] == "chunk_search"
    assert state["diagnostics"]["profile_guided_followup"]["document_ids"] == ["d1"]


class ProfileGuidedFigureRepository(ProfileGuidedChunkRepository):
    def __init__(self) -> None:
        super().__init__()
        self.figure_queries = 0
        self.figure_document_ids = None

    def search_figures(self, query_embedding, top_k=8, document_ids=None):
        self.figure_queries += 1
        self.figure_document_ids = document_ids
        return [
            Evidence(
                chunk_id="figure:img1",
                document_id="d1",
                text="Figure 2. PL spectra of ZnO samples.",
                page_start=5,
                page_end=5,
                section_title="Figure Index",
                score=0.9,
                image_ids=["img1"],
                images=[EvidenceImage(image_id="img1", page=5, label="Figure 2", caption="Figure 2. PL spectra.")],
                evidence_type="figure",
            )
        ]


def test_profile_guided_retrieval_adds_constrained_figure_followup_for_figure_questions():
    repository = ProfileGuidedFigureRepository()
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        PlannedProfileOnlyProvider(),
    )

    state = graph.invoke("Which indexed paper discusses ZnO and which PL spectrum figure supports it?", scope="all_documents")

    assert repository.profile_queries == 1
    assert repository.figure_queries == 1
    assert repository.figure_document_ids == ["d1"]
    assert repository.chunk_queries == 0
    assert [step["route"] for step in state["completed_route_steps"]] == ["document_profile", "figure_index"]
    assert state["selected_images"]["figure:img1"][0].image_id == "img1"
    assert state["diagnostics"]["profile_guided_followup"]["route"] == "figure_index"


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


class PlannedFigureThenChunkProvider(FakeLlmProvider):
    def plan_route(self, question: str, *, conversation_context: str | None, scope: str, document_ids: list[str] | None):
        return RoutePlan(
            plan_type="multi_step",
            rationale="先定位目标图片，再补充文本解释。",
            used_llm=True,
            steps=[
                RouteStep(
                    step_id="step-1",
                    route="figure_index",
                    task_type="figure_query",
                    query="查找PL相关谱图",
                    purpose="定位PL相关图题和图片",
                    max_rounds=1,
                ),
                RouteStep(
                    step_id="step-2",
                    route="chunk_search",
                    task_type="simple_fact",
                    query="解释PL谱图变化",
                    purpose="补充文字解释",
                    max_rounds=1,
                ),
            ],
        )


class FigureThenChunkRepository:
    def search_figures(self, query_embedding, top_k=8, document_ids=None):
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

    def search(self, query_embedding, top_k=5, document_ids=None):
        return [
            Evidence(
                chunk_id="c1",
                document_id="d1",
                text="The PL spectra show stronger green emission after treatment.",
                page_start=5,
                page_end=5,
                section_title="Results",
                score=0.9,
                image_ids=["img2"],
                images=[EvidenceImage(image_id="img2", page=5, caption="Supplementary morphology image.")],
            )
        ]


def test_multi_step_plan_selects_figure_images_even_when_final_route_is_not_figure():
    repository = FigureThenChunkRepository()
    graph = MultiAgentRagGraph(
        RetrievalService(repository, FakeEmbeddingProvider(dimensions=8)),
        PlannedFigureThenChunkProvider(),
    )

    state = graph.invoke("先找PL图，再解释变化。", document_ids=["d1"])

    assert state["route"] == "chunk_search"
    assert [step["route"] for step in state["completed_route_steps"]] == ["figure_index", "chunk_search"]
    assert state["selected_images"]["figure:img1"][0].image_id == "img1"
    assert "c1" not in state["selected_images"]
    assert state["diagnostics"]["image_selection"]["strategy"] == "figure_index"
    assert state["diagnostics"]["llm_call_counts"]["image_selection"] == 0
