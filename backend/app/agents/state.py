from typing import TypedDict

from backend.app.domain.documents import Evidence, EvidenceImage
from backend.app.domain.rag import AnswerVerification, EvidenceAssessment, QueryPlan, RoutePlan, StopReason, TaskDecision


class RagGraphState(TypedDict, total=False):
    # 这个状态对象是 LangGraph 中各节点之间共享的“上下文总线”。
    # total=False 表示每个节点只需要读写自己关心的字段，便于渐进式扩展。
    question: str
    original_question: str
    session_id: str | None
    conversation_context: str | None
    document_ids: list[str] | None
    scope: str | None
    task_decision: TaskDecision | None
    route_plan: RoutePlan | None
    current_step_index: int
    completed_route_steps: list[dict[str, object]]
    plan_verified_evidence: list[Evidence]
    step_has_assessment: bool
    query_plans: list[QueryPlan]
    current_query: str
    retrieval_round: int
    max_retrieval_rounds: int
    continue_retrieval: bool
    stop_reason: StopReason | None
    plan: str
    route: str
    route_scores: dict[str, float]
    route_lexical_scores: dict[str, float]
    route_semantic_scores: dict[str, float]
    route_confidence: float
    route_rationale: list[str]
    needs_llm_router: bool
    is_multi_step: bool
    route_latency_ms: float
    # evidence 表示当前轮检索结果；accumulated_evidence 表示跨轮累积结果。
    # 两者分离后，控制器就能判断“本轮是否新增证据”和“总体证据是否足够回答”。
    evidence: list[Evidence]
    accumulated_evidence: list[Evidence]
    current_new_evidence_count: int
    retrieval_history: list[dict[str, object]]
    assessment_history: list[dict[str, object]]
    verified_evidence: list[Evidence]
    evidence_assessment: EvidenceAssessment
    # 下列 verification_* 字段主要是为了兼容旧消费方，
    # 新逻辑更推荐直接读取 evidence_assessment。
    verification_answerable: bool
    verification_reason: str
    verification_used_llm: bool
    verification_fallback_used: bool
    verification_latency_ms: float
    selected_images: dict[str, list[EvidenceImage]]
    answer: str
    limitations: list[str]
    answer_verification: AnswerVerification | None
    total_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    rerank_enabled: bool
    answer_latency_ms: float
    llm_call_counts: dict[str, int]
    diagnostics: dict[str, object]
    warnings: list[str]
