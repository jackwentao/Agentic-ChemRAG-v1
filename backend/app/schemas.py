from typing import Literal

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    task_id: str | None = None
    status: str
    reused: bool = False
    error: str | None = None


class BatchDocumentUploadItem(BaseModel):
    filename: str
    document_id: str | None = None
    task_id: str | None = None
    status: str
    reused: bool = False
    error: str | None = None


class DocumentBatchUploadResponse(BaseModel):
    results: list[BatchDocumentUploadItem]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: int | None = None
    error: str | None = None


class ReadinessComponent(BaseModel):
    status: Literal["ready", "disabled", "failed"]
    required: bool
    detail: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    complete_qa_online: bool
    can_serve_degraded: bool
    checks: dict[str, ReadinessComponent]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    created_at: str
    chunk_count: int
    is_indexed: bool
    file_hash: str | None = None
    indexing_status: str = "pending"
    indexing_error: str | None = None


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    document_ids: list[str] | None = None
    scope: Literal["current_document", "all_documents"] = "current_document"


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


class ChatMessage(BaseModel):
    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    scope: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    citations: list[dict[str, object]] = Field(default_factory=list)
    images: list[dict[str, object]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    diagnostics_summary: dict[str, object] = Field(default_factory=dict)
    created_at: str | None = None


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class ChatMessagesResponse(BaseModel):
    session: ChatSessionSummary
    messages: list[ChatMessage]


class CitationImage(BaseModel):
    image_id: str
    url: str
    source_url: str
    page: int
    width: int | None = None
    height: int | None = None
    label: str | None = None
    caption: str | None = None
    reason: str | None = None


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    page_start: int
    page_end: int
    source_url: str | None
    section_title: str
    quote: str
    score: float
    evidence_type: str = "chunk"
    source_scope: str = "page_span"
    image_ids: list[str] = []
    images: list[CitationImage] = []


class RouteDiagnostics(BaseModel):
    latency_ms: float = 0.0
    selected_route: str
    confidence: float = 0.0
    needs_llm_router: bool = False
    is_multi_step: bool = False
    proxy_hit: bool = False
    scores: dict[str, float] = Field(default_factory=dict)
    lexical_scores: dict[str, float] = Field(default_factory=dict)
    semantic_scores: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


class OutlineResolutionDiagnostics(BaseModel):
    original_question: str = ""
    rewritten_question: str = ""
    basic_facts: list[str] = Field(default_factory=list)
    decision: str = "pass_through"
    target_document_ids: list[str] = Field(default_factory=list)
    needs_outline_retrieval: bool = False
    reason: str = ""


class OutlineContextDiagnostics(BaseModel):
    injected: bool = False
    evidence_count: int = 0
    document_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    query: str = ""
    reason: str = ""
    evidence: list[dict[str, object]] = Field(default_factory=list)


class RoutePlanStepDiagnostics(BaseModel):
    step_id: str
    route: str
    task_type: str
    query: str
    purpose: str = ""
    max_rounds: int | None = None
    requires_assessment: bool = True
    requires_answer_verification: bool = False


class RoutePlanDiagnostics(BaseModel):
    plan_type: str = "single_step"
    used_llm: bool = False
    fallback_used: bool = False
    current_step_index: int = 0
    rationale: str = ""
    steps: list[RoutePlanStepDiagnostics] = Field(default_factory=list)
    completed_steps: list[dict[str, object]] = Field(default_factory=list)


class RecallDiagnostics(BaseModel):
    mode: str
    latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    candidate_count: int = 0
    returned_count: int = 0
    hit: bool = False
    proxy_hit: bool = False
    top_score: float | None = None
    avg_score: float | None = None


class RerankDiagnostics(BaseModel):
    enabled: bool = False
    latency_ms: float = 0.0
    candidate_build_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    postprocess_latency_ms: float = 0.0
    candidate_count: int = 0
    returned_count: int = 0
    input_char_count: int = 0
    avg_candidate_chars: float = 0.0
    changed_top1: bool = False
    changed_top_k: bool = False
    top1_term_overlap_before: int = 0
    top1_term_overlap_after: int = 0
    skip_reason: str | None = None
    new_candidate_count: int = 0
    excluded_existing_count: int = 0
    incremental_only: bool = False
    proxy_hit: bool = False


class TaskDiagnostics(BaseModel):
    task_type: str
    requires_retrieval: bool = True
    risk_level: str = "low"
    rationale: str = ""


class LlmDiagnostics(BaseModel):
    latency_ms: float = 0.0
    used_llm: bool = False
    answer_chars: int = 0
    evidence_count: int = 0
    citation_count: int = 0
    selected_image_count: int = 0
    proxy_hit: bool = False


class VerifierDiagnostics(BaseModel):
    latency_ms: float = 0.0
    used_llm: bool = False
    sufficient: bool = False
    # 保留 answerable 字段以兼容现有客户端，语义等同于 sufficient。
    answerable: bool = False
    fallback_used: bool = False
    input_evidence_count: int = 0
    verified_evidence_count: int = 0
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    next_search_goal: str | None = None
    confidence: float | None = None
    reason: str = ""
    proxy_hit: bool = False


class RetrievalRoundDiagnostics(BaseModel):
    round_index: int
    query: str
    latency_ms: float = 0.0
    candidate_count: int = 0
    returned_count: int = 0
    new_evidence_count: int = 0
    accumulated_evidence_count: int = 0
    rerank_new_candidate_count: int = 0
    rerank_excluded_existing_count: int = 0
    rerank_incremental_only: bool = False


class AssessmentRoundDiagnostics(BaseModel):
    round_index: int
    latency_ms: float = 0.0
    used_llm: bool = False
    sufficient: bool = False
    fallback_used: bool = False
    input_evidence_count: int = 0
    verified_evidence_count: int = 0
    covered_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    next_search_goal: str | None = None
    confidence: float | None = None


class LlmCallCountsDiagnostics(BaseModel):
    evidence_assessment: int = 0
    answer_generation: int = 0
    answer_verification: int = 0
    answer_revision: int = 0
    image_selection: int = 0


class AnswerVerificationDiagnostics(BaseModel):
    latency_ms: float = 0.0
    triggered: bool = False
    passed: bool = True
    trigger_reasons: list[str] = Field(default_factory=list)
    unsupported_claim_count: int = 0
    missing_disclosure_count: int = 0
    citation_error_count: int = 0
    overstatement_count: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_disclosures: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    overstatements: list[str] = Field(default_factory=list)
    revision_required: bool = False
    revision_applied: bool = False
    revision_instruction: str | None = None
    used_llm: bool = False
    fallback_used: bool = False
    reason: str = ""
    proxy_hit: bool = False


class ImageSelectionDiagnostics(BaseModel):
    strategy: str = "none"
    used_llm: bool = False
    fallback_used: bool = False
    candidate_count: int = 0
    selected_count: int = 0
    latency_ms: float = 0.0
    reason: str = ""
    proxy_hit: bool = False


class ChatDiagnostics(BaseModel):
    total_latency_ms: float = 0.0
    retrieval_round: int = 0
    max_retrieval_rounds: int = 0
    stop_reason: str | None = None
    retrieval_rounds: list[RetrievalRoundDiagnostics] = Field(default_factory=list)
    assessment_rounds: list[AssessmentRoundDiagnostics] = Field(default_factory=list)
    llm_call_counts: LlmCallCountsDiagnostics = Field(default_factory=LlmCallCountsDiagnostics)
    outline_resolution: OutlineResolutionDiagnostics | None = None
    outline_context: OutlineContextDiagnostics | None = None
    route: RouteDiagnostics | None = None
    route_plan: RoutePlanDiagnostics | None = None
    task: TaskDiagnostics | None = None
    recall: RecallDiagnostics | None = None
    rerank: RerankDiagnostics | None = None
    verifier: VerifierDiagnostics | None = None
    answer_verification: AnswerVerificationDiagnostics | None = None
    image_selection: ImageSelectionDiagnostics | None = None
    llm: LlmDiagnostics | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    user_message: ChatMessage
    assistant_message: ChatMessage
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: ChatDiagnostics | None = None
