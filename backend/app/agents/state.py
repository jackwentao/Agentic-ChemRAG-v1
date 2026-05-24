from typing import TypedDict

from backend.app.domain.documents import Evidence, EvidenceImage


class RagGraphState(TypedDict, total=False):
    question: str
    document_ids: list[str] | None
    scope: str | None
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
    evidence: list[Evidence]
    verified_evidence: list[Evidence]
    selected_images: dict[str, list[EvidenceImage]]
    answer: str
    total_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    rerank_enabled: bool
    answer_latency_ms: float
    diagnostics: dict[str, object]
    warnings: list[str]
