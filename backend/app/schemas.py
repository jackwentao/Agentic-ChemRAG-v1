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
    source_url: str
    section_title: str
    quote: str
    score: float
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
    proxy_hit: bool = False


class LlmDiagnostics(BaseModel):
    latency_ms: float = 0.0
    used_llm: bool = False
    answer_chars: int = 0
    evidence_count: int = 0
    citation_count: int = 0
    selected_image_count: int = 0
    proxy_hit: bool = False


class ChatDiagnostics(BaseModel):
    total_latency_ms: float = 0.0
    route: RouteDiagnostics | None = None
    recall: RecallDiagnostics | None = None
    rerank: RerankDiagnostics | None = None
    llm: LlmDiagnostics | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    warnings: list[str] = []
    diagnostics: ChatDiagnostics | None = None
