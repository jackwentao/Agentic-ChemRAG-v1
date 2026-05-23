from typing import Literal

from pydantic import BaseModel


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


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    warnings: list[str] = []
