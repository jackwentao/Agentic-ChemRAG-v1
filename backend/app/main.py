from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache
from hashlib import sha256
import logging
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.config import get_settings
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.providers.embeddings import EmbeddingProvider
from backend.app.providers.llm import LlmProvider, build_llm_provider
from backend.app.providers.rerankers import RerankerProvider, build_reranker_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.routing.query_router import QueryRouter
from backend.app.schemas import (
    BatchDocumentUploadItem,
    ChatRequest,
    ChatResponse,
    Citation,
    CitationImage,
    DocumentBatchUploadResponse,
    DocumentSummary,
    DocumentUploadResponse,
    TaskStatusResponse,
)
from backend.app.services.rag import RetrievalService
from backend.app.services.object_storage import build_object_storage, document_pdf_key
from backend.app.tasks.ingestion_tasks import ingest_document

settings = get_settings()
logger = logging.getLogger(__name__)
DEFAULT_OWNER_ID = "demo"


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_router_on_startup()
    warmup_reranker_on_startup()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider(settings)


@lru_cache(maxsize=1)
def _get_llm_provider() -> LlmProvider:
    return build_llm_provider(settings)


@lru_cache(maxsize=1)
def _get_query_router() -> QueryRouter:
    return QueryRouter(
        embedding_provider=_get_embedding_provider(),
        cache_dir=settings.router_cache_dir,
        cache_enabled=settings.router_cache_enabled,
        provider_fingerprint=_router_provider_fingerprint(),
    )


@lru_cache(maxsize=1)
def _get_reranker_provider() -> RerankerProvider | None:
    return build_reranker_provider(settings)


def warmup_router_on_startup() -> None:
    if not settings.router_warmup_on_startup:
        return
    try:
        status = _get_query_router().warmup()
        logger.info("Query router warmup finished: %s", status.to_dict())
    except Exception:
        logger.exception("Query router warmup failed; continuing without blocking startup.")


def warmup_reranker_on_startup() -> None:
    if not settings.reranker_enabled or not settings.reranker_warmup_on_startup:
        return
    try:
        reranker = _get_reranker_provider()
        if reranker is not None:
            reranker.warmup()
            logger.info("Reranker warmup finished.")
    except Exception:
        logger.exception("Reranker warmup failed; continuing startup but reranked retrieval may fail.")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    return _store_uploaded_pdf(file.filename, content, PgVectorRepository(settings.database_url))


@app.post("/api/documents/batch", response_model=DocumentBatchUploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> DocumentBatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    repository = PgVectorRepository(settings.database_url)
    results: list[BatchDocumentUploadItem] = []
    for file in files:
        filename = file.filename or "unnamed"
        if not filename.lower().endswith(".pdf"):
            results.append(BatchDocumentUploadItem(filename=filename, status="rejected", error="Only PDF files are supported."))
            continue
        try:
            response = _store_uploaded_pdf(filename, await file.read(), repository)
        except Exception as exc:
            results.append(BatchDocumentUploadItem(filename=filename, status="failed", error=str(exc)))
            continue
        results.append(
            BatchDocumentUploadItem(
                filename=filename,
                document_id=response.document_id,
                task_id=response.task_id,
                status=response.status,
                reused=response.reused,
                error=response.error,
            )
        )
    return DocumentBatchUploadResponse(results=results)


def _store_uploaded_pdf(
    filename: str,
    content: bytes,
    repository: PgVectorRepository,
) -> DocumentUploadResponse:
    file_hash = sha256(content).hexdigest()
    file_size = len(content)

    existing = repository.find_document_by_hash(DEFAULT_OWNER_ID, file_hash)
    if existing is None:
        existing = _find_existing_document_by_storage_hash(
            repository,
            file_hash,
            file_size,
            settings.upload_dir,
            filename,
        )

    if existing is not None:
        document_id = str(existing["document_id"])
        pdf_key = document_pdf_key(document_id)
        _store_pdf_object(pdf_key, content)
        repository.update_document_hash(document_id, file_hash, file_size, filename, pdf_object_key=pdf_key)
        if bool(existing["is_indexed"]):
            return DocumentUploadResponse(document_id=document_id, task_id=None, status="already_indexed", reused=True)
        indexing_status = str(existing.get("indexing_status") or "pending")
        if indexing_status in {"pending", "running"}:
            return DocumentUploadResponse(document_id=document_id, task_id=None, status=indexing_status, reused=True)
        target = settings.upload_dir / f"{document_id}.pdf"
        if not target.exists():
            target.write_bytes(content)
        repository.update_document_indexing_status(document_id, "pending")
        return _enqueue_ingestion(repository, document_id, target, filename, file_hash, file_size, reused=True)

    document_id = uuid4().hex
    target = settings.upload_dir / f"{document_id}.pdf"
    target.write_bytes(content)
    pdf_key = document_pdf_key(document_id)
    _store_pdf_object(pdf_key, content)
    repository.add_document(
        document_id,
        filename,
        owner_id=DEFAULT_OWNER_ID,
        file_hash=file_hash,
        file_size=file_size,
        original_filename=filename,
        indexing_status="pending",
        pdf_object_key=pdf_key,
    )
    return _enqueue_ingestion(repository, document_id, target, filename, file_hash, file_size)


def _enqueue_ingestion(
    repository: PgVectorRepository,
    document_id: str,
    target: Path,
    filename: str,
    file_hash: str,
    file_size: int,
    reused: bool = False,
) -> DocumentUploadResponse:
    try:
        task = ingest_document.delay(document_id, str(target), filename, file_hash, file_size)
    except Exception as exc:
        error = f"Failed to enqueue ingestion task: {exc}"
        repository.update_document_indexing_status(document_id, "failed", error=error)
        return DocumentUploadResponse(document_id=document_id, task_id=None, status="failed", reused=reused, error=error)
    return DocumentUploadResponse(document_id=document_id, task_id=task.id, status="pending", reused=reused)


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str) -> TaskStatusResponse:
    try:
        result = ingest_document.AsyncResult(task_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task status backend is unavailable: {exc}") from exc
    error = str(result.result) if result.failed() else None
    value = result.result if result.successful() and isinstance(result.result, int) else None
    return TaskStatusResponse(task_id=task_id, status=result.status.lower(), result=value, error=error)


@app.post("/api/maintenance/indexing/mark-stale-failed")
def mark_stale_indexing_failed(timeout_minutes: int = Query(default=60, ge=1, le=1440)) -> dict[str, object]:
    document_ids = PgVectorRepository(settings.database_url).mark_stale_indexing_documents_failed(
        owner_id=DEFAULT_OWNER_ID,
        stale_after_minutes=timeout_minutes,
    )
    return {"updated": len(document_ids), "document_ids": document_ids}


@app.get("/api/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    rows = PgVectorRepository(settings.database_url).list_documents(owner_id=DEFAULT_OWNER_ID)
    return [
        DocumentSummary(
            document_id=str(row["document_id"]),
            filename=str(row["filename"]),
            created_at=str(row["created_at"]),
            chunk_count=int(row["chunk_count"]),
            is_indexed=bool(row["is_indexed"]),
            file_hash=str(row["file_hash"]) if row["file_hash"] else None,
            indexing_status=str(row["indexing_status"]),
            indexing_error=str(row["indexing_error"]) if row["indexing_error"] else None,
        )
        for row in rows
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    repository = PgVectorRepository(settings.database_url)
    document_ids = _resolve_chat_document_ids(request, repository)
    graph = MultiAgentRagGraph(
        retrieval_service=RetrievalService(
            repository=repository,
            embedding_provider=_get_embedding_provider(),
            reranker_provider=_get_reranker_provider(),
            reranker_enabled=settings.reranker_enabled,
            reranker_chunk_candidates=settings.reranker_chunk_candidates,
            reranker_figure_candidates=settings.reranker_figure_candidates,
        ),
        llm_provider=_get_llm_provider(),
        query_router=_get_query_router(),
    )
    try:
        state = graph.invoke(request.question, document_ids=document_ids, scope=request.scope)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    evidence = state.get("verified_evidence", [])
    selected_images = state.get("selected_images", {})
    return ChatResponse(
        answer=state.get("answer", ""),
        warnings=state.get("warnings", []),
        citations=[
            Citation(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                page_start=item.page_start,
                page_end=item.page_end,
                source_url=_pdf_page_url(item.document_id, item.page_start),
                section_title=item.section_title,
                quote=item.text[:500],
                score=item.score,
                image_ids=[image.image_id for image in selected_images.get(item.chunk_id, [])],
                images=[
                    CitationImage(
                        image_id=image.image_id,
                        url=f"/api/images/{image.image_id}",
                        source_url=_pdf_page_url(item.document_id, image.page),
                        page=image.page,
                        width=image.width,
                        height=image.height,
                        label=image.label,
                        caption=image.caption,
                        reason=image.reason,
                    )
                    for image in selected_images.get(item.chunk_id, [])
                ],
            )
            for item in evidence
        ],
    )


def _resolve_chat_document_ids(request: ChatRequest, repository: PgVectorRepository) -> list[str]:
    if request.scope == "current_document":
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="document_ids is required when scope is current_document.")
        return request.document_ids
    document_ids = repository.list_indexed_document_ids(owner_id=DEFAULT_OWNER_ID)
    if not document_ids:
        raise HTTPException(status_code=400, detail="No indexed documents are available for this user.")
    return document_ids


def _find_existing_document_by_storage_hash(
    repository: PgVectorRepository,
    file_hash: str,
    file_size: int,
    upload_dir: Path,
    original_filename: str | None = None,
) -> dict[str, object] | None:
    for document in repository.list_documents(owner_id=DEFAULT_OWNER_ID):
        document_path = upload_dir / f"{document['document_id']}.pdf"
        if not document_path.exists():
            continue
        if document_path.stat().st_size != file_size:
            continue
        digest = sha256(document_path.read_bytes()).hexdigest()
        if digest == file_hash:
            document_id = str(document["document_id"])
            pdf_key = document_pdf_key(document_id)
            repository.update_document_hash(document_id, file_hash, file_size, original_filename, pdf_object_key=pdf_key)
            document["file_hash"] = file_hash
            if original_filename:
                document["filename"] = original_filename
            return document
    return None


def _store_pdf_object(pdf_key: str, content: bytes) -> None:
    build_object_storage(settings).put_bytes(pdf_key, content, content_type="application/pdf")


def _pdf_page_url(document_id: str, page: int) -> str:
    return f"/api/documents/{document_id}/pdf#page={page}"


def _router_provider_fingerprint() -> str:
    return (
        f"{settings.embedding_provider}:"
        f"{settings.embedding_model}:"
        f"{settings.embedding_device}:"
        f"{settings.embedding_dim}:"
        f"local_files_only={settings.embedding_local_files_only}"
    )


@app.get("/api/documents/{document_id}/pdf")
def get_document_pdf(document_id: str):
    repository = PgVectorRepository(settings.database_url)
    document = repository.get_document_asset(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    object_key = str(document.get("pdf_object_key") or document_pdf_key(document_id))
    filename = str(document.get("filename") or f"{document_id}.pdf")
    storage = build_object_storage(settings)
    if storage.exists(object_key):
        stream = storage.open(object_key)
        return StreamingResponse(
            stream,
            media_type="application/pdf",
            headers={"Content-Disposition": _inline_content_disposition(filename)},
            background=BackgroundTask(stream.close),
        )

    local_path = settings.upload_dir / f"{document_id}.pdf"
    if local_path.exists():
        return FileResponse(local_path, media_type="application/pdf", filename=filename, content_disposition_type="inline")
    raise HTTPException(status_code=404, detail="PDF not found.")


def _inline_content_disposition(filename: str) -> str:
    ascii_name = "".join(char if ord(char) < 128 and char not in {'"', "\\"} else "_" for char in filename)
    return f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    repository = PgVectorRepository(settings.database_url)
    image = repository.get_image_asset(image_id)
    if image is not None:
        object_key = str(image["path"])
        storage = build_object_storage(settings)
        if storage.exists(object_key):
            stream = storage.open(object_key)
            extension = str(image.get("extension") or "png")
            return StreamingResponse(
                stream,
                media_type=f"image/{extension}",
                background=BackgroundTask(stream.close),
            )
    for path in settings.image_dir.glob(f"{image_id}.*"):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")
