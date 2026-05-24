import json
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.api.dependencies import DEFAULT_OWNER_ID, get_rag_graph, get_repository
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.schemas import ChatDiagnostics, ChatRequest, ChatResponse, Citation, CitationImage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    graph: Annotated[MultiAgentRagGraph, Depends(get_rag_graph)],
) -> ChatResponse:
    document_ids = resolve_chat_document_ids(request, repository)
    started_at = perf_counter()
    try:
        state = graph.invoke(request.question, document_ids=document_ids, scope=request.scope)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    evidence = state.get("verified_evidence", [])
    selected_images = state.get("selected_images", {})
    diagnostics_payload = dict(state.get("diagnostics", {}))
    diagnostics_payload.setdefault("total_latency_ms", (perf_counter() - started_at) * 1000)
    logger.info(
        "Chat diagnostics: %s",
        json.dumps(
            {
                "scope": request.scope,
                "route": state.get("route"),
                "document_count": len(document_ids),
                "diagnostics": diagnostics_payload,
            },
            ensure_ascii=False,
        ),
    )
    return ChatResponse(
        answer=state.get("answer", ""),
        warnings=state.get("warnings", []),
        citations=[
            Citation(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                page_start=item.page_start,
                page_end=item.page_end,
                source_url=pdf_page_url(item.document_id, item.page_start),
                section_title=item.section_title,
                quote=item.text[:500],
                score=item.score,
                image_ids=[image.image_id for image in selected_images.get(item.chunk_id, [])],
                images=[
                    CitationImage(
                        image_id=image.image_id,
                        url=f"/api/images/{image.image_id}",
                        source_url=pdf_page_url(item.document_id, image.page),
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
        diagnostics=ChatDiagnostics.model_validate(diagnostics_payload) if diagnostics_payload else None,
    )


def resolve_chat_document_ids(request: ChatRequest, repository) -> list[str]:
    if request.scope == "current_document":
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="document_ids is required when scope is current_document.")
        return request.document_ids
    document_ids = repository.list_indexed_document_ids(owner_id=DEFAULT_OWNER_ID)
    if not document_ids:
        raise HTTPException(status_code=400, detail="No indexed documents are available for this user.")
    return document_ids


def pdf_page_url(document_id: str, page: int) -> str:
    return f"/api/documents/{document_id}/pdf#page={page}"
