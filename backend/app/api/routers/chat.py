import json
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.agents.rag_graph import MultiAgentRagGraph
from backend.app.api.dependencies import (
    DEFAULT_OWNER_ID,
    get_conversation_context_service,
    get_conversation_history_service,
    get_rag_graph,
    get_repository,
)
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.schemas import (
    ChatDiagnostics,
    ChatMessage,
    ChatMessagesResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionSummary,
    ChatSessionsResponse,
    Citation,
    CitationImage,
)
from backend.app.services.conversation_context import ConversationContextService
from backend.app.services.conversation_history import ConversationHistoryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    graph: Annotated[MultiAgentRagGraph, Depends(get_rag_graph)],
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
    context_service: Annotated[ConversationContextService, Depends(get_conversation_context_service)],
) -> ChatResponse:
    # chat 路由本身尽量保持“薄”：
    # 只负责解析 scope、调用 graph，并把内部状态映射为对外响应模型。
    document_ids = resolve_chat_document_ids(request, repository)
    session = history_service.ensure_session(request.session_id, title_seed=request.question)
    conversation_context = context_service.get_context_text(session.session_id)
    started_at = perf_counter()
    try:
        state = _invoke_graph(
            graph,
            request.question,
            document_ids=document_ids,
            scope=request.scope,
            session_id=session.session_id,
            owner_id=DEFAULT_OWNER_ID,
            conversation_context=conversation_context,
        )
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
    citations = _build_citations(evidence, selected_images)
    citation_images = _flatten_citation_images(citations)
    user_message = history_service.append_user_message(
        session_id=session.session_id,
        content=request.question,
        scope=request.scope,
        document_ids=document_ids,
    )
    assistant_message = history_service.append_assistant_message(
        session_id=session.session_id,
        content=state.get("answer", ""),
        scope=request.scope,
        document_ids=document_ids,
        citations=[citation.model_dump() for citation in citations],
        images=[image.model_dump() for image in citation_images],
        limitations=state.get("limitations", []),
        diagnostics_summary=_diagnostics_summary(diagnostics_payload),
    )
    context_service.rebuild_context(session.session_id, DEFAULT_OWNER_ID)
    return ChatResponse(
        session_id=session.session_id,
        answer=state.get("answer", ""),
        citations=citations,
        user_message=_to_chat_message(user_message),
        assistant_message=_to_chat_message(assistant_message),
        limitations=state.get("limitations", []),
        warnings=state.get("warnings", []),
        diagnostics=ChatDiagnostics.model_validate(diagnostics_payload) if diagnostics_payload else None,
    )


@router.get("/chat/sessions", response_model=ChatSessionsResponse)
def list_chat_sessions(
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> ChatSessionsResponse:
    return ChatSessionsResponse(
        sessions=[
            ChatSessionSummary(
                session_id=session.session_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
            for session in history_service.list_sessions()
        ]
    )


@router.get("/chat/sessions/{session_id}/messages", response_model=ChatMessagesResponse)
def list_chat_messages(
    session_id: str,
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> ChatMessagesResponse:
    session = history_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    messages = history_service.list_messages(session.session_id)
    return ChatMessagesResponse(
        session=ChatSessionSummary(
            session_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        ),
        messages=[_to_chat_message(message) for message in messages],
    )


def _to_chat_message(message) -> ChatMessage:
    return ChatMessage(
        message_id=message.message_id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        scope=message.scope,
        document_ids=message.document_ids,
        citations=message.citations,
        images=message.images,
        limitations=message.limitations,
        diagnostics_summary=message.diagnostics_summary,
        created_at=message.created_at,
    )


def resolve_chat_document_ids(request: ChatRequest, repository) -> list[str]:
    # current_document 模式由前端显式提供 document_ids；
    # all_documents 模式则回退到当前用户全部已建索引文档。
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


def _invoke_graph(
    graph,
    question: str,
    *,
    document_ids: list[str],
    scope: str,
    session_id: str,
    owner_id: str,
    conversation_context: str,
):
    kwargs = {
        "document_ids": document_ids,
        "scope": scope,
        "session_id": session_id,
        "owner_id": owner_id,
        "conversation_context": conversation_context,
    }
    try:
        return graph.invoke(question, **kwargs)
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        for key in ["owner_id", "session_id", "conversation_context"]:
            kwargs.pop(key, None)
            try:
                return graph.invoke(question, **kwargs)
            except TypeError as retry_exc:
                if "unexpected keyword" not in str(retry_exc):
                    raise
                exc = retry_exc
        raise exc


def _build_citations(evidence, selected_images) -> list[Citation]:
    return [
        Citation(
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            page_start=item.page_start,
            page_end=item.page_end,
            source_url=None if item.source_scope == "document_level" else pdf_page_url(item.document_id, item.page_start),
            section_title=item.section_title,
            quote=item.text[:500],
            score=item.score,
            evidence_type=item.evidence_type,
            source_scope=item.source_scope,
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
    ]


def _flatten_citation_images(citations: list[Citation]) -> list[CitationImage]:
    seen: set[str] = set()
    images: list[CitationImage] = []
    for citation in citations:
        for image in citation.images:
            if image.image_id in seen:
                continue
            seen.add(image.image_id)
            images.append(image)
    return images


def _diagnostics_summary(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        key: diagnostics.get(key)
        for key in ["route", "retrieval_round", "max_retrieval_rounds", "stop_reason", "answer_verification"]
        if key in diagnostics
    }
