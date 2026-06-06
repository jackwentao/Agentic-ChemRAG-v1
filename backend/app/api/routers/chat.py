import json
import logging
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse

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
    ChatSessionPinRequest,
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
    try:
        return _run_chat_request(
            request,
            repository=repository,
            graph=graph,
            history_service=history_service,
            context_service=context_service,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    graph: Annotated[MultiAgentRagGraph, Depends(get_rag_graph)],
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
    context_service: Annotated[ConversationContextService, Depends(get_conversation_context_service)],
) -> StreamingResponse:
    events: Queue[tuple[str, object]] = Queue()

    def on_answer_delta(chunk: str) -> None:
        events.put(("answer_delta", {"delta": chunk}))

    def on_stage(stage: str, label: str) -> None:
        events.put(("stage", {"stage": stage, "label": label}))

    def run() -> None:
        try:
            response = _run_chat_request(
                request,
                repository=repository,
                graph=graph,
                history_service=history_service,
                context_service=context_service,
                on_answer_delta=on_answer_delta,
                on_stage=on_stage,
            )
            events.put(("final", response.model_dump(mode="json")))
        except TimeoutError as exc:
            events.put(("error", {"status_code": 504, "detail": str(exc)}))
        except HTTPException as exc:
            events.put(("error", {"status_code": exc.status_code, "detail": exc.detail}))
        except Exception:
            logger.exception("Chat stream failed.")
            events.put(("error", {"status_code": 500, "detail": "Chat stream failed."}))
        finally:
            events.put(("done", {}))

    def event_generator():
        worker = Thread(target=run, daemon=True)
        worker.start()
        yield _sse_event("status", {"stage": "started"})
        while True:
            event, payload = events.get()
            if event == "done":
                yield _sse_event("done", {})
                break
            yield _sse_event(event, payload)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/sessions", response_model=ChatSessionsResponse)
def list_chat_sessions(
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> ChatSessionsResponse:
    return ChatSessionsResponse(
        sessions=[
            _to_chat_session_summary(session)
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
        session=_to_chat_session_summary(session),
        messages=[_to_chat_message(message) for message in messages],
    )


@router.patch("/chat/sessions/{session_id}/pin", response_model=ChatSessionSummary)
def set_chat_session_pin(
    session_id: str,
    request: ChatSessionPinRequest,
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> ChatSessionSummary:
    session = history_service.set_session_pinned(session_id, request.pinned)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return _to_chat_session_summary(session)


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat_session(
    session_id: str,
    history_service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> Response:
    if not history_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _run_chat_request(
    request: ChatRequest,
    *,
    repository: PgVectorRepository,
    graph: MultiAgentRagGraph,
    history_service: ConversationHistoryService,
    context_service: ConversationContextService,
    on_answer_delta=None,
    on_stage=None,
) -> ChatResponse:
    document_ids = resolve_chat_document_ids(request, repository)
    session = history_service.ensure_session(request.session_id, title_seed=request.question)
    conversation_context = context_service.get_context_text(session.session_id)
    started_at = perf_counter()
    state = _invoke_graph(
        graph,
        request.question,
        document_ids=document_ids,
        scope=request.scope,
        session_id=session.session_id,
        owner_id=DEFAULT_OWNER_ID,
        conversation_context=conversation_context,
        answer_delta_callback=on_answer_delta,
        stage_callback=on_stage,
    )
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


def _sse_event(event: str, payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


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


def _to_chat_session_summary(session) -> ChatSessionSummary:
    return ChatSessionSummary(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        pinned=session.pinned,
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
    answer_delta_callback=None,
    stage_callback=None,
):
    kwargs = {
        "document_ids": document_ids,
        "scope": scope,
        "session_id": session_id,
        "owner_id": owner_id,
        "conversation_context": conversation_context,
    }
    if answer_delta_callback is not None:
        kwargs["answer_delta_callback"] = answer_delta_callback
    if stage_callback is not None:
        kwargs["stage_callback"] = stage_callback
    try:
        return graph.invoke(question, **kwargs)
    except TypeError as exc:
        if "unexpected keyword" not in str(exc):
            raise
        for key in ["stage_callback", "owner_id", "session_id", "conversation_context", "answer_delta_callback"]:
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
