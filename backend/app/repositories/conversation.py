from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.domain.conversation import ConversationContext, ConversationMessage, ConversationSession


class PgConversationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_session(self, session_id: str, owner_id: str, title: str) -> ConversationSession:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO chat_sessions (session_id, owner_id, title)
                    VALUES (:session_id, :owner_id, :title)
                    ON CONFLICT (session_id) DO UPDATE SET
                        updated_at = chat_sessions.updated_at
                    RETURNING session_id, owner_id, title, created_at, updated_at
                    """
                ),
                {"session_id": session_id, "owner_id": owner_id, "title": title},
            ).mappings().first()
        return _session_from_row(dict(row))

    def touch_session(self, session_id: str, owner_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE chat_sessions
                    SET updated_at = now()
                    WHERE session_id = :session_id
                      AND owner_id = :owner_id
                    """
                ),
                {"session_id": session_id, "owner_id": owner_id},
            )

    def get_session(self, session_id: str, owner_id: str) -> ConversationSession | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT session_id, owner_id, title, created_at, updated_at
                    FROM chat_sessions
                    WHERE session_id = :session_id
                      AND owner_id = :owner_id
                    LIMIT 1
                    """
                ),
                {"session_id": session_id, "owner_id": owner_id},
            ).mappings().first()
        return _session_from_row(dict(row)) if row else None

    def list_sessions(self, owner_id: str, limit: int = 50) -> list[ConversationSession]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT session_id, owner_id, title, created_at, updated_at
                    FROM chat_sessions
                    WHERE owner_id = :owner_id
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT :limit
                    """
                ),
                {"owner_id": owner_id, "limit": limit},
            ).mappings().all()
        return [_session_from_row(dict(row)) for row in rows]

    def add_message(self, message: ConversationMessage) -> ConversationMessage:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO chat_messages (
                        message_id, session_id, role, content, scope, document_ids,
                        citations_json, images_json, limitations_json, diagnostics_summary_json
                    )
                    VALUES (
                        :message_id, :session_id, :role, :content, :scope, CAST(:document_ids AS jsonb),
                        CAST(:citations AS jsonb), CAST(:images AS jsonb), CAST(:limitations AS jsonb),
                        CAST(:diagnostics_summary AS jsonb)
                    )
                    RETURNING message_id, session_id, role, content, scope, document_ids,
                              citations_json, images_json, limitations_json,
                              diagnostics_summary_json, created_at
                    """
                ),
                _message_params(message),
            ).mappings().first()
        return _message_from_row(dict(row))

    def list_messages(self, session_id: str, owner_id: str, limit: int = 100) -> list[ConversationMessage]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT message_id, chat_messages.session_id, role, content, scope, document_ids,
                           citations_json, images_json, limitations_json,
                           diagnostics_summary_json, chat_messages.created_at
                    FROM chat_messages
                    JOIN chat_sessions ON chat_sessions.session_id = chat_messages.session_id
                    WHERE chat_messages.session_id = :session_id
                      AND chat_sessions.owner_id = :owner_id
                    ORDER BY chat_messages.created_at ASC
                    LIMIT :limit
                    """
                ),
                {"session_id": session_id, "owner_id": owner_id, "limit": limit},
            ).mappings().all()
        return [_message_from_row(dict(row)) for row in rows]

    def get_context(self, session_id: str) -> ConversationContext | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT session_id, recent_context, compressed_context,
                           context_char_count, updated_at
                    FROM chat_contexts
                    WHERE session_id = :session_id
                    LIMIT 1
                    """
                ),
                {"session_id": session_id},
            ).mappings().first()
        return _context_from_row(dict(row)) if row else None

    def upsert_context(self, context: ConversationContext) -> ConversationContext:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO chat_contexts (
                        session_id, recent_context, compressed_context, context_char_count, updated_at
                    )
                    VALUES (
                        :session_id, :recent_context, :compressed_context, :context_char_count, now()
                    )
                    ON CONFLICT (session_id) DO UPDATE SET
                        recent_context = EXCLUDED.recent_context,
                        compressed_context = EXCLUDED.compressed_context,
                        context_char_count = EXCLUDED.context_char_count,
                        updated_at = now()
                    RETURNING session_id, recent_context, compressed_context,
                              context_char_count, updated_at
                    """
                ),
                {
                    "session_id": context.session_id,
                    "recent_context": context.recent_context,
                    "compressed_context": context.compressed_context,
                    "context_char_count": context.context_char_count,
                },
            ).mappings().first()
        return _context_from_row(dict(row))


class InMemoryConversationRepository:
    def __init__(self) -> None:
        self.sessions: dict[str, ConversationSession] = {}
        self.messages: dict[str, list[ConversationMessage]] = {}
        self.contexts: dict[str, ConversationContext] = {}

    def create_session(self, session_id: str, owner_id: str, title: str) -> ConversationSession:
        now = _now_iso()
        session = self.sessions.get(session_id)
        if session is None:
            session = ConversationSession(session_id=session_id, owner_id=owner_id, title=title, created_at=now, updated_at=now)
            self.sessions[session_id] = session
        return session

    def touch_session(self, session_id: str, owner_id: str) -> None:
        session = self.sessions.get(session_id)
        if session and session.owner_id == owner_id:
            self.sessions[session_id] = ConversationSession(
                session_id=session.session_id,
                owner_id=session.owner_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=_now_iso(),
            )

    def get_session(self, session_id: str, owner_id: str) -> ConversationSession | None:
        session = self.sessions.get(session_id)
        return session if session and session.owner_id == owner_id else None

    def list_sessions(self, owner_id: str, limit: int = 50) -> list[ConversationSession]:
        sessions = [session for session in self.sessions.values() if session.owner_id == owner_id]
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions[:limit]

    def add_message(self, message: ConversationMessage) -> ConversationMessage:
        stored = ConversationMessage(
            message_id=message.message_id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            scope=message.scope,
            document_ids=list(message.document_ids),
            citations=list(message.citations),
            images=list(message.images),
            limitations=list(message.limitations),
            diagnostics_summary=dict(message.diagnostics_summary),
            created_at=_now_iso(),
        )
        self.messages.setdefault(message.session_id, []).append(stored)
        return stored

    def list_messages(self, session_id: str, owner_id: str, limit: int = 100) -> list[ConversationMessage]:
        session = self.get_session(session_id, owner_id)
        if session is None:
            return []
        return list(self.messages.get(session_id, []))[:limit]

    def get_context(self, session_id: str) -> ConversationContext | None:
        return self.contexts.get(session_id)

    def upsert_context(self, context: ConversationContext) -> ConversationContext:
        stored = ConversationContext(
            session_id=context.session_id,
            recent_context=context.recent_context,
            compressed_context=context.compressed_context,
            context_char_count=context.context_char_count,
            updated_at=_now_iso(),
        )
        self.contexts[context.session_id] = stored
        return stored


def _message_params(message: ConversationMessage) -> dict[str, object]:
    return {
        "message_id": message.message_id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "scope": message.scope,
        "document_ids": json.dumps(message.document_ids, ensure_ascii=False),
        "citations": json.dumps(message.citations, ensure_ascii=False),
        "images": json.dumps(message.images, ensure_ascii=False),
        "limitations": json.dumps(message.limitations, ensure_ascii=False),
        "diagnostics_summary": json.dumps(message.diagnostics_summary, ensure_ascii=False),
    }


def _session_from_row(row: dict[str, object]) -> ConversationSession:
    return ConversationSession(
        session_id=str(row["session_id"]),
        owner_id=str(row["owner_id"]),
        title=str(row["title"]),
        created_at=_stringify_time(row["created_at"]),
        updated_at=_stringify_time(row["updated_at"]),
    )


def _message_from_row(row: dict[str, object]) -> ConversationMessage:
    return ConversationMessage(
        message_id=str(row["message_id"]),
        session_id=str(row["session_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        scope=str(row["scope"]) if row.get("scope") is not None else None,
        document_ids=[str(item) for item in _json_list(row.get("document_ids"))],
        citations=[dict(item) for item in _json_list(row.get("citations_json")) if isinstance(item, dict)],
        images=[dict(item) for item in _json_list(row.get("images_json")) if isinstance(item, dict)],
        limitations=[str(item) for item in _json_list(row.get("limitations_json"))],
        diagnostics_summary=_json_dict(row.get("diagnostics_summary_json")),
        created_at=_stringify_time(row.get("created_at")),
    )


def _context_from_row(row: dict[str, object]) -> ConversationContext:
    return ConversationContext(
        session_id=str(row["session_id"]),
        recent_context=str(row.get("recent_context") or ""),
        compressed_context=str(row.get("compressed_context") or ""),
        context_char_count=int(row.get("context_char_count") or 0),
        updated_at=_stringify_time(row.get("updated_at")),
    )


def _stringify_time(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
