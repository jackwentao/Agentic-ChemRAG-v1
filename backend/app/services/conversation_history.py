from __future__ import annotations

from uuid import uuid4

from backend.app.domain.conversation import ConversationMessage, ConversationSession


class ConversationHistoryService:
    def __init__(self, repository, owner_id: str = "demo") -> None:
        self.repository = repository
        self.owner_id = owner_id

    def ensure_session(self, session_id: str | None, title_seed: str | None = None) -> ConversationSession:
        if session_id:
            session = self.repository.get_session(session_id, self.owner_id)
            if session is not None:
                return session
        new_session_id = session_id or str(uuid4())
        title = _build_title(title_seed)
        return self.repository.create_session(new_session_id, self.owner_id, title)

    def list_sessions(self, limit: int = 50) -> list[ConversationSession]:
        return self.repository.list_sessions(self.owner_id, limit=limit)

    def get_session(self, session_id: str) -> ConversationSession | None:
        return self.repository.get_session(session_id, self.owner_id)

    def set_session_pinned(self, session_id: str, pinned: bool) -> ConversationSession | None:
        return self.repository.set_session_pinned(session_id, self.owner_id, pinned)

    def delete_session(self, session_id: str) -> bool:
        return self.repository.delete_session(session_id, self.owner_id)

    def list_messages(self, session_id: str, limit: int = 100) -> list[ConversationMessage]:
        return self.repository.list_messages(session_id, self.owner_id, limit=limit)

    def append_user_message(
        self,
        *,
        session_id: str,
        content: str,
        scope: str | None,
        document_ids: list[str],
    ) -> ConversationMessage:
        return self.repository.add_message(
            ConversationMessage(
                message_id=str(uuid4()),
                session_id=session_id,
                role="user",
                content=content,
                scope=scope,
                document_ids=list(document_ids),
            )
        )

    def append_assistant_message(
        self,
        *,
        session_id: str,
        content: str,
        scope: str | None,
        document_ids: list[str],
        citations: list[dict[str, object]],
        images: list[dict[str, object]],
        limitations: list[str],
        diagnostics_summary: dict[str, object] | None = None,
    ) -> ConversationMessage:
        message = self.repository.add_message(
            ConversationMessage(
                message_id=str(uuid4()),
                session_id=session_id,
                role="assistant",
                content=content,
                scope=scope,
                document_ids=list(document_ids),
                citations=list(citations),
                images=list(images),
                limitations=list(limitations),
                diagnostics_summary=dict(diagnostics_summary or {}),
            )
        )
        self.repository.touch_session(session_id, self.owner_id)
        return message


def _build_title(seed: str | None) -> str:
    text = " ".join((seed or "新对话").split())
    return text[:32] or "新对话"
