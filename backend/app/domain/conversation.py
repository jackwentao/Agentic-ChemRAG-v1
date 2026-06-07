from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversationSession:
    session_id: str
    owner_id: str
    title: str
    created_at: str
    updated_at: str
    pinned: bool = False


@dataclass(frozen=True)
class ConversationMessage:
    message_id: str
    session_id: str
    role: str
    content: str
    scope: str | None = None
    document_ids: list[str] = field(default_factory=list)
    citations: list[dict[str, object]] = field(default_factory=list)
    images: list[dict[str, object]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    diagnostics_summary: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(frozen=True)
class ConversationContext:
    session_id: str
    recent_context: str = ""
    compressed_context: str = ""
    context_char_count: int = 0
    updated_at: str | None = None

    @property
    def text(self) -> str:
        parts = [self.compressed_context.strip(), self.recent_context.strip()]
        return "\n\n".join(part for part in parts if part)
