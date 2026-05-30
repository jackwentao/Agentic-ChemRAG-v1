from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.conversation import ConversationContext, ConversationMessage


@dataclass(frozen=True)
class ConversationContextPolicy:
    recent_turns: int = 3
    max_context_chars: int = 8000
    user_excerpt_chars: int = 160
    assistant_excerpt_chars: int = 240
    citation_limit_per_turn: int = 3
    limitation_limit_per_turn: int = 2


class ConversationContextService:
    def __init__(self, repository, policy: ConversationContextPolicy | None = None) -> None:
        self.repository = repository
        self.policy = policy or ConversationContextPolicy()

    def get_context_text(self, session_id: str | None) -> str:
        if not session_id:
            return ""
        context = self.repository.get_context(session_id)
        return context.text if context else ""

    def rebuild_context(self, session_id: str, owner_id: str) -> ConversationContext:
        messages = self.repository.list_messages(session_id, owner_id, limit=500)
        turns = _pair_turns(messages)
        recent_turns = turns[-self.policy.recent_turns :] if self.policy.recent_turns > 0 else []
        older_turns = turns[: max(0, len(turns) - len(recent_turns))]

        compressed_context = _format_compressed_turns(older_turns, self.policy)
        recent_context = _format_recent_turns(recent_turns)
        context = _fit_context(
            session_id=session_id,
            recent_context=recent_context,
            compressed_context=compressed_context,
            policy=self.policy,
        )
        return self.repository.upsert_context(context)


def _pair_turns(messages: list[ConversationMessage]) -> list[tuple[ConversationMessage, ConversationMessage | None]]:
    turns: list[tuple[ConversationMessage, ConversationMessage | None]] = []
    pending_user: ConversationMessage | None = None
    for message in messages:
        if message.role == "user":
            if pending_user is not None:
                turns.append((pending_user, None))
            pending_user = message
            continue
        if message.role == "assistant" and pending_user is not None:
            turns.append((pending_user, message))
            pending_user = None
    if pending_user is not None:
        turns.append((pending_user, None))
    return turns


def _format_recent_turns(turns: list[tuple[ConversationMessage, ConversationMessage | None]]) -> str:
    if not turns:
        return ""
    lines = ["最近完整对话："]
    for index, (user_message, assistant_message) in enumerate(turns, start=1):
        lines.append(f"[最近第 {index} 轮]")
        lines.append(f"用户问题：{_clean(user_message.content)}")
        if assistant_message is not None:
            lines.append(f"助手回答：{_clean(assistant_message.content)}")
            lines.extend(_format_structured_metadata(assistant_message, None))
    return "\n".join(lines)


def _format_compressed_turns(
    turns: list[tuple[ConversationMessage, ConversationMessage | None]],
    policy: ConversationContextPolicy,
) -> str:
    if not turns:
        return ""
    lines = ["较早对话压缩摘要："]
    for index, (user_message, assistant_message) in enumerate(turns, start=1):
        lines.append(f"[第 {index} 轮]")
        lines.append(f"用户问题：{_excerpt(user_message.content, policy.user_excerpt_chars)}")
        if assistant_message is not None:
            lines.append(f"助手结论：{_excerpt(assistant_message.content, policy.assistant_excerpt_chars)}")
            lines.extend(_format_structured_metadata(assistant_message, policy))
    return "\n".join(lines)


def _format_structured_metadata(
    message: ConversationMessage,
    policy: ConversationContextPolicy | None,
) -> list[str]:
    lines: list[str] = []
    if message.scope:
        lines.append(f"文档范围：{message.scope}")
    if message.document_ids:
        lines.append(f"相关文档：{', '.join(message.document_ids[:5])}")

    citation_limit = policy.citation_limit_per_turn if policy else 5
    citations = message.citations[:citation_limit]
    if citations:
        citation_lines = []
        for citation in citations:
            section = str(citation.get("section_title") or "未知章节")
            page_start = citation.get("page_start")
            page_end = citation.get("page_end")
            chunk_id = str(citation.get("chunk_id") or "")
            citation_lines.append(f"{section} / p.{page_start}-{page_end} / {chunk_id}")
        lines.append(f"引用证据：{'; '.join(citation_lines)}")

    limitation_limit = policy.limitation_limit_per_turn if policy else 5
    limitations = message.limitations[:limitation_limit]
    if limitations:
        lines.append(f"资料限制：{'; '.join(limitations)}")
    return lines


def _fit_context(
    *,
    session_id: str,
    recent_context: str,
    compressed_context: str,
    policy: ConversationContextPolicy,
) -> ConversationContext:
    compressed = compressed_context
    recent = recent_context
    while len(compressed) + len(recent) > policy.max_context_chars and compressed:
        compressed = _drop_oldest_compressed_turn(compressed)
    if len(compressed) + len(recent) > policy.max_context_chars:
        recent = _trim_recent_context(recent, policy.max_context_chars)
    return ConversationContext(
        session_id=session_id,
        recent_context=recent,
        compressed_context=compressed,
        context_char_count=len(compressed) + len(recent),
    )


def _drop_oldest_compressed_turn(context: str) -> str:
    lines = context.splitlines()
    if not lines or lines[0] != "较早对话压缩摘要：":
        return ""
    next_turn_index = next((index for index, line in enumerate(lines[2:], start=2) if line.startswith("[第 ")), None)
    if next_turn_index is None:
        return ""
    return "\n".join([lines[0], *lines[next_turn_index:]])


def _trim_recent_context(context: str, max_chars: int) -> str:
    heading = "最近完整对话："
    if not context.startswith(heading) or max_chars <= len(heading):
        return context[-max_chars:]
    tail_budget = max_chars - len(heading) - 1
    return f"{heading}\n{context[-tail_budget:]}"


def _excerpt(text: str, max_chars: int) -> str:
    cleaned = _clean(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _clean(text: str) -> str:
    return " ".join(text.split())
