from backend.app.repositories.conversation import InMemoryConversationRepository
from backend.app.services.conversation_context import ConversationContextPolicy, ConversationContextService
from backend.app.services.conversation_history import ConversationHistoryService


def test_conversation_context_keeps_recent_turns_and_compresses_older_turns():
    repository = InMemoryConversationRepository()
    history = ConversationHistoryService(repository)
    context_service = ConversationContextService(
        repository,
        policy=ConversationContextPolicy(recent_turns=1, max_context_chars=2000, user_excerpt_chars=12, assistant_excerpt_chars=16),
    )
    session = history.ensure_session(None, title_seed="第一轮问题")

    for index in range(3):
        history.append_user_message(
            session_id=session.session_id,
            content=f"用户第 {index} 轮提出很长的问题内容，用来测试截断策略。",
            scope="current_document",
            document_ids=["doc1"],
        )
        history.append_assistant_message(
            session_id=session.session_id,
            content=f"助手第 {index} 轮给出很长的回答内容，用来测试确定性压缩。",
            scope="current_document",
            document_ids=["doc1"],
            citations=[
                {
                    "section_title": "Results",
                    "page_start": 2,
                    "page_end": 3,
                    "chunk_id": f"c{index}",
                }
            ],
            images=[],
            limitations=["资料缺口：稳定性"],
        )

    context = context_service.rebuild_context(session.session_id, "demo")

    assert "较早对话压缩摘要" in context.compressed_context
    assert "最近完整对话" in context.recent_context
    assert "用户第 2 轮提出很长的问题内容" in context.recent_context
    assert "用户第 0 轮提出很长的..." in context.compressed_context
    assert "引用证据：Results / p.2-3 / c0" in context.compressed_context
    assert "资料限制：资料缺口：稳定性" in context.compressed_context


def test_conversation_context_limits_total_context_size_without_llm():
    repository = InMemoryConversationRepository()
    history = ConversationHistoryService(repository)
    context_service = ConversationContextService(
        repository,
        policy=ConversationContextPolicy(recent_turns=1, max_context_chars=180, user_excerpt_chars=80, assistant_excerpt_chars=80),
    )
    session = history.ensure_session(None, title_seed="压缩测试")

    for index in range(6):
        history.append_user_message(
            session_id=session.session_id,
            content=f"用户问题 {index} " + "长文本" * 20,
            scope="all_documents",
            document_ids=["doc1"],
        )
        history.append_assistant_message(
            session_id=session.session_id,
            content=f"助手回答 {index} " + "长文本" * 20,
            scope="all_documents",
            document_ids=["doc1"],
            citations=[],
            images=[],
            limitations=[],
        )

    context = context_service.rebuild_context(session.session_id, "demo")

    assert context.context_char_count <= 180
    assert "最近完整对话" in context.text
