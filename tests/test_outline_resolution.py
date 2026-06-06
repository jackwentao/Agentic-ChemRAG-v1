from backend.app.retrieval.outline_resolution import OutlineReferenceResolver


def test_outline_resolver_rewrites_single_current_document_reference():
    resolution = OutlineReferenceResolver().resolve(
        question="它主要研究什么？",
        conversation_context="",
        document_ids=["doc-1"],
        scope="current_document",
    )

    assert resolution.decision == "single_document_outline"
    assert resolution.target_document_ids == ["doc-1"]
    assert resolution.needs_outline_retrieval is False
    assert "文档 doc-1" in resolution.rewritten_question
    assert any("用户问题包含论文指代" in fact for fact in resolution.basic_facts)


def test_outline_resolver_uses_recent_context_document_when_unique():
    resolution = OutlineReferenceResolver().resolve(
        question="这篇论文的大纲是什么？",
        conversation_context="最近完整对话：\n相关文档：doc-ctx",
        document_ids=None,
        scope="all_documents",
    )

    assert resolution.decision == "single_document_outline"
    assert resolution.target_document_ids == ["doc-ctx"]
    assert "文档 doc-ctx" in resolution.rewritten_question


def test_outline_resolver_enters_outline_retrieval_for_cross_document_question():
    resolution = OutlineReferenceResolver().resolve(
        question="哪些论文是讲卟啉相关的？",
        conversation_context="",
        document_ids=["doc-1", "doc-2"],
        scope="all_documents",
    )

    assert resolution.decision == "enter_outline_retrieval"
    assert resolution.needs_outline_retrieval is True
    assert resolution.target_document_ids == ["doc-1", "doc-2"]
