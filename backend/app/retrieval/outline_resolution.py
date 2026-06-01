from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


OutlineResolutionDecision = Literal["pass_through", "enter_outline_retrieval", "single_document_outline"]


@dataclass(frozen=True)
class OutlineReferenceResolution:
    # 大纲检索前置层的稳定输出，供 planner 和诊断统一消费。
    original_question: str
    rewritten_question: str
    basic_facts: list[str] = field(default_factory=list)
    decision: OutlineResolutionDecision = "pass_through"
    target_document_ids: list[str] | None = None
    needs_outline_retrieval: bool = False
    reason: str = ""


class OutlineReferenceResolver:
    """在大纲/画像检索前消解“这篇、它、该论文”等指代。"""

    def resolve(
        self,
        *,
        question: str,
        conversation_context: str | None,
        document_ids: list[str] | None,
        scope: str | None,
    ) -> OutlineReferenceResolution:
        normalized_question = _normalize(question)
        current_scope = scope or ("current_document" if document_ids else "all_documents")
        context_document_ids = _document_ids_from_context(conversation_context or "")
        candidate_document_ids = _dedupe(document_ids or context_document_ids)
        basic_facts = _basic_facts(
            scope=current_scope,
            document_ids=document_ids,
            context_document_ids=context_document_ids,
            question=normalized_question,
        )

        if not _is_outline_intent(normalized_question):
            return OutlineReferenceResolution(
                original_question=question,
                rewritten_question=_rewrite_question(question, candidate_document_ids),
                basic_facts=basic_facts,
                decision="pass_through",
                target_document_ids=document_ids,
                needs_outline_retrieval=False,
                reason="问题没有明显的大纲/文档画像意图。",
            )

        if (
            len(candidate_document_ids) == 1
            and (_has_deictic_reference(normalized_question) or current_scope == "current_document")
            and not _has_non_outline_followup(normalized_question)
        ):
            target_document_ids = candidate_document_ids
            return OutlineReferenceResolution(
                original_question=question,
                rewritten_question=_rewrite_question(question, target_document_ids),
                basic_facts=basic_facts,
                decision="single_document_outline",
                target_document_ids=target_document_ids,
                needs_outline_retrieval=False,
                reason="问题中的论文指代已解析到唯一文档，只需要读取该文档的大纲/画像。",
            )

        return OutlineReferenceResolution(
            original_question=question,
            rewritten_question=_rewrite_question(question, candidate_document_ids),
            basic_facts=basic_facts,
            decision="enter_outline_retrieval",
            target_document_ids=document_ids,
            needs_outline_retrieval=True,
            reason="需要通过大纲/画像检索定位候选论文或补全论文指代。",
        )


def outline_resolution_diagnostics(resolution: OutlineReferenceResolution | None) -> dict[str, object] | None:
    if resolution is None:
        return None
    return {
        "original_question": resolution.original_question,
        "rewritten_question": resolution.rewritten_question,
        "basic_facts": resolution.basic_facts,
        "decision": resolution.decision,
        "target_document_ids": resolution.target_document_ids or [],
        "needs_outline_retrieval": resolution.needs_outline_retrieval,
        "reason": resolution.reason,
    }


def _basic_facts(
    *,
    scope: str,
    document_ids: list[str] | None,
    context_document_ids: list[str],
    question: str,
) -> list[str]:
    facts: list[str] = [f"当前问题范围：{scope}"]
    if document_ids:
        facts.append(f"前端限定文档：{', '.join(document_ids[:5])}")
    if context_document_ids:
        facts.append(f"会话最近引用文档：{', '.join(context_document_ids[:5])}")
    if _has_deictic_reference(question):
        facts.append("用户问题包含论文指代，需要先解析目标论文。")
    return facts


def _rewrite_question(question: str, document_ids: list[str] | None) -> str:
    rewritten = " ".join(question.split())
    if len(document_ids or []) == 1:
        document_fact = f"文档 {document_ids[0]} "
        replacements = [
            "这篇论文",
            "该论文",
            "这篇文献",
            "该文献",
            "这篇文章",
            "该文章",
            "这个论文",
            "这个文献",
            "this paper",
            "the paper",
            "this document",
            "it",
            "它",
        ]
        for marker in replacements:
            rewritten = re.sub(re.escape(marker), document_fact, rewritten, flags=re.IGNORECASE)
    return " ".join(rewritten.split())


def _document_ids_from_context(context: str) -> list[str]:
    ids: list[str] = []
    for line in context.splitlines():
        if "相关文档：" not in line:
            continue
        _, value = line.split("相关文档：", 1)
        ids.extend(item.strip() for item in value.split(",") if item.strip())
    return _dedupe(ids)


def _is_outline_intent(question: str) -> bool:
    return _has_any(
        question,
        [
            "outline",
            "summary",
            "summarize",
            "overview",
            "what is this paper about",
            "what is the paper about",
            "main contribution",
            "main idea",
            "大纲",
            "目录",
            "章节",
            "结构",
            "总结",
            "概述",
            "概括",
            "主要讲",
            "主要研究",
            "讲什么",
            "研究什么",
            "哪些论文",
            "哪几篇论文",
            "相关论文",
            "相关文献",
        ],
    )


def _has_deictic_reference(question: str) -> bool:
    return _has_any(
        question,
        [
            "这篇",
            "该论文",
            "该文献",
            "这个论文",
            "这个文献",
            "它",
            "其中",
            "上面",
            "刚才",
            "this paper",
            "the paper",
            "this document",
            "it",
            "that paper",
        ],
    )


def _has_non_outline_followup(question: str) -> bool:
    return _has_any(
        question,
        [
            "顺便",
            "再",
            "then",
            "and",
            "figure",
            "fig.",
            "plot",
            "chart",
            "图",
            "谱图",
            "曲线",
            "数值",
            "证据",
            "结果",
            "机制",
            "条件",
        ],
    )


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase.lower() in text for phrase in phrases)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
