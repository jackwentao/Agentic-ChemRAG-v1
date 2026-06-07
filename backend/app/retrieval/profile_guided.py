from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.domain.documents import Evidence
from backend.app.domain.rag import TaskDecision


@dataclass(frozen=True)
class ProfileGuidedFollowup:
    route: str
    query: str
    document_ids: list[str]
    reason: str


class ProfileGuidedRetrievalPlanner:
    """根据文档画像命中结果决定是否继续做文档内细节检索。"""

    def plan(
        self,
        *,
        question: str,
        task_decision: TaskDecision | None,
        profile_evidence: list[Evidence],
    ) -> ProfileGuidedFollowup | None:
        documents = _matched_document_ids(profile_evidence)
        if not documents:
            return None
        normalized = " ".join(question.lower().split())
        if _is_profile_level_question(normalized):
            return None
        route = self._route_for_question(normalized)
        if route is None:
            return None
        query = self._query_for_question(question, profile_evidence, route)
        return ProfileGuidedFollowup(
            route=route,
            query=query,
            document_ids=documents,
            reason=self._reason(task_decision, route),
        )

    def _route_for_question(self, normalized_question: str) -> str | None:
        if _has_any(normalized_question, _FIGURE_DETAIL_TERMS):
            return "figure_index"
        if _has_any(normalized_question, _TEXT_DETAIL_TERMS):
            return "chunk_search"
        return None

    def _query_for_question(self, question: str, profile_evidence: list[Evidence], route: str) -> str:
        profile_terms = _profile_terms(profile_evidence)
        route_hint = "figure caption visual evidence" if route == "figure_index" else "results discussion conclusion evidence"
        return " ".join([question.strip(), route_hint, profile_terms]).strip()[:1000]

    def _reason(self, task_decision: TaskDecision | None, route: str) -> str:
        task_type = task_decision.task_type if task_decision else "unknown"
        return f"profile_guided_followup:{task_type}:{route}"


def _matched_document_ids(profile_evidence: list[Evidence]) -> list[str]:
    ids = [
        item.document_id
        for item in profile_evidence
        if item.evidence_type == "document_profile" or item.chunk_id.startswith("profile:")
    ]
    return list(dict.fromkeys(ids))


def _profile_terms(profile_evidence: list[Evidence]) -> str:
    terms: list[str] = []
    for item in profile_evidence[:5]:
        metadata = item.metadata or {}
        for key in ["entities", "topics"]:
            values = metadata.get(key)
            if isinstance(values, list):
                terms.extend(str(value) for value in values if str(value).strip())
        if not terms:
            terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", item.text)[:8])
    return " ".join(list(dict.fromkeys(terms))[:16])


def _is_profile_level_question(normalized_question: str) -> bool:
    return _has_any(normalized_question, _PROFILE_LEVEL_TERMS) and not _has_any(
        normalized_question,
        [*_TEXT_DETAIL_TERMS, *_FIGURE_DETAIL_TERMS],
    )


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


_PROFILE_LEVEL_TERMS = [
    "summarize",
    "summary",
    "what is this paper about",
    "which papers",
    "related to",
    "主要讲",
    "主要研究",
    "总结",
    "概括",
    "哪些论文",
    "相关论文",
]

_TEXT_DETAIL_TERMS = [
    "evidence",
    "support",
    "prove",
    "show",
    "result",
    "results",
    "finding",
    "conclusion",
    "mechanism",
    "method",
    "condition",
    "performance",
    "metric",
    "value",
    "peak",
    "nm",
    "依据",
    "证据",
    "证明",
    "说明",
    "结果",
    "结论",
    "机制",
    "方法",
    "条件",
    "性能",
    "数值",
    "峰",
]

_FIGURE_DETAIL_TERMS = [
    "figure",
    "fig.",
    "image",
    "plot",
    "chart",
    "spectrum",
    "spectra",
    "图",
    "图片",
    "谱图",
    "曲线",
]
