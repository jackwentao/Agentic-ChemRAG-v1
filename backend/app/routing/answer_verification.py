from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.domain.documents import Evidence
from backend.app.domain.rag import EvidenceAssessment, TaskDecision


@dataclass(frozen=True)
class AnswerVerificationDecision:
    """答案核验触发结果，和具体核验模型解耦，便于独立测试策略。"""

    should_verify: bool
    reasons: list[str]


class AnswerVerificationPolicy:
    """按风险、证据状态和答案形态决定是否需要生成后核验。"""

    confidence_threshold = 0.55
    broad_answer_evidence_count = 5

    def decide(
        self,
        *,
        answer: str,
        evidence: list[Evidence],
        limitations: list[str],
        task_decision: TaskDecision | None,
        evidence_assessment: EvidenceAssessment | None,
        route: str,
        scope: str,
        is_multi_step: bool,
    ) -> AnswerVerificationDecision:
        reasons: list[str] = []
        task_type = task_decision.task_type if task_decision else "unknown"
        risk_level = task_decision.risk_level if task_decision else "unknown"
        confidence = evidence_assessment.confidence if evidence_assessment else None
        conflicts = evidence_assessment.conflicts if evidence_assessment else []
        evidence_types = {item.evidence_type for item in evidence}
        profile_fields = _profile_fields(evidence)
        has_profile_evidence = "document_profile" in evidence_types
        has_chunk_evidence = "chunk" in evidence_types
        has_figure_evidence = "figure" in evidence_types
        detail_question = _contains_detail_intent(answer) or _contains_detail_intent(
            evidence_assessment.next_search_goal if evidence_assessment else ""
        )

        if risk_level == "high" or task_type == "high_risk":
            reasons.append("high_risk")
        if conflicts:
            reasons.append("evidence_conflict")
        if confidence is not None and confidence < self.confidence_threshold:
            reasons.append("low_assessment_confidence")
        if is_multi_step:
            reasons.append("multi_step_plan")
        if scope == "all_documents" and task_type == "cross_document_filter":
            reasons.append("cross_document_scope")
        if len(evidence) >= self.broad_answer_evidence_count:
            reasons.append("broad_evidence_set")
        if limitations and not _answer_mentions_limitations(answer):
            reasons.append("limitations_not_disclosed")
        if _contains_strong_claim(answer):
            reasons.append("strong_claim_language")
        if route == "document_profile" and scope == "all_documents":
            reasons.append("profile_cross_document_summary")
        if has_profile_evidence and task_type in {"document_summary", "cross_document_filter", "comparison"}:
            reasons.append("profile_evidence_answer")
        if has_profile_evidence and not has_chunk_evidence and (detail_question or _contains_numeric_or_metric_claim(answer)):
            reasons.append("profile_only_detail_answer")
        if profile_fields & {"properties", "findings", "methods", "conclusion"} and _contains_numeric_or_metric_claim(answer):
            reasons.append("profile_field_specific_claim")
        if has_figure_evidence and (task_type == "figure_query" or _contains_figure_intent(answer)):
            reasons.append("figure_evidence_answer")

        return AnswerVerificationDecision(
            should_verify=bool(reasons),
            reasons=list(dict.fromkeys(reasons)),
        )


def _answer_mentions_limitations(answer: str) -> bool:
    normalized = answer.lower()
    return any(
        marker in normalized
        for marker in [
            "限制",
            "不足",
            "缺口",
            "未检索",
            "无法确认",
            "cannot confirm",
            "not enough evidence",
            "insufficient",
            "limitation",
        ]
    )


def _contains_strong_claim(answer: str) -> bool:
    return bool(
        re.search(
            r"\b(proves?|must|always|never|safe|significantly superior|scale[- ]?up)\b",
            answer,
            re.IGNORECASE,
        )
        or re.search(r"证明|必然|一定|显著优于|安全|可安全|放大|工业化", answer)
    )


def _profile_fields(evidence: list[Evidence]) -> set[str]:
    fields: set[str] = set()
    for item in evidence:
        value = item.metadata.get("profile_field") if item.metadata else None
        if isinstance(value, str) and value:
            fields.add(value)
    return fields


def _contains_detail_intent(text: str | None) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"\b(evidence|support|prove|show|result|finding|conclusion|mechanism|method|condition|performance|metric|value|peak|nm)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(r"依据|证据|证明|说明|结果|结论|机制|方法|条件|性能|数值|峰", text)
    )


def _contains_numeric_or_metric_claim(answer: str) -> bool:
    return bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|ev|%|wt%|mol%|ma\s*h|mah|pa|kpa|mpa|gpa|°c|c|k)\b", answer, re.IGNORECASE)
        or re.search(r"\b(?:peak|band gap|efficiency|yield|capacity|intensity|emission|conductivity)\b", answer, re.IGNORECASE)
        or re.search(r"峰值|带隙|效率|产率|容量|强度|发射|电导|性能", answer)
    )


def _contains_figure_intent(text: str) -> bool:
    return bool(
        re.search(r"\b(?:figure|fig\.|image|plot|chart|spectrum|spectra|curve)\b", text, re.IGNORECASE)
        or re.search(r"图|图片|谱图|曲线", text)
    )
