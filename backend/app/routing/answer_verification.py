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
