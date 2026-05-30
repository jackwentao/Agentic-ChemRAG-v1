from __future__ import annotations

import re

from backend.app.domain.rag import TaskDecision


class TaskClassifier:
    """使用稳定规则识别检索预算相关的任务类型，避免预算由模型自由扩张。"""

    def classify(
        self,
        question: str,
        *,
        route: str,
        scope: str,
        is_multi_step: bool,
        route_rationale: list[str],
    ) -> TaskDecision:
        normalized = " ".join(question.lower().split())
        if _has_any(normalized, _HIGH_RISK_TERMS):
            return TaskDecision(
                task_type="high_risk",
                requires_retrieval=True,
                risk_level="high",
                rationale="问题涉及安全、合成或实验操作建议，需要保留完整检索预算。",
            )
        if "likely_out_of_domain_low_confidence" in route_rationale:
            return TaskDecision(
                task_type="insufficient_evidence",
                requires_retrieval=True,
                risk_level="low",
                rationale="问题主题可能超出当前文档范围，首轮不足时应尽快停止。",
            )
        if route == "figure_index":
            return TaskDecision(
                task_type="figure_query",
                requires_retrieval=True,
                risk_level="low",
                rationale="图片查询允许一次针对图题或图号的补充检索。",
            )
        if route == "document_profile":
            task_type = "cross_document_filter" if scope == "all_documents" else "document_summary"
            return TaskDecision(
                task_type=task_type,
                requires_retrieval=True,
                risk_level="low",
                rationale="文档画像查询需要覆盖主题或跨文档筛选目标。",
            )
        if is_multi_step or _is_multi_condition(normalized):
            return TaskDecision(
                task_type="multi_condition",
                requires_retrieval=True,
                risk_level="medium",
                rationale="问题包含多个条件或需要综合判断，允许按缺口补充检索。",
            )
        return TaskDecision(
            task_type="simple_fact",
            requires_retrieval=True,
            risk_level="low",
            rationale="单一事实问题以首轮直接证据为准，证据不足时保守输出。",
        )


class BudgetPolicy:
    """将冻结的检索预算规则集中在一个可测试组件中。"""

    MAX_ROUNDS_BY_TASK = {
        "simple_fact": 1,
        "insufficient_evidence": 1,
        "figure_query": 2,
        "document_summary": 2,
        "cross_document_filter": 3,
        "multi_condition": 3,
        "comparison": 3,
        "high_risk": 3,
    }

    def max_retrieval_rounds(self, decision: TaskDecision) -> int:
        return self.MAX_ROUNDS_BY_TASK.get(decision.task_type, 1)


def assessment_requirements(decision: TaskDecision) -> list[str]:
    return {
        "simple_fact": ["只需直接支撑用户询问的单一事实，不得扩大覆盖范围。"],
        "insufficient_evidence": ["若候选内容与问题主题无直接关系，应判定不足并停止补查。"],
        "figure_query": ["找到与用户关键词直接相关的图题或图片即可，不要求穷尽全部图片。"],
        "document_summary": ["只覆盖用户要求的论文主题与核心结论，不额外要求作者或期刊元数据。"],
        "cross_document_filter": ["确认相关文档与用户筛选主题的对应关系即可。"],
        "multi_condition": ["用户问题中的每个明确条件均需有证据覆盖。"],
        "comparison": ["比较对象和比较维度都必须获得证据支持。"],
        "high_risk": ["关键结论与限制条件均需证据支持，资料不足时必须保守输出。"],
    }.get(decision.task_type, ["只判断用户明确询问的内容是否有直接证据支持。"])


def _is_multi_condition(text: str) -> bool:
    return bool(
        re.search(r"\b(?:compare|versus| vs |under .* and |at .* and |all requested conditions|continuous operation)\b", text)
        or re.search(r"比较|对比|同时|以及|在.*和.*条件|连续运行|所有.*条件", text)
        or (_has_any(text, ["stability", "stable", "稳定"]) and _has_any(text, ["temperature", "elevated", "高温"]))
    )


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


_HIGH_RISK_TERMS = [
    "safety",
    "hazard",
    "toxic",
    "toxicity",
    "safe to handle",
    "synthesis route",
    "scale-up",
    "reaction procedure",
    "安全",
    "危害",
    "毒性",
    "合成路线",
    "放大操作",
    "实验操作建议",
]
