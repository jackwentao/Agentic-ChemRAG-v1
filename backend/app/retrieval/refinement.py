from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol

from backend.app.domain.rag import EvidenceAssessment, RoutePlan, TaskDecision


RefinementStrategy = Literal["stop", "rule", "llm", "hybrid", "safety_conservative"]
SafetyMode = Literal["normal", "conservative"]
RecallQuality = Literal["good", "weak", "poor", "unknown"]


@dataclass(frozen=True)
class RetrievalPolicyContext:
    """补充检索策略只消费这个归一化上下文，避免误用 planner hint 或诊断字段。"""

    route: str
    scope: str
    policy_task_type: str
    risk_level: str
    retrieval_round: int
    max_retrieval_rounds: int
    step_max_rounds: int | None
    evidence_sufficient: bool
    assessment_fallback_used: bool
    missing_aspects: tuple[str, ...]
    conflicts: tuple[str, ...]
    next_search_goal: str | None
    current_new_evidence_count: int
    attempted_queries: tuple[str, ...]
    rule_route_confidence: float | None = None
    needs_llm_router: bool = False
    recall_quality: RecallQuality = "unknown"
    assessment_confidence: float | None = None
    planner_task_hint: str | None = None

    @property
    def has_clear_gap(self) -> bool:
        return bool(self.missing_aspects or self.conflicts or self.next_search_goal)


@dataclass(frozen=True)
class RetrievalStrategyDecision:
    strategy: RefinementStrategy
    reason: str
    max_queries: int = 1
    allow_route_switch: bool = False
    safety_mode: SafetyMode = "normal"


@dataclass(frozen=True)
class QueryRefinementPlan:
    queries: list[str]
    strategy: RefinementStrategy
    search_goal: str
    reason: str
    safety_notes: list[str] = field(default_factory=list)
    used_llm: bool = False
    fallback_used: bool = False


class SupportsSearchQueryRefinement(Protocol):
    def refine_search_queries(
        self,
        question: str,
        *,
        assessment: EvidenceAssessment,
        task_decision: TaskDecision | None,
        route: str,
        scope: str,
        attempted_queries: list[str],
        max_queries: int,
        safety_mode: str,
    ) -> dict[str, object] | None:
        ...


class RetrievalStrategySelector:
    """把质量、安全、时延和价格原则转换成补充检索策略。"""

    COMPLEX_TASKS = {"multi_condition", "cross_document_filter", "comparison"}

    def select(self, context: RetrievalPolicyContext) -> RetrievalStrategyDecision:
        if context.assessment_fallback_used:
            return RetrievalStrategyDecision("stop", "assessment_fallback")
        if context.evidence_sufficient:
            return RetrievalStrategyDecision("stop", "sufficient_evidence")
        if context.retrieval_round >= context.max_retrieval_rounds:
            return RetrievalStrategyDecision("stop", "max_rounds_reached")
        if context.risk_level == "high":
            return RetrievalStrategyDecision(
                "safety_conservative",
                "high_risk_requires_safety_bounded_refinement",
                max_queries=2,
                safety_mode="conservative",
            )
        if context.retrieval_round > 1 and context.current_new_evidence_count == 0:
            if context.policy_task_type in self.COMPLEX_TASKS:
                return RetrievalStrategyDecision("llm", "complex_task_rule_retry_had_no_new_evidence", max_queries=2)
            return RetrievalStrategyDecision("stop", "no_new_evidence")
        if context.policy_task_type in self.COMPLEX_TASKS:
            return RetrievalStrategyDecision("hybrid", "complex_task_prioritizes_quality", max_queries=3)
        if context.recall_quality == "poor" or context.needs_llm_router:
            return RetrievalStrategyDecision("llm", "weak_route_or_recall_signal", max_queries=2)
        if context.has_clear_gap:
            return RetrievalStrategyDecision("rule", "clear_gap_with_bounded_cost", max_queries=1)
        return RetrievalStrategyDecision("llm", "insufficient_without_clear_rule_direction", max_queries=2)


class RuleQueryRefiner:
    """当前规则补充检索的模块化版本。"""

    def refine(
        self,
        question: str,
        assessment: EvidenceAssessment,
        context: RetrievalPolicyContext,
        decision: RetrievalStrategyDecision,
    ) -> QueryRefinementPlan:
        components = self._components(question, assessment, context)
        query = _compact_query(" ".join(component for component in components if component))
        return QueryRefinementPlan(
            queries=[query] if query else [],
            strategy="rule",
            search_goal=assessment.next_search_goal or "补充尚缺证据",
            reason=decision.reason,
            safety_notes=_safety_notes(context),
        )

    def _components(
        self,
        question: str,
        assessment: EvidenceAssessment,
        context: RetrievalPolicyContext,
    ) -> list[str]:
        gap_terms = [assessment.next_search_goal, *assessment.missing_aspects, *assessment.conflicts]
        if context.route == "figure_index":
            return [question, *_dedupe(gap_terms)]
        if context.route == "document_profile":
            return [question, *_dedupe(gap_terms)]
        return [question, *_dedupe(gap_terms)]


class LlmQueryRefiner:
    """受约束的 LLM query 生成器，只生成检索 query，不生成答案。"""

    def __init__(self, llm_provider: SupportsSearchQueryRefinement) -> None:
        self.llm_provider = llm_provider

    def refine(
        self,
        question: str,
        assessment: EvidenceAssessment,
        context: RetrievalPolicyContext,
        decision: RetrievalStrategyDecision,
        task_decision: TaskDecision | None,
    ) -> QueryRefinementPlan:
        payload = self.llm_provider.refine_search_queries(
            question,
            assessment=assessment,
            task_decision=task_decision,
            route=context.route,
            scope=context.scope,
            attempted_queries=list(context.attempted_queries),
            max_queries=decision.max_queries,
            safety_mode=decision.safety_mode,
        )
        if not payload:
            return QueryRefinementPlan(
                queries=[],
                strategy="llm",
                search_goal=assessment.next_search_goal or "补充尚缺证据",
                reason="llm_refiner_unavailable",
                safety_notes=_safety_notes(context),
                used_llm=False,
                fallback_used=True,
            )
        return _sanitize_llm_refinement_payload(payload, assessment, context, decision)


class HybridQueryRefiner:
    """质量优先的混合策略：复杂/高风险优先尝试 LLM，失败后回退到规则。"""

    def __init__(self, rule_refiner: RuleQueryRefiner, llm_refiner: LlmQueryRefiner) -> None:
        self.rule_refiner = rule_refiner
        self.llm_refiner = llm_refiner

    def refine(
        self,
        question: str,
        assessment: EvidenceAssessment,
        context: RetrievalPolicyContext,
        decision: RetrievalStrategyDecision,
        task_decision: TaskDecision | None,
    ) -> QueryRefinementPlan:
        if decision.strategy == "stop":
            return QueryRefinementPlan([], "stop", "", decision.reason)
        if decision.strategy == "rule":
            return self.rule_refiner.refine(question, assessment, context, decision)
        if decision.strategy in {"llm", "hybrid", "safety_conservative"}:
            llm_plan = self.llm_refiner.refine(question, assessment, context, decision, task_decision)
            if llm_plan.queries:
                return llm_plan
            rule_plan = self.rule_refiner.refine(question, assessment, context, decision)
            return QueryRefinementPlan(
                queries=rule_plan.queries,
                strategy="rule",
                search_goal=rule_plan.search_goal,
                reason=f"{decision.reason}; llm_fallback_to_rule",
                safety_notes=rule_plan.safety_notes,
                fallback_used=True,
            )
        return QueryRefinementPlan([], "stop", "", "unsupported_strategy")


def build_retrieval_policy_context(state: Mapping[str, object]) -> RetrievalPolicyContext:
    assessment = state["evidence_assessment"]
    if not isinstance(assessment, EvidenceAssessment):
        raise TypeError("state['evidence_assessment'] must be an EvidenceAssessment.")
    task_decision = state.get("task_decision")
    task_type = task_decision.task_type if isinstance(task_decision, TaskDecision) else "unknown"
    risk_level = task_decision.risk_level if isinstance(task_decision, TaskDecision) else "unknown"
    route_plan = state.get("route_plan")
    current_step_index = int(state.get("current_step_index", 0))
    step_max_rounds: int | None = None
    planner_task_hint: str | None = None
    if isinstance(route_plan, RoutePlan) and 0 <= current_step_index < len(route_plan.steps):
        step = route_plan.steps[current_step_index]
        step_max_rounds = step.max_rounds
        planner_task_hint = step.task_type
    attempted_queries = tuple(
        query
        for plan in state.get("query_plans", [])
        if hasattr(plan, "queries")
        for query in plan.queries
    )
    diagnostics = state.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    recall = diagnostics.get("recall", {})
    recall = recall if isinstance(recall, dict) else {}
    return RetrievalPolicyContext(
        route=str(state.get("route", "chunk_search")),
        scope=str(state.get("scope") or ("current_document" if state.get("document_ids") else "all_documents")),
        policy_task_type=task_type,
        risk_level=risk_level,
        retrieval_round=int(state.get("retrieval_round", 1)),
        max_retrieval_rounds=int(state.get("max_retrieval_rounds", 1)),
        step_max_rounds=step_max_rounds,
        evidence_sufficient=bool(assessment.sufficient),
        assessment_fallback_used=bool(assessment.fallback_used),
        missing_aspects=tuple(assessment.missing_aspects),
        conflicts=tuple(assessment.conflicts),
        next_search_goal=assessment.next_search_goal,
        current_new_evidence_count=int(state.get("current_new_evidence_count", 0)),
        attempted_queries=attempted_queries,
        rule_route_confidence=_optional_float(state.get("route_confidence")),
        needs_llm_router=bool(state.get("needs_llm_router", False)),
        recall_quality=_recall_quality(recall),
        assessment_confidence=assessment.confidence,
        planner_task_hint=planner_task_hint,
    )


def normalized_query(query: str) -> str:
    return " ".join(query.lower().split())


def refinement_diagnostics(
    context: RetrievalPolicyContext,
    decision: RetrievalStrategyDecision,
    plan: QueryRefinementPlan,
) -> dict[str, object]:
    return {
        "retrieval_strategy": {
            "selected": decision.strategy,
            "reason": decision.reason,
            "max_queries": decision.max_queries,
            "allow_route_switch": decision.allow_route_switch,
            "safety_mode": decision.safety_mode,
            "policy_task_type": context.policy_task_type,
            "risk_level": context.risk_level,
            "recall_quality": context.recall_quality,
            "planner_task_hint": context.planner_task_hint,
        },
        "query_refinement": {
            "strategy": plan.strategy,
            "queries": list(plan.queries),
            "search_goal": plan.search_goal,
            "reason": plan.reason,
            "safety_notes": list(plan.safety_notes),
            "used_llm": plan.used_llm,
            "fallback_used": plan.fallback_used,
        },
    }


def _sanitize_llm_refinement_payload(
    payload: dict[str, object],
    assessment: EvidenceAssessment,
    context: RetrievalPolicyContext,
    decision: RetrievalStrategyDecision,
) -> QueryRefinementPlan:
    raw_queries = payload.get("queries")
    queries = raw_queries if isinstance(raw_queries, list) else []
    sanitized = [
        _compact_query(str(query))
        for query in queries
        if isinstance(query, str) and query.strip()
    ]
    sanitized = _dedupe(sanitized)[: decision.max_queries]
    search_goal = str(payload.get("search_goal") or assessment.next_search_goal or "补充尚缺证据").strip()[:300]
    reason = str(payload.get("reason") or decision.reason).strip()[:500]
    safety_notes = _dedupe(
        [
            str(item).strip()[:240]
            for item in payload.get("safety_notes", [])
            if isinstance(item, str) and item.strip()
        ]
    )
    if context.risk_level == "high" and not safety_notes:
        safety_notes = _safety_notes(context)
    return QueryRefinementPlan(
        queries=sanitized,
        strategy="llm" if decision.strategy != "safety_conservative" else "safety_conservative",
        search_goal=search_goal,
        reason=reason,
        safety_notes=safety_notes,
        used_llm=True,
    )


def _recall_quality(recall: dict[str, object]) -> RecallQuality:
    candidate_count = int(recall.get("candidate_count", 0) or 0)
    returned_count = int(recall.get("returned_count", 0) or 0)
    top_score = _optional_float(recall.get("top_score"))
    if candidate_count == 0 or returned_count == 0:
        return "poor"
    if top_score is None:
        return "unknown"
    if top_score >= 0.65:
        return "good"
    if top_score >= 0.35:
        return "weak"
    return "poor"


def _compact_query(query: str) -> str:
    return " ".join(query.split())[:1000]


def _dedupe(values: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        normalized = normalized_query(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _safety_notes(context: RetrievalPolicyContext) -> list[str]:
    if context.risk_level != "high":
        return []
    return ["高风险问题仅允许补充文献中的限制、风险、概述性条件和证据充分性，不生成可执行实验步骤。"]

