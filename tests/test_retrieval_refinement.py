from backend.app.domain.rag import EvidenceAssessment, QueryPlan, RoutePlan, RouteStep, TaskDecision
from backend.app.retrieval.refinement import (
    HybridQueryRefiner,
    LlmQueryRefiner,
    RetrievalStrategySelector,
    RuleQueryRefiner,
    build_retrieval_policy_context,
    normalized_query,
)


def _assessment(**kwargs) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=kwargs.get("sufficient", False),
        missing_aspects=kwargs.get("missing_aspects", ["high temperature stability"]),
        conflicts=kwargs.get("conflicts", []),
        next_search_goal=kwargs.get("next_search_goal", "search high temperature stability"),
        fallback_used=kwargs.get("fallback_used", False),
    )


def _decision(task_type: str = "simple_fact", risk_level: str = "low") -> TaskDecision:
    return TaskDecision(task_type=task_type, requires_retrieval=True, risk_level=risk_level)


def _state(task_decision: TaskDecision | None = None, assessment: EvidenceAssessment | None = None):
    return {
        "route": "chunk_search",
        "scope": "current_document",
        "task_decision": task_decision or _decision(),
        "route_plan": RoutePlan(
            steps=[
                RouteStep(
                    step_id="step-1",
                    route="chunk_search",
                    task_type="multi_condition",
                    query="planner query",
                    max_rounds=2,
                )
            ],
            used_llm=True,
        ),
        "current_step_index": 0,
        "retrieval_round": 1,
        "max_retrieval_rounds": 3,
        "current_new_evidence_count": 1,
        "query_plans": [QueryPlan("q", "q", ["q"], "initial", 1)],
        "evidence_assessment": assessment or _assessment(),
        "diagnostics": {"recall": {"candidate_count": 2, "returned_count": 2, "top_score": 0.8}},
    }


def test_policy_context_uses_task_decision_not_route_step_task_hint():
    context = build_retrieval_policy_context(_state(_decision("simple_fact")))

    assert context.policy_task_type == "simple_fact"
    assert context.planner_task_hint == "multi_condition"
    assert context.step_max_rounds == 2


def test_strategy_selector_prefers_rule_for_clear_gap_and_good_recall():
    context = build_retrieval_policy_context(_state(_decision("simple_fact")))

    decision = RetrievalStrategySelector().select(context)

    assert decision.strategy == "rule"
    assert decision.reason == "clear_gap_with_bounded_cost"


def test_strategy_selector_uses_safety_conservative_for_high_risk():
    context = build_retrieval_policy_context(_state(_decision("high_risk", "high")))

    decision = RetrievalStrategySelector().select(context)

    assert decision.strategy == "safety_conservative"
    assert decision.safety_mode == "conservative"


def test_rule_query_refiner_deduplicates_and_preserves_missing_goal():
    context = build_retrieval_policy_context(_state())
    decision = RetrievalStrategySelector().select(context)
    assessment = _assessment(missing_aspects=["high temperature stability", "high temperature stability"])

    plan = RuleQueryRefiner().refine("Is catalyst stable?", assessment, context, decision)

    assert plan.strategy == "rule"
    assert "search high temperature stability" in plan.queries[0]
    assert plan.queries[0].count("high temperature stability") == 2


class LlmRefinementProvider:
    def refine_search_queries(self, question: str, **kwargs):
        return {
            "queries": [
                "catalyst high temperature aqueous stability",
                "catalyst elevated temperature water durability",
            ],
            "search_goal": "expand stability terminology",
            "reason": "use synonyms for missing aspect",
            "safety_notes": ["bounded search"],
        }


def test_llm_query_refiner_returns_structured_plan():
    context = build_retrieval_policy_context(_state(_decision("multi_condition")))
    decision = RetrievalStrategySelector().select(context)
    assessment = _assessment()

    plan = LlmQueryRefiner(LlmRefinementProvider()).refine(
        "Is catalyst stable?",
        assessment,
        context,
        decision,
        _decision("multi_condition"),
    )

    assert plan.used_llm is True
    assert plan.strategy == "llm"
    assert plan.queries == [
        "catalyst high temperature aqueous stability",
        "catalyst elevated temperature water durability",
    ]


class EmptyLlmRefinementProvider:
    def refine_search_queries(self, question: str, **kwargs):
        return None


def test_hybrid_refiner_falls_back_to_rule_when_llm_unavailable():
    context = build_retrieval_policy_context(_state(_decision("multi_condition")))
    decision = RetrievalStrategySelector().select(context)
    assessment = _assessment()

    plan = HybridQueryRefiner(RuleQueryRefiner(), LlmQueryRefiner(EmptyLlmRefinementProvider())).refine(
        "Is catalyst stable?",
        assessment,
        context,
        decision,
        _decision("multi_condition"),
    )

    assert plan.strategy == "rule"
    assert plan.fallback_used is True
    assert normalized_query(plan.queries[0]).startswith("is catalyst stable?")

