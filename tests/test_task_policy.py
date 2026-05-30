from backend.app.routing.task_policy import BudgetPolicy, TaskClassifier, assessment_requirements


def _classify(question: str, route: str = "chunk_search", scope: str = "current_document"):
    return TaskClassifier().classify(
        question,
        route=route,
        scope=scope,
        is_multi_step=False,
        route_rationale=[],
    )


def test_simple_fact_is_limited_to_single_retrieval_round():
    decision = _classify("What wavelength is used for the measurement?")

    assert decision.task_type == "simple_fact"
    assert decision.risk_level == "low"
    assert BudgetPolicy().max_retrieval_rounds(decision) == 1


def test_multi_condition_and_high_risk_tasks_retain_three_round_budget():
    multi_condition = _classify("Is catalyst X stable in water at elevated temperature?")
    high_risk = _classify("Suggest a safe synthesis route and scale-up procedure.")

    assert multi_condition.task_type == "multi_condition"
    assert BudgetPolicy().max_retrieval_rounds(multi_condition) == 3
    assert high_risk.task_type == "high_risk"
    assert high_risk.risk_level == "high"
    assert BudgetPolicy().max_retrieval_rounds(high_risk) == 3


def test_route_specific_tasks_receive_limited_budgets_and_assessment_requirements():
    figure = _classify("Which PL spectrum figures are shown?", route="figure_index")
    summary = _classify("Summarize this paper.", route="document_profile")

    assert figure.task_type == "figure_query"
    assert BudgetPolicy().max_retrieval_rounds(figure) == 2
    assert "不要求穷尽全部图片" in assessment_requirements(figure)[0]
    assert summary.task_type == "document_summary"
    assert BudgetPolicy().max_retrieval_rounds(summary) == 2
    assert "不额外要求作者或期刊元数据" in assessment_requirements(summary)[0]


def test_out_of_domain_signal_stops_after_initial_assessment():
    decision = TaskClassifier().classify(
        "What does this paper report about CRISPR efficiency?",
        route="chunk_search",
        scope="current_document",
        is_multi_step=False,
        route_rationale=["likely_out_of_domain_low_confidence"],
    )

    assert decision.task_type == "insufficient_evidence"
    assert BudgetPolicy().max_retrieval_rounds(decision) == 1
