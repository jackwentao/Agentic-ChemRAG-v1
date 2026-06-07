from backend.app.domain.documents import Evidence
from backend.app.domain.rag import EvidenceAssessment, TaskDecision
from backend.app.routing.answer_verification import AnswerVerificationPolicy


def _profile_evidence(field: str = "overview") -> Evidence:
    return Evidence(
        chunk_id=f"profile:doc1:{field}",
        document_id="doc1",
        text="Profile field says ZnO photoluminescence has an emission peak.",
        page_start=1,
        page_end=1,
        section_title=f"Document Profile / {field}",
        score=0.9,
        evidence_type="document_profile",
        source_scope="document_level",
        metadata={"profile_field": field},
    )


def _chunk_evidence() -> Evidence:
    return Evidence(
        chunk_id="c1",
        document_id="doc1",
        text="The PL peak appears at 520 nm.",
        page_start=4,
        page_end=4,
        section_title="Results",
        score=0.9,
    )


def _figure_evidence() -> Evidence:
    return Evidence(
        chunk_id="figure:img1",
        document_id="doc1",
        text="Figure 2. PL spectra of ZnO samples.",
        page_start=5,
        page_end=5,
        section_title="Figure Index",
        score=0.9,
        evidence_type="figure",
    )


def _task(task_type: str = "simple_fact") -> TaskDecision:
    return TaskDecision(task_type=task_type, requires_retrieval=True, risk_level="low")


def test_policy_triggers_for_profile_only_detail_answers():
    decision = AnswerVerificationPolicy().decide(
        answer="The profile indicates a PL peak at 520 nm.",
        evidence=[_profile_evidence("properties")],
        limitations=[],
        task_decision=_task("simple_fact"),
        evidence_assessment=EvidenceAssessment(sufficient=True, next_search_goal="check reported peak value"),
        route="document_profile",
        scope="current_document",
        is_multi_step=False,
    )

    assert decision.should_verify is True
    assert "profile_only_detail_answer" in decision.reasons
    assert "profile_field_specific_claim" in decision.reasons


def test_policy_allows_profile_plus_chunk_detail_without_profile_only_reason():
    decision = AnswerVerificationPolicy().decide(
        answer="The profile locates the topic, and the chunk reports a PL peak at 520 nm.",
        evidence=[_profile_evidence("properties"), _chunk_evidence()],
        limitations=[],
        task_decision=_task("simple_fact"),
        evidence_assessment=EvidenceAssessment(sufficient=True),
        route="chunk_search",
        scope="current_document",
        is_multi_step=False,
    )

    assert "profile_only_detail_answer" not in decision.reasons
    assert "profile_field_specific_claim" in decision.reasons


def test_policy_triggers_for_figure_explanations():
    decision = AnswerVerificationPolicy().decide(
        answer="Figure 2 shows the PL spectrum and supports the ZnO emission trend.",
        evidence=[_figure_evidence()],
        limitations=[],
        task_decision=_task("simple_fact"),
        evidence_assessment=EvidenceAssessment(sufficient=True),
        route="figure_index",
        scope="current_document",
        is_multi_step=False,
    )

    assert decision.should_verify is True
    assert "figure_evidence_answer" in decision.reasons
