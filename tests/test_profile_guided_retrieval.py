from backend.app.domain.documents import Evidence
from backend.app.retrieval.profile_guided import ProfileGuidedRetrievalPlanner


def _profile_evidence() -> list[Evidence]:
    return [
        Evidence(
            chunk_id="profile:d1",
            document_id="d1",
            text="Title: ZnO photoluminescence.",
            page_start=1,
            page_end=1,
            section_title="Document Profile",
            score=0.9,
            evidence_type="document_profile",
            source_scope="document_level",
            metadata={"entities": ["ZnO"], "topics": ["photoluminescence"]},
        )
    ]


def test_profile_guided_planner_skips_profile_level_summary_questions():
    followup = ProfileGuidedRetrievalPlanner().plan(
        question="Summarize this paper.",
        task_decision=None,
        profile_evidence=_profile_evidence(),
    )

    assert followup is None


def test_profile_guided_planner_routes_evidence_questions_to_chunk_search():
    followup = ProfileGuidedRetrievalPlanner().plan(
        question="Which papers discuss ZnO photoluminescence and what evidence do they provide?",
        task_decision=None,
        profile_evidence=_profile_evidence(),
    )

    assert followup is not None
    assert followup.route == "chunk_search"
    assert followup.document_ids == ["d1"]
    assert "ZnO" in followup.query


def test_profile_guided_planner_routes_figure_questions_to_figure_index():
    followup = ProfileGuidedRetrievalPlanner().plan(
        question="Which PL spectrum figure supports the ZnO result?",
        task_decision=None,
        profile_evidence=_profile_evidence(),
    )

    assert followup is not None
    assert followup.route == "figure_index"
    assert followup.document_ids == ["d1"]
