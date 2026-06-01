from pathlib import Path

from backend.app.domain.documents import Evidence
from backend.app.evaluation.cases import GoldenCase, load_golden_cases
from scripts.evaluate_profile_refactor import IndexedCorpus, evaluate_cases, score_text


def test_profile_refactor_cases_are_valid():
    cases = load_golden_cases(Path("eval/profile_refactor_cases.jsonl"))

    assert len(cases) >= 10
    assert all(case.expected_route == "document_profile" for case in cases)
    assert {case.task_category for case in cases} >= {"multi_condition", "high_risk", "context_followup"}


def test_profile_refactor_scoring_handles_domain_aliases():
    score = score_text(
        "Which paper has a PL spectrum for ZnO nanoparticles?",
        "Figure 2 shows photoluminescence spectrum of ZnO nanoparticles.",
    )

    assert score > 0.4


def test_profile_refactor_evaluation_reports_profile_guided_followup():
    case = GoldenCase(
        case_id="profile-guided",
        question="In the ZnO paper, what transmittance measurement is reported?",
        scope="current_document",
        document_ids=["doc-zno"],
        expected_route="document_profile",
        target_type="page",
        expected_documents=["doc-zno"],
        expected_pages=[5],
        expected_answer_terms=["transmittance"],
        expected_retrieval_behavior="retry_if_missing",
        max_acceptable_rounds=2,
        notes=None,
    )
    corpus = IndexedCorpus(
        chunks=[
            Evidence(
                chunk_id="chunk-1",
                document_id="doc-zno",
                text="The optical transmittance measurement used a 1064 nm laser.",
                page_start=5,
                page_end=5,
                section_title="Results",
                score=0.0,
            )
        ],
        figures=[],
        profile_fields=[
            Evidence(
                chunk_id="profile:doc-zno:overview",
                document_id="doc-zno",
                text="ZnO nanoparticle synthesis with photoluminescence and transmittance measurements.",
                page_start=1,
                page_end=1,
                section_title="Document Profile / overview",
                score=0.0,
                evidence_type="document_profile",
                source_scope="document_level",
                metadata={"profile_field": "overview"},
            )
        ],
        parsed_document_count=1,
        chunk_count=1,
        figure_count=0,
        profile_field_count=1,
    )

    report = evaluate_cases([case], corpus, top_k=5)
    result = report["results"][0]

    assert result["actual_route"] == "document_profile"
    assert result["fine_metrics"]["followup_used"] is True
    assert result["fine_metrics"]["followup_route"] == "chunk_search"
    assert result["fine_metrics"]["followup_constrained_document_count"] == 1
    assert result["fine_metrics"]["profile_only_detail_risk"] is False
    assert result["metrics"]["page_recall@k"] == 1.0
