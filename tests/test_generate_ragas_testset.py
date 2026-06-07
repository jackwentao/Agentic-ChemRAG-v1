import json
from pathlib import Path

from scripts.generate_ragas_testset import (
    PdfFileInfo,
    PdfPageSource,
    build_golden_cases,
    cache_is_valid,
    discover_pdfs,
    extract_answer_terms,
    langchain_docs_from_page_sources,
    load_jsonl,
    match_reference_contexts,
    next_case_index,
    read_page_sources,
    rows_from_testset,
    write_cache_manifest,
    write_jsonl,
    write_page_sources,
)


class FakeTestset:
    def to_pandas(self):
        return FakeDataFrame(
            [
                {
                    "user_input": "What PL behavior is reported for ZnO?",
                    "reference": "The paper reports ZnO photoluminescence behavior.",
                    "reference_contexts": ["ZnO samples show photoluminescence emission in the visible region."],
                }
            ]
        )


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return self.rows


def test_discover_pdfs_returns_sorted_pdf_files(tmp_path: Path):
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")

    paths = discover_pdfs(tmp_path, "*.pdf")

    assert [path.name for path in paths] == ["a.pdf", "b.pdf"]


def test_rows_from_testset_uses_pandas_records():
    rows = rows_from_testset(FakeTestset())

    assert rows[0]["user_input"] == "What PL behavior is reported for ZnO?"
    assert rows[0]["reference_contexts"] == ["ZnO samples show photoluminescence emission in the visible region."]


def test_match_reference_contexts_finds_best_pdf_page():
    sources = [
        PdfPageSource("doc1", "hash1", "a.pdf", 1, "General introduction about synthesis."),
        PdfPageSource("doc1", "hash1", "a.pdf", 2, "ZnO samples show photoluminescence emission in the visible region."),
        PdfPageSource("doc2", "hash2", "b.pdf", 1, "Porphyrin nonlinear absorption is discussed."),
    ]

    matches = match_reference_contexts(["photoluminescence emission in the visible region"], sources)

    assert [(match.document_id, match.page) for match in matches] == [("doc1", 2)]


def test_build_golden_cases_converts_ragas_rows_to_project_schema():
    rows = [
        {
            "user_input": "What PL behavior is reported for ZnO?",
            "reference": "The paper reports ZnO photoluminescence behavior.",
            "reference_contexts": ["ZnO samples show photoluminescence emission in the visible region."],
        }
    ]
    sources = [
        PdfPageSource("doc1", "hash1", "a.pdf", 2, "ZnO samples show photoluminescence emission in the visible region.")
    ]

    cases = build_golden_cases(rows, sources, scope="all_documents", case_prefix="ragas-test", max_answer_terms=4)

    assert cases == [
        {
            "case_id": "ragas-test-0001",
            "question": "What PL behavior is reported for ZnO?",
            "scope": "all_documents",
            "document_ids": [],
            "expected_route": "chunk_search",
            "target_type": "mixed",
            "expected_documents": ["doc1"],
            "expected_pages": [2],
            "expected_figures": [],
            "expected_answer_terms": ["ZnO", "photoluminescence", "behavior"],
            "answer_type": "fact_qa",
            "task_category": "simple_fact",
            "risk_level": "low",
            "expected_aspects": [],
            "required_disclosures": [],
            "expected_retrieval_behavior": "single_round_expected",
            "max_acceptable_rounds": 1,
            "notes": "由 Ragas testset generation 基于项目 PDF 自动生成。",
            "reference_answer": "The paper reports ZnO photoluminescence behavior.",
            "reference_contexts": ["ZnO samples show photoluminescence emission in the visible region."],
            "ragas_tags": ["ragas_generated", "pdf", "chemistry"],
        }
    ]
    json.dumps(cases[0], ensure_ascii=False)


def test_current_document_scope_uses_single_resolved_document_id():
    rows = [
        {
            "question": "What is reported?",
            "answer": "ZnO photoluminescence is reported.",
            "contexts": ["ZnO photoluminescence evidence."],
        }
    ]
    sources = [PdfPageSource("doc1", "hash1", "a.pdf", 1, "ZnO photoluminescence evidence.")]

    cases = build_golden_cases(rows, sources, scope="current_document", case_prefix="ragas-test", max_answer_terms=3)

    assert cases[0]["document_ids"] == ["doc1"]
    assert cases[0]["scope"] == "current_document"


def test_extract_answer_terms_deduplicates_and_limits_terms():
    terms = extract_answer_terms("ZnO ZnO photoluminescence and nonlinear optical behavior were reported.", max_terms=3)

    assert terms == ["ZnO", "photoluminescence", "nonlinear"]


def test_page_source_cache_round_trips_without_reparsing(tmp_path: Path):
    cache = tmp_path / "pages.json"
    sources = [PdfPageSource("doc1", "hash1", "a.pdf", 2, "ZnO page text")]

    write_page_sources(cache, sources)
    loaded = read_page_sources(cache)
    docs = langchain_docs_from_page_sources(loaded)

    assert loaded == sources
    assert docs[0].page_content == "ZnO page text"
    assert docs[0].metadata["document_id"] == "doc1"


def test_cache_manifest_validation_uses_exact_signature(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    signature = {
        "version": 1,
        "pdfs": [PdfFileInfo("a.pdf", "hash1").__dict__],
        "embedding_model": "BAAI/bge-m3",
    }

    write_cache_manifest(manifest, signature)

    assert cache_is_valid(manifest, signature)
    assert not cache_is_valid(manifest, {**signature, "embedding_model": "other"})


def test_append_helpers_continue_case_numbering(tmp_path: Path):
    output = tmp_path / "cases.jsonl"
    write_jsonl(
        output,
        [
            {"case_id": "ragas-test-0001", "question": "q1"},
            {"case_id": "other-0009", "question": "q2"},
            {"case_id": "ragas-test-0003", "question": "q3"},
        ],
    )

    rows = load_jsonl(output)

    assert [row["case_id"] for row in rows] == ["ragas-test-0001", "other-0009", "ragas-test-0003"]
    assert next_case_index(rows, "ragas-test") == 4
