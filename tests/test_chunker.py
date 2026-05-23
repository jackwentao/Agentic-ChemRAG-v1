from backend.app.domain.documents import ParsedDocument, ParsedPage
from backend.app.parsing.chunker import ParentChildChunker


def test_parent_child_chunker_preserves_sections_and_children():
    parsed = ParsedDocument(
        document_id="doc1",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text=(
                    "Abstract\n"
                    "Polymer composites show improved conductivity after chemical bonding.\n"
                    "Introduction\n"
                    + " ".join(["The interface controls charge transport."] * 80)
                ),
            )
        ],
    )

    chunks = ParentChildChunker(child_words=40, overlap_words=10).chunk(parsed)

    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]
    assert parents
    assert children
    assert any(chunk.section_title.lower() == "abstract" for chunk in chunks)
    assert all(child.parent_id for child in children)


def test_parent_child_chunker_does_not_create_reference_children():
    parsed = ParsedDocument(
        document_id="doc1",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text=(
                    "Results and Discussion\n"
                    "The doped ZnO nanoparticles show photoluminescence changes.\n"
                    "References\n"
                    "[1] Example reference title journal year.\n"
                    "[2] Another reference title journal year.\n"
                ),
            )
        ],
    )

    chunks = ParentChildChunker(child_words=20, overlap_words=5).chunk(parsed)
    reference_children = [
        chunk for chunk in chunks if chunk.chunk_type == "child" and chunk.section_title.lower() == "references"
    ]

    assert reference_children == []


def test_parent_child_chunker_uses_character_windows_for_chinese_text():
    parsed = ParsedDocument(
        document_id="doc1",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text="摘要\n" + "酞菁基类高分子的设计制备及光电性能研究" * 120,
            )
        ],
    )

    chunks = ParentChildChunker(child_chars=120, overlap_chars=20).chunk(parsed)
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert len(children) > 1
    assert all(len(child.text) <= 120 for child in children)
