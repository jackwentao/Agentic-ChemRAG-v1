from backend.app.domain.documents import ParsedDocument, ParsedImage, ParsedPage
from backend.app.services.document_profile import DocumentProfileBuilder


def test_document_profile_builder_extracts_holistic_signals():
    parsed = ParsedDocument(
        document_id="doc1",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text=(
                    "Porphyrin polymer for photoluminescence\n"
                    "Abstract\n"
                    "This work reports porphyrin polymers with strong photoluminescence.\n"
                    "Introduction\n"
                    "The polymer is designed for emission.\n"
                    "Conclusions\n"
                    "The porphyrin material shows stable luminescence."
                ),
                images=[
                    ParsedImage(
                        page=1,
                        index=1,
                        extension="png",
                        bytes_data=b"image",
                        caption="Figure 1. Photoluminescence spectra of the porphyrin polymer.",
                    )
                ],
            )
        ],
    )

    profile = DocumentProfileBuilder().build(parsed)

    assert profile.title == "Porphyrin polymer for photoluminescence"
    assert "porphyrin" in profile.entities
    assert "photoluminescence" in profile.topics
    assert "Figure 1" in (profile.figure_summary or "")
    assert "Abstract:" in profile.summary_text


def test_document_profile_builder_recognizes_chinese_domain_terms():
    parsed = ParsedDocument(
        document_id="doc2",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text=(
                    "酞菁高分子的非线性光学性能研究\n"
                    "摘要\n"
                    "本文研究卟啉和酞菁高分子的荧光发光与非线性光学性能。\n"
                    "结论\n"
                    "材料表现出稳定的非线性响应。"
                ),
                images=[],
            )
        ],
    )

    profile = DocumentProfileBuilder().build(parsed)

    assert "porphyrin" in profile.entities
    assert "phthalocyanine" in profile.entities
    assert "polymer" in profile.entities
    assert "luminescence" in profile.entities
    assert "nonlinear optics" in profile.entities
    assert "phthalocyanine chemistry" in profile.topics
    assert "nonlinear optical response" in profile.topics
