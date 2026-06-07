from dataclasses import replace

from backend.app.domain.documents import DocumentSection, ParsedDocument, ParsedImage, ParsedPage, SourceRef
from backend.app.services.document_outline import DocumentOutlineBuilder
from backend.app.services.document_profile import DocumentProfileBuilder
from backend.app.services.profile_normalization import ProfileNormalizer
from backend.app.services.profile_rendering import ProfileTextRenderer
from backend.app.services.profile_schema import ProfileExtraction


class LlmProfileProvider:
    def extract_document_profile(self, *, document_id: str, document_text: str, outline, figure_captions):
        return ProfileExtraction(
            document_id=document_id,
            title="LLM title",
            authors=["A. Author", "B. Researcher"],
            affiliations=["Department of Chemistry, Test University"],
            corresponding_author="A. Author",
            journal="Journal of Test Materials",
            publication_year=2024,
            doi="10.1234/test.2024.001",
            keywords=["photocatalysis", "porphyrin"],
            abstract="LLM abstract.",
            conclusion="LLM conclusion.",
            topics=["catalysis", "photoluminescence"],
            entities=["TiO2", "porphyrin"],
            key_findings=["LLM extracted finding."],
            methods=["solvothermal synthesis"],
            properties=["emission intensity"],
            applications=["photocatalysis"],
            limitations=["No durability test."],
            figure_summary="\n".join(figure_captions[:1]) or None,
            outline=[
                DocumentSection(
                    title="LLM Results and Discussion",
                    normalized_type="results",
                    page_start=3,
                    page_end=5,
                    summary="LLM section summary.",
                    keywords=["llm_outline"],
                )
            ],
            source_refs=[],
            extraction_method="llm",
        )


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
    assert profile.conclusion == "The porphyrin material shows stable luminescence."
    assert "porphyrin" in profile.entities
    assert "photoluminescence" in profile.topics
    assert "Figure 1" in (profile.figure_summary or "")
    assert "Abstract:" in profile.summary_text
    assert "Conclusion:" in profile.summary_text
    assert any(section.normalized_type == "abstract" for section in profile.outline)


def test_document_profile_builder_keeps_publication_metadata_llm_only():
    parsed = ParsedDocument(
        document_id="doc-meta",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text=(
                    "Band alignment and charge transport characteristics of TPP/ZnO hybrid\n"
                    "B.A. Taleatu a,b,*, W.O. Makinde c, M.A. Eleruja b\n"
                    "a International Center for Theoretical Physics, Trieste, Italy\n"
                    "b Department of Physics, Obafemi Awolowo University, Ile-Ife, Nigeria\n"
                    "Current Applied Physics 13 (2013) 97-102\n"
                    "Keywords:\n"
                    "Core hole; Photoelectron; Macrocycle; Spectroscopy\n"
                    "Abstract\n"
                    "Growth of ZnO nanocrystals has been carried out.\n"
                ),
                images=[],
            )
        ],
    )

    profile = DocumentProfileBuilder().build(parsed)

    assert profile.authors == []
    assert profile.affiliations == []
    assert profile.corresponding_author is None
    assert profile.journal is None
    assert profile.publication_year is None
    assert profile.doi is None
    assert profile.keywords == []
    assert "Authors:" not in profile.summary_text


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
    assert profile.abstract == "本文研究卟啉和酞菁高分子的荧光发光与非线性光学性能。"
    assert profile.conclusion == "材料表现出稳定的非线性响应。"
    assert any(section.normalized_type == "conclusion" for section in profile.outline)


def test_document_outline_builder_dedupes_and_normalizes_sections():
    parsed = ParsedDocument(
        document_id="doc3",
        filename="paper.pdf",
        pages=[
            ParsedPage(page=1, text="1 Introduction\nIntroduction\n2 Results and Discussion"),
            ParsedPage(page=2, text="3 Conclusions\nReferences"),
        ],
    )

    outline = DocumentOutlineBuilder().build(parsed)

    assert [section.normalized_type for section in outline] == [
        "introduction",
        "results",
        "conclusion",
        "references",
    ]
    assert outline[0].page_start == 1
    assert outline[2].page_start == 2


def test_profile_renderer_keeps_embedding_text_generation_separate():
    parsed = ParsedDocument(document_id="doc4", filename="paper.pdf", pages=[ParsedPage(page=1, text="Fallback text only.")])
    extraction = ProfileExtraction(
        document_id="doc4",
        title="A profile title",
        abstract="Abstract text.",
        conclusion="Conclusion text.",
        topics=["photoluminescence"],
        entities=["ZnO"],
        figure_summary="Figure 1. PL spectra.",
    )

    summary = ProfileTextRenderer().render_summary_text(extraction, parsed)

    assert "Title: A profile title" in summary
    assert "Conclusion: Conclusion text." in summary
    assert "Entities: ZnO" in summary


def test_profile_renderer_splits_profile_into_field_texts():
    extraction = ProfileExtraction(
        document_id="doc4",
        title="A profile title",
        abstract="Abstract text.",
        conclusion="Conclusion text.",
        topics=["photoluminescence"],
        entities=["ZnO"],
        key_findings=["Finding text."],
        methods=["sol-gel"],
        properties=["emission"],
    )
    parsed = ParsedDocument(document_id="doc4", filename="paper.pdf", pages=[ParsedPage(page=1, text="Fallback.")])
    renderer = ProfileTextRenderer()
    profile = DocumentProfileBuilder(renderer=renderer).build(parsed)
    profile = replace(
        profile,
        title=extraction.title,
        abstract=extraction.abstract,
        conclusion=extraction.conclusion,
        topics=extraction.topics,
        entities=extraction.entities,
        key_findings=extraction.key_findings,
        methods=extraction.methods,
        properties=extraction.properties,
        summary_text=renderer.render_summary_text(extraction, parsed),
    )

    fields = renderer.render_field_texts(profile)

    assert {"overview", "abstract", "conclusion", "findings", "methods", "properties"} <= {
        field.field_name for field in fields
    }


def test_profile_normalizer_dedupes_and_types_entities():
    normalizer = ProfileNormalizer()

    entities = normalizer.normalize_entities(["ZnO", "ZnO", "polymer"])
    typed = normalizer.typed_entities(entities)

    assert entities == ["ZnO", "polymer"]
    assert [entity.entity_type for entity in typed] == ["material", "material"]


def test_document_profile_builder_merges_llm_extraction_with_rule_fallback():
    parsed = ParsedDocument(
        document_id="doc5",
        filename="paper.pdf",
        pages=[
            ParsedPage(
                page=1,
                text=(
                    "Porphyrin photocatalyst\n"
                    "Abstract\n"
                    "Rule abstract.\n"
                    "Conclusions\n"
                    "Rule conclusion."
                ),
                images=[
                    ParsedImage(
                        page=1,
                        index=1,
                        extension="png",
                        bytes_data=b"image",
                        caption="Figure 1. Emission spectra.",
                    )
                ],
            )
        ],
    )

    profile = DocumentProfileBuilder(llm_provider=LlmProfileProvider()).build(parsed)

    assert profile.extraction_method == "hybrid"
    assert profile.title == "LLM title"
    assert profile.authors == ["A. Author", "B. Researcher"]
    assert profile.affiliations == ["Department of Chemistry, Test University"]
    assert profile.corresponding_author == "A. Author"
    assert profile.journal == "Journal of Test Materials"
    assert profile.publication_year == 2024
    assert profile.doi == "10.1234/test.2024.001"
    assert "photocatalysis" in profile.keywords
    assert profile.abstract == "LLM abstract."
    assert profile.conclusion == "LLM conclusion."
    assert "TiO2" in profile.entities
    assert "porphyrin" in profile.entities
    assert profile.key_findings == ["LLM extracted finding."]
    assert "Key findings: LLM extracted finding." in profile.summary_text
    assert [section.title for section in profile.outline] == ["LLM Results and Discussion"]
    assert profile.outline[0].keywords == ["llm_outline"]
