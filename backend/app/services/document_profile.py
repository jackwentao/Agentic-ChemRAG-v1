from typing import TYPE_CHECKING

from backend.app.domain.documents import DocumentProfile, DocumentProfileField, ParsedDocument
from backend.app.services.profile_extraction import RuleDocumentProfileExtractor
from backend.app.services.profile_normalization import ProfileNormalizer
from backend.app.services.profile_rendering import ProfileTextRenderer
from backend.app.services.profile_schema import ProfileExtraction

if TYPE_CHECKING:
    from backend.app.providers.llm import LlmProvider


class DocumentProfileBuilder:
    def __init__(
        self,
        extractor: RuleDocumentProfileExtractor | None = None,
        normalizer: ProfileNormalizer | None = None,
        renderer: ProfileTextRenderer | None = None,
        llm_provider: "LlmProvider | None" = None,
    ) -> None:
        self.extractor = extractor or RuleDocumentProfileExtractor()
        self.normalizer = normalizer or ProfileNormalizer()
        self.renderer = renderer or ProfileTextRenderer()
        self.llm_provider = llm_provider

    def build(self, parsed: ParsedDocument) -> DocumentProfile:
        rule_extraction = self.extractor.extract(parsed)
        extraction = self._extract_with_optional_llm(parsed, rule_extraction)
        normalized = self._normalize(extraction)
        summary_text = self.renderer.render_summary_text(normalized, parsed)
        return DocumentProfile(
            document_id=parsed.document_id,
            title=normalized.title,
            abstract=normalized.abstract,
            conclusion=normalized.conclusion,
            summary_text=summary_text,
            topics=normalized.topics,
            entities=normalized.entities,
            typed_entities=self.normalizer.typed_entities(normalized.entities),
            key_findings=normalized.key_findings,
            methods=normalized.methods,
            properties=normalized.properties,
            applications=normalized.applications,
            limitations=normalized.limitations,
            outline=normalized.outline,
            figure_summary=normalized.figure_summary,
            source_refs=normalized.source_refs,
            extraction_method=normalized.extraction_method,
            extraction_warnings=normalized.extraction_warnings,
        )

    def field_texts(self, profile: DocumentProfile) -> list[DocumentProfileField]:
        return self.renderer.render_field_texts(profile)

    def _extract_with_optional_llm(self, parsed: ParsedDocument, rule_extraction: ProfileExtraction) -> ProfileExtraction:
        if self.llm_provider is None:
            return rule_extraction
        warnings = list(rule_extraction.extraction_warnings)
        try:
            llm_extraction = self.llm_provider.extract_document_profile(
                document_id=parsed.document_id,
                document_text=parsed.text,
                outline=rule_extraction.outline,
                figure_captions=self._figure_captions(parsed),
            )
        except Exception as exc:
            return self._with_warning(rule_extraction, f"llm_profile_extraction_failed: {exc}")
        if llm_extraction is None:
            return self._with_warning(rule_extraction, "llm_profile_extraction_unavailable")
        return ProfileExtraction(
            document_id=parsed.document_id,
            title=llm_extraction.title or rule_extraction.title,
            abstract=llm_extraction.abstract or rule_extraction.abstract,
            conclusion=llm_extraction.conclusion or rule_extraction.conclusion,
            topics=[*llm_extraction.topics, *rule_extraction.topics],
            entities=[*llm_extraction.entities, *rule_extraction.entities],
            key_findings=llm_extraction.key_findings,
            methods=llm_extraction.methods,
            properties=llm_extraction.properties,
            applications=llm_extraction.applications,
            limitations=llm_extraction.limitations,
            figure_summary=llm_extraction.figure_summary or rule_extraction.figure_summary,
            outline=rule_extraction.outline or llm_extraction.outline,
            source_refs=[*llm_extraction.source_refs, *rule_extraction.source_refs],
            extraction_method="hybrid",
            extraction_warnings=[*warnings, *llm_extraction.extraction_warnings],
        )

    def _normalize(self, extraction: ProfileExtraction) -> ProfileExtraction:
        return ProfileExtraction(
            document_id=extraction.document_id,
            title=extraction.title,
            abstract=extraction.abstract,
            conclusion=extraction.conclusion,
            topics=self.normalizer.normalize_topics(extraction.topics),
            entities=self.normalizer.normalize_entities(extraction.entities),
            key_findings=self.normalizer.normalize_topics(extraction.key_findings),
            methods=self.normalizer.normalize_topics(extraction.methods),
            properties=self.normalizer.normalize_topics(extraction.properties),
            applications=self.normalizer.normalize_topics(extraction.applications),
            limitations=self.normalizer.normalize_topics(extraction.limitations),
            figure_summary=extraction.figure_summary,
            outline=extraction.outline,
            source_refs=extraction.source_refs,
            extraction_method=extraction.extraction_method,
            extraction_warnings=extraction.extraction_warnings,
        )

    def _with_warning(self, extraction: ProfileExtraction, warning: str) -> ProfileExtraction:
        return ProfileExtraction(
            document_id=extraction.document_id,
            title=extraction.title,
            abstract=extraction.abstract,
            conclusion=extraction.conclusion,
            topics=extraction.topics,
            entities=extraction.entities,
            key_findings=extraction.key_findings,
            methods=extraction.methods,
            properties=extraction.properties,
            applications=extraction.applications,
            limitations=extraction.limitations,
            figure_summary=extraction.figure_summary,
            outline=extraction.outline,
            source_refs=extraction.source_refs,
            extraction_method=extraction.extraction_method,
            extraction_warnings=[*extraction.extraction_warnings, warning],
        )

    def _figure_captions(self, parsed: ParsedDocument) -> list[str]:
        return [
            image.caption
            for page in parsed.pages
            for image in page.images
            if image.caption
        ]
