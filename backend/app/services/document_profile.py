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
            authors=normalized.authors,
            affiliations=normalized.affiliations,
            corresponding_author=normalized.corresponding_author,
            journal=normalized.journal,
            publication_year=normalized.publication_year,
            doi=normalized.doi,
            keywords=normalized.keywords,
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
            authors=llm_extraction.authors,
            affiliations=llm_extraction.affiliations,
            corresponding_author=llm_extraction.corresponding_author,
            journal=llm_extraction.journal,
            publication_year=llm_extraction.publication_year,
            doi=llm_extraction.doi,
            keywords=llm_extraction.keywords,
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
            outline=llm_extraction.outline or rule_extraction.outline,
            source_refs=[*llm_extraction.source_refs, *rule_extraction.source_refs],
            extraction_method="hybrid",
            extraction_warnings=[*warnings, *llm_extraction.extraction_warnings],
        )

    def _normalize(self, extraction: ProfileExtraction) -> ProfileExtraction:
        return ProfileExtraction(
            document_id=extraction.document_id,
            title=extraction.title,
            authors=self._normalize_authors(extraction.authors, extraction.title),
            affiliations=self._normalize_metadata_list(extraction.affiliations, limit=20),
            corresponding_author=extraction.corresponding_author,
            journal=self._normalize_journal(extraction.journal),
            publication_year=extraction.publication_year,
            doi=extraction.doi,
            keywords=self._normalize_metadata_list(extraction.keywords, limit=20),
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
            authors=extraction.authors,
            affiliations=extraction.affiliations,
            corresponding_author=extraction.corresponding_author,
            journal=extraction.journal,
            publication_year=extraction.publication_year,
            doi=extraction.doi,
            keywords=extraction.keywords,
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

    def _normalize_authors(self, authors: list[str], title: str | None) -> list[str]:
        title_terms = {term.lower() for term in (title or "").replace("/", " ").replace("-", " ").split() if len(term) > 3}
        cleaned = []
        for author in authors:
            value = " ".join(author.replace("\n", " ").split()).strip(" ,;")
            if not value:
                continue
            words = value.split()
            while len(words) > 2 and words[0].lower().strip(":,;") in title_terms:
                words.pop(0)
            value = " ".join(words).strip(" ,;")
            if self._looks_like_non_author(value):
                continue
            cleaned.append(value)
        return list(dict.fromkeys(cleaned))[:30]

    def _looks_like_non_author(self, value: str) -> bool:
        lowered = value.lower()
        if len(value) < 3 or len(value) > 120:
            return True
        if any(marker in lowered for marker in ["abstract", "keyword", "journal", "university", "department", "institute"]):
            return True
        if lowered in {"synthesis", "characterization", "luminescent", "properties", "study"}:
            return True
        return False

    def _normalize_journal(self, journal: str | None) -> str | None:
        if not journal:
            return None
        value = " ".join(journal.split()).strip(" ,;")
        lowered = value.lower()
        if len(value) > 160 or any(marker in lowered for marker in ["department", "university", "institute", "abstract:", "zinc oxide is"]):
            return None
        return value

    def _normalize_metadata_list(self, values: list[str], limit: int) -> list[str]:
        cleaned = []
        for value in values:
            text = " ".join(value.split()).strip(" ,;")
            if not text or len(text) > 120:
                continue
            cleaned.append(text)
        return list(dict.fromkeys(cleaned))[:limit]
