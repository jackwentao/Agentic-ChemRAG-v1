from backend.app.domain.documents import DocumentProfile, DocumentProfileField, ParsedDocument
from backend.app.services.document_outline import compact_text
from backend.app.services.profile_schema import ProfileExtraction


class ProfileTextRenderer:
    def render_summary_text(self, extraction: ProfileExtraction, parsed: ParsedDocument) -> str:
        summary_parts = [
            f"Title: {extraction.title}" if extraction.title else None,
            f"Abstract: {extraction.abstract}" if extraction.abstract else None,
            f"Conclusion: {extraction.conclusion}" if extraction.conclusion else None,
            f"Topics: {', '.join(extraction.topics)}" if extraction.topics else None,
            f"Entities: {', '.join(extraction.entities)}" if extraction.entities else None,
            f"Key findings: {'; '.join(extraction.key_findings)}" if extraction.key_findings else None,
            f"Methods: {', '.join(extraction.methods)}" if extraction.methods else None,
            f"Properties: {', '.join(extraction.properties)}" if extraction.properties else None,
            f"Applications: {', '.join(extraction.applications)}" if extraction.applications else None,
            f"Limitations: {'; '.join(extraction.limitations)}" if extraction.limitations else None,
            f"Figures: {extraction.figure_summary}" if extraction.figure_summary else None,
        ]
        summary_text = "\n".join(part for part in summary_parts if part)
        return summary_text or compact_text(parsed.text)[:2000]

    def render_field_texts(self, profile: DocumentProfile) -> list[DocumentProfileField]:
        fields = [
            self._field(profile, "overview", profile.summary_text),
            self._field(profile, "abstract", profile.abstract),
            self._field(profile, "conclusion", profile.conclusion),
            self._field(profile, "topics", ", ".join(profile.topics)),
            self._field(profile, "entities", ", ".join(profile.entities)),
            self._field(profile, "findings", "; ".join(profile.key_findings)),
            self._field(profile, "methods", ", ".join(profile.methods)),
            self._field(profile, "properties", ", ".join(profile.properties)),
            self._field(profile, "applications", ", ".join(profile.applications)),
            self._field(profile, "limitations", "; ".join(profile.limitations)),
            self._field(profile, "figures", profile.figure_summary),
            self._field(
                profile,
                "outline",
                "\n".join(
                    f"{section.normalized_type or 'section'}: {section.title}"
                    for section in profile.outline
                    if section.title
                ),
            ),
        ]
        return [field for field in fields if field and field.text.strip()]

    def _field(self, profile: DocumentProfile, field_name: str, text: str | None) -> DocumentProfileField | None:
        if not text or not text.strip():
            return None
        return DocumentProfileField(
            document_id=profile.document_id,
            field_name=field_name,
            text=compact_text(text),
            source_refs=[ref for ref in profile.source_refs if ref.field_name in {field_name, None}],
            metadata={"profile_title": profile.title},
        )
