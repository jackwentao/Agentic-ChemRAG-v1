import re

from backend.app.domain.documents import DocumentSection, ParsedDocument, SourceRef


class DocumentOutlineBuilder:
    def build(self, parsed: ParsedDocument) -> list[DocumentSection]:
        candidates: list[DocumentSection] = []
        for page in parsed.pages:
            for line in page.text.splitlines():
                heading = compact_text(line)
                if not heading or len(heading) > 120:
                    continue
                normalized_type = self.normalized_section_type(heading)
                if not normalized_type:
                    continue
                candidates.append(
                    DocumentSection(
                        title=heading[:120],
                        normalized_type=normalized_type,
                        page_start=page.page,
                        page_end=page.page,
                        source_refs=[
                            SourceRef(
                                source_type="page_text",
                                page_start=page.page,
                                page_end=page.page,
                                section_title=heading[:120],
                                field_name="outline",
                                text=heading[:120],
                            )
                        ],
                    )
                )
        return _dedupe_sections(candidates)[:40]

    def normalized_section_type(self, heading: str) -> str | None:
        normalized = compact_text(heading).lower()
        normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", normalized)
        patterns = [
            ("abstract", r"^(abstract|摘要)$"),
            ("introduction", r"^(introduction|引言|绪论)$"),
            ("methods", r"^(experimental|experiment|methods?|materials and methods|实验|方法|材料与方法)$"),
            ("results", r"^(results|results and discussion|结果|结果与讨论)$"),
            ("discussion", r"^(discussion|讨论)$"),
            ("conclusion", r"^(conclusions?|结论|总结)$"),
            ("references", r"^(references|参考文献)$"),
        ]
        for section_type, pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return section_type
        return None


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
    deduped: list[DocumentSection] = []
    seen: set[tuple[str | None, int | None]] = set()
    for section in sections:
        key = (section.normalized_type, section.page_start)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(section)
    return deduped
