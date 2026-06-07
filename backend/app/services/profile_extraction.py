import re

from backend.app.domain.documents import ParsedDocument, SourceRef
from backend.app.services.document_outline import DocumentOutlineBuilder, compact_text
from backend.app.services.profile_schema import ProfileExtraction


class RuleDocumentProfileExtractor:
    def __init__(self, outline_builder: DocumentOutlineBuilder | None = None) -> None:
        self.outline_builder = outline_builder or DocumentOutlineBuilder()

    def extract(self, parsed: ParsedDocument) -> ProfileExtraction:
        title, title_ref = self._title(parsed)
        abstract, abstract_ref = self._section_text(parsed, ["abstract", "摘要"], "abstract")
        conclusion, conclusion_ref = self._section_text(parsed, ["conclusions?", "conclusion", "结论", "总结"], "conclusion")
        captions = [
            image.caption
            for page in parsed.pages
            for image in page.images
            if image.caption
        ]
        figure_summary = "\n".join(captions[:20]) or None
        source_refs = [ref for ref in [title_ref, abstract_ref, conclusion_ref] if ref]
        if figure_summary:
            source_refs.append(SourceRef(source_type="figure_caption", field_name="figure_summary", text=figure_summary[:500]))
        return ProfileExtraction(
            document_id=parsed.document_id,
            title=title,
            abstract=abstract,
            conclusion=conclusion,
            topics=self._topics(parsed.text, captions),
            entities=self._entities(parsed.text),
            figure_summary=figure_summary,
            outline=self.outline_builder.build(parsed),
            source_refs=source_refs,
            extraction_method="rule",
        )

    def _title(self, parsed: ParsedDocument) -> tuple[str | None, SourceRef | None]:
        for page in parsed.pages[:2]:
            for line in page.text.splitlines():
                value = compact_text(line)
                if not value:
                    continue
                if len(value) < 8 or self._looks_like_metadata(value):
                    continue
                title = value[:300]
                return title, SourceRef(
                    source_type="page_text",
                    page_start=page.page,
                    page_end=page.page,
                    field_name="title",
                    text=title,
                )
        return None, None

    def _section_text(self, parsed: ParsedDocument, section_names: list[str], field_name: str) -> tuple[str | None, SourceRef | None]:
        text = parsed.text
        pattern = re.compile(
            rf"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:{'|'.join(section_names)})\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            return None, None
        start = match.end()
        next_heading = re.search(
            r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:introduction|experimental|experiment|methods?|materials?|results|discussion|conclusions?|references|引言|绪论|实验|方法|材料|结果|讨论|结论|参考文献)\s*$",
            text[start:],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        end = start + next_heading.start() if next_heading else min(len(text), start + 2500)
        value = compact_text(text[start:end])
        if not value:
            return None, None
        page_number = self._page_for_offset(parsed, match.start())
        section_text = value[:1600]
        return section_text, SourceRef(
            source_type="page_text",
            page_start=page_number,
            page_end=page_number,
            field_name=field_name,
            text=section_text[:500],
        )

    def _entities(self, text: str) -> list[str]:
        terms = {
            "porphyrin": ["porphyrin", "卟啉"],
            "phthalocyanine": ["phthalocyanine", "酞菁"],
            "zinc": ["zinc", "Zn", "锌"],
            "ZnO": ["ZnO"],
            "polymer": ["polymer", "聚合物", "高分子"],
            "COF": ["COF", "covalent organic framework", "共价有机框架"],
            "luminescence": ["luminescence", "photoluminescence", "发光", "荧光"],
            "nonlinear optics": ["nonlinear optical", "nonlinear optics", "NLO", "非线性光学"],
        }
        normalized = text.lower()
        entities = []
        for canonical, aliases in terms.items():
            if any(alias.lower() in normalized for alias in aliases):
                entities.append(canonical)
        return entities

    def _topics(self, text: str, captions: list[str]) -> list[str]:
        source = f"{text}\n{' '.join(captions)}".lower()
        topic_patterns = {
            "synthesis": r"synthesis|synthetic route|制备|合成",
            "characterization": r"characterization|xrd|sem|tem|nmr|表征",
            "photoluminescence": r"photoluminescence|pl spectra|emission|荧光|发光",
            "nonlinear optical response": r"nonlinear optical|nonlinear optics|nlo|z-scan|非线性",
            "porphyrin chemistry": r"porphyrin|卟啉",
            "phthalocyanine chemistry": r"phthalocyanine|酞菁",
            "COF materials": r"covalent organic framework|cof|共价有机框架",
        }
        topics = [topic for topic, pattern in topic_patterns.items() if re.search(pattern, source, flags=re.IGNORECASE)]
        return topics[:12]

    def _looks_like_metadata(self, line: str) -> bool:
        return bool(re.search(r"^(doi|abstract|keywords?|received|accepted)\b|^(摘要|关键词)$", line, flags=re.IGNORECASE))

    def _page_for_offset(self, parsed: ParsedDocument, target_offset: int) -> int | None:
        offset = 0
        for page in parsed.pages:
            page_text = page.text
            next_offset = offset + len(page_text) + 2
            if target_offset < next_offset:
                return page.page
            offset = next_offset
        return parsed.pages[0].page if parsed.pages else None
