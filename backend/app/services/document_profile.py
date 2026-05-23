import re

from backend.app.domain.documents import DocumentProfile, ParsedDocument


class DocumentProfileBuilder:
    def build(self, parsed: ParsedDocument) -> DocumentProfile:
        title = self._title(parsed)
        abstract = self._section_text(parsed.text, "abstract")
        conclusion = self._section_text(parsed.text, "conclusions?") or self._section_text(parsed.text, "conclusion")
        captions = [
            image.caption
            for page in parsed.pages
            for image in page.images
            if image.caption
        ]
        figure_summary = "\n".join(captions[:20]) or None
        entities = self._entities(parsed.text)
        topics = self._topics(parsed.text, captions)
        summary_parts = [
            f"Title: {title}" if title else None,
            f"Abstract: {abstract}" if abstract else None,
            f"Conclusion: {conclusion}" if conclusion else None,
            f"Topics: {', '.join(topics)}" if topics else None,
            f"Entities: {', '.join(entities)}" if entities else None,
            f"Figures: {figure_summary}" if figure_summary else None,
        ]
        summary_text = "\n".join(part for part in summary_parts if part)
        if not summary_text:
            summary_text = self._compact(parsed.text)[:2000]
        return DocumentProfile(
            document_id=parsed.document_id,
            title=title,
            abstract=abstract,
            summary_text=summary_text,
            topics=topics,
            entities=entities,
            figure_summary=figure_summary,
        )

    def _title(self, parsed: ParsedDocument) -> str | None:
        for page in parsed.pages[:2]:
            for line in page.text.splitlines():
                value = self._compact(line)
                if not value:
                    continue
                if len(value) < 8 or self._looks_like_metadata(value):
                    continue
                return value[:300]
        return None

    def _section_text(self, text: str, section_name: str) -> str | None:
        pattern = re.compile(
            rf"^\s*(?:\d+(?:\.\d+)*\.?\s+)?{section_name}\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            return None
        start = match.end()
        next_heading = re.search(
            r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:introduction|experimental|experiment|methods|results|discussion|conclusions?|references)\s*$",
            text[start:],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        end = start + next_heading.start() if next_heading else min(len(text), start + 2500)
        value = self._compact(text[start:end])
        return value[:1600] if value else None

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
        return bool(re.search(r"^(doi|abstract|keywords?|received|accepted)\b", line, flags=re.IGNORECASE))

    def _compact(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
