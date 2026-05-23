import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha1

from backend.app.domain.documents import Chunk, ParsedDocument


SECTION_PATTERNS = [
    r"^\s*(\d+(\.\d+)*\.?\s+)?abstract\s*$",
    r"^\s*(\d+(\.\d+)*\.?\s+)?introduction\s*$",
    r"^\s*(\d+(\.\d+)*\.?\s+)?(experimental|experiment|experimental details|materials and methods|methods)\s*$",
    r"^\s*(\d+(\.\d+)*\.?\s+)?(characterization techniques|characterization)\s*$",
    r"^\s*(\d+(\.\d+)*\.?\s+)?(results and discussion|results|discussion)\s*$",
    r"^\s*(\d+(\.\d+)*\.?\s+)?(conclusion|conclusions)\s*$",
    r"^\s*(\d+(\.\d+)*\.?\s+)?(references|bibliography)\s*$",
]

REFERENCE_SECTION_PATTERN = r"^\s*(\d+(\.\d+)*\.?\s+)?(references|bibliography)\s*$"


@dataclass(frozen=True)
class Section:
    title: str
    text: str
    page_start: int
    page_end: int


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        raise NotImplementedError


class ParentChildChunker(ChunkingStrategy):
    def __init__(
        self,
        child_words: int = 220,
        overlap_words: int = 40,
        child_chars: int = 1200,
        overlap_chars: int = 160,
    ) -> None:
        self.child_words = child_words
        self.overlap_words = overlap_words
        self.child_chars = child_chars
        self.overlap_chars = overlap_chars

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        sections = self._sections(parsed)
        chunks: list[Chunk] = []
        for section_index, section in enumerate(sections):
            parent_id = self._stable_id(parsed.document_id, "parent", section_index, section.title)
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    document_id=parsed.document_id,
                    parent_id=None,
                    text=section.text,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section_title=section.title,
                    section_path=section.title,
                    chunk_type="parent",
                )
            )
            if self._is_reference_section(section.title):
                continue
            for child_index, child_text in enumerate(self._window(section.text)):
                chunks.append(
                    Chunk(
                        chunk_id=self._stable_id(parsed.document_id, "child", section_index, child_index, child_text[:80]),
                        document_id=parsed.document_id,
                        parent_id=parent_id,
                        text=child_text,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title=section.title,
                        section_path=section.title,
                        chunk_type="child",
                    )
                )
        return chunks

    def _sections(self, parsed: ParsedDocument) -> list[Section]:
        sections: list[Section] = []
        current_title = "Document"
        current_lines: list[str] = []
        start_page = parsed.pages[0].page if parsed.pages else 1
        end_page = start_page

        for page in parsed.pages:
            for raw_line in page.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if self._is_section_heading(line) and current_lines:
                    sections.append(Section(current_title, "\n".join(current_lines), start_page, end_page))
                    current_title = line
                    current_lines = []
                    start_page = page.page
                elif self._is_section_heading(line):
                    current_title = line
                    start_page = page.page
                else:
                    current_lines.append(line)
                end_page = page.page

        if current_lines:
            sections.append(Section(current_title, "\n".join(current_lines), start_page, end_page))
        return sections or [Section("Document", parsed.text, 1, 1)]

    def _is_section_heading(self, line: str) -> bool:
        if len(line) > 90:
            return False
        return any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in SECTION_PATTERNS)

    def _is_reference_section(self, title: str) -> bool:
        return re.match(REFERENCE_SECTION_PATTERN, title, flags=re.IGNORECASE) is not None

    def _window(self, text: str) -> list[str]:
        if self._should_use_char_windows(text):
            return self._char_window(text)
        words = text.split()
        if len(words) <= self.child_words:
            return [text] if text.strip() else []
        step = max(1, self.child_words - self.overlap_words)
        windows = []
        for start in range(0, len(words), step):
            window = words[start : start + self.child_words]
            if len(window) < max(30, self.overlap_words):
                break
            windows.append(" ".join(window))
        return windows

    def _should_use_char_windows(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return False
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", compact))
        return cjk_chars / len(compact) > 0.25

    def _char_window(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        if len(compact) <= self.child_chars:
            return [compact]
        step = max(1, self.child_chars - self.overlap_chars)
        windows = []
        min_tail = max(120, self.overlap_chars)
        for start in range(0, len(compact), step):
            window = compact[start : start + self.child_chars].strip()
            if len(window) < min_tail:
                break
            windows.append(window)
        return windows

    def _stable_id(self, *parts: object) -> str:
        raw = "::".join(str(part) for part in parts)
        return sha1(raw.encode("utf-8")).hexdigest()
