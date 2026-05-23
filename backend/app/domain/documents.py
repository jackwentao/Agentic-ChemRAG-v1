from dataclasses import dataclass, field
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ParsedImage:
    page: int
    index: int
    extension: str
    bytes_data: bytes
    width: int | None = None
    height: int | None = None
    label: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    source_type: str = "embedded"


@dataclass(frozen=True)
class ParsedPage:
    page: int
    text: str
    images: list[ParsedImage] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    filename: str
    pages: list[ParsedPage]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    parent_id: str | None
    text: str
    page_start: int
    page_end: int
    section_title: str
    section_path: str
    chunk_type: str


@dataclass(frozen=True)
class EvidenceImage:
    image_id: str
    page: int
    width: int | None = None
    height: int | None = None
    label: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    section_title: str
    score: float
    image_ids: list[str] = field(default_factory=list)
    images: list[EvidenceImage] = field(default_factory=list)


@dataclass(frozen=True)
class StoredImage:
    image_id: str
    document_id: str
    page: int
    path: str
    extension: str
    width: int | None = None
    height: int | None = None
    label: str | None = None
    caption: str | None = None
    nearby_text: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    source_type: str = "embedded"


@dataclass(frozen=True)
class DocumentProfile:
    document_id: str
    title: str | None
    abstract: str | None
    summary_text: str
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    figure_summary: str | None = None
