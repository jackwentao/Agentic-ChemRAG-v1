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
    owner_id: str | None = None


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
class SourceRef:
    source_type: str
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    field_name: str | None = None
    text: str | None = None


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
    evidence_type: str = "chunk"
    source_scope: str = "page_span"
    source_refs: list[SourceRef] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


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
class ProfileEntity:
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    normalized_name: str | None = None
    source_refs: list[SourceRef] = field(default_factory=list)
    confidence: float | None = None


@dataclass(frozen=True)
class DocumentSection:
    title: str
    normalized_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None
    keywords: list[str] = field(default_factory=list)
    linked_figures: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentProfile:
    document_id: str
    title: str | None
    abstract: str | None
    conclusion: str | None
    summary_text: str
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    corresponding_author: str | None = None
    journal: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    keywords: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    typed_entities: list[ProfileEntity] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    outline: list[DocumentSection] = field(default_factory=list)
    figure_summary: str | None = None
    source_refs: list[SourceRef] = field(default_factory=list)
    extraction_method: str = "rule"
    extraction_warnings: list[str] = field(default_factory=list)
    owner_id: str | None = None


@dataclass(frozen=True)
class DocumentProfileField:
    document_id: str
    field_name: str
    text: str
    source_refs: list[SourceRef] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    owner_id: str | None = None
