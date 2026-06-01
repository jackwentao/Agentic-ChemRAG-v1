from dataclasses import dataclass, field

from backend.app.domain.documents import DocumentSection, SourceRef


@dataclass(frozen=True)
class ProfileExtraction:
    document_id: str
    title: str | None = None
    abstract: str | None = None
    conclusion: str | None = None
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    figure_summary: str | None = None
    outline: list[DocumentSection] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    extraction_method: str = "rule"
    extraction_warnings: list[str] = field(default_factory=list)
