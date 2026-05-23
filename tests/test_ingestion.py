from pathlib import Path

from backend.app.domain.documents import ParsedDocument, ParsedPage
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import DocumentParser
from backend.app.providers.embeddings import FakeEmbeddingProvider
from backend.app.repositories.memory import InMemoryVectorRepository
from backend.app.services.ingestion import IngestionService


class FakeParser(DocumentParser):
    def parse(self, document_id: str, file_path: Path) -> ParsedDocument:
        return ParsedDocument(
            document_id=document_id,
            filename=file_path.name,
            pages=[
                ParsedPage(
                    page=1,
                    text="Abstract\nChemical bonding improves interface stability.\nResults\nConductivity increases after bonding.",
                )
            ],
        )


class DocumentTrackingRepository(InMemoryVectorRepository):
    def __init__(self) -> None:
        super().__init__()
        self.document = None

    def add_document(
        self,
        document_id: str,
        filename: str,
        owner_id: str = "demo",
        file_hash: str | None = None,
        file_size: int | None = None,
        original_filename: str | None = None,
    ) -> None:
        self.document = {
            "document_id": document_id,
            "filename": filename,
            "owner_id": owner_id,
            "file_hash": file_hash,
            "file_size": file_size,
            "original_filename": original_filename,
        }


def test_ingestion_service_indexes_chunks(tmp_path):
    repository = InMemoryVectorRepository()
    service = IngestionService(
        parser=FakeParser(),
        chunker=ParentChildChunker(child_words=20, overlap_words=5),
        embedding_provider=FakeEmbeddingProvider(dimensions=8),
        repository=repository,
    )

    count = service.ingest("doc1", tmp_path / "paper.pdf")

    assert count > 0
    assert repository.search("bonding conductivity", FakeEmbeddingProvider(dimensions=8), top_k=2)


def test_ingestion_service_assigns_document_to_demo_owner(tmp_path):
    repository = DocumentTrackingRepository()
    service = IngestionService(
        parser=FakeParser(),
        chunker=ParentChildChunker(child_words=20, overlap_words=5),
        embedding_provider=FakeEmbeddingProvider(dimensions=8),
        repository=repository,
    )

    service.ingest("doc1", tmp_path / "paper.pdf", filename="original.pdf")

    assert repository.document == {
        "document_id": "doc1",
        "filename": "original.pdf",
        "owner_id": "demo",
        "file_hash": None,
        "file_size": None,
        "original_filename": "original.pdf",
    }
