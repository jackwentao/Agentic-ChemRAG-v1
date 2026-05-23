from pathlib import Path

from backend.app.config import get_settings
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.image_store import ObjectImageStore
from backend.app.services.ingestion import IngestionService
from backend.app.services.object_storage import build_object_storage
from backend.app.tasks.celery_app import celery_app


@celery_app.task(name="ingest_document")
def ingest_document(
    document_id: str,
    file_path: str,
    filename: str | None = None,
    file_hash: str | None = None,
    file_size: int | None = None,
) -> int:
    settings = get_settings()
    repository = PgVectorRepository(settings.database_url)
    repository.update_document_indexing_status(document_id, "running")
    service = IngestionService(
        parser=PyMuPDFParser(),
        chunker=ParentChildChunker(),
        embedding_provider=build_embedding_provider(settings),
        repository=repository,
        image_store=ObjectImageStore(build_object_storage(settings)),
    )
    try:
        chunk_count = service.ingest(document_id, Path(file_path), filename=filename, file_hash=file_hash, file_size=file_size)
    except Exception as exc:
        repository.update_document_indexing_status(document_id, "failed", error=str(exc))
        raise
    repository.update_document_indexing_status(document_id, "succeeded")
    return chunk_count
