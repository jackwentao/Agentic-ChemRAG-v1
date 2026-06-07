from pathlib import Path
from tempfile import NamedTemporaryFile

from backend.app.config import get_settings
from backend.app.database import build_database_engine
from backend.app.domain.errors import object_not_found
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.providers.llm import build_llm_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.document_profile import DocumentProfileBuilder
from backend.app.services.image_store import ObjectImageStore
from backend.app.services.ingestion import IngestionService
from backend.app.services.object_storage import build_object_storage
from backend.app.tasks.celery_app import celery_app


@celery_app.task(name="ingest_document")
def ingest_document(
    document_id: str,
    file_path: str | None = None,
    filename: str | None = None,
    file_hash: str | None = None,
    file_size: int | None = None,
) -> int:
    settings = get_settings()
    engine = build_database_engine(settings.database_url)
    repository = PgVectorRepository(engine)
    temp_path: Path | None = None
    try:
        repository.update_document_indexing_status(document_id, "running")
        document = repository.get_document_for_ingestion(document_id)
        if document is None:
            raise object_not_found(f"Document {document_id} was not found for ingestion.")
        filename = filename or str(document.get("filename") or f"{document_id}.pdf")
        file_hash = file_hash or (str(document["file_hash"]) if document.get("file_hash") else None)
        raw_file_size = document.get("file_size")
        file_size = file_size or (int(raw_file_size) if raw_file_size is not None else None)
        pdf_object_key = str(document.get("pdf_object_key") or "")
        if not pdf_object_key:
            raise object_not_found(f"Document {document_id} does not have a PDF object key.")
        object_storage = build_object_storage(settings)
        if not object_storage.exists(pdf_object_key):
            raise object_not_found(f"PDF object {pdf_object_key} was not found.")
        temp_path = _download_pdf_to_temp(object_storage, pdf_object_key)
        service = IngestionService(
            parser=PyMuPDFParser(
                max_extracted_images=settings.pdf_max_extracted_images,
                max_image_bytes=settings.pdf_max_image_bytes,
            ),
            chunker=ParentChildChunker(),
            embedding_provider=build_embedding_provider(settings),
            repository=repository,
            image_store=ObjectImageStore(object_storage),
            profile_builder=DocumentProfileBuilder(llm_provider=_build_optional_profile_llm_provider(settings)),
        )
        chunk_count = service.ingest(document_id, temp_path, filename=filename, file_hash=file_hash, file_size=file_size)
        repository.update_document_indexing_status(document_id, "succeeded")
        return chunk_count
    except Exception as exc:
        repository.update_document_indexing_status(document_id, "failed", error=str(exc))
        raise
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        engine.dispose()


def _build_optional_profile_llm_provider(settings):
    if not settings.llm_profile_extraction_enabled:
        return None
    try:
        return build_llm_provider(settings)
    except Exception:
        return None


def _download_pdf_to_temp(object_storage, object_key: str) -> Path:
    stream = object_storage.open(object_key)
    temp = NamedTemporaryFile(delete=False, suffix=".pdf")
    path = Path(temp.name)
    try:
        with stream, temp:
            while chunk := stream.read(1024 * 1024):
                temp.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
