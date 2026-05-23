import argparse
from hashlib import sha256
from pathlib import Path

from sqlalchemy import create_engine, text

from backend.app.config import get_settings
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.image_store import ObjectImageStore
from backend.app.services.ingestion import IngestionService
from backend.app.services.object_storage import build_object_storage


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    repository = PgVectorRepository(settings.database_url)

    documents = _list_documents(engine, resume=args.resume)
    print(f"Found {len(documents)} documents to rebuild.", flush=True)
    if not args.resume:
        print("Deleting all derived parsing/index data: figures, document profiles, images and chunks.", flush=True)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM figures"))
            conn.execute(text("DELETE FROM document_profiles"))
            conn.execute(text("DELETE FROM images"))
            conn.execute(text("DELETE FROM chunks"))
            conn.execute(
                text(
                    """
                    UPDATE documents
                    SET indexing_status = 'pending',
                        indexing_error = NULL,
                        indexed_at = NULL,
                        indexing_updated_at = now()
                    """
                )
            )

    service = IngestionService(
        parser=PyMuPDFParser(),
        chunker=ParentChildChunker(),
        embedding_provider=build_embedding_provider(settings),
        repository=repository,
        image_store=ObjectImageStore(build_object_storage(settings)),
    )

    succeeded = 0
    failed = 0
    for index, document in enumerate(documents, start=1):
        document_id = str(document["document_id"])
        filename = str(document["filename"])
        path = settings.upload_dir / f"{document_id}.pdf"
        if not path.exists():
            failed += 1
            repository.update_document_indexing_status(document_id, "failed", f"Missing PDF: {path}")
            print(f"[{index}/{len(documents)}] FAILED {filename}: missing {path}", flush=True)
            continue

        print(f"[{index}/{len(documents)}] Rebuilding {filename}", flush=True)
        _delete_document_derived_data(engine, document_id)
        repository.update_document_indexing_status(document_id, "running")
        try:
            content = path.read_bytes()
            chunk_count = service.ingest(
                document_id,
                path,
                filename=filename,
                file_hash=sha256(content).hexdigest(),
                file_size=len(content),
            )
        except Exception as exc:
            failed += 1
            repository.update_document_indexing_status(document_id, "failed", str(exc))
            print(f"[{index}/{len(documents)}] FAILED {filename}: {exc}", flush=True)
            continue
        succeeded += 1
        repository.update_document_indexing_status(document_id, "succeeded")
        print(f"[{index}/{len(documents)}] OK {filename}: {chunk_count} chunks", flush=True)

    print(f"Rebuild complete. succeeded={succeeded}, failed={failed}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete and rebuild parsing/index artifacts.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only rebuild documents that are not already marked succeeded.",
    )
    return parser.parse_args()


def _list_documents(engine, resume: bool) -> list[dict[str, object]]:
    status_filter = "WHERE indexing_status <> 'succeeded'" if resume else ""
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT document_id,
                       COALESCE(original_filename, filename) AS filename
                FROM documents
                {status_filter}
                ORDER BY created_at
                """
            )
        ).mappings().all()
    return [dict(row) for row in rows]


def _delete_document_derived_data(engine, document_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM figures WHERE document_id = :document_id"), {"document_id": document_id})
        conn.execute(text("DELETE FROM document_profiles WHERE document_id = :document_id"), {"document_id": document_id})
        conn.execute(text("DELETE FROM images WHERE document_id = :document_id"), {"document_id": document_id})
        conn.execute(text("DELETE FROM chunks WHERE document_id = :document_id"), {"document_id": document_id})


if __name__ == "__main__":
    main()
