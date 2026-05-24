import argparse
from hashlib import sha256
from pathlib import Path

import psycopg

from backend.app.config import get_settings
from backend.app.database import build_database_engine
from backend.app.parsing.chunker import ParentChildChunker
from backend.app.parsing.parser import PyMuPDFParser
from backend.app.providers.embeddings import build_embedding_provider
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.image_store import LocalImageStore
from backend.app.services.ingestion import IngestionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean duplicate documents and reindex the kept document.")
    parser.add_argument("--keep-document-id", required=True)
    parser.add_argument("--title-contains", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    upload_path = settings.upload_dir / f"{args.keep_document_id}.pdf"
    if not upload_path.exists():
        raise FileNotFoundError(upload_path)

    file_bytes = upload_path.read_bytes()
    file_hash = sha256(file_bytes).hexdigest()
    file_size = len(file_bytes)
    dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://")

    with psycopg.connect(dsn) as conn:
        candidates = collect_duplicate_candidates(conn, settings.upload_dir, args.keep_document_id, args.title_contains, file_hash, file_size)
        delete_ids = [document_id for document_id in candidates if document_id != args.keep_document_id]
        print(f"keep_document_id: {args.keep_document_id}")
        print(f"file_hash: {file_hash}")
        print(f"candidates: {candidates}")
        print(f"delete_ids: {delete_ids}")
        if not args.execute:
            print("dry_run: true")
            return

        delete_document_records(conn, delete_ids, settings.upload_dir)
        clear_index_for_document(conn, args.keep_document_id)
        conn.commit()

    repository = PgVectorRepository(build_database_engine(settings.database_url))
    repository.update_document_hash(args.keep_document_id, file_hash, file_size, upload_path.name)
    service = IngestionService(
        parser=PyMuPDFParser(),
        chunker=ParentChildChunker(),
        embedding_provider=build_embedding_provider(settings),
        repository=repository,
        image_store=LocalImageStore(settings.image_dir),
    )
    chunk_count = service.ingest(
        args.keep_document_id,
        upload_path,
        filename=upload_path.name,
        file_hash=file_hash,
        file_size=file_size,
    )
    print(f"reindexed_child_chunks: {chunk_count}")


def collect_duplicate_candidates(
    conn,
    upload_dir: Path,
    keep_document_id: str,
    title_contains: str,
    file_hash: str,
    file_size: int,
) -> list[str]:
    rows = conn.execute(
        """
        SELECT document_id, COALESCE(original_filename, filename), file_hash
        FROM documents
        ORDER BY created_at DESC
        """
    ).fetchall()
    candidates: list[str] = []
    title_key = title_contains.lower()
    for document_id, filename, stored_hash in rows:
        upload_path = upload_dir / f"{document_id}.pdf"
        matches_upload = upload_path.exists() and upload_path.stat().st_size == file_size and sha256(upload_path.read_bytes()).hexdigest() == file_hash
        matches_hash = stored_hash == file_hash
        matches_title = title_key in filename.lower()
        if document_id == keep_document_id or matches_hash or matches_upload or matches_title:
            candidates.append(document_id)
    return candidates


def delete_document_records(conn, document_ids: list[str], upload_dir: Path) -> None:
    for document_id in document_ids:
        image_paths = [Path(row[0]) for row in conn.execute("SELECT path FROM images WHERE document_id = %s", (document_id,)).fetchall()]
        conn.execute("DELETE FROM images WHERE document_id = %s", (document_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        conn.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
        delete_files(image_paths)
        upload_path = upload_dir / f"{document_id}.pdf"
        if upload_path.exists():
            upload_path.unlink()


def clear_index_for_document(conn, document_id: str) -> None:
    image_paths = [Path(row[0]) for row in conn.execute("SELECT path FROM images WHERE document_id = %s", (document_id,)).fetchall()]
    conn.execute("DELETE FROM images WHERE document_id = %s", (document_id,))
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    delete_files(image_paths)


def delete_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            print(f"warning: failed to delete {path}: {exc}")


if __name__ == "__main__":
    main()
