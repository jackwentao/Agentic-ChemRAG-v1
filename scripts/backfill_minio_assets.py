from pathlib import Path

from sqlalchemy import create_engine, text

from backend.app.config import get_settings
from backend.app.services.object_storage import build_object_storage, document_pdf_key, image_object_key


def main() -> None:
    settings = get_settings()
    storage = build_object_storage(settings)
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    pdf_count = 0
    image_count = 0
    with engine.begin() as conn:
        documents = conn.execute(
            text("SELECT document_id FROM documents ORDER BY created_at")
        ).mappings().all()
        for document in documents:
            document_id = str(document["document_id"])
            path = settings.upload_dir / f"{document_id}.pdf"
            if not path.exists():
                continue
            key = document_pdf_key(document_id)
            storage.put_bytes(key, path.read_bytes(), "application/pdf")
            conn.execute(
                text("UPDATE documents SET pdf_object_key = :key WHERE document_id = :document_id"),
                {"key": key, "document_id": document_id},
            )
            pdf_count += 1

        images = conn.execute(
            text("SELECT image_id, path, extension FROM images ORDER BY image_id")
        ).mappings().all()
        for image in images:
            image_id = str(image["image_id"])
            extension = str(image["extension"] or "png")
            source_path = _resolve_image_path(settings.image_dir, image_id, extension, str(image["path"]))
            if source_path is None:
                continue
            key = image_object_key(image_id, extension)
            storage.put_bytes(key, source_path.read_bytes(), _image_content_type(extension))
            conn.execute(
                text("UPDATE images SET path = :key WHERE image_id = :image_id"),
                {"key": key, "image_id": image_id},
            )
            image_count += 1

    print(f"Backfilled {pdf_count} PDFs and {image_count} images to {settings.object_storage_provider}.")


def _resolve_image_path(image_dir: Path, image_id: str, extension: str, stored_path: str) -> Path | None:
    candidates = [Path(stored_path), image_dir / f"{image_id}.{extension}"]
    candidates.extend(image_dir.glob(f"{image_id}.*"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _image_content_type(extension: str) -> str:
    normalized = extension.lower().lstrip(".")
    if normalized == "jpg":
        normalized = "jpeg"
    return f"image/{normalized}"


if __name__ == "__main__":
    main()
