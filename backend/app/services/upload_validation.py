from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

from backend.app.config import Settings
from backend.app.domain.errors import (
    DomainError,
    pdf_encrypted,
    pdf_invalid,
    pdf_too_many_pages,
    upload_file_too_large,
    upload_invalid_type,
)


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    path: Path
    file_hash: str
    file_size: int
    content_type: str
    page_count: int


async def receive_pdf_upload(file: UploadFile, settings: Settings) -> ValidatedUpload:
    filename = file.filename or "unnamed"
    _validate_declared_type(filename, file.content_type, settings)
    temp = NamedTemporaryFile(delete=False, suffix=".pdf", dir=settings.upload_dir)
    temp_path = Path(temp.name)
    digest = sha256()
    total = 0
    try:
        with temp:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.upload_max_file_bytes:
                    raise upload_file_too_large(settings.upload_max_file_bytes)
                digest.update(chunk)
                temp.write(chunk)
        _validate_pdf_header(temp_path)
        page_count = inspect_pdf_safety(temp_path, settings)
        return ValidatedUpload(
            filename=filename,
            path=temp_path,
            file_hash=digest.hexdigest(),
            file_size=total,
            content_type=file.content_type or "application/pdf",
            page_count=page_count,
        )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_declared_type(filename: str, content_type: str | None, settings: Settings) -> None:
    if not filename.lower().endswith(".pdf"):
        raise upload_invalid_type()
    allowed_types = {item.strip().lower() for item in settings.upload_allowed_mime_types.split(",") if item.strip()}
    if allowed_types and content_type and content_type.lower() not in allowed_types:
        raise upload_invalid_type("Uploaded file MIME type is not allowed.")


def _validate_pdf_header(path: Path) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(1024).lstrip()
    if not prefix.startswith(b"%PDF-"):
        raise pdf_invalid("Uploaded file does not look like a PDF.")


def inspect_pdf_safety(path: Path, settings: Settings) -> int:
    try:
        import fitz

        with fitz.open(path) as pdf:
            if pdf.is_encrypted:
                raise pdf_encrypted()
            page_count = int(pdf.page_count)
            if page_count <= 0:
                raise pdf_invalid("PDF does not contain any pages.")
            if page_count > settings.pdf_max_pages:
                raise pdf_too_many_pages(page_count, settings.pdf_max_pages)
            return page_count
    except DomainError:
        raise
    except Exception as exc:
        raise pdf_invalid("PDF cannot be parsed safely.") from exc
