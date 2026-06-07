from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainError(Exception):
    code: str
    user_message: str
    status_code: int = 400
    details: dict[str, Any] = field(default_factory=dict)
    cause: Exception | None = None

    def __post_init__(self) -> None:
        super().__init__(self.user_message)


def upload_invalid_type(message: str = "Only PDF files are supported.") -> DomainError:
    return DomainError(code="UPLOAD_INVALID_TYPE", user_message=message, status_code=400)


def upload_file_too_large(limit_bytes: int) -> DomainError:
    return DomainError(
        code="UPLOAD_FILE_TOO_LARGE",
        user_message=f"PDF file is larger than the configured limit of {limit_bytes} bytes.",
        status_code=413,
        details={"limit_bytes": limit_bytes},
    )


def upload_batch_too_large(limit: int) -> DomainError:
    return DomainError(
        code="UPLOAD_BATCH_TOO_LARGE",
        user_message=f"At most {limit} PDF files can be uploaded at once.",
        status_code=400,
        details={"limit": limit},
    )


def pdf_invalid(message: str = "PDF file is invalid or corrupted.") -> DomainError:
    return DomainError(code="PDF_INVALID", user_message=message, status_code=400)


def pdf_encrypted() -> DomainError:
    return DomainError(code="PDF_ENCRYPTED", user_message="Encrypted PDFs are not supported.", status_code=400)


def pdf_too_many_pages(page_count: int, limit: int) -> DomainError:
    return DomainError(
        code="PDF_TOO_MANY_PAGES",
        user_message=f"PDF has {page_count} pages, exceeding the configured limit of {limit}.",
        status_code=413,
        details={"page_count": page_count, "limit": limit},
    )


def object_not_found(message: str = "Required object was not found in object storage.") -> DomainError:
    return DomainError(code="OBJECT_NOT_FOUND", user_message=message, status_code=404)


def object_storage_unavailable(message: str = "Object storage is unavailable.") -> DomainError:
    return DomainError(code="OBJECT_STORAGE_UNAVAILABLE", user_message=message, status_code=503)


def task_queue_unavailable(message: str = "Task queue is unavailable.") -> DomainError:
    return DomainError(code="TASK_QUEUE_UNAVAILABLE", user_message=message, status_code=503)
