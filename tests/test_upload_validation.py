from io import BytesIO

import pytest
from fastapi import UploadFile

from backend.app.config import Settings
from backend.app.domain.errors import DomainError
from backend.app.services.upload_validation import inspect_pdf_safety, receive_pdf_upload


def _pdf_bytes(page_count: int = 1) -> bytes:
    import fitz

    document = fitz.open()
    for _ in range(page_count):
        document.new_page()
    return document.tobytes()


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        upload_dir=tmp_path,
        upload_max_file_bytes=overrides.pop("upload_max_file_bytes", 1024 * 1024),
        pdf_max_pages=overrides.pop("pdf_max_pages", 10),
        **overrides,
    )


@pytest.mark.asyncio
async def test_receive_pdf_upload_streams_to_temp_file(tmp_path):
    upload = UploadFile(filename="paper.pdf", file=BytesIO(_pdf_bytes()), headers={"content-type": "application/pdf"})

    result = await receive_pdf_upload(upload, _settings(tmp_path))

    try:
        assert result.filename == "paper.pdf"
        assert result.file_size > 0
        assert result.path.exists()
        assert result.page_count == 1
    finally:
        result.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_receive_pdf_upload_rejects_non_pdf_header(tmp_path):
    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"not a pdf"), headers={"content-type": "application/pdf"})

    with pytest.raises(DomainError) as exc:
        await receive_pdf_upload(upload, _settings(tmp_path))

    assert exc.value.code == "PDF_INVALID"


@pytest.mark.asyncio
async def test_receive_pdf_upload_rejects_oversized_file(tmp_path):
    upload = UploadFile(filename="paper.pdf", file=BytesIO(_pdf_bytes()), headers={"content-type": "application/pdf"})

    with pytest.raises(DomainError) as exc:
        await receive_pdf_upload(upload, _settings(tmp_path, upload_max_file_bytes=10))

    assert exc.value.code == "UPLOAD_FILE_TOO_LARGE"


def test_pdf_safety_rejects_too_many_pages(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(_pdf_bytes(page_count=2))

    with pytest.raises(DomainError) as exc:
        inspect_pdf_safety(path, _settings(tmp_path, pdf_max_pages=1))

    assert exc.value.code == "PDF_TOO_MANY_PAGES"
