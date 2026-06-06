from io import BytesIO

from backend.app.tasks.ingestion_tasks import _download_pdf_to_temp


class FakeObjectStorage:
    def open(self, key: str):
        assert key == "documents/doc1.pdf"
        return BytesIO(b"%PDF-1.7\n")


def test_download_pdf_to_temp_reads_from_object_storage():
    path = _download_pdf_to_temp(FakeObjectStorage(), "documents/doc1.pdf")

    try:
        assert path.exists()
        assert path.read_bytes() == b"%PDF-1.7\n"
    finally:
        path.unlink(missing_ok=True)
