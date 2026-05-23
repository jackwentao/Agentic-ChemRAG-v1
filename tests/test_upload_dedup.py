from hashlib import sha256

import backend.app.main as main_module
from backend.app.main import _find_existing_document_by_storage_hash, _store_uploaded_pdf


class FakeDocumentRepository:
    def __init__(self, documents):
        self.documents = documents
        self.updated = None
        self.added = None
        self.status_updates = []

    def list_documents(self, owner_id: str = "demo"):
        return self.documents

    def find_document_by_hash(self, owner_id: str, file_hash: str):
        return next((document for document in self.documents if document.get("file_hash") == file_hash), None)

    def update_document_hash(
        self,
        document_id: str,
        file_hash: str,
        file_size: int,
        original_filename: str | None = None,
        pdf_object_key: str | None = None,
    ) -> None:
        self.updated = {
            "document_id": document_id,
            "file_hash": file_hash,
            "file_size": file_size,
            "original_filename": original_filename,
            "pdf_object_key": pdf_object_key,
        }

    def add_document(
        self,
        document_id: str,
        filename: str,
        owner_id: str = "demo",
        file_hash: str | None = None,
        file_size: int | None = None,
        original_filename: str | None = None,
        indexing_status: str | None = None,
        pdf_object_key: str | None = None,
    ) -> None:
        self.added = {
            "document_id": document_id,
            "filename": filename,
            "owner_id": owner_id,
            "file_hash": file_hash,
            "file_size": file_size,
            "original_filename": original_filename,
            "indexing_status": indexing_status,
            "pdf_object_key": pdf_object_key,
        }

    def update_document_indexing_status(self, document_id: str, status: str, error: str | None = None) -> None:
        self.status_updates.append((document_id, status, error))


class FakeTaskQueue:
    def __init__(self, should_fail: bool = False) -> None:
        self.calls = []
        self.should_fail = should_fail

    def delay(self, *args):
        self.calls.append(args)
        if self.should_fail:
            raise RuntimeError("redis unavailable")

        class Result:
            id = "task-1"

        return Result()


def test_find_existing_document_by_storage_hash_backfills_hash(tmp_path):
    content = b"pdf bytes"
    file_hash = sha256(content).hexdigest()
    upload_dir = tmp_path
    (upload_dir / "doc1.pdf").write_bytes(content)
    repository = FakeDocumentRepository(
        [{"document_id": "doc1", "filename": "paper.pdf", "chunk_count": 3, "is_indexed": True}]
    )

    document = _find_existing_document_by_storage_hash(repository, file_hash, len(content), upload_dir, "paper.pdf")

    assert document is not None
    assert document["document_id"] == "doc1"
    assert document["file_hash"] == file_hash
    assert document["filename"] == "paper.pdf"
    assert repository.updated == {
        "document_id": "doc1",
        "file_hash": file_hash,
        "file_size": len(content),
        "original_filename": "paper.pdf",
        "pdf_object_key": "documents/doc1.pdf",
    }


def test_find_existing_document_by_storage_hash_ignores_different_content(tmp_path):
    content = b"pdf bytes"
    upload_dir = tmp_path
    (upload_dir / "doc1.pdf").write_bytes(b"other bytes")
    repository = FakeDocumentRepository([{"document_id": "doc1", "filename": "paper.pdf"}])

    document = _find_existing_document_by_storage_hash(repository, sha256(content).hexdigest(), len(content), upload_dir)

    assert document is None
    assert repository.updated is None


def test_store_uploaded_pdf_schedules_new_document(tmp_path, monkeypatch):
    task_queue = FakeTaskQueue()
    repository = FakeDocumentRepository([])
    monkeypatch.setattr(main_module.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main_module, "ingest_document", task_queue)
    monkeypatch.setattr(main_module, "_store_pdf_object", lambda pdf_key, content: None)

    response = _store_uploaded_pdf("paper.pdf", b"pdf bytes", repository)

    assert response.status == "pending"
    assert response.task_id == "task-1"
    assert repository.added is not None
    assert (tmp_path / f"{response.document_id}.pdf").read_bytes() == b"pdf bytes"
    assert len(task_queue.calls) == 1


def test_store_uploaded_pdf_reuses_pending_document_without_duplicate_task(tmp_path, monkeypatch):
    file_hash = sha256(b"pdf bytes").hexdigest()
    task_queue = FakeTaskQueue()
    repository = FakeDocumentRepository(
        [
            {
                "document_id": "doc1",
                "filename": "paper.pdf",
                "file_hash": file_hash,
                "is_indexed": False,
                "indexing_status": "running",
            }
        ]
    )
    monkeypatch.setattr(main_module.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main_module, "ingest_document", task_queue)
    monkeypatch.setattr(main_module, "_store_pdf_object", lambda pdf_key, content: None)

    response = _store_uploaded_pdf("paper.pdf", b"pdf bytes", repository)

    assert response.document_id == "doc1"
    assert response.status == "running"
    assert response.reused is True
    assert response.task_id is None
    assert task_queue.calls == []


def test_store_uploaded_pdf_marks_document_failed_when_task_enqueue_fails(tmp_path, monkeypatch):
    task_queue = FakeTaskQueue(should_fail=True)
    repository = FakeDocumentRepository([])
    monkeypatch.setattr(main_module.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main_module, "ingest_document", task_queue)
    monkeypatch.setattr(main_module, "_store_pdf_object", lambda pdf_key, content: None)

    response = _store_uploaded_pdf("paper.pdf", b"pdf bytes", repository)

    assert response.status == "failed"
    assert response.task_id is None
    assert response.error == "Failed to enqueue ingestion task: redis unavailable"
    assert repository.status_updates == [
        (response.document_id, "failed", "Failed to enqueue ingestion task: redis unavailable")
    ]
