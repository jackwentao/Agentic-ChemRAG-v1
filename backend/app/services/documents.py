from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from backend.app.schemas import DocumentSummary, DocumentUploadResponse
from backend.app.services.object_storage import ObjectStorage, document_pdf_key


class DocumentService:
    """处理文档入库、重复上传检测与索引任务提交。"""

    def __init__(
        self,
        repository,
        object_storage: ObjectStorage,
        upload_dir: Path,
        ingestion_task,
        owner_id: str = "demo",
    ) -> None:
        self.repository = repository
        self.object_storage = object_storage
        self.upload_dir = upload_dir
        self.ingestion_task = ingestion_task
        self.owner_id = owner_id

    def store_uploaded_pdf(self, filename: str, content: bytes) -> DocumentUploadResponse:
        file_hash = sha256(content).hexdigest()
        file_size = len(content)

        existing = self.repository.find_document_by_hash(self.owner_id, file_hash)
        if existing is None:
            existing = self.find_existing_document_by_storage_hash(file_hash, file_size, filename)

        if existing is not None:
            document_id = str(existing["document_id"])
            pdf_key = document_pdf_key(document_id)
            self._store_pdf_object(pdf_key, content)
            self.repository.update_document_hash(document_id, file_hash, file_size, filename, pdf_object_key=pdf_key)
            if bool(existing["is_indexed"]):
                return DocumentUploadResponse(document_id=document_id, task_id=None, status="already_indexed", reused=True)
            indexing_status = str(existing.get("indexing_status") or "pending")
            if indexing_status in {"pending", "running"}:
                return DocumentUploadResponse(document_id=document_id, task_id=None, status=indexing_status, reused=True)
            target = self.upload_dir / f"{document_id}.pdf"
            if not target.exists():
                target.write_bytes(content)
            self.repository.update_document_indexing_status(document_id, "pending")
            return self._enqueue_ingestion(document_id, target, filename, file_hash, file_size, reused=True)

        document_id = uuid4().hex
        target = self.upload_dir / f"{document_id}.pdf"
        target.write_bytes(content)
        pdf_key = document_pdf_key(document_id)
        self._store_pdf_object(pdf_key, content)
        self.repository.add_document(
            document_id,
            filename,
            owner_id=self.owner_id,
            file_hash=file_hash,
            file_size=file_size,
            original_filename=filename,
            indexing_status="pending",
            pdf_object_key=pdf_key,
        )
        return self._enqueue_ingestion(document_id, target, filename, file_hash, file_size)

    def list_documents(self) -> list[DocumentSummary]:
        rows = self.repository.list_documents(owner_id=self.owner_id)
        return [
            DocumentSummary(
                document_id=str(row["document_id"]),
                filename=str(row["filename"]),
                created_at=str(row["created_at"]),
                chunk_count=int(row["chunk_count"]),
                is_indexed=bool(row["is_indexed"]),
                file_hash=str(row["file_hash"]) if row["file_hash"] else None,
                indexing_status=str(row["indexing_status"]),
                indexing_error=str(row["indexing_error"]) if row["indexing_error"] else None,
            )
            for row in rows
        ]

    def find_existing_document_by_storage_hash(
        self,
        file_hash: str,
        file_size: int,
        original_filename: str | None = None,
    ) -> dict[str, object] | None:
        for document in self.repository.list_documents(owner_id=self.owner_id):
            document_path = self.upload_dir / f"{document['document_id']}.pdf"
            if not document_path.exists() or document_path.stat().st_size != file_size:
                continue
            digest = sha256(document_path.read_bytes()).hexdigest()
            if digest != file_hash:
                continue
            document_id = str(document["document_id"])
            pdf_key = document_pdf_key(document_id)
            self.repository.update_document_hash(
                document_id,
                file_hash,
                file_size,
                original_filename,
                pdf_object_key=pdf_key,
            )
            document["file_hash"] = file_hash
            if original_filename:
                document["filename"] = original_filename
            return document
        return None

    def _enqueue_ingestion(
        self,
        document_id: str,
        target: Path,
        filename: str,
        file_hash: str,
        file_size: int,
        reused: bool = False,
    ) -> DocumentUploadResponse:
        try:
            task = self.ingestion_task.delay(document_id, str(target), filename, file_hash, file_size)
        except Exception as exc:
            error = f"Failed to enqueue ingestion task: {exc}"
            self.repository.update_document_indexing_status(document_id, "failed", error=error)
            return DocumentUploadResponse(document_id=document_id, task_id=None, status="failed", reused=reused, error=error)
        return DocumentUploadResponse(document_id=document_id, task_id=task.id, status="pending", reused=reused)

    def _store_pdf_object(self, pdf_key: str, content: bytes) -> None:
        self.object_storage.put_bytes(pdf_key, content, content_type="application/pdf")
