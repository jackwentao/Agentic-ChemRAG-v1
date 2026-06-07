from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.api.dependencies import get_document_service
from backend.app.config import Settings, get_settings
from backend.app.domain.errors import DomainError, upload_batch_too_large
from backend.app.schemas import BatchDocumentUploadItem, DocumentBatchUploadResponse, DocumentSummary, DocumentUploadResponse
from backend.app.services.documents import DocumentService
from backend.app.services.upload_validation import receive_pdf_upload

router = APIRouter(prefix="/api/documents", tags=["documents"])
DocumentServiceDependency = Annotated[DocumentService, Depends(get_document_service)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    service: DocumentServiceDependency,
    settings: SettingsDependency,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if hasattr(service, "store_uploaded_pdf_file"):
        upload = await receive_pdf_upload(file, settings)
        try:
            return service.store_uploaded_pdf_file(upload)
        finally:
            upload.path.unlink(missing_ok=True)
    return service.store_uploaded_pdf(file.filename, await file.read())


@router.post("/batch", response_model=DocumentBatchUploadResponse)
async def upload_documents(
    service: DocumentServiceDependency,
    settings: SettingsDependency,
    files: list[UploadFile] = File(...),
) -> DocumentBatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if len(files) > settings.upload_max_batch_files:
        raise upload_batch_too_large(settings.upload_max_batch_files)

    results: list[BatchDocumentUploadItem] = []
    for file in files:
        filename = file.filename or "unnamed"
        if not filename.lower().endswith(".pdf"):
            results.append(BatchDocumentUploadItem(filename=filename, status="rejected", error="Only PDF files are supported."))
            continue
        try:
            if hasattr(service, "store_uploaded_pdf_file"):
                upload = await receive_pdf_upload(file, settings)
                try:
                    response = service.store_uploaded_pdf_file(upload)
                finally:
                    upload.path.unlink(missing_ok=True)
            else:
                response = service.store_uploaded_pdf(filename, await file.read())
        except DomainError as exc:
            results.append(BatchDocumentUploadItem(filename=filename, status="rejected", error=exc.user_message))
            continue
        except Exception as exc:
            results.append(BatchDocumentUploadItem(filename=filename, status="failed", error=str(exc)))
            continue
        results.append(
            BatchDocumentUploadItem(
                filename=filename,
                document_id=response.document_id,
                task_id=response.task_id,
                status=response.status,
                reused=response.reused,
                error=response.error,
            )
        )
    return DocumentBatchUploadResponse(results=results)


@router.get("", response_model=list[DocumentSummary])
def list_documents(service: DocumentServiceDependency) -> list[DocumentSummary]:
    return service.list_documents()
