from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.api.dependencies import get_document_service
from backend.app.schemas import BatchDocumentUploadItem, DocumentBatchUploadResponse, DocumentSummary, DocumentUploadResponse
from backend.app.services.documents import DocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])
DocumentServiceDependency = Annotated[DocumentService, Depends(get_document_service)]


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(service: DocumentServiceDependency, file: UploadFile = File(...)) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    return service.store_uploaded_pdf(file.filename, await file.read())


@router.post("/batch", response_model=DocumentBatchUploadResponse)
async def upload_documents(
    service: DocumentServiceDependency,
    files: list[UploadFile] = File(...),
) -> DocumentBatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    results: list[BatchDocumentUploadItem] = []
    for file in files:
        filename = file.filename or "unnamed"
        if not filename.lower().endswith(".pdf"):
            results.append(BatchDocumentUploadItem(filename=filename, status="rejected", error="Only PDF files are supported."))
            continue
        try:
            response = service.store_uploaded_pdf(filename, await file.read())
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
