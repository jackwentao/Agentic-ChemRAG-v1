from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from backend.app.api.dependencies import get_object_storage, get_repository
from backend.app.config import Settings, get_settings
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.services.object_storage import ObjectStorage, document_pdf_key

router = APIRouter(prefix="/api", tags=["assets"])


@router.get("/documents/{document_id}/pdf")
def get_document_pdf(
    document_id: str,
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    storage: Annotated[ObjectStorage, Depends(get_object_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    document = repository.get_document_asset(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    object_key = str(document.get("pdf_object_key") or document_pdf_key(document_id))
    filename = str(document.get("filename") or f"{document_id}.pdf")
    if storage.exists(object_key):
        stream = storage.open(object_key)
        return StreamingResponse(
            stream,
            media_type="application/pdf",
            headers={"Content-Disposition": inline_content_disposition(filename)},
            background=BackgroundTask(stream.close),
        )

    local_path = settings.upload_dir / f"{document_id}.pdf"
    if local_path.exists():
        return FileResponse(local_path, media_type="application/pdf", filename=filename, content_disposition_type="inline")
    raise HTTPException(status_code=404, detail="PDF not found.")


@router.get("/images/{image_id}")
def get_image(
    image_id: str,
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    storage: Annotated[ObjectStorage, Depends(get_object_storage)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    image = repository.get_image_asset(image_id)
    if image is not None:
        object_key = str(image["path"])
        if storage.exists(object_key):
            stream = storage.open(object_key)
            extension = str(image.get("extension") or "png")
            return StreamingResponse(
                stream,
                media_type=f"image/{extension}",
                background=BackgroundTask(stream.close),
            )
    for path in settings.image_dir.glob(f"{image_id}.*"):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")


def inline_content_disposition(filename: str) -> str:
    ascii_name = "".join(char if ord(char) < 128 and char not in {'"', "\\"} else "_" for char in filename)
    return f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
