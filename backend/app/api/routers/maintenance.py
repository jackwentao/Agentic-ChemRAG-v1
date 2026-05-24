from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.dependencies import DEFAULT_OWNER_ID, get_ingestion_task, get_repository
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.schemas import TaskStatusResponse

router = APIRouter(prefix="/api", tags=["maintenance"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str, ingestion_task=Depends(get_ingestion_task)) -> TaskStatusResponse:
    try:
        result = ingestion_task.AsyncResult(task_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task status backend is unavailable: {exc}") from exc
    error = str(result.result) if result.failed() else None
    value = result.result if result.successful() and isinstance(result.result, int) else None
    return TaskStatusResponse(task_id=task_id, status=result.status.lower(), result=value, error=error)


@router.post("/maintenance/indexing/mark-stale-failed")
def mark_stale_indexing_failed(
    repository: Annotated[PgVectorRepository, Depends(get_repository)],
    timeout_minutes: int = Query(default=60, ge=1, le=1440),
) -> dict[str, object]:
    document_ids = repository.mark_stale_indexing_documents_failed(
        owner_id=DEFAULT_OWNER_ID,
        stale_after_minutes=timeout_minutes,
    )
    return {"updated": len(document_ids), "document_ids": document_ids}
