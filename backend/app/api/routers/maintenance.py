from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.app.api.dependencies import DEFAULT_OWNER_ID, get_ingestion_task, get_readiness_service, get_repository
from backend.app.config import Settings, get_settings
from backend.app.repositories.postgres import PgVectorRepository
from backend.app.schemas import ReadinessResponse, TaskStatusResponse
from backend.app.services.readiness import ReadinessService

router = APIRouter(prefix="/api", tags=["maintenance"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
def version(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    return {"app_version": settings.app_version}


@router.get("/ready", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}})
def readiness(service: Annotated[ReadinessService, Depends(get_readiness_service)]) -> ReadinessResponse | JSONResponse:
    report = service.inspect()
    response = ReadinessResponse(
        status=report.status,
        complete_qa_online=report.complete_qa_online,
        can_serve_degraded=report.can_serve_degraded,
        checks={
            name: {"status": check.status, "required": check.required, "detail": check.detail}
            for name, check in report.checks.items()
        },
    )
    if report.complete_qa_online:
        return response
    return JSONResponse(status_code=503, content=response.model_dump())


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
