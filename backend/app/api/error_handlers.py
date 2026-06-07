import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.domain.errors import DomainError

logger = logging.getLogger(__name__)


def error_payload(code: str, message: str, request_id: str | None, details: dict[str, Any] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if details:
        payload["error"]["details"] = details
    return payload


async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "Domain error",
        extra={
            "request_id": request_id,
            "error_code": exc.code,
            "details": exc.details,
        },
        exc_info=exc.cause,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.code, exc.user_message, request_id, exc.details),
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content=error_payload("REQUEST_VALIDATION_FAILED", "Request validation failed.", request_id, {"errors": exc.errors()}),
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "Unhandled API error",
        extra={"request_id": request_id, "error_code": "INTERNAL_SERVER_ERROR"},
    )
    return JSONResponse(
        status_code=500,
        content=error_payload("INTERNAL_SERVER_ERROR", "Server error. Please check the API logs.", request_id),
    )
