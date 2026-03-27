from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

# ======================================================================================
def _trace_id_from_request(request: Request) -> str:
    trace_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    return trace_id or str(uuid4())

# ======================================================================================
def _error_payload(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "status_code": status_code,
            "trace_id": _trace_id_from_request(request),
            "path": request.url.path,
            "method": request.method,
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return payload

# ======================================================================================
@dataclass(eq=False)
class AppError(Exception):
    """
    Exception applicative (métier) sérialisée de façon homogène.

    - status_code: code HTTP renvoyé
    - code: code applicatif stable (utile côté frontend)
    - message: message lisible
    - details: payload optionnel (dict/str/list/...)
    """

    message: str
    code: str = "app_error"
    status_code: int = 400
    details: Any | None = None

# ======================================================================================
class NotFoundError(AppError):
    status_code = 404
    code = "not_found"

# ======================================================================================
class ConflictError(AppError):
    status_code = 409
    code = "conflict"

# ======================================================================================
class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"

# ======================================================================================
class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"

# ======================================================================================
class ValidationError(AppError):
    status_code = 422
    code = "validation_error"

# ======================================================================================
class DatabaseError(AppError):
    status_code = 500
    code = "database_error"

# ======================================================================================
def add_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request=request,
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                request=request,
                status_code=422,
                code="request_validation_error",
                message="Requête invalide.",
                details={"errors": exc.errors()},
            ),
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else "Erreur HTTP."
        details = None if isinstance(detail, str) else detail
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request=request,
                status_code=exc.status_code,
                code="http_exception",
                message=message,
                details=details,
            ),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                request=request,
                status_code=500,
                code="sqlalchemy_error",
                message="Erreur base de données.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        from app.core.logger import logger
        logger.exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                request=request,
                status_code=500,
                code="internal_server_error",
                message="Erreur interne du serveur.",
            ),
        )
