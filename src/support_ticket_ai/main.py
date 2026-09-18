"""FastAPI application factory, resource lifecycle, and safe error boundary."""

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError
from starlette.responses import Response

from support_ticket_ai.agent.context import ApplicationError
from support_ticket_ai.agent.execution import build_agent
from support_ticket_ai.api import router
from support_ticket_ai.config import Settings
from support_ticket_ai.dataset import Dataset

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, model: BaseChatModel | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configuration = settings or Settings()
        app.state.settings = configuration
        app.state.dataset = Dataset(configuration.data_path, configuration.data_timezone)
        with httpx.Client(timeout=configuration.provider_timeout) as sync_client:
            async with httpx.AsyncClient(timeout=configuration.provider_timeout) as async_client:
                app.state.agent = build_agent(
                    configuration, model, http_client=sync_client, http_async_client=async_client
                )
                try:
                    yield
                finally:
                    app.state.agent = None
                    app.state.dataset = None

    app = FastAPI(title="Support Ticket AI", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def request_identity(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = uuid4().hex
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request.state.request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
        )
        return response

    def error_response(
        request: Request, code: str, message: str, status: int, retry_after: str | None = None
    ) -> JSONResponse:
        headers = {"Retry-After": retry_after} if retry_after else None
        return JSONResponse(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            },
            status_code=status,
            headers=headers,
        )

    @app.exception_handler(ApplicationError)
    async def application_error(request: Request, error: ApplicationError) -> JSONResponse:
        return error_response(
            request, error.code, error.message, error.status_code, error.retry_after
        )

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation_error(request: Request, error: Exception) -> JSONResponse:
        return error_response(
            request,
            "invalid_input",
            "Input does not match the documented schema. Check values, lengths, and field names.",
            422,
        )

    @app.exception_handler(ValueError)
    async def calculation_error(request: Request, error: ValueError) -> JSONResponse:
        return error_response(
            request,
            "invalid_calculation",
            "The calculation arguments or date range are invalid.",
            422,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.error(
            json.dumps(
                {
                    "event": "unhandled_error",
                    "error_type": type(error).__name__,
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            )
        )
        return error_response(
            request,
            "internal_error",
            "An unexpected error occurred. Check the server logs using the request ID.",
            500,
        )

    return app


app = create_app()
