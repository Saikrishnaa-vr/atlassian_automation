"""FastAPI application entry point for the Atlassian Automation service.

This module is the sole bootstrap layer.  It creates the :data:`app` object,
registers the requirements workflow router, mounts health endpoints, wires
global exception handlers, and logs startup/shutdown via a lifespan context.

It contains **no** business logic, LLM logic, Jira logic, prompt logic, or
workflow logic.  All of that lives in the layers below
(``src.agents``, ``src.graph``, ``src.services``).

Running the application
-----------------------
::

    # development — reload on file changes
    uv run uvicorn src.main:app --reload

    # production
    uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

Registered routes
-----------------
=================================  ========  ===================================
Path                               Method    Description
=================================  ========  ===================================
``/``                              ``GET``   Root — service identity and version
``/health``                        ``GET``   Liveness probe — no external calls
``/requirements``                  ``POST``  Submit requirement, run pipeline
``/requirements/{workflow_id}``    ``GET``   Retrieve workflow state
``/requirements/{id}/approval``    ``POST``  Record approval, resume pipeline
=================================  ========  ===================================

Exception handling
------------------
Two global handlers sit at the outermost boundary:

* :class:`~src.exceptions.ApplicationError` handler — maps each subtype to the
  semantically correct HTTP status code (400 / 401 / 404 / 429 / 500) and
  returns a structured ``{"error": …, "detail": …}`` JSON body.
* Bare :class:`Exception` handler — catches any exception that escapes all
  route handlers, logs it with a full traceback (Loguru only — never in the
  response), and returns a generic HTTP 500 without leaking internal detail.

``HTTPException`` and ``RequestValidationError`` are handled by FastAPI's own
built-in handlers (which are more specific than the :class:`Exception` handler)
so they are not overridden here.

Lifespan
--------
A minimal :func:`_lifespan` context manager logs startup / shutdown.  It
attempts to read the current environment label from :func:`~src.config.get_settings`
and degrades gracefully when the settings cannot be loaded (e.g. in test
environments where some required env vars are absent).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.api.routes import router as _requirements_router
from src.exceptions import (
    ApplicationError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)

# --------------------------------------------------------------------------- #
# Module-level constants
# --------------------------------------------------------------------------- #

#: Application name — matches the ``APP_NAME`` settings default.
_APP_NAME: str = "atlassian-automation"

#: Semantic version, kept in sync with ``pyproject.toml``.
_APP_VERSION: str = "0.1.0"

_APP_DESCRIPTION: str = (
    "Multi-agent AI system: converts raw requirements into structured, "
    "review-ready Jira work items — with a mandatory human approval gate "
    "before any Jira write. "
    "Built with **LangGraph**, **Claude (Anthropic)**, **FastAPI**, and the "
    "**Jira REST API**."
)

_STATIC_DIR: Path = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------- #
# Response models (health / root)
# --------------------------------------------------------------------------- #


class _RootResponse(BaseModel):
    """Response body for ``GET /``."""

    status: str
    service: str
    version: str


class _HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str


# --------------------------------------------------------------------------- #
# Exception status mapping
# --------------------------------------------------------------------------- #


def _http_status_for(exc: ApplicationError) -> int:
    """Return the most semantically appropriate HTTP status code for *exc*.

    Maps the :mod:`src.exceptions` hierarchy to HTTP status codes:

    * 400 — :class:`~src.exceptions.ValidationError`,
      :class:`~src.exceptions.InvalidRequestError`
    * 401 — :class:`~src.exceptions.AuthenticationError`
    * 404 — :class:`~src.exceptions.NotFoundError`
    * 429 — :class:`~src.exceptions.RateLimitError`
    * 500 — all other :class:`~src.exceptions.ApplicationError` subtypes

    Args:
        exc: The application exception to classify.

    Returns:
        An HTTP status code integer.
    """
    if isinstance(exc, (ValidationError, InvalidRequestError)):
        return 400
    if isinstance(exc, AuthenticationError):
        return 401
    if isinstance(exc, NotFoundError):
        return 404
    if isinstance(exc, RateLimitError):
        return 429
    return 500


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Log startup and shutdown; degrade gracefully when settings are absent.

    Startup attempts to read the current environment label from
    :func:`~src.config.get_settings` so the log line includes the deployment
    context.  If settings are unavailable (e.g. required env vars are missing
    in a stripped test environment) the exception is swallowed and the label
    falls back to ``"unknown"`` so the application still boots.
    """
    env_label = "unknown"
    try:
        from src.config import get_settings  # local import — settings may be absent

        env_label = get_settings().app.environment
    except Exception:
        pass  # credentials not required for the bootstrap itself
    logger.info(f"Starting {_APP_NAME} v{_APP_VERSION} [{env_label}]")
    yield
    logger.info(f"Shutdown complete — {_APP_NAME}")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title=_APP_NAME,
    description=_APP_DESCRIPTION,
    version=_APP_VERSION,
    lifespan=_lifespan,
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Global exception handlers
# --------------------------------------------------------------------------- #


@app.exception_handler(ApplicationError)
async def _application_error_handler(
    request: Request, exc: ApplicationError
) -> JSONResponse:
    """Convert :class:`~src.exceptions.ApplicationError` to a structured HTTP response.

    Route handlers in :mod:`src.api.routes` already wrap application errors in
    ``HTTPException`` before they reach here.  This handler therefore acts as a
    safety net for any application error that escapes a route — for example from
    middleware, background tasks, or future routes that omit explicit handling.

    The exception message is included in the response; secrets must **never**
    appear in :attr:`~src.exceptions.ApplicationError.message`.
    """
    status = _http_status_for(exc)
    logger.bind(
        error_type=type(exc).__name__,
        status=status,
        path=str(request.url.path),
    ).warning("Application error reached global handler")
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": exc.message},
    )


@app.exception_handler(Exception)
async def _unexpected_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch all unexpected exceptions and return a generic HTTP 500.

    Logs the full exception (including traceback) via Loguru so it appears in
    server logs without leaking any internal detail to the HTTP caller.  The
    response body contains only a generic message.

    ``HTTPException`` and ``RequestValidationError`` are handled by FastAPI's
    own built-in handlers, which are registered for more specific types and
    therefore take precedence over this catch-all.
    """
    logger.bind(
        error_type=type(exc).__name__,
        path=str(request.url.path),
    ).exception("Unexpected error reached global handler")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred.",
        },
    )


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #

app.include_router(_requirements_router)


# --------------------------------------------------------------------------- #
# Health / meta endpoints
# --------------------------------------------------------------------------- #


@app.get("/", response_model=_RootResponse, tags=["meta"])
def root() -> _RootResponse:
    """Return service identity and version.

    Intended as a lightweight "is anything running?" check.  Does not contact
    Anthropic, Jira, or any other external service.
    """
    return _RootResponse(status="ok", service=_APP_NAME, version=_APP_VERSION)


@app.get("/health", response_model=_HealthResponse, tags=["meta"])
def health() -> _HealthResponse:
    """Return a liveness probe response.

    Always returns ``{"status": "healthy"}`` as long as the process is alive.
    Does not contact Anthropic, Jira, or any other external service.
    """
    return _HealthResponse(status="healthy")


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    """Serve the interactive frontend for manual workflow testing."""
    return FileResponse(_STATIC_DIR / "index.html")
