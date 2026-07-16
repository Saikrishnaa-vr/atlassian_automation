"""Infrastructure-only client wrapper for the Anthropic Claude API.

This module encapsulates *all* communication with the Anthropic Python SDK
behind a small, strongly typed interface. It is deliberately free of any
business logic, prompts, agent behaviour, workflow/LangGraph orchestration, or
Jira concerns — callers pass in fully-formed messages and receive a typed
response.

Responsibilities:
    * build and own the Anthropic SDK client (configured from ``get_settings``),
    * expose a minimal public API (:meth:`AnthropicService.generate` and
      :meth:`AnthropicService.health_check`),
    * retry transient/recoverable failures with bounded exponential backoff,
    * translate raw SDK exceptions into project-specific exceptions so callers
      never depend on ``anthropic`` internals,
    * emit structured, secret-free Loguru logs.

The project-specific exception hierarchy lives here for now; it can be promoted
to a shared ``src/exceptions`` module once other services need it.
"""

from __future__ import annotations

import time
from typing import Literal

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_SECONDS: float = 0.5
MAX_BACKOFF_SECONDS: float = 8.0

Role = Literal["user", "assistant"]


# --------------------------------------------------------------------------- #
# Project-specific exceptions
#
# No raw Anthropic SDK exception is ever allowed to escape this service. Every
# public method raises a subclass of ``AnthropicServiceError`` instead.
# --------------------------------------------------------------------------- #
class AnthropicServiceError(Exception):
    """Base class for all errors raised by :class:`AnthropicService`."""


class AnthropicConfigurationError(AnthropicServiceError):
    """Raised when the service cannot be constructed from configuration."""


class AnthropicAuthenticationError(AnthropicServiceError):
    """Raised on authentication/authorization failures (HTTP 401/403)."""


class AnthropicInvalidRequestError(AnthropicServiceError):
    """Raised on non-retryable client errors (HTTP 400/404/409/422)."""


class AnthropicRateLimitError(AnthropicServiceError):
    """Raised when the API rate limit is exhausted after retries (HTTP 429)."""


class AnthropicConnectionError(AnthropicServiceError):
    """Raised on network/timeout failures after retries are exhausted."""


class AnthropicServerError(AnthropicServiceError):
    """Raised on upstream server errors after retries (HTTP 5xx)."""


class AnthropicResponseError(AnthropicServiceError):
    """Raised when a response is received but cannot be interpreted."""


# --------------------------------------------------------------------------- #
# Typed request / response contracts
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    """A single conversational turn passed to the model."""

    role: Role
    content: str = Field(min_length=1)


class UsageInfo(BaseModel):
    """Token accounting for a single generation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by the request and response."""
        return self.input_tokens + self.output_tokens


class GenerationRequest(BaseModel):
    """Typed, validated input for :meth:`AnthropicService.generate`.

    Only transport-level parameters live here — no prompt templating or business
    logic. ``model`` defaults to the configured model when left unset.
    """

    model_config = ConfigDict(protected_namespaces=())

    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    model: str | None = None
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    # Optional by design: some models (e.g. claude-opus-4-8) reject an explicit
    # ``temperature``. When ``None`` the parameter is omitted and the provider
    # default applies.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class GenerationResponse(BaseModel):
    """Typed, SDK-agnostic result of a generation."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    model: str
    text: str
    stop_reason: str | None
    usage: UsageInfo


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class AnthropicService:
    """Thin, resilient wrapper around the Anthropic Claude API.

    The service owns a single SDK client and exposes a minimal interface. SDK
    retries are disabled so that retry behaviour is governed solely by this
    class and remains observable in the logs.

    Args:
        settings: Application settings. Defaults to :func:`get_settings`.
        client: A pre-built Anthropic client. Primarily an injection seam for
            testing; when omitted a client is constructed from ``settings``.
        max_retries: Maximum number of retries for transient failures.
        backoff_seconds: Base delay for exponential backoff between retries.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: Anthropic | None = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        if max_retries < 0:
            raise AnthropicConfigurationError("max_retries must be non-negative")
        if backoff_seconds < 0:
            raise AnthropicConfigurationError("backoff_seconds must be non-negative")

        self._settings: Settings = settings or get_settings()
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._default_model = self._settings.anthropic.model
        self._client = client or self._build_client(self._settings)

    # -- Construction ------------------------------------------------------- #
    @staticmethod
    def _build_client(settings: Settings) -> Anthropic:
        """Build an SDK client from settings, with SDK-level retries disabled."""
        return Anthropic(
            api_key=settings.anthropic.api_key.get_secret_value(),
            timeout=float(settings.app.request_timeout),
            max_retries=0,
        )

    # -- Public API --------------------------------------------------------- #
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for the given request.

        Args:
            request: A validated :class:`GenerationRequest`.

        Returns:
            A :class:`GenerationResponse` with the concatenated text output and
            token usage.

        Raises:
            AnthropicServiceError: A mapped, project-specific error. Raw SDK
                exceptions are never propagated.
        """
        model = request.model or self._default_model
        log = logger.bind(
            service="anthropic",
            operation="generate",
            model=model,
            message_count=len(request.messages),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        log.info("Anthropic generation started")

        payload = self._build_payload(request, model)
        raw = self._call_with_retries(
            lambda: self._client.messages.create(**payload),  # type: ignore[call-overload]
            log=log,
            operation="generate",
        )
        response = self._to_response(raw)

        log.bind(
            response_id=response.id,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        ).info("Anthropic generation succeeded")
        return response

    def health_check(self) -> bool:
        """Verify authentication and connectivity without generating content.

        Performs a lightweight model listing call. Returns ``True`` when the API
        is reachable and the credentials are valid; ``False`` otherwise. This
        method never raises and never logs secrets.

        Returns:
            Whether the Anthropic API is reachable and authenticated.
        """
        log = logger.bind(service="anthropic", operation="health_check")
        log.info("Anthropic health check started")
        try:
            self._client.models.list(limit=1)
        except APIStatusError as exc:
            log.bind(status_code=exc.status_code).warning(
                "Anthropic health check failed with API status error"
            )
            return False
        except APIConnectionError:
            log.warning("Anthropic health check failed with connection error")
            return False

        log.info("Anthropic health check succeeded")
        return True

    # -- Internals ---------------------------------------------------------- #
    @staticmethod
    def _build_payload(request: GenerationRequest, model: str) -> dict[str, object]:
        """Translate a request into SDK keyword arguments."""
        payload: dict[str, object] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.system is not None:
            payload["system"] = request.system
        return payload

    def _call_with_retries(
        self,
        operation_fn,  # type: ignore[no-untyped-def]  # local callable, no public exposure
        *,
        log,  # type: ignore[no-untyped-def]  # bound loguru logger
        operation: str,
    ) -> object:
        """Invoke ``operation_fn`` with bounded exponential backoff on transient errors.

        Only recoverable API/network errors are retried. Non-retryable errors
        (validation, auth, not-found) are mapped and raised immediately.
        """
        attempt = 0
        while True:
            try:
                return operation_fn()
            except (
                APIConnectionError,
                RateLimitError,
                InternalServerError,
            ) as exc:
                # Recoverable: retry until the budget is exhausted.
                if attempt >= self._max_retries:
                    log.bind(attempts=attempt + 1, error_type=type(exc).__name__).error(
                        "Anthropic request failed after exhausting retries"
                    )
                    raise self._map_exception(exc) from exc
                delay = self._backoff_delay(attempt)
                log.bind(
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    retry_in=delay,
                    error_type=type(exc).__name__,
                ).warning("Anthropic request failed with transient error; retrying")
                time.sleep(delay)
                attempt += 1
            except APIStatusError as exc:
                # Non-retryable client-side status errors (4xx other than 429).
                log.bind(status_code=exc.status_code, error_type=type(exc).__name__).error(
                    "Anthropic request failed with non-retryable status error"
                )
                raise self._map_exception(exc) from exc

    def _backoff_delay(self, attempt: int) -> float:
        """Compute the exponential backoff delay for a given attempt index."""
        return min(self._backoff_seconds * (2**attempt), MAX_BACKOFF_SECONDS)

    @staticmethod
    def _to_response(raw: object) -> GenerationResponse:
        """Convert a raw SDK message into a typed :class:`GenerationResponse`."""
        try:
            blocks = getattr(raw, "content", []) or []
            text = "".join(
                getattr(block, "text", "")
                for block in blocks
                if getattr(block, "type", None) == "text"
            )
            usage_obj = getattr(raw, "usage", None)
            usage = UsageInfo(
                input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
                output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
            )
            return GenerationResponse(
                id=getattr(raw, "id", "") or "",
                model=getattr(raw, "model", "") or "",
                text=text,
                stop_reason=getattr(raw, "stop_reason", None),
                usage=usage,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise AnthropicResponseError("Failed to parse Anthropic response") from exc

    @staticmethod
    def _map_exception(exc: APIStatusError | APIConnectionError) -> AnthropicServiceError:
        """Map a raw SDK exception onto a project-specific exception.

        The raw exception is preserved as the ``__cause__`` (via ``raise ... from``
        at the call site) but is never exposed as the public error type.
        """
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return AnthropicAuthenticationError("Anthropic authentication failed")
        if isinstance(exc, RateLimitError):
            return AnthropicRateLimitError("Anthropic rate limit exceeded")
        if isinstance(exc, InternalServerError):
            return AnthropicServerError("Anthropic server error")
        if isinstance(
            exc, (BadRequestError, NotFoundError, ConflictError, UnprocessableEntityError)
        ):
            return AnthropicInvalidRequestError("Anthropic rejected the request")
        if isinstance(exc, APIConnectionError):
            return AnthropicConnectionError("Anthropic connection error")
        # Any other APIStatusError (unexpected status code).
        return AnthropicServiceError("Unexpected Anthropic API error")
