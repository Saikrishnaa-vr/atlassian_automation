"""Shared, application-wide exception hierarchy.

This module is the single source of truth for the project's error types. Every
layer — configuration, services, agents, and the workflow — should raise (and
catch) the exceptions defined here rather than declaring duplicate local
classes.

The module is intentionally pure Python with no third-party dependencies and no
business, service, or workflow logic: it defines *types* only.

Hierarchy::

    ApplicationError                     (root of every application error)
    ├── ConfigurationError               (invalid / missing configuration)
    ├── ValidationError                  (domain data-integrity failure)
    ├── ServiceError                     (external-service interaction failure)
    │   ├── AnthropicServiceError        (Anthropic provider base)
    │   ├── JiraServiceError             (Jira provider base)
    │   ├── AuthenticationError          (auth/authorization, e.g. 401/403)
    │   ├── ConnectionError              (network/timeout failure)
    │   ├── RateLimitError               (rate limit exceeded, e.g. 429)
    │   ├── NotFoundError                (resource not found, e.g. 404)
    │   ├── InvalidRequestError          (rejected request, e.g. 400/422)
    │   └── ServerError                  (upstream server error, e.g. 5xx)
    ├── AgentError                       (reasoning-agent failure)
    ├── WorkflowError                    (orchestration failure)
    └── ApprovalError                    (human-approval gate failure)

Note: :class:`ConnectionError` and :class:`ValidationError` intentionally shadow
the builtin ``ConnectionError`` and Pydantic's ``ValidationError`` *within this
namespace*. Import them explicitly (e.g. ``from src.exceptions import
ServiceError``) to avoid ambiguity at call sites.
"""

from __future__ import annotations

__all__ = [
    "ApplicationError",
    "ConfigurationError",
    "ValidationError",
    "ServiceError",
    "AnthropicServiceError",
    "JiraServiceError",
    "AuthenticationError",
    "ConnectionError",
    "RateLimitError",
    "NotFoundError",
    "InvalidRequestError",
    "ServerError",
    "AgentError",
    "WorkflowError",
    "ApprovalError",
]


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class ApplicationError(Exception):
    """Base class for every error raised by this application.

    Catching :class:`ApplicationError` catches any error the application raises
    on purpose, while letting unexpected (programming) errors propagate.

    Args:
        message: A human-readable description. When omitted, the subclass's
            :attr:`default_message` is used.
        cause: The originating exception, if any. When provided it is attached
            as ``__cause__`` so the traceback shows the full chain even when the
            error is not raised with an explicit ``raise ... from`` statement.
    """

    #: Default message used when no explicit ``message`` is supplied. Subclasses
    #: override this to provide a meaningful, type-specific default.
    default_message: str = "An application error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self.message: str = message if message is not None else self.default_message
        super().__init__(self.message)
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return the human-readable message."""
        return self.message


# --------------------------------------------------------------------------- #
# Configuration & validation
# --------------------------------------------------------------------------- #
class ConfigurationError(ApplicationError):
    """Raised when application configuration is missing or invalid."""

    default_message = "Invalid or missing configuration."


class ValidationError(ApplicationError):
    """Raised when domain data fails an application-level integrity check.

    Distinct from Pydantic's ``ValidationError``; use this for validation
    performed by application code rather than by a Pydantic model.
    """

    default_message = "Domain validation failed."


# --------------------------------------------------------------------------- #
# Service layer
# --------------------------------------------------------------------------- #
class ServiceError(ApplicationError):
    """Base class for failures interacting with an external service."""

    default_message = "An external service error occurred."


class AnthropicServiceError(ServiceError):
    """Base class for errors originating from the Anthropic service layer."""

    default_message = "An Anthropic service error occurred."


class JiraServiceError(ServiceError):
    """Base class for errors originating from the Jira service layer."""

    default_message = "A Jira service error occurred."


class AuthenticationError(ServiceError):
    """Raised on authentication or authorization failure (e.g. HTTP 401/403)."""

    default_message = "Authentication with the service failed."


class ConnectionError(ServiceError):
    """Raised on network connectivity or timeout failures.

    Shadows the builtin ``ConnectionError`` within this module's namespace.
    """

    default_message = "Failed to connect to the service."


class RateLimitError(ServiceError):
    """Raised when a service rate limit has been exceeded (e.g. HTTP 429)."""

    default_message = "Service rate limit exceeded."


class NotFoundError(ServiceError):
    """Raised when a requested resource does not exist (e.g. HTTP 404)."""

    default_message = "The requested resource was not found."


class InvalidRequestError(ServiceError):
    """Raised when a service rejects a request as invalid (e.g. HTTP 400/422)."""

    default_message = "The service rejected the request as invalid."


class ServerError(ServiceError):
    """Raised on an upstream server-side error (e.g. HTTP 5xx)."""

    default_message = "The service reported a server-side error."


# --------------------------------------------------------------------------- #
# Agent / workflow layer
# --------------------------------------------------------------------------- #
class AgentError(ApplicationError):
    """Raised when a reasoning agent fails to produce a valid result."""

    default_message = "An agent error occurred."


class WorkflowError(ApplicationError):
    """Raised when workflow orchestration fails."""

    default_message = "A workflow error occurred."


class ApprovalError(ApplicationError):
    """Raised when the human-approval gate is violated or cannot be resolved.

    Examples include attempting to proceed past the gate without an approved
    decision, or receiving an unrecognized approval state.
    """

    default_message = "A human-approval error occurred."
