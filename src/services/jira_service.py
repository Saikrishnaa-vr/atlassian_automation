"""Infrastructure-only client wrapper for the Jira REST API.

This service encapsulates *all* communication with the ``atlassian-python-api``
SDK behind a small, strongly typed interface. It is deliberately free of any
business, agent, prompt, or workflow logic: callers pass in domain models from
:mod:`src.schemas.jira` and receive domain models back. No raw SDK object or
SDK/``requests`` exception is ever exposed to callers.

Responsibilities:
    * build and own the Jira SDK client (configured from ``get_settings``),
    * expose a minimal read/write API returning only typed schema models,
    * translate SDK/transport exceptions into project-specific exceptions,
    * emit structured, secret-free Loguru logs.

The project-specific exception hierarchy lives here for now; it can be promoted
to a shared ``src/exceptions`` module once more services need it (mirroring the
approach taken in ``anthropic_service``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from atlassian import Jira
from loguru import logger
from pydantic import HttpUrl, TypeAdapter
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout

from src.config import Settings, get_settings
from src.schemas.jira import (
    JiraHealthResponse,
    JiraIssueCreateRequest,
    JiraIssueResponse,
    JiraIssueType,
    JiraProject,
    JiraSearchRequest,
    JiraSearchResponse,
    JiraStatus,
    JiraUser,
)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Project-specific exceptions
#
# No raw SDK / requests exception is allowed to escape this service. Every
# public method raises a subclass of ``JiraServiceError`` instead.
# --------------------------------------------------------------------------- #
class JiraServiceError(Exception):
    """Base class for all errors raised by :class:`JiraService`."""


class JiraConfigurationError(JiraServiceError):
    """Raised when the service cannot be constructed from configuration."""


class JiraAuthenticationError(JiraServiceError):
    """Raised on authentication/authorization failures (HTTP 401/403)."""


class JiraNotFoundError(JiraServiceError):
    """Raised when a requested resource does not exist (HTTP 404)."""


class JiraInvalidRequestError(JiraServiceError):
    """Raised on non-retryable client errors (HTTP 400/422)."""


class JiraRateLimitError(JiraServiceError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


class JiraConnectionError(JiraServiceError):
    """Raised on network/timeout failures."""


class JiraServerError(JiraServiceError):
    """Raised on upstream server errors (HTTP 5xx)."""


class JiraResponseError(JiraServiceError):
    """Raised when a response is received but cannot be interpreted."""


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class JiraService:
    """Thin, typed wrapper around the Jira REST API.

    The service owns a single SDK client and exposes a minimal interface. All
    inputs and outputs are :mod:`src.schemas.jira` models; SDK internals never
    cross the boundary.

    Args:
        settings: Application settings. Defaults to :func:`get_settings`.
        client: A pre-built Jira client. Primarily an injection seam for
            testing; when omitted a client is constructed from ``settings``.
    """

    def __init__(self, settings: Settings | None = None, client: Jira | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._base_url: str = str(self._settings.jira.base_url).rstrip("/")
        self._typed_base_url: HttpUrl = self._settings.jira.base_url
        self._client: Jira = client or self._build_client(self._settings)

    # -- Construction ------------------------------------------------------- #
    @staticmethod
    def _build_client(settings: Settings) -> Jira:
        """Build an authenticated Jira Cloud client from settings."""
        try:
            return Jira(
                url=str(settings.jira.base_url),
                username=settings.jira.email,
                password=settings.jira.api_token.get_secret_value(),
                cloud=True,
                timeout=settings.app.request_timeout,
            )
        except (ValueError, TypeError) as exc:  # defensive: never a generic Exception
            raise JiraConfigurationError("Failed to construct the Jira client") from exc

    # -- Public API --------------------------------------------------------- #
    def health_check(self) -> JiraHealthResponse:
        """Verify authentication and connectivity without touching business data.

        Performs a lightweight authenticated ``myself`` call. Returns a
        :class:`JiraHealthResponse` with ``success=True`` when Jira is reachable
        and the credentials are valid; ``success=False`` otherwise. This method
        never raises and never logs secrets.

        Returns:
            A typed health result.
        """
        log = logger.bind(service="jira", operation="health_check", base_url=self._base_url)
        log.info("Jira health check started")
        try:
            data = self._call(log, self._client.myself)
        except JiraServiceError as exc:
            log.bind(error_type=type(exc).__name__).warning("Jira health check failed")
            return JiraHealthResponse(
                success=False,
                base_url=self._typed_base_url,
                message=type(exc).__name__,
                checked_at=self._now(),
            )

        account_id = str(data.get("accountId", "")) or None
        log.bind(account_id=account_id).info("Jira health check succeeded")
        return JiraHealthResponse(
            success=True,
            base_url=self._typed_base_url,
            account_id=account_id,
            checked_at=self._now(),
        )

    def get_current_user(self) -> JiraUser:
        """Return the authenticated account.

        Returns:
            The current :class:`JiraUser`.

        Raises:
            JiraServiceError: A mapped, project-specific error.
        """
        log = logger.bind(service="jira", operation="get_current_user")
        log.info("Fetching authenticated Jira user")
        data = self._call(log, self._client.myself)
        user = self._map_user(data)
        if user is None:
            raise JiraResponseError("Jira returned an empty current-user payload")
        log.bind(account_id=user.account_id).info("Fetched authenticated Jira user")
        return user

    def get_project(self, project_key: str) -> JiraProject:
        """Fetch a project by key.

        Args:
            project_key: The project key, e.g. ``"ATLAS"``.

        Returns:
            The typed :class:`JiraProject`.

        Raises:
            JiraServiceError: A mapped, project-specific error.
        """
        log = logger.bind(service="jira", operation="get_project", project_key=project_key)
        log.info("Fetching Jira project")
        data = self._call(log, lambda: self._client.project(project_key))
        project = self._map_project(data)
        log.bind(project_id=project.id).info("Fetched Jira project")
        return project

    def create_issue(self, request: JiraIssueCreateRequest) -> JiraIssueResponse:
        """Create an issue and return the fully materialized result.

        Args:
            request: The typed creation intent.

        Returns:
            The created issue as a :class:`JiraIssueResponse`.

        Raises:
            JiraServiceError: A mapped, project-specific error.
        """
        log = logger.bind(
            service="jira",
            operation="create_issue",
            project_key=request.project_key,
            issue_type=request.issue_type,
        )
        log.info("Creating Jira issue")
        fields = self._build_create_fields(request)
        created = self._call(log, lambda: self._client.create_issue(fields=fields))

        key = created.get("key") if isinstance(created, dict) else None
        if not key:
            raise JiraResponseError("Jira create response did not include an issue key")

        log.bind(issue_key=key).info("Created Jira issue; fetching full representation")
        return self.get_issue(str(key))

    def get_issue(self, issue_key: str) -> JiraIssueResponse:
        """Fetch a single issue by id or key.

        Args:
            issue_key: The issue key or id, e.g. ``"ATLAS-123"``.

        Returns:
            The typed :class:`JiraIssueResponse`.

        Raises:
            JiraServiceError: A mapped, project-specific error.
        """
        log = logger.bind(service="jira", operation="get_issue", issue_key=issue_key)
        log.info("Fetching Jira issue")
        data = self._call(log, lambda: self._client.get_issue(issue_key))
        issue = self._map_issue(data)
        log.bind(issue_id=issue.id, status=issue.status.name).info("Fetched Jira issue")
        return issue

    def search_issues(self, request: JiraSearchRequest) -> JiraSearchResponse:
        """Run a JQL search and return a typed page of results.

        Args:
            request: The typed search request.

        Returns:
            A :class:`JiraSearchResponse` page.

        Raises:
            JiraServiceError: A mapped, project-specific error.
        """
        log = logger.bind(
            service="jira",
            operation="search_issues",
            start_at=request.start_at,
            max_results=request.max_results,
        )
        log.info("Searching Jira issues")
        fields: str | list[str] = list(request.fields) if request.fields else "*all"
        raw = self._call(
            log,
            lambda: self._client.jql(
                jql=request.jql,
                fields=fields,
                start=request.start_at,
                limit=request.max_results,
            ),
        )
        response = self._map_search(raw, request)
        log.bind(total=response.total, returned=len(response.issues)).info(
            "Jira search completed"
        )
        return response

    # -- Execution / error mapping ----------------------------------------- #
    def _call(self, log: "logger.__class__", fn: Callable[[], T]) -> T:  # type: ignore[name-defined]
        """Execute an SDK call, mapping transport errors to domain exceptions."""
        try:
            return fn()
        except Timeout as exc:
            log.bind(error_type="Timeout").error("Jira request timed out")
            raise JiraConnectionError("Jira request timed out") from exc
        except RequestsConnectionError as exc:
            log.bind(error_type="ConnectionError").error("Jira connection error")
            raise JiraConnectionError("Jira connection error") from exc
        except HTTPError as exc:
            mapped = self._map_http_error(exc)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            log.bind(status_code=status, error_type=type(mapped).__name__).error(
                "Jira request failed with HTTP error"
            )
            raise mapped from exc
        except RequestException as exc:
            log.bind(error_type=type(exc).__name__).error("Jira transport error")
            raise JiraConnectionError("Jira transport error") from exc

    @staticmethod
    def _map_http_error(exc: HTTPError) -> JiraServiceError:
        """Map a ``requests`` HTTP error onto a project-specific exception."""
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403):
            return JiraAuthenticationError(f"Jira authentication failed (HTTP {status})")
        if status == 404:
            return JiraNotFoundError("Jira resource not found (HTTP 404)")
        if status == 429:
            return JiraRateLimitError("Jira rate limit exceeded (HTTP 429)")
        if status in (400, 422):
            return JiraInvalidRequestError(f"Jira rejected the request (HTTP {status})")
        if status is not None and status >= 500:
            return JiraServerError(f"Jira server error (HTTP {status})")
        return JiraServiceError(f"Unexpected Jira API error (HTTP {status})")

    # -- Mapping helpers ---------------------------------------------------- #
    @staticmethod
    def _now() -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc)

    @staticmethod
    def _map_user(data: dict[str, Any] | None) -> JiraUser | None:
        """Map a raw Jira user payload to a :class:`JiraUser` (or None)."""
        if not data:
            return None
        return JiraUser(
            account_id=str(data.get("accountId", "")),
            display_name=str(data.get("displayName", "")),
            email=data.get("emailAddress") or None,
            active=bool(data.get("active", True)),
        )

    def _map_project(self, data: dict[str, Any]) -> JiraProject:
        """Map a raw Jira project payload to a :class:`JiraProject`."""
        try:
            return JiraProject(
                id=str(data["id"]),
                key=str(data["key"]),
                name=str(data.get("name", data["key"])),
                project_type_key=str(data.get("projectTypeKey", "software")),
                lead=self._map_user(data.get("lead")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JiraResponseError("Failed to parse Jira project payload") from exc

    def _map_issue(self, data: dict[str, Any]) -> JiraIssueResponse:
        """Map a raw Jira issue payload to a :class:`JiraIssueResponse`."""
        try:
            fields = data.get("fields", {}) or {}
            issue_type = fields.get("issuetype", {}) or {}
            status = fields.get("status", {}) or {}
            category = (status.get("statusCategory", {}) or {}).get("key", "undefined")
            key = str(data["key"])
            issue_url = TypeAdapter(HttpUrl).validate_python(f"{self._base_url}/browse/{key}")
            return JiraIssueResponse(
                id=str(data["id"]),
                key=key,
                summary=str(fields.get("summary", "")),
                description=self._adf_to_text(fields.get("description")),
                issue_type=JiraIssueType(
                    id=str(issue_type.get("id", "")),
                    name=str(issue_type.get("name", "")),
                    subtask=bool(issue_type.get("subtask", False)),
                ),
                status=JiraStatus(
                    name=str(status.get("name", "Unknown")),
                    category_key=category,
                ),
                project_key=str((fields.get("project", {}) or {}).get("key", "")),
                labels=tuple(fields.get("labels", []) or ()),
                assignee=self._map_user(fields.get("assignee")),
                reporter=self._map_user(fields.get("reporter")),
                url=issue_url,
                created=fields["created"],
                updated=fields["updated"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JiraResponseError("Failed to parse Jira issue payload") from exc

    def _map_search(
        self, raw: dict[str, Any] | None, request: JiraSearchRequest
    ) -> JiraSearchResponse:
        """Map a raw JQL search payload to a :class:`JiraSearchResponse`."""
        if not isinstance(raw, dict):
            raise JiraResponseError("Jira search returned an unexpected payload")
        issues = tuple(self._map_issue(item) for item in raw.get("issues", []) or [])
        total = int(raw.get("total", len(issues)))
        start_at = int(raw.get("startAt", request.start_at))
        max_results = int(raw.get("maxResults", request.max_results))
        is_last = bool(raw.get("isLast", start_at + len(issues) >= total))
        return JiraSearchResponse(
            total=total,
            start_at=start_at,
            max_results=max_results if max_results >= 1 else request.max_results,
            issues=issues,
            is_last=is_last,
        )

    @staticmethod
    def _build_create_fields(request: JiraIssueCreateRequest) -> dict[str, Any]:
        """Translate a create request into a Jira ``fields`` payload."""
        fields: dict[str, Any] = {
            "project": {"key": request.project_key},
            "summary": request.summary,
            "issuetype": {"name": request.issue_type},
        }
        if request.description is not None:
            fields["description"] = request.description
        if request.labels:
            fields["labels"] = list(request.labels)
        if request.assignee_account_id is not None:
            fields["assignee"] = {"accountId": request.assignee_account_id}
        if request.priority is not None:
            fields["priority"] = {"name": request.priority}
        if request.parent_key is not None:
            fields["parent"] = {"key": request.parent_key}
        return fields

    @classmethod
    def _adf_to_text(cls, value: Any) -> str | None:
        """Best-effort extraction of plain text from a Jira description field.

        Handles plain strings (API v2), Atlassian Document Format dicts
        (API v3), and ``None``. Returns ``None`` when no text is present.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value or None
        if isinstance(value, dict):
            parts: list[str] = []
            if value.get("type") == "text" and isinstance(value.get("text"), str):
                parts.append(value["text"])
            for child in value.get("content", []) or []:
                extracted = cls._adf_to_text(child)
                if extracted:
                    parts.append(extracted)
            text = "".join(parts).strip()
            return text or None
        return None
