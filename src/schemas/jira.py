"""Domain contracts exchanged between the agents and the Jira service.

These are *application-level* models. They are deliberately decoupled from any
transport or SDK representation: they do not import or expose ``atlassian``,
``anthropic``, FastAPI, or LangGraph types. The Jira service is responsible for
translating raw Jira REST payloads into (and out of) these models, so the rest
of the application depends only on this stable, strongly typed vocabulary.

The module contains contracts only — no business logic, service logic, API
routing, or workflow orchestration.

Naming convention: fields use application-native ``snake_case`` (e.g.
``account_id``), never the Jira wire format (``accountId``). Mapping between the
two lives in the service layer, not here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl, model_validator

# --------------------------------------------------------------------------- #
# Shared base
# --------------------------------------------------------------------------- #
class _DomainModel(BaseModel):
    """Base for every Jira domain contract.

    All models are immutable (``frozen``) and reject unknown fields
    (``extra="forbid"``), so a contract mismatch surfaces as a validation error
    rather than silently passing untyped data through the application.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# Jira's ``statusCategory.key`` is a small, well-defined enumeration, unlike the
# per-project status/issue-type *names*, which are free-form and left as ``str``.
StatusCategoryKey = Literal["new", "indeterminate", "done", "undefined"]


# --------------------------------------------------------------------------- #
# Reference / value objects
# --------------------------------------------------------------------------- #
class JiraUser(_DomainModel):
    """A Jira account, as referenced by issues or authentication checks."""

    account_id: str = Field(description="Opaque, stable Atlassian account identifier.")
    display_name: str = Field(description="Human-readable name shown in the Jira UI.")
    # Email is frequently withheld by Jira privacy settings, so it is optional.
    email: EmailStr | None = Field(
        default=None, description="Account email, when exposed by Jira privacy settings."
    )
    active: bool = Field(default=True, description="Whether the account is active.")


class JiraProject(_DomainModel):
    """A Jira project."""

    id: str = Field(description="Numeric project id, as an opaque string.")
    key: str = Field(min_length=1, description="Project key, e.g. 'ATLAS'.")
    name: str = Field(min_length=1, description="Human-readable project name.")
    project_type_key: str = Field(
        default="software", description="Jira project type, e.g. 'software'."
    )
    lead: JiraUser | None = Field(default=None, description="Project lead, when available.")


class JiraIssueType(_DomainModel):
    """The type of a Jira issue (e.g. 'Story', 'Task', 'Epic', 'Bug').

    Names are configurable per project, so this is intentionally a free-form
    string rather than a ``Literal``.
    """

    id: str = Field(description="Numeric issue-type id, as an opaque string.")
    name: str = Field(min_length=1, description="Issue-type name, e.g. 'Story'.")
    subtask: bool = Field(default=False, description="Whether this type is a subtask type.")


class JiraStatus(_DomainModel):
    """The workflow status of an issue."""

    name: str = Field(min_length=1, description="Status name, e.g. 'In Progress'.")
    category_key: StatusCategoryKey = Field(
        description="Status category key: new, indeterminate, done, or undefined."
    )


class JiraIssueReference(_DomainModel):
    """A lightweight pointer to an issue, without its full field set.

    Used wherever an issue must be referenced (e.g. a parent link or the result
    of a create call) but the complete issue payload is unnecessary.
    """

    id: str = Field(description="Numeric issue id, as an opaque string.")
    key: str = Field(min_length=1, description="Issue key, e.g. 'ATLAS-123'.")
    url: HttpUrl | None = Field(
        default=None, description="Canonical browse URL for the issue, when known."
    )


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class JiraIssueCreateRequest(_DomainModel):
    """Application intent to create a Jira issue.

    Carries only the data required to create an issue; it does not perform the
    creation. All fields except ``labels`` that are optional are genuinely
    optional in Jira.
    """

    project_key: str = Field(min_length=1, description="Key of the target project.")
    issue_type: str = Field(min_length=1, description="Issue-type name, e.g. 'Story'.")
    summary: str = Field(
        min_length=1, max_length=255, description="Issue summary (Jira caps this at 255)."
    )
    description: str | None = Field(default=None, description="Optional issue description.")
    labels: tuple[str, ...] = Field(default=(), description="Labels to attach to the issue.")
    assignee_account_id: str | None = Field(
        default=None, description="Assignee account id; unset leaves the issue unassigned."
    )
    priority: str | None = Field(
        default=None, description="Priority name, e.g. 'High'; unset uses the project default."
    )
    parent_key: str | None = Field(
        default=None, description="Parent issue key for subtasks or epic children."
    )


class JiraIssueUpdateRequest(_DomainModel):
    """Application intent to update fields on an existing Jira issue.

    Every field is optional; only the fields that are set are updated. A
    validator guarantees the request is non-empty so a no-op update cannot be
    dispatched to the service.
    """

    summary: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None)
    labels: tuple[str, ...] | None = Field(
        default=None, description="Replacement label set; None leaves labels unchanged."
    )
    assignee_account_id: str | None = Field(default=None)
    priority: str | None = Field(default=None)

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> "JiraIssueUpdateRequest":
        """Reject an update request that would change nothing."""
        if all(value is None for value in self.__dict__.values()):
            raise ValueError("JiraIssueUpdateRequest must set at least one field to update")
        return self


class JiraSearchRequest(_DomainModel):
    """A JQL search request with pagination and optional field projection."""

    jql: str = Field(min_length=1, description="Jira Query Language expression.")
    start_at: int = Field(default=0, ge=0, description="Zero-based index of the first result.")
    max_results: int = Field(
        default=50, ge=1, le=100, description="Page size (Jira caps this at 100)."
    )
    fields: tuple[str, ...] = Field(
        default=(), description="Field names to return; empty means the service default set."
    )


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class JiraIssueResponse(_DomainModel):
    """A fully materialized Jira issue returned to the application."""

    id: str = Field(description="Numeric issue id, as an opaque string.")
    key: str = Field(min_length=1, description="Issue key, e.g. 'ATLAS-123'.")
    summary: str = Field(description="Issue summary.")
    description: str | None = Field(default=None, description="Issue description, if any.")
    issue_type: JiraIssueType = Field(description="The issue's type.")
    status: JiraStatus = Field(description="The issue's current workflow status.")
    project_key: str = Field(min_length=1, description="Key of the owning project.")
    labels: tuple[str, ...] = Field(default=(), description="Labels attached to the issue.")
    assignee: JiraUser | None = Field(
        default=None, description="Assignee, or None when unassigned."
    )
    reporter: JiraUser | None = Field(
        default=None, description="Reporter, or None when unavailable."
    )
    url: HttpUrl = Field(description="Canonical browse URL for the issue.")
    created: datetime = Field(description="Creation timestamp (timezone-aware).")
    updated: datetime = Field(description="Last-update timestamp (timezone-aware).")


class JiraSearchResponse(_DomainModel):
    """A page of JQL search results."""

    total: int = Field(ge=0, description="Total number of issues matching the query.")
    start_at: int = Field(ge=0, description="Zero-based index of the first returned issue.")
    max_results: int = Field(ge=1, description="Page size used for this response.")
    issues: tuple[JiraIssueResponse, ...] = Field(
        default=(), description="Issues on this page."
    )
    is_last: bool = Field(description="Whether this is the final page of results.")


class JiraHealthResponse(_DomainModel):
    """Result of a lightweight Jira connectivity/authentication check.

    Contains no issue content; it only reports whether the service can reach and
    authenticate against Jira.
    """

    success: bool = Field(description="Whether Jira is reachable and authenticated.")
    base_url: HttpUrl = Field(description="Base URL that was checked.")
    account_id: str | None = Field(
        default=None, description="Authenticated account id, when the check succeeds."
    )
    message: str | None = Field(
        default=None, description="Optional human-readable detail (never a secret)."
    )
    checked_at: datetime = Field(description="When the check was performed (timezone-aware).")
