"""Domain models for Sprint planning and execution.

These are *application-level* contracts shared by the future Sprint Planning
Agent, the Reporting Agent, the LangGraph workflow, and the JiraService. They
are intentionally free of any SDK, transport, or framework dependency (no
Anthropic, Jira SDK, FastAPI, or LangGraph imports) so the same vocabulary flows
unchanged across every layer.

The module contains contracts only — no business, service, or workflow logic,
no planning algorithms, and no reporting calculations. Validators here enforce
*data integrity* only (non-blank name, ordered dates, no duplicates, completed
subsets of planned, consistent timestamps, non-negative velocity, positive
capacity).

A Sprint references its stories and tasks only by opaque identifier, never by
importing the story/task models, so the schemas stay decoupled. The base model
and helper validators mirror the conventions in ``story.py``, ``epic.py``, and
``task.py`` and are candidates for later consolidation into a shared
``schemas/base`` module.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import Field, computed_field, field_validator, model_validator

from src.schemas.base import (
    DomainModel as _DomainModel,
    clean_unique_tokens as _clean_unique_tokens,
    require_non_blank as _require_non_blank,
)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class SprintStatus(str, Enum):
    """Lifecycle state of a sprint.

    ``PLANNED`` — scoped but not yet started (Jira 'future').
    ``ACTIVE`` — currently in progress.
    ``COMPLETED`` — work finished, pending closure.
    ``CLOSED`` — formally closed (Jira 'closed').
    ``CANCELLED`` — abandoned before completion.
    """

    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
class SprintGoal(_DomainModel):
    """The objective a sprint commits to achieving."""

    description: str = Field(min_length=1, description="The sprint goal, in plain language.")
    success_criteria: str | None = Field(
        default=None, description="Optional statement of what success looks like."
    )

    @field_validator("description")
    @classmethod
    def _description_not_blank(cls, value: str) -> str:
        """Sprint goal description cannot be blank."""
        return _require_non_blank(value, field="description")


class SprintMetrics(_DomainModel):
    """Immutable reporting snapshot for a sprint.

    This is a pure data container: every value is supplied by the caller. It
    performs no calculations and holds no business logic — validators only
    enforce that the recorded numbers are internally non-contradictory.
    """

    total_stories: int = Field(ge=0, description="Number of stories in the sprint.")
    completed_stories: int = Field(ge=0, description="Number of stories completed.")
    total_tasks: int = Field(ge=0, description="Number of tasks in the sprint.")
    completed_tasks: int = Field(ge=0, description="Number of tasks completed.")
    completion_percentage: float = Field(
        ge=0.0, le=100.0, description="Recorded completion percentage (0–100)."
    )
    velocity: float = Field(ge=0.0, description="Recorded velocity (completed effort).")
    spillover_count: int = Field(
        ge=0, description="Number of items that spilled over to a later sprint."
    )

    @model_validator(mode="after")
    def _counts_consistent(self) -> "SprintMetrics":
        """Completed counts cannot exceed their totals (integrity, not calculation)."""
        if self.completed_stories > self.total_stories:
            raise ValueError("completed_stories cannot exceed total_stories")
        if self.completed_tasks > self.total_tasks:
            raise ValueError("completed_tasks cannot exceed total_tasks")
        return self


# --------------------------------------------------------------------------- #
# Aggregate models
# --------------------------------------------------------------------------- #
class Sprint(_DomainModel):
    """A time-boxed iteration grouping planned stories and tasks.

    ``id`` is optional because a sprint is drafted before it is assigned any
    identifier (local or Jira). ``velocity``, ``capacity``, ``metrics``,
    ``created_at``, and ``updated_at`` are optional because they are frequently
    unknown at planning time. Completed id sets must be subsets of the
    corresponding planned id sets.
    """

    id: str | None = Field(default=None, description="Optional sprint identifier.")
    name: str = Field(min_length=1, max_length=255, description="Human-readable sprint name.")
    goal: SprintGoal = Field(description="The objective the sprint commits to.")
    status: SprintStatus = Field(
        default=SprintStatus.PLANNED, description="Lifecycle state of the sprint."
    )
    start_date: date = Field(description="Sprint start date (inclusive).")
    end_date: date = Field(description="Sprint end date (inclusive).")
    planned_story_ids: tuple[str, ...] = Field(
        default=(), description="Identifiers of stories planned into the sprint; unique."
    )
    planned_task_ids: tuple[str, ...] = Field(
        default=(), description="Identifiers of tasks planned into the sprint; unique."
    )
    completed_story_ids: tuple[str, ...] = Field(
        default=(), description="Identifiers of completed stories; subset of planned."
    )
    completed_task_ids: tuple[str, ...] = Field(
        default=(), description="Identifiers of completed tasks; subset of planned."
    )
    velocity: float | None = Field(
        default=None, ge=0.0, description="Optional recorded velocity for the sprint."
    )
    capacity: float | None = Field(
        default=None, gt=0.0, description="Optional team capacity for the sprint."
    )
    metrics: SprintMetrics | None = Field(
        default=None, description="Optional reporting snapshot for the sprint."
    )
    created_at: datetime | None = Field(
        default=None, description="Optional creation timestamp."
    )
    updated_at: datetime | None = Field(
        default=None, description="Optional last-update timestamp."
    )

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        """Sprint name cannot be blank."""
        return _require_non_blank(value, field="name")

    @field_validator(
        "planned_story_ids",
        "planned_task_ids",
        "completed_story_ids",
        "completed_task_ids",
    )
    @classmethod
    def _ids_unique(cls, value: tuple[str, ...], info) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
        """Each id collection must be unique and non-blank."""
        return _clean_unique_tokens(value, field=info.field_name)

    @model_validator(mode="after")
    def _dates_ordered(self) -> "Sprint":
        """The sprint end date cannot precede its start date."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        return self

    @model_validator(mode="after")
    def _completed_subset_of_planned(self) -> "Sprint":
        """Completed id sets must be subsets of the planned id sets."""
        if not set(self.completed_story_ids) <= set(self.planned_story_ids):
            raise ValueError("completed_story_ids must be a subset of planned_story_ids")
        if not set(self.completed_task_ids) <= set(self.planned_task_ids):
            raise ValueError("completed_task_ids must be a subset of planned_task_ids")
        return self

    @model_validator(mode="after")
    def _timestamps_consistent(self) -> "Sprint":
        """When both timestamps are present, updated_at cannot precede created_at."""
        if (
            self.created_at is not None
            and self.updated_at is not None
            and self.updated_at < self.created_at
        ):
            raise ValueError("updated_at must not be earlier than created_at")
        return self


class SprintCollection(_DomainModel):
    """An ordered set of sprints, e.g. a release train or program increment."""

    sprints: tuple[Sprint, ...] = Field(
        default=(), description="The sprints in this collection."
    )
    program_id: str | None = Field(
        default=None, description="Optional identifier of the owning program/roadmap."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of sprints in the collection."""
        return len(self.sprints)

    @model_validator(mode="after")
    def _unique_sprint_ids(self) -> "SprintCollection":
        """Sprint identifiers, when present, must be unique within the collection."""
        ids = [sprint.id for sprint in self.sprints if sprint.id is not None]
        if len(set(ids)) != len(ids):
            raise ValueError("sprint identifiers must be unique within a collection")
        return self
