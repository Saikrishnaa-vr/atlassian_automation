"""Workflow state — the single source of truth for graph execution.

This module defines the typed state that flows through every LangGraph node in
the MVP pipeline (Requirement → RequirementAgent → StoryAgent → Approval →
Jira). Each node receives a :class:`WorkflowState` snapshot and returns a new
one; nothing is mutated in place.

The module contains state contracts and supporting enumerations only: no
business logic, no service calls, no agent invocations, and no LangGraph-
specific imports. All agent-output fields reuse the strongly typed models from
:mod:`src.schemas` so the same vocabulary flows unchanged across every layer.

Key design decisions
--------------------
* :class:`ApprovalStatus` is an enum with three explicit values — ``PENDING``,
  ``APPROVED``, ``REJECTED`` — because boolean approval is explicitly prohibited
  by the project constraints.
* Downstream nodes **must** verify ``approval_status == ApprovalStatus.APPROVED``
  before calling any Jira service method. That check is enforced at the graph
  level in :mod:`src.graph.workflow`; it is documented here as the canonical
  reference for all nodes.
* Epic and task collections are present as empty placeholders; their agent
  implementations are deferred to a later phase, but the state shape remains
  stable so the graph wiring does not need to change when those phases land.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.epic import EpicCollection
from src.schemas.jira import JiraIssueResponse
from src.schemas.story import StoryCollection
from src.schemas.task import TaskCollection


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ApprovalStatus(str, Enum):
    """Explicit outcome of the human-approval gate.

    Using an enum (rather than a boolean) makes the three-way distinction
    between "not yet decided", "approved", and "rejected" unambiguous and
    prevents accidental truthiness checks on the approval field.

    ``PENDING``  — no approval decision has been recorded (the initial state).
    ``APPROVED`` — a human has approved the generated stories for Jira creation.
    ``REJECTED`` — a human has rejected the generated stories; the workflow stops.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class WorkflowStep(str, Enum):
    """Coarse-grained label for the step the workflow is currently executing.

    Each value corresponds to one logical phase of the MVP pipeline. ``FAILED``
    is a terminal error state; ``COMPLETED`` is the successful terminal state.
    """

    PENDING = "pending"
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    STORY_GENERATION = "story_generation"
    AWAITING_APPROVAL = "awaiting_approval"
    JIRA_CREATION = "jira_creation"
    COMPLETED = "completed"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Nested state models
# --------------------------------------------------------------------------- #


class ExecutionMetadata(BaseModel):
    """Timing and progress counters for a single workflow run."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime | None = Field(
        default=None,
        description="When the workflow started executing (timezone-aware).",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the workflow reached a terminal step (timezone-aware).",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of node executions completed so far in this run.",
    )

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed time in seconds, or None when start or completion is unknown."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class ErrorInfo(BaseModel):
    """Structured record of an error that halted or degraded the workflow.

    Captured by the node that caught the error and stored on the state so that
    callers can inspect failure details without parsing log output. The message
    field must never contain secrets.
    """

    model_config = ConfigDict(extra="forbid")

    step: WorkflowStep = Field(
        description="The step at which the error was recorded.",
    )
    error_type: str = Field(
        min_length=1,
        description="Qualified exception class name, e.g. 'AgentError'.",
    )
    message: str = Field(
        min_length=1,
        description="Human-readable error message. Must not contain secrets.",
    )
    timestamp: datetime = Field(
        description="When the error was recorded (timezone-aware).",
    )


# --------------------------------------------------------------------------- #
# Workflow state
# --------------------------------------------------------------------------- #


class WorkflowState(BaseModel):
    """Complete state snapshot for a single workflow execution.

    This is the single source of truth for everything that has happened or
    needs to happen in the current run. LangGraph nodes receive this object,
    derive an updated copy, and return it; nodes must never mutate a received
    state in place.

    Approval gate contract
    ~~~~~~~~~~~~~~~~~~~~~~
    The ``approval_status`` field is the gate that separates pre-Jira
    computation from any Jira service interaction. Nodes operating in the
    post-approval path **must** verify::

        state.approval_status == ApprovalStatus.APPROVED

    before calling any Jira service method. This check is enforced at the
    graph level in :mod:`src.graph.workflow`; it is documented here as the
    canonical reference for all nodes.
    """

    model_config = ConfigDict(extra="forbid")

    # -- Identity ------------------------------------------------------------ #
    workflow_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this workflow run.",
    )

    # -- Requirement --------------------------------------------------------- #
    raw_requirement: str = Field(
        min_length=1,
        description="Raw requirement text submitted by the user; never modified after creation.",
    )
    normalized_requirement: str | None = Field(
        default=None,
        description="Requirement text after RequirementAgent analysis and normalization.",
    )

    # -- Agent outputs ------------------------------------------------------- #
    generated_stories: StoryCollection = Field(
        default_factory=StoryCollection,
        description="User stories produced by the StoryAgent from the normalized requirement.",
    )
    # Placeholders for future agents — populated in later phases.
    epics: EpicCollection = Field(
        default_factory=EpicCollection,
        description="Epic collection placeholder; populated by the future EpicAgent.",
    )
    tasks: TaskCollection = Field(
        default_factory=TaskCollection,
        description="Task collection placeholder; populated by the future TaskAgent.",
    )

    # -- Approval ------------------------------------------------------------ #
    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description=(
            "Human-approval gate outcome. Must be APPROVED before the Jira creation "
            "node runs. PENDING means no decision has been recorded yet."
        ),
    )

    # -- Jira results -------------------------------------------------------- #
    jira_results: tuple[JiraIssueResponse, ...] = Field(
        default=(),
        description=(
            "Jira issues created during the JIRA_CREATION step, one per approved story. "
            "Empty until the workflow is approved and the Jira node completes."
        ),
    )

    # -- Execution tracking -------------------------------------------------- #
    current_step: WorkflowStep = Field(
        default=WorkflowStep.PENDING,
        description="The step the workflow is currently executing.",
    )
    metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Timing and progress metadata for this run.",
    )
    error: ErrorInfo | None = Field(
        default=None,
        description=(
            "Structured error detail populated when the workflow transitions to FAILED. "
            "None while the workflow is running normally."
        ),
    )
