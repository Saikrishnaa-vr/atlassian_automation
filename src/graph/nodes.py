"""Orchestration nodes for the LangGraph workflow.

Each function in this module is a *node* in the MVP pipeline::

    raw_requirement
        → requirement_node          (RequirementAgent)
        → story_generation_node     (StoryAgent)
        → approval_gate_node        (hard-stop gate)
        → jira_node                 (JiraService — only when APPROVED)
        → COMPLETED

Nodes are pure Python callables: they receive a :class:`~src.graph.state.WorkflowState`
snapshot and return an updated snapshot via ``model_copy``.  They never mutate
the received state in place.

Responsibility boundaries
-------------------------
* Nodes are **thin orchestration only**.  Business logic belongs in agents;
  transport logic belongs in services.  A node should fit in 20–40 lines.
* Nodes **must not** contain prompt text, JSON parsing, or HTTP calls.
* The approval hard-stop is enforced at two levels: ``approval_gate_node`` blocks
  any APPROVED-path advancement until the flag is set, and ``jira_node``
  (and ``make_jira_node``) redundantly refuses to proceed without APPROVED status.
  No Jira service method is ever called on an unapproved state.

``make_jira_node`` factory
--------------------------
``jira_node`` requires a Jira project key that is not stored in
:class:`~src.graph.state.WorkflowState` or in
:class:`~src.config.Settings` (a known Phase-3 gap).  The node is therefore
exposed as a factory ``make_jira_node(project_key)`` that returns a fully bound
``(state: WorkflowState) -> WorkflowState`` callable.  The workflow wires this
once at startup::

    jira_node = make_jira_node(project_key="ATLAS")

``normalized_requirement`` serialization
-----------------------------------------
:class:`~src.graph.state.WorkflowState` stores ``normalized_requirement`` as
``str | None`` (serialized JSON).  ``requirement_node`` serializes the
:class:`~src.agents.requirement_agent.NormalizedRequirement` result via
``model_dump_json()``; ``story_generation_node`` deserializes it back via
``model_validate_json()`` before invoking the :class:`~src.agents.story_agent.StoryAgent`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from loguru import logger

from src.agents.requirement_agent import RequirementAgent
from src.agents.story_agent import StoryAgent
from src.schemas.requirement import NormalizedRequirement
from src.exceptions import ApprovalError, WorkflowError
from src.graph.state import ApprovalStatus, ErrorInfo, WorkflowState, WorkflowStep
from src.schemas.jira import JiraIssueCreateRequest, JiraIssueResponse
from src.schemas.story import UserStory
from src.services.factory import get_jira_service
from src.services.jira_service import JiraServiceError as _LocalJiraServiceError


# --------------------------------------------------------------------------- #
# Module-private helpers
# --------------------------------------------------------------------------- #


def _story_to_create_request(story: UserStory, project_key: str) -> JiraIssueCreateRequest:
    """Map a :class:`~src.schemas.story.UserStory` to a Jira create request.

    Composes the story's persona/goal/benefit narrative and acceptance criteria
    into a plain-text description.  The service layer is responsible for any
    Atlassian Document Format (ADF) transformation required by Jira Cloud v3.

    Args:
        story: The approved user story to create in Jira.
        project_key: Destination Jira project key (e.g. ``"ATLAS"``).

    Returns:
        A typed :class:`~src.schemas.jira.JiraIssueCreateRequest` ready for
        :meth:`~src.services.jira_service.JiraService.create_issue`.
    """
    criteria_text = "\n".join(f"- {ac.text}" for ac in story.acceptance_criteria)
    description = (
        f"As a {story.persona}, I want {story.goal}, so that {story.benefit}.\n\n"
        f"Acceptance Criteria:\n{criteria_text}"
    )
    return JiraIssueCreateRequest(
        project_key=project_key,
        issue_type="Story",
        summary=story.title,
        description=description,
        labels=story.labels,
        priority=story.priority.value.capitalize(),
    )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def requirement_node(state: WorkflowState) -> WorkflowState:
    """Execute :class:`~src.agents.requirement_agent.RequirementAgent` and store the result.

    Reads ``raw_requirement`` from *state*, passes it to a fresh
    :class:`~src.agents.requirement_agent.RequirementAgent`, and stores the
    resulting :class:`~src.agents.requirement_agent.NormalizedRequirement` as
    serialized JSON in ``normalized_requirement``.

    On success, ``current_step`` advances to
    :attr:`~src.graph.state.WorkflowStep.REQUIREMENT_ANALYSIS`.
    On any failure, delegates to :func:`error_node` and returns a FAILED state.

    Args:
        state: Current workflow state snapshot.

    Returns:
        Updated state with ``normalized_requirement`` populated and
        ``current_step`` set to ``REQUIREMENT_ANALYSIS``, or a FAILED state.
    """
    log = logger.bind(node="requirement_node", workflow_id=state.workflow_id)
    log.info("Requirement node started")
    try:
        normalized = RequirementAgent().execute(state.raw_requirement)
        new_state = state.model_copy(update={
            "normalized_requirement": normalized.model_dump_json(),
            "current_step": WorkflowStep.REQUIREMENT_ANALYSIS,
            "metadata": state.metadata.model_copy(
                update={"step_count": state.metadata.step_count + 1}
            ),
        })
        log.info("Requirement node completed")
        return new_state
    except Exception as exc:
        log.bind(error_type=type(exc).__name__).error("Requirement node failed")
        return error_node(state, exc)


def story_generation_node(state: WorkflowState) -> WorkflowState:
    """Execute :class:`~src.agents.story_agent.StoryAgent` and store the result.

    Deserializes ``normalized_requirement`` from the state JSON string, passes
    it to a fresh :class:`~src.agents.story_agent.StoryAgent`, and stores the
    resulting :class:`~src.schemas.story.StoryCollection` on the state.

    On success, ``current_step`` advances to
    :attr:`~src.graph.state.WorkflowStep.AWAITING_APPROVAL` because the
    generated stories must be reviewed before any Jira action is taken.
    On any failure, delegates to :func:`error_node` and returns a FAILED state.

    Args:
        state: Current workflow state snapshot; ``normalized_requirement`` must
            be non-None (populated by a prior :func:`requirement_node` call).

    Returns:
        Updated state with ``generated_stories`` populated and ``current_step``
        set to ``AWAITING_APPROVAL``, or a FAILED state.
    """
    log = logger.bind(node="story_generation_node", workflow_id=state.workflow_id)
    log.info("Story generation node started")
    try:
        if not state.normalized_requirement:
            raise WorkflowError(
                "Story generation requires a normalized requirement — "
                "run requirement_node first"
            )
        normalized = NormalizedRequirement.model_validate_json(state.normalized_requirement)
        stories = StoryAgent().execute(normalized)
        new_state = state.model_copy(update={
            "generated_stories": stories,
            "current_step": WorkflowStep.AWAITING_APPROVAL,
            "metadata": state.metadata.model_copy(
                update={"step_count": state.metadata.step_count + 1}
            ),
        })
        log.bind(story_count=stories.count).info("Story generation node completed")
        return new_state
    except Exception as exc:
        log.bind(error_type=type(exc).__name__).error("Story generation node failed")
        return error_node(state, exc)


def approval_gate_node(state: WorkflowState) -> WorkflowState:
    """Enforce the human-approval hard-stop before any Jira interaction.

    Reads ``approval_status`` and branches on its value:

    * :attr:`~src.graph.state.ApprovalStatus.APPROVED` — advances
      ``current_step`` to :attr:`~src.graph.state.WorkflowStep.JIRA_CREATION`.
    * :attr:`~src.graph.state.ApprovalStatus.PENDING` — returns state
      **unchanged**; the workflow remains paused at ``AWAITING_APPROVAL``.
    * :attr:`~src.graph.state.ApprovalStatus.REJECTED` — transitions
      ``current_step`` to :attr:`~src.graph.state.WorkflowStep.FAILED`.

    This node never calls any Jira service method.  It only reads and routes on
    :attr:`~src.graph.state.ApprovalStatus`; boolean flags are explicitly
    prohibited by the project constraints.

    Args:
        state: Current workflow state snapshot.

    Returns:
        Updated state reflecting the approval decision, or the original state
        when the decision is still PENDING.
    """
    log = logger.bind(
        node="approval_gate_node",
        workflow_id=state.workflow_id,
        approval_status=state.approval_status.value,
    )
    log.info("Approval gate node evaluating")

    if state.approval_status == ApprovalStatus.APPROVED:
        log.info("Approval gate: APPROVED — advancing to Jira creation")
        return state.model_copy(update={
            "current_step": WorkflowStep.JIRA_CREATION,
            "metadata": state.metadata.model_copy(
                update={"step_count": state.metadata.step_count + 1}
            ),
        })

    if state.approval_status == ApprovalStatus.REJECTED:
        log.info("Approval gate: REJECTED — workflow halted by human decision")
        return state.model_copy(update={
            "current_step": WorkflowStep.FAILED,
            "metadata": state.metadata.model_copy(
                update={"step_count": state.metadata.step_count + 1}
            ),
        })

    # PENDING — no human decision has been recorded; leave state unchanged.
    log.info("Approval gate: PENDING — awaiting human decision; state unchanged")
    return state


def make_jira_node(project_key: str) -> Callable[[WorkflowState], WorkflowState]:
    """Return a configured Jira creation node bound to *project_key*.

    ``jira_node`` requires a Jira project key that is absent from both
    :class:`~src.graph.state.WorkflowState` and :class:`~src.config.Settings`
    (a known Phase-3 gap).  This factory captures the key once at wiring time
    so the returned callable satisfies the ``(WorkflowState) -> WorkflowState``
    contract expected by LangGraph::

        jira_node = make_jira_node(project_key="ATLAS")
        graph.add_node("jira", jira_node)

    The returned ``jira_node``:

    * Enforces the approval hard-stop: refuses to proceed unless
      ``approval_status == ApprovalStatus.APPROVED``.
    * Creates one Jira issue per story in ``generated_stories`` via
      :meth:`~src.services.jira_service.JiraService.create_issue`.
    * Stores all created :class:`~src.schemas.jira.JiraIssueResponse` objects
      in ``jira_results``.
    * On success, advances ``current_step`` to
      :attr:`~src.graph.state.WorkflowStep.COMPLETED`.
    * On any failure, delegates to :func:`error_node` and returns a FAILED state.

    Note: if any single ``create_issue`` call fails, the entire node fails.
    Partial results are **not** stored.  This is a known MVP limitation.

    Args:
        project_key: Target Jira project key, e.g. ``"ATLAS"``.

    Returns:
        A ``(state: WorkflowState) -> WorkflowState`` callable suitable for
        use as a LangGraph node.
    """

    def jira_node(state: WorkflowState) -> WorkflowState:
        log = logger.bind(
            node="jira_node",
            workflow_id=state.workflow_id,
            project_key=project_key,
        )
        log.info("Jira creation node started")

        if state.approval_status != ApprovalStatus.APPROVED:
            violation = ApprovalError(
                f"Jira creation node requires APPROVED status; "
                f"got {state.approval_status.value!r} — hard-stop enforced"
            )
            log.error("Approval hard-stop: Jira creation blocked")
            return error_node(state, violation)

        try:
            service = get_jira_service()
            results: list[JiraIssueResponse] = []
            for story in state.generated_stories.stories:
                log.debug(f"Creating Jira issue for story: {story.title!r}")
                request = _story_to_create_request(story, project_key)
                results.append(service.create_issue(request))

            now = datetime.now(timezone.utc)
            log.bind(issues_created=len(results)).info("Jira creation node completed")
            return state.model_copy(update={
                "jira_results": tuple(results),
                "current_step": WorkflowStep.COMPLETED,
                "metadata": state.metadata.model_copy(update={
                    "step_count": state.metadata.step_count + 1,
                    "completed_at": now,
                }),
            })
        except _LocalJiraServiceError as exc:
            log.bind(error_type=type(exc).__name__).error("Jira creation node: service error")
            return error_node(
                state,
                WorkflowError(f"Jira issue creation failed: {exc}", cause=exc),
            )
        except Exception as exc:
            log.bind(error_type=type(exc).__name__).error(
                "Jira creation node: unexpected error"
            )
            return error_node(state, exc)

    return jira_node


def error_node(state: WorkflowState, error: Exception) -> WorkflowState:
    """Populate structured error information and transition the workflow to FAILED.

    Called by other nodes when they catch an exception.  Records an
    :class:`~src.graph.state.ErrorInfo` snapshot (step, error type, message,
    timestamp) on the state and advances ``current_step`` to
    :attr:`~src.graph.state.WorkflowStep.FAILED`.

    Logs the failure at ERROR level without revealing secret content.  The
    message is taken from ``str(error)``; if that is empty, the exception class
    name is used as the fallback so :attr:`~src.graph.state.ErrorInfo.message`
    satisfies its ``min_length=1`` constraint.

    Args:
        state: State snapshot at the moment the error was caught.
        error: The exception that caused the failure.

    Returns:
        A new state with ``current_step = FAILED``, ``error`` populated, and
        ``metadata.completed_at`` set to the current UTC time.
    """
    now = datetime.now(timezone.utc)
    message = str(error) or type(error).__name__
    error_info = ErrorInfo(
        step=state.current_step,
        error_type=type(error).__name__,
        message=message,
        timestamp=now,
    )
    logger.bind(
        node="error_node",
        workflow_id=state.workflow_id,
        error_type=error_info.error_type,
    ).error(
        f"Workflow error in step {state.current_step.value!r}: {error_info.message}"
    )
    return state.model_copy(update={
        "current_step": WorkflowStep.FAILED,
        "error": error_info,
        "metadata": state.metadata.model_copy(update={
            "step_count": state.metadata.step_count + 1,
            "completed_at": now,
        }),
    })
