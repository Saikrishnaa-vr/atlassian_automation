"""FastAPI router for the requirements workflow API.

This module is the sole API boundary between HTTP callers and the LangGraph
workflow pipeline. It is intentionally thin: it validates incoming requests,
delegates all computation to :mod:`src.graph.workflow`, and converts application
exceptions to HTTP responses. It contains **no** business logic, LLM logic,
Jira logic, or prompt logic.

Endpoints
---------
``POST /requirements``
    Submit a raw requirement. The workflow runs through requirement analysis
    and story generation, then halts at the approval gate with
    ``approval_status = PENDING``. Returns the :class:`~src.graph.state.WorkflowState`
    including a ``workflow_id`` for subsequent calls.

``GET /requirements/{workflow_id}``
    Retrieve the latest :class:`~src.graph.state.WorkflowState` for a submitted
    requirement.

``POST /requirements/{workflow_id}/approval``
    Record a human approval decision (``APPROVED``, ``REJECTED``, or ``PENDING``)
    and resume the workflow. When ``APPROVED`` the pipeline proceeds through the
    approval gate to Jira issue creation.

In-memory registry (temporary MVP placeholder)
-----------------------------------------------
Two module-level dicts act as an ephemeral workflow store:

``_registry`` (``dict[str, WorkflowState]``)
    Maps ``workflow_id`` → latest :class:`~src.graph.state.WorkflowState` snapshot.

``_project_keys`` (``dict[str, str]``)
    Maps ``workflow_id`` → the Jira project key supplied at submission time.
    Retained so the approval endpoint can re-use the same key without the
    caller repeating it.

**Both dicts are explicit MVP placeholders. They are lost on process restart,
carry no eviction policy, and are not safe for concurrent access or multi-process
deployments. They must be replaced by a persistent store before any production
use.**

Compiled-workflow cache
-----------------------
:func:`_get_workflow` wraps :func:`~src.graph.workflow.build_workflow` with
:func:`~functools.lru_cache`. The graph is compiled once per unique
``project_key`` and reused for all subsequent requests targeting that project.
Graph compilation is pure CPU work (no I/O), so the cache is safe across
requests.

Error handling
--------------
Application exceptions are caught at the endpoint level and converted to
:class:`~fastapi.HTTPException`:

* :class:`~src.exceptions.ValidationError` → 400 Bad Request
* :class:`~src.exceptions.ApplicationError` (all other subtypes) → 500
* Unexpected :class:`Exception` → 500 with a generic message (detail is not
  forwarded to avoid leaking internal context)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.exceptions import ApplicationError, ValidationError
from src.graph.state import ApprovalStatus, WorkflowState
from src.graph.workflow import build_workflow, run_workflow

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# --------------------------------------------------------------------------- #
# Module-level registry — temporary MVP placeholder
# --------------------------------------------------------------------------- #

#: Latest WorkflowState snapshot for every submitted requirement.
#: TEMPORARY: lost on restart; no eviction; not concurrency-safe.
_registry: dict[str, WorkflowState] = {}

#: Jira project key recorded at submission time, keyed by workflow_id.
#: TEMPORARY: same limitations as _registry.
_project_keys: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Compiled-workflow cache
# --------------------------------------------------------------------------- #


@lru_cache
def _get_workflow(project_key: str) -> CompiledStateGraph:
    """Return a compiled LangGraph workflow, building it once per project key.

    Graph compilation involves only Python object construction (no I/O or
    network calls). The :func:`~functools.lru_cache` ensures the same compiled
    graph instance is reused for all requests targeting ``project_key``, which
    avoids redundant construction in the common case where all requests target
    the same Jira project.

    Args:
        project_key: Target Jira project key, e.g. ``"ATLAS"``.

    Returns:
        A compiled :class:`~langgraph.graph.state.CompiledStateGraph` bound to
        ``project_key``.
    """
    logger.debug(f"Compiling workflow for project_key={project_key!r}")
    return build_workflow(project_key)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class RequirementRequest(BaseModel):
    """Request body for ``POST /requirements``.

    ``project_key`` is required because
    :func:`~src.graph.workflow.build_workflow` binds it into the Jira creation
    node at wiring time. There is no application-level default; callers must
    supply the target Jira project explicitly.
    """

    requirement: str = Field(
        min_length=1,
        description=(
            "Raw requirement text to analyze and convert into Jira user stories. "
            "Must be non-empty."
        ),
    )
    project_key: str = Field(
        min_length=1,
        description=(
            "Target Jira project key for issue creation, e.g. ``'ATLAS'``. "
            "Bound into the workflow at compile time and stored for the approval step."
        ),
    )


class ApprovalRequest(BaseModel):
    """Request body for ``POST /requirements/{workflow_id}/approval``.

    Accepts only the three values of :class:`~src.graph.state.ApprovalStatus`
    (``"approved"``, ``"rejected"``, ``"pending"``). Boolean values are
    explicitly prohibited by the project approval-gate constraint.
    """

    approval: ApprovalStatus = Field(
        description=(
            "Human approval decision for the generated stories. "
            "Use ``'approved'`` to proceed to Jira creation, ``'rejected'`` to "
            "halt the workflow, or ``'pending'`` to leave the gate unchanged."
        ),
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/requirements", tags=["requirements"])


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("", response_model=WorkflowState, status_code=201)
def submit_requirement(body: RequirementRequest) -> WorkflowState:
    """Submit a raw requirement and start the workflow pipeline.

    Creates a new :class:`~src.graph.state.WorkflowState`, executes the
    requirement-analysis and story-generation nodes, then halts at the approval
    gate (``approval_status = PENDING``, ``current_step = AWAITING_APPROVAL``).
    Stores the final state and associated project key in the in-memory registry
    so subsequent endpoints can retrieve and resume the run.

    The ``workflow_id`` in the response is generated automatically by the state
    model. Callers must retain it to use the approval and retrieval endpoints.

    Args:
        body: The raw requirement text and target Jira project key.

    Returns:
        The :class:`~src.graph.state.WorkflowState` snapshot after the workflow
        first halts, including ``workflow_id``, ``generated_stories``, and
        ``current_step``.

    Raises:
        HTTPException 422: FastAPI/Pydantic rejects a malformed or missing field.
        HTTPException 400: The requirement text fails an application-level check.
        HTTPException 500: An error occurs during workflow execution.
    """
    log = logger.bind(endpoint="POST /requirements", project_key=body.project_key)
    log.info("Requirement submission received")
    try:
        initial_state = WorkflowState(raw_requirement=body.requirement)
        compiled = _get_workflow(body.project_key)
        final_state = run_workflow(compiled, initial_state)

        _registry[final_state.workflow_id] = final_state
        _project_keys[final_state.workflow_id] = body.project_key

        log.bind(
            workflow_id=final_state.workflow_id,
            step=final_state.current_step.value,
        ).info("Requirement submission completed")
        return final_state

    except ValidationError as exc:
        log.bind(error=str(exc)).warning("Validation error in requirement submission")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ApplicationError as exc:
        log.bind(error_type=type(exc).__name__, error=str(exc)).error(
            "Application error in requirement submission"
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        log.bind(error_type=type(exc).__name__).error(
            "Unexpected error in requirement submission"
        )
        raise HTTPException(status_code=500, detail="An unexpected error occurred") from exc


@router.get("/{workflow_id}", response_model=WorkflowState)
def get_requirement(workflow_id: str) -> WorkflowState:
    """Retrieve the current workflow state for a submitted requirement.

    Looks up ``workflow_id`` in the in-memory registry and returns the latest
    :class:`~src.graph.state.WorkflowState` snapshot. The snapshot reflects the
    state after the most recent workflow execution (initial submission or a
    subsequent approval call).

    Note: because the registry is in-memory, workflow states are lost on process
    restart. A 404 response after a restart does not indicate the workflow never
    existed.

    Args:
        workflow_id: The ``workflow_id`` returned by ``POST /requirements``.

    Returns:
        The latest :class:`~src.graph.state.WorkflowState` for the run.

    Raises:
        HTTPException 404: ``workflow_id`` is not present in the registry.
    """
    state = _registry.get(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Workflow '{workflow_id}' not found. "
                "The in-memory registry is lost on process restart — "
                "resubmit the requirement if the process has restarted."
            ),
        )
    return state


@router.post("/{workflow_id}/approval", response_model=WorkflowState)
def set_approval(workflow_id: str, body: ApprovalRequest) -> WorkflowState:
    """Record a human approval decision and resume the workflow.

    Fetches the current :class:`~src.graph.state.WorkflowState` from the
    registry, applies the approval decision via
    :meth:`~pydantic.BaseModel.model_copy`, then re-runs the full workflow
    graph from the updated state.

    When ``approval = "approved"`` the workflow passes through the approval gate
    and proceeds to the Jira creation node, creating one issue per generated
    story. When ``"rejected"`` or ``"pending"`` the workflow halts at the gate
    without calling Jira.

    MVP limitation — full pipeline re-execution
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Because the current implementation has no LangGraph checkpointer, the
    entire graph re-executes from START on every call, including the LLM-backed
    requirement analysis and story generation nodes. This results in two LLM
    calls per approval and means the generated stories may differ slightly
    between runs due to model non-determinism. A future phase can introduce
    ``graph.compile(interrupt_before=["jira"])`` with a persistent checkpointer
    to enable true resume-from-gate behaviour.

    Args:
        workflow_id: The ``workflow_id`` returned by ``POST /requirements``.
        body: The approval decision (``ApprovalStatus`` enum value).

    Returns:
        The updated :class:`~src.graph.state.WorkflowState` after the workflow
        resumes. ``current_step`` will be ``COMPLETED`` for an approved run,
        ``FAILED`` for a rejected run, or ``AWAITING_APPROVAL`` when
        ``approval = "pending"`` leaves the gate unchanged.

    Raises:
        HTTPException 404: ``workflow_id`` is not present in the registry.
        HTTPException 400: The approval decision fails application validation.
        HTTPException 500: An error occurs during workflow execution.
    """
    log = logger.bind(
        endpoint=f"POST /requirements/{workflow_id}/approval",
        approval=body.approval.value,
    )

    current_state = _registry.get(workflow_id)
    if current_state is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Workflow '{workflow_id}' not found. "
                "The in-memory registry is lost on process restart — "
                "resubmit the requirement if the process has restarted."
            ),
        )

    project_key = _project_keys.get(workflow_id)
    if project_key is None:
        # Registry inconsistency: project key should always be written
        # alongside the state in submit_requirement. This branch is unreachable
        # in normal operation.
        log.error("Registry inconsistency: project_key missing for known workflow_id")
        raise HTTPException(
            status_code=500,
            detail="Internal registry inconsistency — project key is missing.",
        )

    log.info("Processing approval decision")
    try:
        updated_state = current_state.model_copy(
            update={"approval_status": body.approval}
        )
        compiled = _get_workflow(project_key)
        final_state = run_workflow(compiled, updated_state)

        _registry[workflow_id] = final_state

        log.bind(step=final_state.current_step.value).info(
            "Approval processing completed"
        )
        return final_state

    except ValidationError as exc:
        log.bind(error=str(exc)).warning("Validation error during approval")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ApplicationError as exc:
        log.bind(error_type=type(exc).__name__, error=str(exc)).error(
            "Application error during approval"
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        log.bind(error_type=type(exc).__name__).error(
            "Unexpected error during approval"
        )
        raise HTTPException(status_code=500, detail="An unexpected error occurred") from exc
