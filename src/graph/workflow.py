"""LangGraph orchestration workflow for the MVP pipeline.

This module is the single wiring point for the entire graph.  It imports
nodes and state from their respective modules and connects them into a
compiled, executable LangGraph graph.  It contains **no** business logic,
prompt text, Jira logic, or LLM logic of its own.

Topology
--------
::

    START
      │
      ▼
    requirement          (RequirementAgent — raw text → NormalizedRequirement)
      │
      ▼
    story_generation     (StoryAgent — NormalizedRequirement → StoryCollection)
      │
      ▼ ── INTERRUPT ──  (graph pauses here; caller stores state and waits for human decision)
      ▼
    approval_gate        (reads ApprovalStatus; routes without calling Jira)
      ├── APPROVED ──▶ jira  (JiraService — creates one issue per story)
      │                  │
      │                  ▼
      │                 END
      │
      └── PENDING / REJECTED ──▶ END

Interrupt / resume
------------------
The graph is compiled with ``interrupt_before=[_NODE_APPROVAL]`` and a
:class:`~langgraph.checkpoint.memory.MemorySaver` checkpointer.

* **First invocation** — ``run_workflow(compiled, initial_state)`` runs
  ``requirement`` and ``story_generation``, then pauses before
  ``approval_gate``.  The checkpoint saves state keyed by
  ``initial_state.workflow_id`` (the LangGraph ``thread_id``).  The caller
  receives the state with ``current_step = AWAITING_APPROVAL``.

* **Resume** — once a human decision is recorded, the caller updates the
  stored state's ``approval_status`` and calls
  ``run_workflow(compiled, updated_state)`` again.
  :func:`run_workflow` detects the pending interrupt via
  ``compiled.get_state(config).next``, applies the approval via
  ``compiled.update_state``, and resumes execution from ``approval_gate``
  onward — **without re-running** ``requirement_node`` or
  ``story_generation_node``.

* **Explicit resume helper** — :func:`resume_workflow` is a named entry
  point for the future API refactor.  It accepts a ``workflow_id`` and
  :class:`~src.graph.state.ApprovalStatus` directly, making the intent
  unambiguous.

Approval hard-stop
------------------
The Jira node is structurally unreachable unless ``_route_after_approval``
returns ``_ROUTE_JIRA``, which only happens when
``state.approval_status == ApprovalStatus.APPROVED``.  The ``jira_node``
itself also refuses to proceed without APPROVED status (double guard), so
no Jira service call is ever made on an unapproved state.

LangGraph state behaviour
--------------------------
``StateGraph(WorkflowState)`` passes Pydantic model instances into nodes and
returns a plain :class:`dict` whose values are the live model instances from
the final node.  :func:`_reconstruct` converts that dict to a typed
:class:`~src.graph.state.WorkflowState` via ``WorkflowState(**result)``.
Fields arrive as the correct Pydantic types, so no round-trip validation
occurs and the :class:`~src.schemas.story.StoryCollection` ``@computed_field``
hazard is avoided.

Project key binding
-------------------
The Jira node is produced by :func:`~src.graph.nodes.make_jira_node` and
bound to a project key once at wiring time::

    compiled = build_workflow(project_key="ATLAS")

Pass a different key to build independent workflow instances for different
projects.  The compiled graph and its :class:`~langgraph.checkpoint.memory.MemorySaver`
are typically cached via :func:`~functools.lru_cache` in the API layer, so
checkpoint state persists across requests for the same project.

Public API
----------
.. code-block:: python

    from src.graph.workflow import build_workflow, run_workflow, resume_workflow

    compiled = build_workflow(project_key="ATLAS")

    # Initial run — pauses before approval_gate
    halted_state = run_workflow(compiled, initial_state)

    # After human decision — resume without re-running LLM agents
    final_state = resume_workflow(compiled, halted_state.workflow_id, ApprovalStatus.APPROVED)
    # or equivalently (for backwards-compatible callers):
    final_state = run_workflow(compiled, halted_state.model_copy(update={"approval_status": ApprovalStatus.APPROVED}))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from src.infrastructure.checkpoint_serializer import make_memory_saver

from src.exceptions import WorkflowError
from src.graph.nodes import (
    approval_gate_node,
    make_jira_node,
    requirement_node,
    story_generation_node,
)
from src.graph.state import ApprovalStatus, WorkflowState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# --------------------------------------------------------------------------- #
# Module-private constants
# --------------------------------------------------------------------------- #

_NODE_REQUIREMENT: str = "requirement"
_NODE_STORY: str = "story_generation"
_NODE_APPROVAL: str = "approval_gate"
_NODE_JIRA: str = "jira"

# Routing return values for _route_after_approval.
# These must match the keys in the path_map passed to add_conditional_edges.
_ROUTE_JIRA: str = "jira"
_ROUTE_END: str = "end"

# LangGraph thread-ID key used in all invoke/get_state/update_state configs.
_THREAD_ID_KEY: str = "thread_id"


# --------------------------------------------------------------------------- #
# Routing helper
# --------------------------------------------------------------------------- #


def _route_after_approval(state: WorkflowState) -> str:
    """Return the name of the next edge after the approval gate node.

    Called by LangGraph immediately after ``approval_gate_node`` returns.
    Reads ``approval_status`` from the returned state snapshot and maps it
    to one of two routes:

    * :attr:`~src.graph.state.ApprovalStatus.APPROVED` → ``_ROUTE_JIRA``
      (proceeds to Jira issue creation).
    * :attr:`~src.graph.state.ApprovalStatus.PENDING` or
      :attr:`~src.graph.state.ApprovalStatus.REJECTED` → ``_ROUTE_END``
      (terminates the graph).

    The returned string is looked up in the ``path_map`` supplied to
    :meth:`~langgraph.graph.StateGraph.add_conditional_edges`.

    Args:
        state: The state snapshot returned by ``approval_gate_node``.

    Returns:
        ``_ROUTE_JIRA`` when approved; ``_ROUTE_END`` otherwise.
    """
    if state.approval_status == ApprovalStatus.APPROVED:
        return _ROUTE_JIRA
    return _ROUTE_END


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _make_config(workflow_id: str) -> dict:
    """Build the LangGraph invocation config for *workflow_id* as thread."""
    return {"configurable": {_THREAD_ID_KEY: workflow_id}}


def _reconstruct(result: object) -> WorkflowState:
    """Convert a LangGraph invoke result to a typed :class:`WorkflowState`.

    LangGraph returns either a :class:`WorkflowState` instance (some code
    paths) or a plain :class:`dict` whose values are live Pydantic model
    instances from the last-executed node.  Both are handled here so that
    callers always receive a fully typed object.

    Raises:
        WorkflowError: If *result* is neither a WorkflowState nor a dict.
    """
    if isinstance(result, WorkflowState):
        return result
    if isinstance(result, dict):
        return WorkflowState(**result)
    raise WorkflowError(
        f"Compiled graph returned an unexpected type: {type(result).__name__}"
    )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_workflow(project_key: str) -> CompiledStateGraph:
    """Build and compile the MVP orchestration graph with interrupt/resume support.

    Wires the four pipeline nodes into a LangGraph :class:`~langgraph.graph.StateGraph`,
    applies conditional routing after the approval gate, and compiles the
    result with a :class:`~langgraph.checkpoint.memory.MemorySaver` checkpointer
    and ``interrupt_before=[_NODE_APPROVAL]``.

    The interrupt causes execution to pause **after** ``story_generation`` and
    **before** ``approval_gate``, holding state in the in-memory checkpointer
    keyed by ``workflow_id`` (the LangGraph ``thread_id``).  :func:`run_workflow`
    and :func:`resume_workflow` use this checkpoint to resume without
    re-running the LLM-backed nodes.

    Node sequence::

        requirement → story_generation → [INTERRUPT] → approval_gate
            ├── APPROVED  → jira → END
            └── PENDING / REJECTED → END

    Args:
        project_key: Jira project key where stories will be created as issues
            (e.g. ``"ATLAS"``).  Must be non-empty; validation is the caller's
            responsibility.

    Returns:
        A compiled :class:`~langgraph.graph.state.CompiledStateGraph` with an
        embedded :class:`~langgraph.checkpoint.memory.MemorySaver` and an
        interrupt registered before ``approval_gate``.  Use via
        :func:`run_workflow` or :func:`resume_workflow`.
    """
    jira_node = make_jira_node(project_key)

    graph: StateGraph = StateGraph(WorkflowState)

    # -- Register nodes ------------------------------------------------------ #
    graph.add_node(_NODE_REQUIREMENT, requirement_node)
    graph.add_node(_NODE_STORY, story_generation_node)
    graph.add_node(_NODE_APPROVAL, approval_gate_node)
    graph.add_node(_NODE_JIRA, jira_node)

    # -- Wire edges ---------------------------------------------------------- #
    graph.add_edge(START, _NODE_REQUIREMENT)
    graph.add_edge(_NODE_REQUIREMENT, _NODE_STORY)
    graph.add_edge(_NODE_STORY, _NODE_APPROVAL)

    # Conditional branch after approval_gate: APPROVED → jira, else → END.
    graph.add_conditional_edges(
        _NODE_APPROVAL,
        _route_after_approval,
        {_ROUTE_JIRA: _NODE_JIRA, _ROUTE_END: END},
    )
    graph.add_edge(_NODE_JIRA, END)

    return graph.compile(
        checkpointer=make_memory_saver(),
        interrupt_before=[_NODE_APPROVAL],
    )


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #


def run_workflow(
    compiled: CompiledStateGraph,
    initial_state: WorkflowState,
) -> WorkflowState:
    """Invoke the compiled workflow and return the final :class:`~src.graph.state.WorkflowState`.

    This function handles **both** the initial run and the approval resume
    transparently, so existing callers do not need to change.

    **Initial run** (no checkpoint for ``initial_state.workflow_id``):
        Runs ``requirement_node`` and ``story_generation_node``, then pauses
        at the ``interrupt_before=[approval_gate]`` boundary.  Returns the
        state with ``current_step = AWAITING_APPROVAL``.

    **Resume** (checkpoint exists and graph is at the interrupt boundary):
        Detected via ``compiled.get_state(config).next`` being non-empty.
        Applies ``initial_state.approval_status`` to the checkpoint via
        ``compiled.update_state``, then resumes with
        ``compiled.invoke(None, config)``.  Neither ``RequirementAgent`` nor
        ``StoryAgent`` runs again.

    The resume detection relies on ``len(snapshot.next) > 0``, which returns
    0 for :class:`unittest.mock.MagicMock` objects, preserving mock-based test
    compatibility.

    Args:
        compiled: A graph produced by :func:`build_workflow`.
        initial_state: Starting :class:`~src.graph.state.WorkflowState` with
            ``raw_requirement`` populated.  On resume, this should be the
            previously halted state with ``approval_status`` updated.

    Returns:
        The :class:`~src.graph.state.WorkflowState` after the graph pauses
        (first call) or terminates (second call).

    Raises:
        WorkflowError: If the graph returns an unexpected type.
    """
    config = _make_config(initial_state.workflow_id)

    try:
        snapshot = compiled.get_state(config)
        is_interrupted = len(snapshot.next) > 0
    except Exception:
        is_interrupted = False

    if is_interrupted:
        compiled.update_state(config, {"approval_status": initial_state.approval_status})
        result = compiled.invoke(None, config=config)
    else:
        result = compiled.invoke(initial_state, config=config)

    return _reconstruct(result)


def resume_workflow(
    compiled: CompiledStateGraph,
    workflow_id: str,
    approval: ApprovalStatus,
) -> WorkflowState:
    """Resume a workflow that is paused at the approval interrupt.

    An explicit, named entry point for the approval-resume path.  Prefer this
    over calling :func:`run_workflow` a second time in new code — the intent
    is unambiguous and the ``workflow_id`` / ``approval`` parameters map
    directly to the API approval endpoint's inputs.

    Applies *approval* to the checkpoint identified by *workflow_id*, then
    resumes execution from ``approval_gate``.  Neither ``RequirementAgent``
    nor ``StoryAgent`` runs again.

    Args:
        compiled: A graph produced by :func:`build_workflow` that already
            holds a checkpoint for *workflow_id* from a prior
            :func:`run_workflow` call.
        workflow_id: The ``workflow_id`` of the halted run (used as the
            LangGraph ``thread_id``).
        approval: The human decision to record before resuming.

    Returns:
        The final :class:`~src.graph.state.WorkflowState` after the graph
        runs ``approval_gate`` (and ``jira`` when ``APPROVED``).

    Raises:
        WorkflowError: If the graph returns an unexpected type.
    """
    config = _make_config(workflow_id)
    compiled.update_state(config, {"approval_status": approval})
    result = compiled.invoke(None, config=config)
    return _reconstruct(result)
