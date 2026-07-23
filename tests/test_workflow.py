from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import WorkflowError
from src.graph.state import ApprovalStatus, WorkflowState, WorkflowStep
from src.graph.workflow import _route_after_approval, build_workflow, resume_workflow, run_workflow


# ------------------------------------------------------------------ #
# _route_after_approval
# ------------------------------------------------------------------ #

def test_route_after_approval_approved_returns_jira():
    state = WorkflowState(raw_requirement="req").model_copy(
        update={"approval_status": ApprovalStatus.APPROVED}
    )
    assert _route_after_approval(state) == "jira"


def test_route_after_approval_pending_returns_end():
    state = WorkflowState(raw_requirement="req")
    # default approval_status is PENDING
    assert _route_after_approval(state) == "end"


def test_route_after_approval_rejected_returns_end():
    state = WorkflowState(raw_requirement="req").model_copy(
        update={"approval_status": ApprovalStatus.REJECTED}
    )
    assert _route_after_approval(state) == "end"


def test_route_after_approval_only_approved_leads_to_jira():
    for status in [ApprovalStatus.PENDING, ApprovalStatus.REJECTED]:
        state = WorkflowState(raw_requirement="req").model_copy(
            update={"approval_status": status}
        )
        assert _route_after_approval(state) == "end"


# ------------------------------------------------------------------ #
# run_workflow
# ------------------------------------------------------------------ #

def test_run_workflow_returns_workflow_state_directly():
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = state
    result = run_workflow(mock_compiled, state)
    assert result is state


def test_run_workflow_calls_compiled_invoke():
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = state
    run_workflow(mock_compiled, state)
    expected_config = {"configurable": {"thread_id": state.workflow_id}}
    mock_compiled.invoke.assert_called_once_with(state, config=expected_config)


def test_run_workflow_reconstructs_from_dict():
    state = WorkflowState(raw_requirement="req")
    # Simulate LangGraph dict result: values are live Pydantic instances
    result_dict = {
        "workflow_id": state.workflow_id,
        "raw_requirement": state.raw_requirement,
        "normalized_requirement": state.normalized_requirement,
        "generated_stories": state.generated_stories,
        "epics": state.epics,
        "tasks": state.tasks,
        "approval_status": state.approval_status,
        "jira_results": state.jira_results,
        "current_step": state.current_step,
        "metadata": state.metadata,
        "error": state.error,
    }
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = result_dict
    result = run_workflow(mock_compiled, state)
    assert isinstance(result, WorkflowState)
    assert result.raw_requirement == "req"
    assert result.workflow_id == state.workflow_id


def test_run_workflow_preserves_approval_status_from_dict():
    state = WorkflowState(raw_requirement="req").model_copy(
        update={"approval_status": ApprovalStatus.APPROVED}
    )
    result_dict = {
        "workflow_id": state.workflow_id,
        "raw_requirement": state.raw_requirement,
        "normalized_requirement": state.normalized_requirement,
        "generated_stories": state.generated_stories,
        "epics": state.epics,
        "tasks": state.tasks,
        "approval_status": state.approval_status,
        "jira_results": state.jira_results,
        "current_step": state.current_step,
        "metadata": state.metadata,
        "error": state.error,
    }
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = result_dict
    result = run_workflow(mock_compiled, state)
    assert result.approval_status == ApprovalStatus.APPROVED


def test_run_workflow_raises_workflow_error_on_unexpected_type():
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = "unexpected string result"
    with pytest.raises(WorkflowError):
        run_workflow(mock_compiled, state)


def test_run_workflow_raises_workflow_error_on_list_result():
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = [state]
    with pytest.raises(WorkflowError):
        run_workflow(mock_compiled, state)


def test_run_workflow_raises_workflow_error_on_none_result():
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = None
    with pytest.raises(WorkflowError):
        run_workflow(mock_compiled, state)


# ------------------------------------------------------------------ #
# build_workflow
# ------------------------------------------------------------------ #

def test_build_workflow_returns_compiled_graph():
    compiled = build_workflow("PROJ")
    assert compiled is not None


def test_build_workflow_has_invoke_method():
    compiled = build_workflow("PROJ")
    assert hasattr(compiled, "invoke")
    assert callable(compiled.invoke)


def test_build_workflow_different_project_keys():
    compiled_a = build_workflow("PROJ_A")
    compiled_b = build_workflow("PROJ_B")
    assert hasattr(compiled_a, "invoke")
    assert hasattr(compiled_b, "invoke")


def test_build_workflow_has_checkpointer():
    compiled = build_workflow("PROJ")
    assert compiled.checkpointer is not None


# ------------------------------------------------------------------ #
# run_workflow — interrupted-state detection (mock-based)
# ------------------------------------------------------------------ #

def test_run_workflow_takes_resume_path_when_graph_is_interrupted():
    """When get_state reports pending nodes, run_workflow resumes instead of fresh-starting."""
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    final_state = WorkflowState(raw_requirement="req")

    mock_snapshot = MagicMock()
    mock_snapshot.next = ("approval_gate",)          # non-empty → interrupted
    mock_compiled.get_state.return_value = mock_snapshot
    mock_compiled.invoke.return_value = final_state

    result = run_workflow(mock_compiled, state)

    expected_config = {"configurable": {"thread_id": state.workflow_id}}
    mock_compiled.update_state.assert_called_once_with(
        expected_config, {"approval_status": state.approval_status}
    )
    mock_compiled.invoke.assert_called_once_with(None, config=expected_config)
    assert result is final_state


def test_run_workflow_uses_workflow_id_as_thread_id():
    """The config thread_id must equal the state's workflow_id."""
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = state
    run_workflow(mock_compiled, state)
    _, kwargs = mock_compiled.invoke.call_args
    assert kwargs["config"]["configurable"]["thread_id"] == state.workflow_id


# ------------------------------------------------------------------ #
# resume_workflow (mock-based)
# ------------------------------------------------------------------ #

def test_resume_workflow_calls_update_state_then_invoke():
    workflow_id = "wf-resume-test"
    mock_compiled = MagicMock()
    final_state = WorkflowState(raw_requirement="req")
    mock_compiled.invoke.return_value = final_state

    result = resume_workflow(mock_compiled, workflow_id, ApprovalStatus.APPROVED)

    expected_config = {"configurable": {"thread_id": workflow_id}}
    mock_compiled.update_state.assert_called_once_with(
        expected_config, {"approval_status": ApprovalStatus.APPROVED}
    )
    mock_compiled.invoke.assert_called_once_with(None, config=expected_config)
    assert result is final_state


def test_resume_workflow_returns_workflow_state_from_dict():
    workflow_id = "wf-resume-dict"
    state = WorkflowState(raw_requirement="req")
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = {
        "workflow_id": state.workflow_id,
        "raw_requirement": state.raw_requirement,
        "normalized_requirement": state.normalized_requirement,
        "generated_stories": state.generated_stories,
        "epics": state.epics,
        "tasks": state.tasks,
        "approval_status": ApprovalStatus.APPROVED,
        "jira_results": state.jira_results,
        "current_step": WorkflowStep.COMPLETED,
        "metadata": state.metadata,
        "error": state.error,
    }
    result = resume_workflow(mock_compiled, workflow_id, ApprovalStatus.APPROVED)
    assert isinstance(result, WorkflowState)
    assert result.approval_status == ApprovalStatus.APPROVED
    assert result.current_step == WorkflowStep.COMPLETED


def test_resume_workflow_raises_workflow_error_on_unexpected_type():
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = 42
    with pytest.raises(WorkflowError):
        resume_workflow(mock_compiled, "some-id", ApprovalStatus.APPROVED)


# ------------------------------------------------------------------ #
# Integration — interrupt / resume with real compiled graph
# ------------------------------------------------------------------ #

def test_run_workflow_pauses_before_approval_gate(normalized_requirement, story_collection):
    """First run_workflow call must return with current_step = AWAITING_APPROVAL."""
    initial_state = WorkflowState(raw_requirement="Build login system.")
    compiled = build_workflow("PROJ")

    with (
        patch("src.graph.nodes.RequirementAgent") as MockReq,
        patch("src.graph.nodes.StoryAgent") as MockStory,
    ):
        req_inst = MagicMock()
        req_inst.execute.return_value = normalized_requirement
        MockReq.return_value = req_inst

        story_inst = MagicMock()
        story_inst.execute.return_value = story_collection
        MockStory.return_value = story_inst

        halted = run_workflow(compiled, initial_state)

    assert halted.current_step == WorkflowStep.AWAITING_APPROVAL


def test_requirement_and_story_agents_execute_exactly_once_across_interrupt_and_resume(
    normalized_requirement, story_collection
):
    """Neither LLM agent must run on the resume pass — only approval_gate executes."""
    initial_state = WorkflowState(raw_requirement="Build login system.")
    compiled = build_workflow("PROJ")

    with (
        patch("src.graph.nodes.RequirementAgent") as MockReq,
        patch("src.graph.nodes.StoryAgent") as MockStory,
    ):
        req_inst = MagicMock()
        req_inst.execute.return_value = normalized_requirement
        MockReq.return_value = req_inst

        story_inst = MagicMock()
        story_inst.execute.return_value = story_collection
        MockStory.return_value = story_inst

        # First invocation — should interrupt before approval_gate
        halted = run_workflow(compiled, initial_state)
        assert req_inst.execute.call_count == 1
        assert story_inst.execute.call_count == 1

        # Resume with REJECTED approval so jira service is not needed
        state_with_approval = halted.model_copy(
            update={"approval_status": ApprovalStatus.REJECTED}
        )
        run_workflow(compiled, state_with_approval)

        # Agents must NOT have been called again
        assert req_inst.execute.call_count == 1, "RequirementAgent re-ran on resume"
        assert story_inst.execute.call_count == 1, "StoryAgent re-ran on resume"


def test_resume_workflow_does_not_rerun_agents(normalized_requirement, story_collection):
    """resume_workflow is the explicit entry point; agents still run only once."""
    initial_state = WorkflowState(raw_requirement="Build login system.")
    compiled = build_workflow("PROJ")

    with (
        patch("src.graph.nodes.RequirementAgent") as MockReq,
        patch("src.graph.nodes.StoryAgent") as MockStory,
    ):
        req_inst = MagicMock()
        req_inst.execute.return_value = normalized_requirement
        MockReq.return_value = req_inst

        story_inst = MagicMock()
        story_inst.execute.return_value = story_collection
        MockStory.return_value = story_inst

        halted = run_workflow(compiled, initial_state)
        assert halted.current_step == WorkflowStep.AWAITING_APPROVAL

        resume_workflow(compiled, halted.workflow_id, ApprovalStatus.REJECTED)

        assert req_inst.execute.call_count == 1, "RequirementAgent re-ran via resume_workflow"
        assert story_inst.execute.call_count == 1, "StoryAgent re-ran via resume_workflow"


def test_get_state_shows_pending_approval_gate_after_first_run(
    normalized_requirement, story_collection
):
    """After the first run, get_state must show approval_gate as the pending node."""
    initial_state = WorkflowState(raw_requirement="Build login system.")
    compiled = build_workflow("PROJ")

    with (
        patch("src.graph.nodes.RequirementAgent") as MockReq,
        patch("src.graph.nodes.StoryAgent") as MockStory,
    ):
        req_inst = MagicMock()
        req_inst.execute.return_value = normalized_requirement
        MockReq.return_value = req_inst

        story_inst = MagicMock()
        story_inst.execute.return_value = story_collection
        MockStory.return_value = story_inst

        halted = run_workflow(compiled, initial_state)

    config = {"configurable": {"thread_id": halted.workflow_id}}
    snapshot = compiled.get_state(config)
    assert "approval_gate" in snapshot.next
