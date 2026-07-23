from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import AgentError
from src.graph.nodes import (
    approval_gate_node,
    error_node,
    make_jira_node,
    requirement_node,
    story_generation_node,
)
from src.graph.state import ApprovalStatus, WorkflowState, WorkflowStep


# ------------------------------------------------------------------ #
# requirement_node
# ------------------------------------------------------------------ #

def test_requirement_node_advances_to_requirement_analysis(workflow_state, normalized_requirement):
    with patch("src.graph.nodes.RequirementAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = normalized_requirement
        MockAgent.return_value = instance
        result = requirement_node(workflow_state)
    assert result.current_step == WorkflowStep.REQUIREMENT_ANALYSIS


def test_requirement_node_stores_serialized_normalized_requirement(workflow_state, normalized_requirement):
    with patch("src.graph.nodes.RequirementAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = normalized_requirement
        MockAgent.return_value = instance
        result = requirement_node(workflow_state)
    assert result.normalized_requirement is not None
    from src.schemas.requirement import NormalizedRequirement
    restored = NormalizedRequirement.model_validate_json(result.normalized_requirement)
    assert restored == normalized_requirement


def test_requirement_node_increments_step_count(workflow_state, normalized_requirement):
    with patch("src.graph.nodes.RequirementAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = normalized_requirement
        MockAgent.return_value = instance
        result = requirement_node(workflow_state)
    assert result.metadata.step_count == workflow_state.metadata.step_count + 1


def test_requirement_node_error_transitions_to_failed(workflow_state):
    with patch("src.graph.nodes.RequirementAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.side_effect = AgentError("LLM failed")
        MockAgent.return_value = instance
        result = requirement_node(workflow_state)
    assert result.current_step == WorkflowStep.FAILED
    assert result.error is not None


def test_requirement_node_error_sets_error_type(workflow_state):
    with patch("src.graph.nodes.RequirementAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.side_effect = AgentError("LLM failed")
        MockAgent.return_value = instance
        result = requirement_node(workflow_state)
    assert result.error.error_type == "AgentError"


def test_requirement_node_preserves_workflow_id(workflow_state, normalized_requirement):
    with patch("src.graph.nodes.RequirementAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = normalized_requirement
        MockAgent.return_value = instance
        result = requirement_node(workflow_state)
    assert result.workflow_id == workflow_state.workflow_id


# ------------------------------------------------------------------ #
# story_generation_node
# ------------------------------------------------------------------ #

def test_story_generation_node_advances_to_awaiting_approval(
    workflow_state, normalized_requirement, story_collection
):
    state = workflow_state.model_copy(update={
        "normalized_requirement": normalized_requirement.model_dump_json(),
    })
    with patch("src.graph.nodes.StoryAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = story_collection
        MockAgent.return_value = instance
        result = story_generation_node(state)
    assert result.current_step == WorkflowStep.AWAITING_APPROVAL


def test_story_generation_node_stores_stories(
    workflow_state, normalized_requirement, story_collection
):
    state = workflow_state.model_copy(update={
        "normalized_requirement": normalized_requirement.model_dump_json(),
    })
    with patch("src.graph.nodes.StoryAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = story_collection
        MockAgent.return_value = instance
        result = story_generation_node(state)
    assert result.generated_stories.count == 1


def test_story_generation_node_increments_step_count(
    workflow_state, normalized_requirement, story_collection
):
    state = workflow_state.model_copy(update={
        "normalized_requirement": normalized_requirement.model_dump_json(),
    })
    with patch("src.graph.nodes.StoryAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.return_value = story_collection
        MockAgent.return_value = instance
        result = story_generation_node(state)
    assert result.metadata.step_count == state.metadata.step_count + 1


def test_story_generation_node_missing_normalized_requirement_transitions_to_failed(workflow_state):
    # normalized_requirement is None by default
    result = story_generation_node(workflow_state)
    assert result.current_step == WorkflowStep.FAILED
    assert result.error is not None


def test_story_generation_node_error_transitions_to_failed(workflow_state, normalized_requirement):
    state = workflow_state.model_copy(update={
        "normalized_requirement": normalized_requirement.model_dump_json(),
    })
    with patch("src.graph.nodes.StoryAgent") as MockAgent:
        instance = MagicMock()
        instance.execute.side_effect = AgentError("Story generation failed")
        MockAgent.return_value = instance
        result = story_generation_node(state)
    assert result.current_step == WorkflowStep.FAILED
    assert result.error is not None


# ------------------------------------------------------------------ #
# approval_gate_node
# ------------------------------------------------------------------ #

def test_approval_gate_node_approved_advances_to_jira_creation(workflow_state):
    state = workflow_state.model_copy(update={
        "approval_status": ApprovalStatus.APPROVED,
        "current_step": WorkflowStep.AWAITING_APPROVAL,
    })
    result = approval_gate_node(state)
    assert result.current_step == WorkflowStep.JIRA_CREATION


def test_approval_gate_node_approved_increments_step_count(workflow_state):
    state = workflow_state.model_copy(update={"approval_status": ApprovalStatus.APPROVED})
    result = approval_gate_node(state)
    assert result.metadata.step_count == state.metadata.step_count + 1


def test_approval_gate_node_approved_has_no_error(workflow_state):
    state = workflow_state.model_copy(update={"approval_status": ApprovalStatus.APPROVED})
    result = approval_gate_node(state)
    assert result.error is None


def test_approval_gate_node_approved_does_not_set_jira_results(workflow_state):
    state = workflow_state.model_copy(update={"approval_status": ApprovalStatus.APPROVED})
    result = approval_gate_node(state)
    assert len(result.jira_results) == 0


def test_approval_gate_node_rejected_transitions_to_failed(workflow_state):
    state = workflow_state.model_copy(update={"approval_status": ApprovalStatus.REJECTED})
    result = approval_gate_node(state)
    assert result.current_step == WorkflowStep.FAILED


def test_approval_gate_node_rejected_increments_step_count(workflow_state):
    state = workflow_state.model_copy(update={"approval_status": ApprovalStatus.REJECTED})
    result = approval_gate_node(state)
    assert result.metadata.step_count == state.metadata.step_count + 1


def test_approval_gate_node_pending_step_unchanged(workflow_state):
    state = workflow_state.model_copy(update={
        "current_step": WorkflowStep.AWAITING_APPROVAL,
    })
    result = approval_gate_node(state)
    assert result.current_step == WorkflowStep.AWAITING_APPROVAL


def test_approval_gate_node_pending_returns_same_approval_status(workflow_state):
    result = approval_gate_node(workflow_state)
    assert result.approval_status == ApprovalStatus.PENDING


def test_approval_gate_node_pending_step_count_unchanged(workflow_state):
    result = approval_gate_node(workflow_state)
    assert result.metadata.step_count == workflow_state.metadata.step_count


# ------------------------------------------------------------------ #
# make_jira_node
# ------------------------------------------------------------------ #

def test_jira_node_blocks_pending_approval(workflow_state):
    # approval_status defaults to PENDING
    jira_node = make_jira_node("PROJ")
    result = jira_node(workflow_state)
    assert result.current_step == WorkflowStep.FAILED
    assert result.error is not None


def test_jira_node_blocks_rejected_approval(workflow_state):
    state = workflow_state.model_copy(update={"approval_status": ApprovalStatus.REJECTED})
    jira_node = make_jira_node("PROJ")
    result = jira_node(state)
    assert result.current_step == WorkflowStep.FAILED


def test_jira_node_hard_stop_error_mentions_approved(workflow_state):
    jira_node = make_jira_node("PROJ")
    result = jira_node(workflow_state)
    assert result.error is not None
    error_text = result.error.message.lower()
    assert "approved" in error_text or "approval" in error_text


def test_jira_node_does_not_call_service_without_approval(workflow_state):
    jira_node = make_jira_node("PROJ")
    with patch("src.graph.nodes.get_jira_service") as mock_get_svc:
        jira_node(workflow_state)
    mock_get_svc.assert_not_called()


def test_jira_node_success(workflow_state, story_collection, jira_issue_response):
    state = workflow_state.model_copy(update={
        "approval_status": ApprovalStatus.APPROVED,
        "generated_stories": story_collection,
        "current_step": WorkflowStep.JIRA_CREATION,
    })
    with patch("src.graph.nodes.get_jira_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.create_issue.return_value = jira_issue_response
        mock_get_svc.return_value = mock_svc
        jira_node = make_jira_node("PROJ")
        result = jira_node(state)
    assert result.current_step == WorkflowStep.COMPLETED
    assert result.error is None


def test_jira_node_stores_jira_results(workflow_state, story_collection, jira_issue_response):
    state = workflow_state.model_copy(update={
        "approval_status": ApprovalStatus.APPROVED,
        "generated_stories": story_collection,
    })
    with patch("src.graph.nodes.get_jira_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.create_issue.return_value = jira_issue_response
        mock_get_svc.return_value = mock_svc
        jira_node = make_jira_node("PROJ")
        result = jira_node(state)
    assert len(result.jira_results) == 1


def test_jira_node_creates_one_issue_per_story(workflow_state, story_collection, jira_issue_response):
    state = workflow_state.model_copy(update={
        "approval_status": ApprovalStatus.APPROVED,
        "generated_stories": story_collection,
    })
    with patch("src.graph.nodes.get_jira_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.create_issue.return_value = jira_issue_response
        mock_get_svc.return_value = mock_svc
        jira_node = make_jira_node("PROJ")
        jira_node(state)
    assert mock_svc.create_issue.call_count == 1


def test_jira_node_sets_completed_at(workflow_state, story_collection, jira_issue_response):
    state = workflow_state.model_copy(update={
        "approval_status": ApprovalStatus.APPROVED,
        "generated_stories": story_collection,
    })
    with patch("src.graph.nodes.get_jira_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.create_issue.return_value = jira_issue_response
        mock_get_svc.return_value = mock_svc
        jira_node = make_jira_node("PROJ")
        result = jira_node(state)
    assert result.metadata.completed_at is not None


def test_jira_node_service_error_transitions_to_failed(workflow_state, story_collection):
    from src.services.jira_service import JiraServiceError
    state = workflow_state.model_copy(update={
        "approval_status": ApprovalStatus.APPROVED,
        "generated_stories": story_collection,
    })
    with patch("src.graph.nodes.get_jira_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.create_issue.side_effect = JiraServiceError("Jira unavailable")
        mock_get_svc.return_value = mock_svc
        jira_node = make_jira_node("PROJ")
        result = jira_node(state)
    assert result.current_step == WorkflowStep.FAILED
    assert result.error is not None


# ------------------------------------------------------------------ #
# error_node
# ------------------------------------------------------------------ #

def test_error_node_transitions_to_failed(workflow_state):
    result = error_node(workflow_state, ValueError("test error"))
    assert result.current_step == WorkflowStep.FAILED


def test_error_node_populates_error_info(workflow_state):
    exc = AgentError("agent failed")
    result = error_node(workflow_state, exc)
    assert result.error is not None
    assert result.error.error_type == "AgentError"
    assert result.error.message == "agent failed"


def test_error_node_records_step_at_failure(workflow_state):
    state = workflow_state.model_copy(update={"current_step": WorkflowStep.REQUIREMENT_ANALYSIS})
    result = error_node(state, RuntimeError("unexpected"))
    assert result.error.step == WorkflowStep.REQUIREMENT_ANALYSIS


def test_error_node_sets_completed_at(workflow_state):
    result = error_node(workflow_state, RuntimeError("test"))
    assert result.metadata.completed_at is not None


def test_error_node_increments_step_count(workflow_state):
    result = error_node(workflow_state, RuntimeError("test"))
    assert result.metadata.step_count == workflow_state.metadata.step_count + 1


def test_error_node_preserves_workflow_id(workflow_state):
    result = error_node(workflow_state, RuntimeError("test"))
    assert result.workflow_id == workflow_state.workflow_id


def test_error_node_uses_class_name_when_str_is_empty(workflow_state):
    exc = Exception()  # str(Exception()) == ""
    result = error_node(workflow_state, exc)
    assert result.error.message == "Exception"


def test_error_node_uses_exception_message(workflow_state):
    exc = ValueError("specific error message")
    result = error_node(workflow_state, exc)
    assert result.error.message == "specific error message"
