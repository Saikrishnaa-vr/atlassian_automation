from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.graph.state import (
    ApprovalStatus,
    ErrorInfo,
    ExecutionMetadata,
    WorkflowState,
    WorkflowStep,
)
from src.schemas.story import StoryCollection


# ------------------------------------------------------------------ #
# ApprovalStatus
# ------------------------------------------------------------------ #

def test_approval_status_pending_value():
    assert ApprovalStatus.PENDING.value == "pending"


def test_approval_status_approved_value():
    assert ApprovalStatus.APPROVED.value == "approved"


def test_approval_status_rejected_value():
    assert ApprovalStatus.REJECTED.value == "rejected"


def test_approval_status_is_str_enum():
    assert isinstance(ApprovalStatus.PENDING, str)
    assert isinstance(ApprovalStatus.APPROVED, str)


# ------------------------------------------------------------------ #
# WorkflowStep
# ------------------------------------------------------------------ #

def test_workflow_step_pending_value():
    assert WorkflowStep.PENDING.value == "pending"


def test_workflow_step_failed_value():
    assert WorkflowStep.FAILED.value == "failed"


def test_workflow_step_completed_value():
    assert WorkflowStep.COMPLETED.value == "completed"


def test_workflow_step_awaiting_approval_value():
    assert WorkflowStep.AWAITING_APPROVAL.value == "awaiting_approval"


def test_workflow_step_jira_creation_value():
    assert WorkflowStep.JIRA_CREATION.value == "jira_creation"


def test_workflow_step_requirement_analysis_value():
    assert WorkflowStep.REQUIREMENT_ANALYSIS.value == "requirement_analysis"


# ------------------------------------------------------------------ #
# WorkflowState construction
# ------------------------------------------------------------------ #

def test_workflow_state_requires_raw_requirement():
    with pytest.raises(Exception):
        WorkflowState()


def test_workflow_state_rejects_blank_requirement():
    with pytest.raises(Exception):
        WorkflowState(raw_requirement="")


def test_workflow_state_constructs_with_raw_requirement():
    state = WorkflowState(raw_requirement="Build login.")
    assert state.raw_requirement == "Build login."


def test_workflow_state_default_approval_status():
    state = WorkflowState(raw_requirement="req")
    assert state.approval_status == ApprovalStatus.PENDING


def test_workflow_state_default_current_step():
    state = WorkflowState(raw_requirement="req")
    assert state.current_step == WorkflowStep.PENDING


def test_workflow_state_default_normalized_requirement_is_none():
    state = WorkflowState(raw_requirement="req")
    assert state.normalized_requirement is None


def test_workflow_state_default_error_is_none():
    state = WorkflowState(raw_requirement="req")
    assert state.error is None


def test_workflow_state_default_generated_stories_is_empty():
    state = WorkflowState(raw_requirement="req")
    assert isinstance(state.generated_stories, StoryCollection)
    assert state.generated_stories.count == 0


def test_workflow_state_rejects_extra_fields():
    with pytest.raises(Exception):
        WorkflowState(raw_requirement="req", unknown_field="extra")


def test_workflow_state_generates_unique_ids():
    state1 = WorkflowState(raw_requirement="req")
    state2 = WorkflowState(raw_requirement="req")
    assert state1.workflow_id != state2.workflow_id


def test_workflow_state_workflow_id_is_string():
    state = WorkflowState(raw_requirement="req")
    assert isinstance(state.workflow_id, str)
    assert len(state.workflow_id) > 0


# ------------------------------------------------------------------ #
# WorkflowState.model_copy
# ------------------------------------------------------------------ #

def test_model_copy_updates_approval_status():
    state = WorkflowState(raw_requirement="req")
    updated = state.model_copy(update={"approval_status": ApprovalStatus.APPROVED})
    assert updated.approval_status == ApprovalStatus.APPROVED


def test_model_copy_does_not_mutate_original():
    state = WorkflowState(raw_requirement="req")
    state.model_copy(update={"approval_status": ApprovalStatus.APPROVED})
    assert state.approval_status == ApprovalStatus.PENDING


def test_model_copy_preserves_workflow_id():
    state = WorkflowState(raw_requirement="req")
    updated = state.model_copy(update={"current_step": WorkflowStep.FAILED})
    assert updated.workflow_id == state.workflow_id


def test_model_copy_updates_current_step():
    state = WorkflowState(raw_requirement="req")
    updated = state.model_copy(update={"current_step": WorkflowStep.COMPLETED})
    assert updated.current_step == WorkflowStep.COMPLETED


# ------------------------------------------------------------------ #
# ExecutionMetadata
# ------------------------------------------------------------------ #

def test_execution_metadata_default_step_count():
    meta = ExecutionMetadata()
    assert meta.step_count == 0


def test_execution_metadata_default_started_at_is_none():
    meta = ExecutionMetadata()
    assert meta.started_at is None


def test_execution_metadata_default_completed_at_is_none():
    meta = ExecutionMetadata()
    assert meta.completed_at is None


def test_execution_metadata_duration_seconds_none_when_no_timestamps():
    meta = ExecutionMetadata()
    assert meta.duration_seconds is None


def test_execution_metadata_duration_seconds_none_when_only_started():
    meta = ExecutionMetadata(started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert meta.duration_seconds is None


def test_execution_metadata_duration_seconds_calculated():
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    meta = ExecutionMetadata(started_at=start, completed_at=end)
    assert meta.duration_seconds == 5.0


def test_execution_metadata_rejects_extra_fields():
    with pytest.raises(Exception):
        ExecutionMetadata(unknown_field="x")


# ------------------------------------------------------------------ #
# ErrorInfo
# ------------------------------------------------------------------ #

def test_error_info_construction():
    now = datetime.now(timezone.utc)
    error = ErrorInfo(
        step=WorkflowStep.REQUIREMENT_ANALYSIS,
        error_type="AgentError",
        message="LLM invocation failed",
        timestamp=now,
    )
    assert error.step == WorkflowStep.REQUIREMENT_ANALYSIS
    assert error.error_type == "AgentError"
    assert error.message == "LLM invocation failed"
    assert error.timestamp == now


def test_error_info_rejects_blank_message():
    with pytest.raises(Exception):
        ErrorInfo(
            step=WorkflowStep.PENDING,
            error_type="AgentError",
            message="",
            timestamp=datetime.now(timezone.utc),
        )


def test_error_info_rejects_blank_error_type():
    with pytest.raises(Exception):
        ErrorInfo(
            step=WorkflowStep.PENDING,
            error_type="",
            message="some error",
            timestamp=datetime.now(timezone.utc),
        )


def test_error_info_rejects_extra_fields():
    with pytest.raises(Exception):
        ErrorInfo(
            step=WorkflowStep.PENDING,
            error_type="AgentError",
            message="error",
            timestamp=datetime.now(timezone.utc),
            extra="bad",
        )
