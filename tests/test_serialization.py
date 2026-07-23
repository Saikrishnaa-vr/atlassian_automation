from __future__ import annotations

import json

import pytest

from src.graph.state import ApprovalStatus, WorkflowState, WorkflowStep
from src.schemas.requirement import NormalizedRequirement
from src.schemas.story import AcceptanceCriterion, StoryCollection, UserStory


# ------------------------------------------------------------------ #
# NormalizedRequirement round-trip
# ------------------------------------------------------------------ #

def test_normalized_requirement_json_round_trip(normalized_requirement):
    serialized = normalized_requirement.model_dump_json()
    restored = NormalizedRequirement.model_validate_json(serialized)
    assert restored == normalized_requirement


def test_normalized_requirement_json_contains_title(normalized_requirement):
    data = json.loads(normalized_requirement.model_dump_json())
    assert "title" in data
    assert data["title"] == normalized_requirement.title


def test_normalized_requirement_json_contains_functional_requirements(normalized_requirement):
    data = json.loads(normalized_requirement.model_dump_json())
    assert "functional_requirements" in data
    assert isinstance(data["functional_requirements"], list)
    assert len(data["functional_requirements"]) > 0


def test_normalized_requirement_preserves_tuple_types(normalized_requirement):
    restored = NormalizedRequirement.model_validate_json(
        normalized_requirement.model_dump_json()
    )
    assert isinstance(restored.functional_requirements, tuple)
    assert isinstance(restored.primary_actors, tuple)


# ------------------------------------------------------------------ #
# StoryCollection computed field constraints
# ------------------------------------------------------------------ #

def test_story_collection_count_matches_stories_length(story_collection):
    assert story_collection.count == len(story_collection.stories)


def test_story_collection_model_dump_includes_count(story_collection):
    data = story_collection.model_dump()
    assert "count" in data
    assert data["count"] == 1


def test_story_collection_round_trip_via_model_validate_fails(story_collection):
    # model_dump() includes 'count' (computed_field); model_validate rejects it
    # because extra="forbid" treats computed field keys as extra inputs
    data = story_collection.model_dump()
    with pytest.raises(Exception):
        StoryCollection.model_validate(data)


def test_story_collection_model_copy_preserves_count(story_collection):
    copied = story_collection.model_copy(update={})
    assert copied.count == story_collection.count


def test_empty_story_collection_count_is_zero():
    collection = StoryCollection()
    assert collection.count == 0


def test_story_collection_count_updates_with_stories():
    ac = AcceptanceCriterion(text="Test criterion.")
    story = UserStory(
        title="A story",
        description="Description here.",
        persona="user",
        goal="do something",
        benefit="get value",
        acceptance_criteria=(ac,),
    )
    collection = StoryCollection(stories=(story,))
    assert collection.count == 1


# ------------------------------------------------------------------ #
# WorkflowState serialization
# ------------------------------------------------------------------ #

def test_workflow_state_approval_status_serializes_as_string():
    state = WorkflowState(raw_requirement="req")
    data = state.model_dump()
    assert data["approval_status"] == "pending"


def test_workflow_state_current_step_serializes_as_string():
    state = WorkflowState(raw_requirement="req")
    data = state.model_dump()
    assert data["current_step"] == "pending"


def test_workflow_state_approved_status_serializes_correctly():
    state = WorkflowState(raw_requirement="req").model_copy(
        update={"approval_status": ApprovalStatus.APPROVED}
    )
    data = state.model_dump()
    assert data["approval_status"] == "approved"


def test_workflow_state_model_dump_json_contains_raw_requirement():
    state = WorkflowState(raw_requirement="my requirement text")
    serialized = state.model_dump_json()
    assert "my requirement text" in serialized


def test_workflow_state_model_dump_json_is_valid_json():
    state = WorkflowState(raw_requirement="req")
    serialized = state.model_dump_json()
    data = json.loads(serialized)
    assert isinstance(data, dict)
    assert "workflow_id" in data


# ------------------------------------------------------------------ #
# Enum serialization
# ------------------------------------------------------------------ #

def test_approval_status_values_are_strings():
    for status in ApprovalStatus:
        assert isinstance(status.value, str)


def test_workflow_step_values_are_strings():
    for step in WorkflowStep:
        assert isinstance(step.value, str)


def test_approval_status_round_trip_via_value():
    for status in ApprovalStatus:
        restored = ApprovalStatus(status.value)
        assert restored == status


def test_workflow_step_round_trip_via_value():
    for step in WorkflowStep:
        restored = WorkflowStep(step.value)
        assert restored == step
