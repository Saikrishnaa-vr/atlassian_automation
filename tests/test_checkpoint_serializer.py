from __future__ import annotations

import ormsgpack
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import EXT_PYDANTIC_V2 as _EXT_PYDANTIC_V2
from langgraph.graph import StateGraph

from src.graph.state import ApprovalStatus, ExecutionMetadata, WorkflowState, WorkflowStep
from src.infrastructure.checkpoint_serializer import (
    LangGraphCheckpointSerde,
    make_memory_saver,
)
from src.schemas.epic import Epic, EpicCollection
from src.schemas.story import AcceptanceCriterion, StoryCollection, UserStory
from src.schemas.task import Task, TaskCollection


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_kwargs(msgpack_bytes: bytes) -> dict:
    """Unpack the kwargs dict from the outermost EXT_PYDANTIC_V2 envelope.

    The EXT payload format is a 4-tuple:
    (module: str, classname: str, kwargs: dict, hint: str)

    We pass a recursive ext_hook so that nested EXT_PYDANTIC_V2 objects (e.g.
    UserStory instances inside StoryCollection.stories) are unpacked to their
    own kwargs dicts, and non-Pydantic EXT codes (enums, UUIDs, datetimes) are
    replaced with None sentinels — safe because we only inspect top-level keys.
    """
    def _ext_hook(code: int, data: bytes) -> Any:
        if code == _EXT_PYDANTIC_V2:
            inner = ormsgpack.unpackb(data, ext_hook=_ext_hook)
            return inner[2]  # kwargs dict, nested EXT already resolved
        return None  # sentinel for enum/UUID/datetime EXT codes

    return ormsgpack.unpackb(msgpack_bytes, ext_hook=_ext_hook)


# --------------------------------------------------------------------------- #
# Basic API
# --------------------------------------------------------------------------- #

def test_serde_is_importable():
    from src.infrastructure.checkpoint_serializer import LangGraphCheckpointSerde  # noqa: F401


def test_make_memory_saver_returns_memory_saver():
    saver = make_memory_saver()
    assert isinstance(saver, InMemorySaver)


def test_memory_saver_has_serde():
    saver = make_memory_saver()
    assert isinstance(saver.serde, LangGraphCheckpointSerde)


# --------------------------------------------------------------------------- #
# Computed fields excluded from payload
# --------------------------------------------------------------------------- #

def test_story_collection_count_not_in_serialized_payload(story_collection):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    assert type_tag == "msgpack"
    kwargs = _extract_kwargs(data)
    assert "count" not in kwargs, "computed field 'count' must not appear in msgpack payload"


def test_epic_collection_count_not_in_serialized_payload():
    from src.schemas.epic import EpicGoal
    epic = Epic(
        title="Auth Platform",
        summary="Build the authentication platform.",
        business_value="Enables secure user access.",
        objective="Deliver a secure login flow.",
        success_metrics=(EpicGoal(description="Login success rate >= 99%"),),
        related_story_ids=("story-001",),
    )
    collection = EpicCollection(epics=(epic,))
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(collection)
    assert type_tag == "msgpack"
    kwargs = _extract_kwargs(data)
    assert "count" not in kwargs


def test_task_collection_count_not_in_serialized_payload():
    from src.schemas.task import TaskPriority, TaskType
    task = Task(
        title="Write unit tests",
        description="Cover all edge cases for the login handler.",
        parent_story_id="story-001",
        task_type=TaskType.TEST,
        priority=TaskPriority.MEDIUM,
    )
    collection = TaskCollection(tasks=(task,))
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(collection)
    assert type_tag == "msgpack"
    kwargs = _extract_kwargs(data)
    assert "count" not in kwargs


# --------------------------------------------------------------------------- #
# Round-trip: StoryCollection
# --------------------------------------------------------------------------- #

def test_story_collection_round_trip(story_collection):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result, StoryCollection)


def test_story_collection_stories_are_typed_after_round_trip(story_collection):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result, StoryCollection)
    assert isinstance(result.stories[0], UserStory), (
        f"expected UserStory but got {type(result.stories[0])!r}"
    )


def test_story_collection_count_matches_after_round_trip(story_collection):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    result = serde.loads_typed((type_tag, data))
    assert result.count == story_collection.count


def test_nested_acceptance_criteria_are_typed(story_collection):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    result = serde.loads_typed((type_tag, data))
    ac = result.stories[0].acceptance_criteria[0]
    assert isinstance(ac, AcceptanceCriterion), (
        f"expected AcceptanceCriterion but got {type(ac)!r}"
    )


def test_story_title_preserved_after_round_trip(story_collection):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    result = serde.loads_typed((type_tag, data))
    assert result.stories[0].title == story_collection.stories[0].title


def test_optional_requirement_id_none_preserved():
    story = UserStory(
        title="Register account",
        description="A new user can register.",
        persona="visitor",
        goal="create an account",
        benefit="access the platform",
        acceptance_criteria=(AcceptanceCriterion(text="Account created."),),
    )
    collection = StoryCollection(stories=(story,), requirement_id=None)
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(collection)
    result = serde.loads_typed((type_tag, data))
    assert result.requirement_id is None


# --------------------------------------------------------------------------- #
# Round-trip: WorkflowState
# --------------------------------------------------------------------------- #

def test_workflow_state_round_trip(workflow_state):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(workflow_state)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result, WorkflowState)


def test_enum_approval_status_preserved_after_round_trip():
    state = WorkflowState(raw_requirement="test req").model_copy(
        update={"approval_status": ApprovalStatus.APPROVED}
    )
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(state)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result.approval_status, ApprovalStatus)
    assert result.approval_status == ApprovalStatus.APPROVED


def test_enum_current_step_preserved_after_round_trip():
    state = WorkflowState(raw_requirement="test req").model_copy(
        update={"current_step": WorkflowStep.AWAITING_APPROVAL}
    )
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(state)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result.current_step, WorkflowStep)
    assert result.current_step == WorkflowStep.AWAITING_APPROVAL


def test_uuid_workflow_id_preserved(workflow_state):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(workflow_state)
    result = serde.loads_typed((type_tag, data))
    assert result.workflow_id == workflow_state.workflow_id


def test_raw_requirement_preserved(workflow_state):
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(workflow_state)
    result = serde.loads_typed((type_tag, data))
    assert result.raw_requirement == workflow_state.raw_requirement


def test_workflow_state_with_stories_round_trips(story_collection):
    state = WorkflowState(raw_requirement="Build auth.").model_copy(
        update={"generated_stories": story_collection}
    )
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(state)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result.generated_stories, StoryCollection)
    assert isinstance(result.generated_stories.stories[0], UserStory)


# --------------------------------------------------------------------------- #
# Round-trip: datetime via ExecutionMetadata
# --------------------------------------------------------------------------- #

def test_datetime_preserved_in_round_trip():
    from datetime import datetime, timezone
    started = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    metadata = ExecutionMetadata(started_at=started, step_count=3)
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(metadata)
    result = serde.loads_typed((type_tag, data))
    assert isinstance(result, ExecutionMetadata)
    assert result.started_at == started
    assert result.step_count == 3


# --------------------------------------------------------------------------- #
# No model_construct fallback
# --------------------------------------------------------------------------- #

def test_no_model_construct_fallback_story_collection(story_collection):
    """model_construct fallback leaves stories as plain dicts; typed instances prove it didn't happen."""
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    result = serde.loads_typed((type_tag, data))
    assert type(result.stories[0]) is UserStory, (
        f"model_construct fallback occurred: got {type(result.stories[0])!r} instead of UserStory"
    )


def test_story_collection_validator_does_not_raise_after_round_trip(story_collection):
    """_unique_story_ids accesses story.id — raises AttributeError if stories are plain dicts."""
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(story_collection)
    # If model_construct was used, reconstructing WorkflowState triggers the validator and crashes.
    result = serde.loads_typed((type_tag, data))
    # Re-validate by constructing a WorkflowState that holds the round-tripped collection.
    state = WorkflowState(
        raw_requirement="test",
        generated_stories=result,
    )
    assert state.generated_stories.stories[0].id == result.stories[0].id


# --------------------------------------------------------------------------- #
# Invalid checkpoint raises
# --------------------------------------------------------------------------- #

def test_invalid_msgpack_raises():
    serde = LangGraphCheckpointSerde()
    with pytest.raises(Exception):
        # \x8a = fixmap claiming 10 entries with no data following — incomplete msgpack
        serde.loads_typed(("msgpack", b"\x8a"))


def test_bytes_passthrough_round_trip():
    raw = b"\x00\x01\x02\x03"
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(raw)
    assert type_tag == "bytes"
    result = serde.loads_typed((type_tag, data))
    assert result == raw


def test_none_passthrough_round_trip():
    serde = LangGraphCheckpointSerde()
    type_tag, data = serde.dumps_typed(None)
    assert type_tag == "null"
    result = serde.loads_typed((type_tag, data))
    assert result is None


# --------------------------------------------------------------------------- #
# make_memory_saver integration
# --------------------------------------------------------------------------- #

def test_make_memory_saver_checkpointer_enables_interrupt():
    """A compiled graph using make_memory_saver() has a non-None checkpointer."""
    from pydantic import BaseModel

    class _S(BaseModel):
        x: str = ""

    graph = StateGraph(_S)
    graph.add_node("n", lambda s: s)
    graph.set_entry_point("n")
    graph.set_finish_point("n")
    compiled = graph.compile(checkpointer=make_memory_saver())
    assert compiled.checkpointer is not None
