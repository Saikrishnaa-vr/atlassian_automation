from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, create_autospec

import pytest
from pydantic import TypeAdapter, HttpUrl as _HttpUrl

from src.schemas.jira import JiraIssueResponse, JiraIssueType, JiraStatus
from src.schemas.requirement import NormalizedRequirement
from src.schemas.story import AcceptanceCriterion, StoryCollection, UserStory
from src.graph.state import ApprovalStatus, WorkflowState, WorkflowStep
from src.services.anthropic_service import AnthropicService, GenerationResponse, UsageInfo


@pytest.fixture
def normalized_requirement() -> NormalizedRequirement:
    return NormalizedRequirement(
        title="User Authentication",
        summary="Allow users to authenticate with email and password.",
        business_objective="Enable secure user access to the application.",
        primary_actors=("registered user",),
        functional_requirements=(
            "User can log in with email and password",
            "User receives a session token on successful login",
        ),
    )


@pytest.fixture
def user_story() -> UserStory:
    return UserStory(
        title="Login with credentials",
        description="A user logs in using their email and password to gain access.",
        persona="registered user",
        goal="log in with my credentials",
        benefit="access the application securely",
        acceptance_criteria=(
            AcceptanceCriterion(text="Login succeeds with valid credentials."),
        ),
    )


@pytest.fixture
def story_collection(user_story: UserStory) -> StoryCollection:
    return StoryCollection(stories=(user_story,))


@pytest.fixture
def workflow_state() -> WorkflowState:
    return WorkflowState(raw_requirement="Build a login system for users.")


@pytest.fixture
def make_generation_response():
    """Factory: returns a callable that builds a GenerationResponse with custom text."""
    def _make(text: str) -> GenerationResponse:
        return GenerationResponse(
            id="msg_test",
            model="claude-sonnet-4-6",
            text=text,
            stop_reason="end_turn",
            usage=UsageInfo(input_tokens=10, output_tokens=20),
        )
    return _make


@pytest.fixture
def mock_anthropic_service() -> MagicMock:
    return create_autospec(AnthropicService, instance=True)


@pytest.fixture
def jira_issue_response() -> JiraIssueResponse:
    url = TypeAdapter(_HttpUrl).validate_python(
        "https://example.atlassian.net/browse/PROJ-1"
    )
    return JiraIssueResponse(
        id="10001",
        key="PROJ-1",
        summary="Login with credentials",
        description=None,
        issue_type=JiraIssueType(id="1", name="Story"),
        status=JiraStatus(name="To Do", category_key="new"),
        project_key="PROJ",
        labels=(),
        url=url,
        created=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
