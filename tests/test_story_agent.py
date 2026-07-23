from __future__ import annotations

import json
from unittest.mock import MagicMock, create_autospec

import pytest

from src.agents.story_agent import StoryAgent
from src.exceptions import AgentError, ValidationError
from src.schemas.requirement import NormalizedRequirement
from src.schemas.story import StoryCollection
from src.services.anthropic_service import (
    AnthropicService,
    AnthropicServiceError,
    GenerationResponse,
    UsageInfo,
)


def _make_response(text: str) -> GenerationResponse:
    return GenerationResponse(
        id="msg_test",
        model="claude-sonnet-4-6",
        text=text,
        stop_reason="end_turn",
        usage=UsageInfo(input_tokens=10, output_tokens=20),
    )


def _valid_story_json() -> str:
    return json.dumps({
        "stories": [
            {
                "title": "Login with credentials",
                "description": "User logs in using email and password.",
                "persona": "registered user",
                "goal": "log in with my credentials",
                "benefit": "access the application securely",
                "acceptance_criteria": [
                    {"text": "Login succeeds with valid credentials."}
                ],
            }
        ]
    })


def _two_stories_json() -> str:
    return json.dumps({
        "stories": [
            {
                "title": "Login with credentials",
                "description": "User logs in using email and password.",
                "persona": "registered user",
                "goal": "log in",
                "benefit": "get access",
                "acceptance_criteria": [{"text": "Login works."}],
            },
            {
                "title": "Logout from application",
                "description": "User logs out of the application.",
                "persona": "registered user",
                "goal": "log out",
                "benefit": "secure my session",
                "acceptance_criteria": [{"text": "Logout terminates the session."}],
            },
        ]
    })


@pytest.fixture
def mock_service() -> MagicMock:
    return create_autospec(AnthropicService, instance=True)


# ------------------------------------------------------------------ #
# Happy path
# ------------------------------------------------------------------ #

def test_execute_returns_story_collection(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response(_valid_story_json())
    agent = StoryAgent(service=mock_service)
    result = agent.execute(normalized_requirement)
    assert isinstance(result, StoryCollection)


def test_execute_returns_correct_story_count(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response(_valid_story_json())
    agent = StoryAgent(service=mock_service)
    result = agent.execute(normalized_requirement)
    assert result.count == 1


def test_execute_multiple_stories(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response(_two_stories_json())
    agent = StoryAgent(service=mock_service)
    result = agent.execute(normalized_requirement)
    assert result.count == 2


def test_execute_calls_service_generate(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response(_valid_story_json())
    agent = StoryAgent(service=mock_service)
    agent.execute(normalized_requirement)
    assert mock_service.generate.call_count == 1


def test_execute_story_has_correct_title(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response(_valid_story_json())
    agent = StoryAgent(service=mock_service)
    result = agent.execute(normalized_requirement)
    assert result.stories[0].title == "Login with credentials"


# ------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------ #

def test_none_input_raises_validation_error(mock_service):
    agent = StoryAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute(None)


def test_empty_functional_requirements_raises_validation_error(mock_service):
    # Bypass Pydantic construction constraints with a mock
    mock_req = MagicMock(spec=NormalizedRequirement)
    mock_req.functional_requirements = ()
    agent = StoryAgent(service=mock_service)
    with pytest.raises(ValidationError, match="functional requirement"):
        agent.execute(mock_req)


def test_input_validation_does_not_call_service(mock_service):
    agent = StoryAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute(None)
    mock_service.generate.assert_not_called()


# ------------------------------------------------------------------ #
# Service error handling
# ------------------------------------------------------------------ #

def test_service_error_raises_agent_error(mock_service, normalized_requirement):
    mock_service.generate.side_effect = AnthropicServiceError("API down")
    agent = StoryAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute(normalized_requirement)


def test_agent_error_wraps_service_error(mock_service, normalized_requirement):
    original = AnthropicServiceError("rate limited")
    mock_service.generate.side_effect = original
    agent = StoryAgent(service=mock_service)
    with pytest.raises(AgentError) as exc_info:
        agent.execute(normalized_requirement)
    assert exc_info.value.__cause__ is original


# ------------------------------------------------------------------ #
# Response parsing errors
# ------------------------------------------------------------------ #

def test_invalid_json_raises_agent_error(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response("not json")
    agent = StoryAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute(normalized_requirement)


def test_wrong_schema_raises_agent_error(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response('{"wrong": "schema"}')
    agent = StoryAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute(normalized_requirement)


def test_empty_response_text_raises_agent_error(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response("   ")
    agent = StoryAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute(normalized_requirement)


# ------------------------------------------------------------------ #
# Output validation
# ------------------------------------------------------------------ #

def test_empty_stories_raises_validation_error(mock_service, normalized_requirement):
    mock_service.generate.return_value = _make_response('{"stories": []}')
    agent = StoryAgent(service=mock_service)
    with pytest.raises(ValidationError, match="empty"):
        agent.execute(normalized_requirement)


def test_duplicate_story_titles_raises_validation_error(mock_service, normalized_requirement):
    duplicate_json = json.dumps({
        "stories": [
            {
                "title": "Duplicate title",
                "description": "First story.",
                "persona": "user",
                "goal": "do something",
                "benefit": "get value",
                "acceptance_criteria": [{"text": "First criterion."}],
            },
            {
                "title": "Duplicate title",
                "description": "Second story.",
                "persona": "user",
                "goal": "do another thing",
                "benefit": "get more value",
                "acceptance_criteria": [{"text": "Second criterion."}],
            },
        ]
    })
    mock_service.generate.return_value = _make_response(duplicate_json)
    agent = StoryAgent(service=mock_service)
    with pytest.raises(ValidationError, match="duplicate"):
        agent.execute(normalized_requirement)


# ------------------------------------------------------------------ #
# JSON fence stripping
# ------------------------------------------------------------------ #

def test_strip_json_fences_removes_json_code_block():
    fenced = "```json\n{\"stories\": []}\n```"
    result = StoryAgent._strip_json_fences(fenced)
    assert result == '{"stories": []}'


def test_strip_json_fences_leaves_plain_json_unchanged():
    text = '{"stories": []}'
    result = StoryAgent._strip_json_fences(text)
    assert result == text


def test_execute_strips_json_fences_before_parsing(mock_service, normalized_requirement):
    json_str = _valid_story_json()
    mock_service.generate.return_value = _make_response(f"```json\n{json_str}\n```")
    agent = StoryAgent(service=mock_service)
    result = agent.execute(normalized_requirement)
    assert isinstance(result, StoryCollection)
    assert result.count == 1
