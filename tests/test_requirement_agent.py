from __future__ import annotations

import json
from unittest.mock import MagicMock, create_autospec

import pytest

from src.agents.requirement_agent import RequirementAgent
from src.exceptions import AgentError, ValidationError
from src.schemas.requirement import NormalizedRequirement
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


def _valid_requirement_json() -> str:
    return json.dumps({
        "title": "User Authentication",
        "summary": "Allow users to authenticate with email and password.",
        "business_objective": "Enable secure access to the application.",
        "primary_actors": ["registered user"],
        "functional_requirements": ["User can log in with valid credentials"],
    })


@pytest.fixture
def mock_service() -> MagicMock:
    return create_autospec(AnthropicService, instance=True)


# ------------------------------------------------------------------ #
# Happy path
# ------------------------------------------------------------------ #

def test_execute_returns_normalized_requirement(mock_service):
    mock_service.generate.return_value = _make_response(_valid_requirement_json())
    agent = RequirementAgent(service=mock_service)
    result = agent.execute("Build a login system.")
    assert isinstance(result, NormalizedRequirement)
    assert result.title == "User Authentication"


def test_execute_passes_requirement_to_service(mock_service):
    mock_service.generate.return_value = _make_response(_valid_requirement_json())
    agent = RequirementAgent(service=mock_service)
    agent.execute("Build a login system.")
    assert mock_service.generate.call_count == 1


def test_execute_returns_correct_functional_requirements(mock_service):
    mock_service.generate.return_value = _make_response(_valid_requirement_json())
    agent = RequirementAgent(service=mock_service)
    result = agent.execute("Build a login system.")
    assert len(result.functional_requirements) == 1


# ------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------ #

def test_blank_input_raises_validation_error(mock_service):
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute("   ")


def test_empty_string_input_raises_validation_error(mock_service):
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute("")


def test_none_input_raises_validation_error(mock_service):
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute(None)


def test_whitespace_only_input_raises_validation_error(mock_service):
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute("\t\n  ")


def test_input_validation_does_not_call_service(mock_service):
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(ValidationError):
        agent.execute("")
    mock_service.generate.assert_not_called()


# ------------------------------------------------------------------ #
# Service error handling
# ------------------------------------------------------------------ #

def test_service_error_raises_agent_error(mock_service):
    mock_service.generate.side_effect = AnthropicServiceError("API unavailable")
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute("Build a login system.")


def test_agent_error_wraps_service_error(mock_service):
    original = AnthropicServiceError("rate limited")
    mock_service.generate.side_effect = original
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(AgentError) as exc_info:
        agent.execute("Build a login system.")
    assert exc_info.value.__cause__ is original


# ------------------------------------------------------------------ #
# Response parsing errors
# ------------------------------------------------------------------ #

def test_invalid_json_response_raises_agent_error(mock_service):
    mock_service.generate.return_value = _make_response("not json at all")
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute("Build a login system.")


def test_valid_json_wrong_schema_raises_agent_error(mock_service):
    mock_service.generate.return_value = _make_response('{"wrong_field": "value"}')
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute("Build a login system.")


def test_empty_response_text_raises_agent_error(mock_service):
    mock_service.generate.return_value = _make_response("   ")
    agent = RequirementAgent(service=mock_service)
    with pytest.raises(AgentError):
        agent.execute("Build a login system.")


# ------------------------------------------------------------------ #
# JSON fence stripping
# ------------------------------------------------------------------ #

def test_strip_json_fences_removes_json_code_block():
    fenced = "```json\n{\"key\": \"value\"}\n```"
    result = RequirementAgent._strip_json_fences(fenced)
    assert result == '{"key": "value"}'


def test_strip_json_fences_removes_plain_code_block():
    fenced = "```\n{\"key\": \"value\"}\n```"
    result = RequirementAgent._strip_json_fences(fenced)
    assert result == '{"key": "value"}'


def test_strip_json_fences_leaves_plain_json_unchanged():
    text = '{"key": "value"}'
    result = RequirementAgent._strip_json_fences(text)
    assert result == text


def test_strip_json_fences_leaves_text_unchanged():
    text = "some plain text"
    result = RequirementAgent._strip_json_fences(text)
    assert result == text


def test_execute_strips_json_fences_before_parsing(mock_service):
    json_str = _valid_requirement_json()
    mock_service.generate.return_value = _make_response(f"```json\n{json_str}\n```")
    agent = RequirementAgent(service=mock_service)
    result = agent.execute("Build a login system.")
    assert isinstance(result, NormalizedRequirement)
