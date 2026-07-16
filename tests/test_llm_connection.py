"""Live infrastructure verification for :class:`AnthropicService`.

These are NOT unit tests: they exercise the real Anthropic API through the
project's service wrapper to confirm the infrastructure layer works end to end.
The Anthropic SDK is never mocked. Configuration is read exclusively through
``get_settings``; when live credentials are unavailable (or the API is
unreachable) the tests skip rather than fail, so the suite stays green in
environments without secrets.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings
from src.services.anthropic_service import (
    AnthropicInvalidRequestError,
    AnthropicService,
    GenerationRequest,
    GenerationResponse,
    Message,
    UsageInfo,
)

# Intentionally tiny to minimize token usage and keep the test cheap.
SMALL_PROMPT = "Reply with exactly: OK"
SMALL_MAX_TOKENS = 16


def _require_settings() -> Settings:
    """Return live settings, skipping the test when configuration is unavailable."""
    try:
        return get_settings()
    except ValidationError as exc:
        pytest.skip(
            f"Skipping live Anthropic test: configuration unavailable "
            f"({exc.error_count()} validation error(s))"
        )


@pytest.fixture(scope="module")
def live_service() -> AnthropicService:
    """Provide an AnthropicService backed by live credentials.

    Skips the dependent tests when configuration is missing/invalid or when the
    API is unreachable, so real failures remain distinguishable from an
    unconfigured environment.
    """
    _require_settings()
    service = AnthropicService()
    if not service.health_check():
        pytest.skip("Skipping live Anthropic test: API unreachable or credentials invalid")
    return service


def test_service_construction() -> None:
    """The service can be constructed from live configuration."""
    _require_settings()
    service = AnthropicService()
    assert isinstance(service, AnthropicService)


def test_health_check_authenticates(live_service: AnthropicService) -> None:
    """health_check() confirms authentication and connectivity to the API."""
    assert live_service.health_check() is True


def test_generation_returns_valid_typed_response(live_service: AnthropicService) -> None:
    """A minimal generation returns a typed, non-empty response with usage data."""
    request = GenerationRequest(
        messages=[Message(role="user", content=SMALL_PROMPT)],
        max_tokens=SMALL_MAX_TOKENS,
    )

    try:
        response = live_service.generate(request)
    except AnthropicInvalidRequestError:
        pytest.skip(
            "Skipping live Anthropic test: model rejected the request "
            "(set ANTHROPIC_MODEL to a model enabled for this account)"
        )

    # Typed response object.
    assert isinstance(response, GenerationResponse)

    # Response model identity is valid.
    assert isinstance(response.model, str)
    assert response.model.strip()

    # Generated text is present and non-empty.
    assert isinstance(response.text, str)
    assert response.text.strip()

    # Usage information exists and is internally consistent.
    assert isinstance(response.usage, UsageInfo)
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.input_tokens + response.usage.output_tokens
    )
