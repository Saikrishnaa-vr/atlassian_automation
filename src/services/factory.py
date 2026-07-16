"""Cached service factories for infrastructure clients.

Factories centralize service construction and expose singleton accessors for
runtime use. Service classes themselves remain unchanged and still support
constructor injection for tests and advanced wiring.
"""

from __future__ import annotations

from functools import lru_cache

from src.services.anthropic_service import AnthropicService
from src.services.jira_service import JiraService


@lru_cache
def get_anthropic_service() -> AnthropicService:
    """Return the cached Anthropic service singleton."""
    return AnthropicService()


@lru_cache
def get_jira_service() -> JiraService:
    """Return the cached Jira service singleton."""
    return JiraService()
