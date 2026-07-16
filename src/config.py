"""Application configuration — the single source of truth.

This module defines the strongly typed settings for the Atlassian Automation
system and loads them exclusively from environment variables / the ``.env``
file via :mod:`pydantic_settings`.

Configuration is organized into cohesive, independently validated nested
settings groups (:class:`AppSettings`, :class:`AnthropicSettings`,
:class:`JiraSettings`, :class:`LoggingSettings`) composed onto the top-level
:class:`Settings` object.

It contains configuration only: no business logic, no API code, no workflow or
LangGraph code. Consumers obtain the validated, cached settings singleton via
:func:`get_settings`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Typed value domains
# --------------------------------------------------------------------------- #
Environment = Literal["development", "testing", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# Shared source configuration for every settings group. Each group is its own
# ``BaseSettings`` so it reads and validates its own environment variables
# independently while remaining composable on the top-level ``Settings``.
_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",  # ignore unrelated keys in .env (e.g. TAVILY_API_KEY)
    populate_by_name=True,
)


def _not_blank(value: str) -> str:
    """Reject empty or whitespace-only strings, returning the trimmed value."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("value must not be empty or whitespace")
    return cleaned


# --------------------------------------------------------------------------- #
# Nested settings groups
# --------------------------------------------------------------------------- #
class AppSettings(BaseSettings):
    """General application identity and runtime behaviour."""

    model_config = _ENV_CONFIG

    name: str = Field(
        default="atlassian-automation",
        validation_alias="APP_NAME",
        description="Human-readable application name.",
    )
    environment: Environment = Field(
        default="development",
        validation_alias="ENVIRONMENT",
        description="Deployment environment: development, testing, or production.",
    )
    request_timeout: int = Field(
        default=30,
        gt=0,
        validation_alias="REQUEST_TIMEOUT",
        description="Outbound request timeout in seconds for service clients.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Ensure APP_NAME is not blank."""
        return _not_blank(value)


class AnthropicSettings(BaseSettings):
    """Credentials and model selection for the Anthropic Claude API."""

    model_config = _ENV_CONFIG

    api_key: SecretStr = Field(
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key (secret).",
    )
    model: str = Field(
        default="claude-sonnet-5",
        validation_alias="ANTHROPIC_MODEL",
        description="Default Claude model identifier.",
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        """Ensure ANTHROPIC_MODEL is not blank."""
        return _not_blank(value)


class JiraSettings(BaseSettings):
    """Connection details for the Jira REST API."""

    model_config = _ENV_CONFIG

    base_url: HttpUrl = Field(
        validation_alias="JIRA_BASE_URL",
        description="Base URL of the Jira instance, e.g. https://org.atlassian.net.",
    )
    email: EmailStr = Field(
        validation_alias="JIRA_EMAIL",
        description="Account email used for Jira Basic auth.",
    )
    api_token: SecretStr = Field(
        validation_alias="JIRA_API_TOKEN",
        description="Jira API token (secret).",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration (consumed by the Loguru setup layer)."""

    model_config = _ENV_CONFIG

    level: LogLevel = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Minimum log level.",
    )


# --------------------------------------------------------------------------- #
# Top-level settings
# --------------------------------------------------------------------------- #
class Settings(BaseModel):
    """Composed application settings.

    ``Settings`` holds no environment variables of its own; it composes the
    nested groups, each of which reads and validates its own environment
    variables. An invalid or missing required value therefore fails fast at
    startup with a :class:`pydantic.ValidationError` rather than surfacing
    later at runtime.
    """

    app: AppSettings = Field(default_factory=AppSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    jira: JiraSettings = Field(default_factory=JiraSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # -- Environment predicates -------------------------------------------- #
    @property
    def is_development(self) -> bool:
        """True when running in the development environment."""
        return self.app.environment == "development"

    @property
    def is_testing(self) -> bool:
        """True when running in the testing environment."""
        return self.app.environment == "testing"

    @property
    def is_production(self) -> bool:
        """True when running in the production environment."""
        return self.app.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the cached, validated settings singleton.

    The first call reads the environment and validates all fields; subsequent
    calls return the same instance. Call ``get_settings.cache_clear()`` to force
    a reload (primarily useful in tests).
    """
    return Settings()
