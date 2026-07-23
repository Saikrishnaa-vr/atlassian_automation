"""Domain model for a normalized software requirement.

This is an *application-level* contract produced by
:class:`~src.agents.requirement_agent.RequirementAgent` and consumed by
:class:`~src.agents.story_agent.StoryAgent` and the LangGraph workflow nodes.
It is intentionally free of any SDK, transport, or framework dependency (no
Anthropic, Jira SDK, FastAPI, or LangGraph imports) so the same vocabulary
flows unchanged across every layer.

The module contains contracts only — no business logic, service logic, or
workflow orchestration. Validators here enforce *data integrity* (non-blank
text, non-empty required collections, unique entries), not domain behaviour.

:class:`NormalizedRequirement` was originally defined in
``src/agents/requirement_agent.py`` as an interim model and moved here during
the Phase-3 architectural refactor to enforce the correct dependency direction::

    src.schemas  ←  consumed by  →  src.agents  →  src.graph.nodes  →  src.graph.workflow

Agents must never define domain models that are shared with other agents or
with the graph layer; all shared contracts belong in this schema layer.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from src.schemas.base import (
    DomainModel as _DomainModel,
    clean_unique_tokens as _clean_unique_tokens,
    require_non_blank as _require_non_blank,
)


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #


class NormalizedRequirement(_DomainModel):
    """Structured, normalized representation of a raw requirement string.

    Produced by :class:`~src.agents.requirement_agent.RequirementAgent` and
    consumed by :class:`~src.agents.story_agent.StoryAgent` and the LangGraph
    workflow nodes.  All fields carry application-native semantics; none are
    tied to any SDK or external service.

    Fields with ``min_length=1`` on :func:`~pydantic.Field` enforce that the
    LLM provides at least one entry.  Fields with ``default=()`` may be empty
    when the requirement does not supply that information.
    """

    title: str = Field(
        min_length=1,
        max_length=255,
        description="Short title capturing the core requirement (max 255 chars).",
    )
    summary: str = Field(
        min_length=1,
        description="Concise 2-3 sentence summary of the requirement.",
    )
    business_objective: str = Field(
        min_length=1,
        description="The primary business goal this requirement addresses.",
    )
    primary_actors: tuple[str, ...] = Field(
        min_length=1,
        description="Users or systems that directly interact with the feature; at least one required.",
    )
    functional_requirements: tuple[str, ...] = Field(
        min_length=1,
        description="Distinct things the system must do; at least one required.",
    )
    non_functional_requirements: tuple[str, ...] = Field(
        default=(),
        description="Performance, reliability, security, or other quality attributes.",
    )
    assumptions: tuple[str, ...] = Field(
        default=(),
        description="Assumptions relied on during analysis.",
    )
    constraints: tuple[str, ...] = Field(
        default=(),
        description="Known limitations or boundaries that apply to the requirement.",
    )
    missing_information: tuple[str, ...] = Field(
        default=(),
        description="Gaps or ambiguities requiring stakeholder clarification.",
    )

    @field_validator("title", "summary", "business_objective")
    @classmethod
    def _text_fields_not_blank(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        """Core text fields cannot be blank or whitespace-only."""
        return _require_non_blank(value, field=info.field_name)

    @field_validator(
        "primary_actors",
        "functional_requirements",
        "non_functional_requirements",
        "assumptions",
        "constraints",
        "missing_information",
    )
    @classmethod
    def _items_unique_and_non_blank(
        cls,
        value: tuple[str, ...],
        info,  # type: ignore[no-untyped-def]
    ) -> tuple[str, ...]:
        """Tuple fields must not contain blank or duplicate entries."""
        return _clean_unique_tokens(value, field=info.field_name)
