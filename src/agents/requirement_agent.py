"""Requirement normalization agent.

:class:`RequirementAgent` is the first reasoning agent in the MVP pipeline
(``Requirement -> RequirementAgent -> StoryAgent -> Approval -> Jira``).
It converts raw, potentially ambiguous requirement text into a structured,
fully typed :class:`NormalizedRequirement` that downstream agents consume.

Responsibility boundary
-----------------------
``RequirementAgent`` has exactly one responsibility: normalize a raw
requirement string.  It must **not**:

* generate stories, epics, or tasks,
* call Jira or any other external service,
* modify or update workflow state,
* invoke other agents,
* contain workflow routing or approval logic.

Output model
------------
:class:`NormalizedRequirement` is defined here as an interim domain model.
Once the schema layer is extended, it **must** be moved to
``src/schemas/requirement.py`` and imported from there.  The agent
implementation will not need to change when that migration happens.

Prompt templates
----------------
Prompt text lives in :mod:`src.prompts.requirement_prompt` and is imported
as private module-level names.  To evolve the prompt wording, edit that
module — this agent class does not need to change.
"""

from __future__ import annotations

import re

from src.agents.base_agent import BaseAgent, GenerationRequest, GenerationResponse
from src.exceptions import ValidationError
from src.prompts.requirement_prompt import (
    SYSTEM_PROMPT as _SYSTEM_PROMPT,
    build_user_prompt as _build_user_prompt,
)
from src.schemas.requirement import NormalizedRequirement
from src.services.anthropic_service import AnthropicService


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

_MAX_TOKENS: int = 2048
"""Default token budget for :class:`RequirementAgent`.

2048 tokens is sufficient for detailed structured JSON output across all nine
output fields, including verbose functional requirement lists.
"""

_JSON_FENCE_RE: re.Pattern[str] = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
"""Compiled pattern for stripping optional markdown code fence wrappers from LLM output."""


class RequirementAgent(BaseAgent[str, NormalizedRequirement]):
    """Agent that normalizes raw requirement text into a structured representation.

    Takes a raw requirement string and returns a fully validated
    :class:`NormalizedRequirement` covering title, summary, business objective,
    primary actors, functional and non-functional requirements, assumptions,
    constraints, and identified information gaps.

    This agent has exactly one responsibility.  It must not generate stories,
    call Jira, modify workflow state, invoke other agents, or contain any
    workflow or approval logic.

    Args:
        service: Optional :class:`~src.services.anthropic_service.AnthropicService`
            injection.  When omitted the factory singleton is used, which is
            the correct choice for production.  Pass a custom instance in tests.
        max_tokens: Token budget for generation.  Defaults to
            :data:`_MAX_TOKENS` (2048).
        temperature: Sampling temperature.  ``None`` uses the model default.
            Lower values (e.g. 0.2) produce more deterministic JSON output.
    """

    def __init__(
        self,
        *,
        service: AnthropicService | None = None,
        max_tokens: int = _MAX_TOKENS,
        temperature: float | None = None,
    ) -> None:
        super().__init__(service=service, max_tokens=max_tokens, temperature=temperature)

    # -- Validation override ------------------------------------------------- #

    def _validate_input(self, input: str) -> None:  # noqa: A002
        """Extend the base None-check with a blank/whitespace-only check.

        Args:
            input: The raw requirement text.

        Raises:
            ValidationError: If the input is ``None``, empty, or whitespace-only.
        """
        super()._validate_input(input)
        if not input.strip():
            raise ValidationError(
                "RequirementAgent input must not be blank or whitespace-only"
            )

    # -- Abstract implementations -------------------------------------------- #

    def _build_prompt(self, input: str) -> GenerationRequest:  # noqa: A002
        """Format the requirement text into a :class:`GenerationRequest`.

        Args:
            input: The validated raw requirement text.

        Returns:
            A :class:`GenerationRequest` with system and user messages.
        """
        user_message = _build_user_prompt(input.strip())
        return self._make_request(
            [self._make_user_message(user_message)],
            system=_SYSTEM_PROMPT,
        )

    def _parse_response(
        self,
        response: GenerationResponse,
        input: str,  # noqa: A002
    ) -> NormalizedRequirement:
        """Parse the LLM response into a :class:`NormalizedRequirement`.

        Extracts text, strips any markdown code fences the model may have
        inserted despite instructions, then parses JSON via
        :meth:`~BaseAgent._parse_model`.

        Args:
            response: The :class:`GenerationResponse` from the LLM.
            input: Original requirement text; unused here, retained for
                signature consistency and future correlation if needed.

        Returns:
            A validated :class:`NormalizedRequirement`.

        Raises:
            AgentError: If the response is empty, not valid JSON, or does not
                conform to the :class:`NormalizedRequirement` schema.
        """
        text = self._extract_text(response)
        cleaned = self._strip_json_fences(text)
        return self._parse_model(cleaned, NormalizedRequirement)

    # -- Parsing helpers ----------------------------------------------------- #

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Strip markdown code fence wrappers from LLM output, if present.

        The model occasionally wraps JSON in a code block (e.g. ```json...```)
        despite being instructed not to.  This method extracts the inner content
        when a fence is detected and returns the text unchanged otherwise.

        Args:
            text: Raw text extracted from the LLM response.

        Returns:
            The inner JSON string, or ``text`` unchanged.
        """
        match = _JSON_FENCE_RE.search(text)
        return match.group(1).strip() if match else text
