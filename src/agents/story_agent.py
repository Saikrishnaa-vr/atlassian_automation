"""Story generation agent.

:class:`StoryAgent` is the second reasoning agent in the MVP pipeline
(``Requirement -> RequirementAgent -> StoryAgent -> Approval -> Jira``).
It converts a structured :class:`~src.agents.requirement_agent.NormalizedRequirement`
into a :class:`~src.schemas.story.StoryCollection` of INVEST-aligned Agile user
stories suitable for Jira.

Responsibility boundary
-----------------------
``StoryAgent`` has exactly one responsibility: convert a normalized requirement
into a collection of user stories.  It must **not**:

* call Jira or any other external service,
* modify or update workflow state,
* perform or check human approval,
* invoke other agents,
* contain workflow routing or LangGraph logic,
* know about FastAPI or HTTP concerns.

Input
-----
A :class:`~src.agents.requirement_agent.NormalizedRequirement` produced by
:class:`~src.agents.requirement_agent.RequirementAgent`.  The requirement is
serialized to JSON and embedded into the LLM prompt; the agent never forwards
raw string text to the model.

Output
------
A :class:`~src.schemas.story.StoryCollection` containing one or more validated
:class:`~src.schemas.story.UserStory` instances.  The collection is returned
directly — no wrappers, no dictionaries, no raw JSON.

Prompt templates
----------------
Prompt text lives in :mod:`src.prompts.story_prompt` and is imported as
private module-level names.  To evolve the prompt wording, edit that module —
this agent class does not need to change.

Isolation
---------
``StoryAgent`` can be instantiated and exercised in isolation without LangGraph,
FastAPI, Jira, or any workflow infrastructure.  Pass a custom
:class:`~src.services.anthropic_service.AnthropicService` instance for testing.
"""

from __future__ import annotations

import re

from src.agents.base_agent import BaseAgent, GenerationRequest, GenerationResponse
from src.exceptions import ValidationError
from src.schemas.requirement import NormalizedRequirement
from src.prompts.story_prompt import (
    SYSTEM_PROMPT as _SYSTEM_PROMPT,
    build_user_prompt as _build_user_prompt,
)
from src.schemas.story import StoryCollection
from src.services.anthropic_service import AnthropicService


# --------------------------------------------------------------------------- #
# Module-private constants
# --------------------------------------------------------------------------- #

_MAX_TOKENS: int = 4096
"""Default token budget for :class:`StoryAgent`.

4096 tokens accommodates a full :class:`~src.schemas.story.StoryCollection`
for complex requirements — typically 4-10 stories with persona, goal, benefit,
and multiple acceptance criteria each.
"""

_JSON_FENCE_RE: re.Pattern[str] = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
"""Compiled pattern for stripping optional markdown code fence wrappers from LLM output."""


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class StoryAgent(BaseAgent[NormalizedRequirement, StoryCollection]):
    """Agent that converts a normalized requirement into a StoryCollection.

    Takes a :class:`~src.agents.requirement_agent.NormalizedRequirement` and
    returns a fully validated :class:`~src.schemas.story.StoryCollection`
    containing one or more INVEST-aligned user stories, each with a persona,
    goal, benefit, and at least one testable acceptance criterion.

    This agent has exactly one responsibility.  It must not call Jira, modify
    workflow state, perform approval, invoke other agents, or contain any
    workflow or routing logic.

    Args:
        service: Optional :class:`~src.services.anthropic_service.AnthropicService`
            injection.  When omitted the factory singleton is used, which is
            the correct choice for production.  Pass a custom instance in tests.
        max_tokens: Token budget for generation.  Defaults to
            :data:`_MAX_TOKENS` (4096).
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

    # -- Validation overrides ------------------------------------------------ #

    def _validate_input(self, input: NormalizedRequirement) -> None:  # noqa: A002
        """Extend the base None-check with a meaningful-requirement check.

        A :class:`~src.agents.requirement_agent.NormalizedRequirement` is
        structurally valid (Pydantic enforces that) but we additionally verify
        that it contains at least one functional requirement, because story
        generation from a purely non-functional requirement set is undefined.

        Args:
            input: The normalized requirement to validate.

        Raises:
            ValidationError: If the input is ``None`` or has no functional
                requirements.
        """
        super()._validate_input(input)
        if not input.functional_requirements:
            raise ValidationError(
                "StoryAgent requires at least one functional requirement to generate stories"
            )

    def _validate_output(self, output: StoryCollection) -> None:
        """Extend the base None-check with semantic story-quality checks.

        Structural validation (field types, min_length, uniqueness of ids) is
        Pydantic's responsibility and has already run by the time this hook is
        called.  These checks enforce semantic invariants that Pydantic does not
        cover:

        * The collection must be non-empty.
        * Every story must have at least one acceptance criterion.
        * Story titles must be unique within the collection.
        * Story descriptions must not be whitespace-only.

        Args:
            output: The :class:`StoryCollection` to validate.

        Raises:
            ValidationError: If any of the above checks fail.
        """
        super()._validate_output(output)

        if not output.stories:
            raise ValidationError("StoryAgent produced an empty StoryCollection")

        for story in output.stories:
            if not story.acceptance_criteria:
                raise ValidationError(
                    f"Story '{story.title}' has no acceptance criteria"
                )
            if not story.description.strip():
                raise ValidationError(
                    f"Story '{story.title}' has a blank description"
                )

        titles = [story.title for story in output.stories]
        if len(set(titles)) != len(titles):
            raise ValidationError(
                "StoryAgent produced stories with duplicate titles"
            )

    # -- Abstract implementations -------------------------------------------- #

    def _build_prompt(self, input: NormalizedRequirement) -> GenerationRequest:  # noqa: A002
        """Serialize the requirement and construct the LLM request.

        The :class:`~src.agents.requirement_agent.NormalizedRequirement` is
        serialized to indented JSON via :meth:`~pydantic.BaseModel.model_dump_json`
        and embedded into the user message using :func:`~src.prompts.story_prompt.build_user_prompt`.
        The system message is the constant from :mod:`src.prompts.story_prompt`.

        Indented JSON (``indent=2``) is used because it materially improves LLM
        comprehension of nested structured data.

        Args:
            input: The validated :class:`~src.agents.requirement_agent.NormalizedRequirement`.

        Returns:
            A :class:`GenerationRequest` with system and user messages.
        """
        requirement_json = input.model_dump_json(indent=2)
        user_message = _build_user_prompt(requirement_json)
        return self._make_request(
            [self._make_user_message(user_message)],
            system=_SYSTEM_PROMPT,
        )

    def _parse_response(
        self,
        response: GenerationResponse,
        input: NormalizedRequirement,  # noqa: A002
    ) -> StoryCollection:
        """Parse the LLM response into a :class:`~src.schemas.story.StoryCollection`.

        Extracts text from the response, strips any markdown code fences the
        model may have inserted despite instructions, then parses the JSON
        into :class:`~src.schemas.story.StoryCollection` via
        :meth:`~src.agents.base_agent.BaseAgent._parse_model`.

        Args:
            response: The :class:`GenerationResponse` from the LLM.
            input: The original normalized requirement; unused here, retained
                for signature consistency and future correlation if needed.

        Returns:
            A validated :class:`~src.schemas.story.StoryCollection`.

        Raises:
            AgentError: If the response is empty, not valid JSON, or does not
                conform to the :class:`~src.schemas.story.StoryCollection` schema.
        """
        text = self._extract_text(response)
        cleaned = self._strip_json_fences(text)
        return self._parse_model(cleaned, StoryCollection)

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
