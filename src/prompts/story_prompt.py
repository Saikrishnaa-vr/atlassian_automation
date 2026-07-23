"""Prompt templates for the story generation agent.

This module is the single source of truth for all prompt text used by
:class:`~src.agents.story_agent.StoryAgent`.  It contains only prompt-related
constants and a thin formatting helper — no SDK imports, no service calls, no
business logic.

Contents
--------
:data:`SYSTEM_PROMPT`
    The system-turn instruction that governs LLM behavior for story generation:
    role definition, INVEST-aligned story rules, acceptance criteria rules,
    output format contract, and field constraints for the
    :class:`~src.schemas.story.StoryCollection` schema.

:data:`USER_TEMPLATE`
    A ``str.format``-style template for the user-turn message.  The single
    placeholder ``{requirement_json}`` is filled by :func:`build_user_prompt`.

:func:`build_user_prompt`
    Formats :data:`USER_TEMPLATE` with the caller-supplied normalized
    requirement JSON string.  Deterministic and side-effect free.

Separation of concerns
-----------------------
Keeping prompt text here means prompt evolution (wording, schema changes,
additional constraints) never touches agent implementation code.  When the
wording needs to change, edit the constants below;
:class:`~src.agents.story_agent.StoryAgent` will pick up the change
automatically without modification.

Schema alignment
----------------
The output contract in this module must stay in sync with the
:class:`~src.schemas.story.StoryCollection` schema defined in
``src/schemas/story.py``.  Key invariants enforced by that schema:

* ``StoryCollection.count`` is a ``@computed_field`` — it is derived at
  serialization time and must **never** appear in JSON passed to the model.
* ``StoryCollection.stories`` is a tuple; JSON arrays are coerced correctly.
* ``UserStory.acceptance_criteria`` items are objects ``{"text": "..."}``
  — **not** plain strings.
* ``UserStory.status`` must be ``"draft"`` for all freshly generated stories.
* ``UserStory.priority`` must be one of: ``lowest``, ``low``, ``medium``,
  ``high``, ``highest``.

Public API
----------
.. code-block:: python

    from src.prompts.story_prompt import (
        SYSTEM_PROMPT,
        USER_TEMPLATE,
        build_user_prompt,
    )
"""

from __future__ import annotations

__all__ = ["SYSTEM_PROMPT", "USER_TEMPLATE", "build_user_prompt"]


# --------------------------------------------------------------------------- #
# System prompt
#
# Establishes the LLM's role as an Agile Business Analyst and specifies the
# full output contract for StoryCollection generation.
#
# Structure:
#   1. Role statement      — what the model is and what its task is.
#   2. Story generation rules — INVEST, one feature per story, no impl detail.
#   3. Acceptance criteria rules — testable, concise, business-language.
#   4. Output format       — strict JSON-only, no markdown, no prose.
#   5. Schema example      — exact JSON structure the model must produce.
#   6. Field constraints   — type, cardinality, and value rules per field.
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT: str = """
You are an experienced Agile Business Analyst specializing in converting structured
software requirements into high-quality Agile user stories suitable for Jira.

Story generation rules:
1. Each story must represent exactly one independently deliverable feature or capability.
2. Follow INVEST principles where practical: Independent, Negotiable, Valuable, Estimable, Small, Testable.
3. Do not combine unrelated functionality into a single story.
4. Preserve the business intent of the requirement. Do not add or invent functionality.
5. Avoid duplicate stories. Each story must cover a distinct capability.
6. Avoid implementation details (e.g. database schema, class names, algorithms).
7. Write stories from the perspective of the user or stakeholder, not the developer.
8. Assign a meaningful priority based on the business value described in the requirement.
9. Assign status "draft" to all generated stories — they have not been reviewed yet.

Acceptance criteria rules:
1. Each criterion must be a single, testable statement with a clear pass/fail outcome.
2. Write in plain business language. Avoid technical implementation language.
3. Criteria must reflect the original requirement — do not extend scope.
4. Every story must have at least one acceptance criterion.

Output format:
Respond with ONLY a valid JSON object. Do not include markdown, code blocks, or any other text.
Do not explain, summarize, or add prose before or after the JSON.

The JSON object must conform exactly to this structure:

{
  "stories": [
    {
      "title": "<short story title, max 255 characters>",
      "description": "<full narrative description of the story>",
      "persona": "<the user role — fills 'As a <persona>'...>",
      "goal": "<the capability — fills '...I want <goal>'...>",
      "benefit": "<the value — fills '...so that <benefit>'>",
      "acceptance_criteria": [
        {"text": "<a single, verifiable acceptance statement>"}
      ],
      "priority": "<lowest | low | medium | high | highest>",
      "labels": [],
      "dependencies": [],
      "status": "draft"
    }
  ],
  "requirement_id": null
}

Field constraints:
- stories: an array of story objects; include at least one story when the requirement is actionable.
- title: non-empty string, max 255 characters.
- description: non-empty string; the full user story narrative.
- persona, goal, benefit: non-empty strings; together they form "As a <persona>, I want <goal>, so that <benefit>".
- acceptance_criteria: array of objects {"text": "<criterion>"}; at least one object required per story.
  Each "text" value must be a non-empty, unique string within the story.
- priority: one of exactly these values — "lowest", "low", "medium", "high", "highest".
- labels: array of unique, non-blank strings; use an empty array when no labels apply.
- dependencies: array of story identifier strings; use an empty array when there are no dependencies.
- status: always "draft" for newly generated stories.
- requirement_id: null unless an explicit requirement identifier was provided.

Do NOT include these fields — they are not part of the input contract:
- count (computed automatically from the stories array length)
- id on individual stories (assigned later; omit the field entirely)
- estimate (assigned during planning; omit the field entirely)
""".strip()


# --------------------------------------------------------------------------- #
# User message template
#
# The single placeholder {requirement_json} is substituted by
# build_user_prompt().  The XML-style <normalized_requirement> tags delineate
# the structured input from the instruction prose, which helps the model
# clearly distinguish the content it is asked to process.
# --------------------------------------------------------------------------- #

USER_TEMPLATE: str = """\
The following normalized requirement has already been analyzed and structured.
Convert it into a StoryCollection by generating Agile user stories.

<normalized_requirement>
{requirement_json}
</normalized_requirement>
"""


# --------------------------------------------------------------------------- #
# Formatting helper
# --------------------------------------------------------------------------- #


def build_user_prompt(requirement_json: str) -> str:
    """Format the user-turn message from a normalized requirement JSON string.

    Substitutes ``{requirement_json}`` in :data:`USER_TEMPLATE` with the
    supplied text.  The caller is responsible for any pre-processing (e.g.
    serializing the requirement model to JSON) before passing the value in.

    This function is deterministic and has no side effects.

    Args:
        requirement_json: The serialized normalized requirement JSON string to
            embed in the prompt.

    Returns:
        The formatted user message string ready to pass to the LLM.
    """
    return USER_TEMPLATE.format(requirement_json=requirement_json)
