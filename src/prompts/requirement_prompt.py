"""Prompt templates for the requirement normalization agent.

This module is the single source of truth for all prompt text used by
:class:`~src.agents.requirement_agent.RequirementAgent`.  It contains only
prompt-related constants and a thin formatting helper — no SDK imports, no
service calls, no business logic.

Contents
--------
:data:`SYSTEM_PROMPT`
    The system-turn instruction that governs LLM behavior for requirement
    analysis: role definition, normalization rules, output format contract,
    and field constraints.

:data:`USER_TEMPLATE`
    A ``str.format``-style template for the user-turn message.  The single
    placeholder ``{requirement}`` is filled by :func:`build_user_prompt`.

:func:`build_user_prompt`
    Formats :data:`USER_TEMPLATE` with the caller-supplied requirement text.
    Deterministic and side-effect free.

Separation of concerns
-----------------------
Keeping prompt text here means prompt evolution (wording, examples, schema
changes) never touches agent implementation code.  When the wording needs to
change, edit the constants below; :class:`~src.agents.requirement_agent.RequirementAgent`
will pick up the change automatically without modification.

Public API
----------
.. code-block:: python

    from src.prompts.requirement_prompt import (
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
# Establishes the LLM's role as a business analyst, defines the normalization
# rules it must follow, and specifies the exact JSON output contract.
#
# Structure:
#   1. Role statement      — what the model is and what its task is.
#   2. Behavior rules      — how to analyze and normalize the requirement.
#   3. Output format       — strict JSON-only instruction with schema example.
#   4. Field constraints   — cardinality and non-blank rules per field.
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT: str = """
You are an expert business analyst specializing in software requirements analysis.

Your task is to analyze the raw requirement text provided and return a single structured
JSON object that normalizes, organizes, and enriches the requirement.

Behavior rules:
1. Normalize ambiguous, informal, or redundant wording into clear, precise language.
2. Remove duplicate or overlapping requirements.
3. Organize requirements logically within each category.
4. Preserve all business intent from the original text. Do not alter or omit any stated requirement.
5. Identify any missing or ambiguous information and list it in missing_information.
6. Never invent business requirements. Only extract what is explicitly stated or strongly implied.
7. Avoid unsupported assumptions. If you must assume something, document it in assumptions.

Output format:
Respond with ONLY a valid JSON object. Do not include markdown, code blocks, or any other text.

The JSON object must have exactly these fields:

{
  "title": "<short title capturing the core requirement, max 255 characters>",
  "summary": "<2-3 sentence summary of the requirement>",
  "business_objective": "<the primary business goal this requirement addresses>",
  "primary_actors": ["<user or system role>"],
  "functional_requirements": ["<something the system must do>"],
  "non_functional_requirements": ["<quality attribute such as performance or security>"],
  "assumptions": ["<assumption made during analysis>"],
  "constraints": ["<known limitation or boundary>"],
  "missing_information": ["<gap or ambiguity that needs stakeholder clarification>"]
}

Field constraints:
- title, summary, and business_objective must be non-empty strings.
- primary_actors must contain at least one entry.
- functional_requirements must contain at least one entry.
- non_functional_requirements, assumptions, constraints, and missing_information may be empty lists.
- All list entries must be non-empty strings and unique within their respective list.
""".strip()


# --------------------------------------------------------------------------- #
# User message template
#
# The single placeholder {requirement} is substituted by build_user_prompt().
# The XML-style <requirement> tags delineate the user-supplied text from the
# instruction prose, which helps the model distinguish its instructions from
# the content it is asked to analyze.
# --------------------------------------------------------------------------- #

USER_TEMPLATE: str = """\
Analyze the following raw requirement and return the structured JSON:

<requirement>
{requirement}
</requirement>
"""


# --------------------------------------------------------------------------- #
# Formatting helper
# --------------------------------------------------------------------------- #


def build_user_prompt(requirement: str) -> str:
    """Format the user-turn message from a raw requirement string.

    Substitutes ``{requirement}`` in :data:`USER_TEMPLATE` with the supplied
    text.  The caller is responsible for any pre-processing (e.g. stripping
    whitespace) before passing the value in.

    This function is deterministic and has no side effects.

    Args:
        requirement: The pre-validated requirement text to embed in the prompt.

    Returns:
        The formatted user message string ready to pass to the LLM.
    """
    return USER_TEMPLATE.format(requirement=requirement)
