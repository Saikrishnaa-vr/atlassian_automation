"""Shared base agent — common execution infrastructure for all reasoning agents.

Every concrete agent in this project inherits from :class:`BaseAgent`. The
base class is responsible for all behaviour that is not domain-specific:

* Acquiring the :class:`~src.services.anthropic_service.AnthropicService`
  through the service factory so callers never wire dependencies manually.
* Orchestrating the fixed execution sequence: validate input → build prompt →
  invoke LLM → parse response → validate output.
* Emitting structured, secret-free Loguru log entries with agent name, timing,
  and failure context at each step.
* Translating every :class:`~src.services.anthropic_service.AnthropicServiceError`
  and unexpected exception into :class:`~src.exceptions.AgentError` so nothing
  below the agent layer ever propagates to callers.

Concrete agents implement only two abstract methods:

* :meth:`BaseAgent._build_prompt` — translate typed input into a
  :class:`~src.services.anthropic_service.GenerationRequest`.
* :meth:`BaseAgent._parse_response` — translate a
  :class:`~src.services.anthropic_service.GenerationResponse` into the typed
  output model.

Everything else — timing, logging, exception mapping, validation hooks, and
convenience helpers — is inherited automatically.

Execution sequence::

    execute(input)
        ↓  _validate_input(input)            # hook — no-op base, overrideable
        ↓  _build_prompt(input)              # abstract — subclass must implement
        ↓  _invoke(request)                  # calls service, maps all errors
        ↓  _parse_response(response, input)  # abstract — subclass must implement
        ↓  _validate_output(output)          # hook — no-op base, overrideable
        ↓  return output

Dependency contract
-------------------
Allowed imports: ``src.services.factory``, ``src.services.anthropic_service``,
``src.exceptions``, standard library, ``pydantic``.

Prohibited imports: ``src.graph``, ``src.services.jira_service``, ``fastapi``,
``langgraph``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from loguru import logger
from pydantic import BaseModel as _BaseModel
from pydantic import ValidationError as _PydanticValidationError

from src.exceptions import AgentError, ApplicationError, ValidationError
from src.services.anthropic_service import (
    AnthropicService,
    AnthropicServiceError as _AnthropicServiceError,
    GenerationRequest,
    GenerationResponse,
    Message,
)
from src.services.factory import get_anthropic_service


# --------------------------------------------------------------------------- #
# Type variables
# --------------------------------------------------------------------------- #

InputT = TypeVar("InputT")
"""Generic input type bound to a concrete agent's :meth:`~BaseAgent.execute` call."""

OutputT = TypeVar("OutputT")
"""Generic output type returned by a concrete agent's :meth:`~BaseAgent.execute` call."""

_ModelT = TypeVar("_ModelT", bound=_BaseModel)
"""Module-private bound type variable used by :meth:`BaseAgent._parse_model`."""


# --------------------------------------------------------------------------- #
# Module-private constants
# --------------------------------------------------------------------------- #

_DEFAULT_MAX_TOKENS: int = 1024
"""Default token budget passed to every :class:`GenerationRequest` built by an agent."""


# --------------------------------------------------------------------------- #
# Base agent
# --------------------------------------------------------------------------- #


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base class providing the common agent execution infrastructure.

    All reasoning agents in this project inherit from :class:`BaseAgent`.
    Concrete agents supply the two domain-specific abstract methods; every
    other concern — service wiring, timing, logging, error translation, and
    validation hooks — is handled here and inherited automatically.

    Type parameters
    ~~~~~~~~~~~~~~~
    Declare concrete types when subclassing::

        class RequirementAgent(BaseAgent[str, NormalizedRequirement]):
            ...

    ``InputT`` and ``OutputT`` propagate into :meth:`execute`,
    :meth:`_validate_input`, :meth:`_validate_output`,
    :meth:`_build_prompt`, and :meth:`_parse_response` so the full execution
    chain is statically typed end-to-end.

    Args:
        service: A pre-built :class:`AnthropicService`. When omitted the
            cached factory singleton is used, which is the correct choice for
            production. Pass a custom instance in tests.
        max_tokens: Default token budget applied to every
            :class:`GenerationRequest` produced by :meth:`_make_request`.
            Individual calls can override this per-request.
        temperature: Default sampling temperature. ``None`` omits the
            parameter from the request and lets the provider apply its own
            default — the correct behaviour for most agents.
    """

    def __init__(
        self,
        *,
        service: AnthropicService | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> None:
        self._service: AnthropicService = service or get_anthropic_service()
        self._max_tokens: int = max_tokens
        self._temperature: float | None = temperature
        # Bind the agent name once; all operation logs inherit this context.
        self._log = logger.bind(agent=self._agent_name)

    # -- Identity ----------------------------------------------------------- #

    @property
    def _agent_name(self) -> str:
        """The canonical display name for this agent, derived from the class name.

        Used in every log entry. Automatically reflects the concrete subclass
        name (e.g. ``"RequirementAgent"``, ``"StoryAgent"``).
        """
        return type(self).__name__

    # -- Public execution interface ----------------------------------------- #

    def execute(self, input: InputT) -> OutputT:  # noqa: A002
        """Run the full agent execution sequence for the given input.

        This is the only public method callers should invoke. The execution
        sequence is fixed and cannot be overridden by subclasses — concrete
        agents customise behaviour through the two abstract methods and the
        two validation hooks.

        The method logs a start entry at the beginning, a completion entry
        with elapsed time on success, and a failure entry with elapsed time
        on any error path.

        Args:
            input: The typed input for this agent. The concrete type is
                determined by the ``InputT`` type parameter of the subclass.

        Returns:
            The typed output produced by :meth:`_parse_response`. The concrete
            type is determined by the ``OutputT`` type parameter.

        Raises:
            ValidationError: If :meth:`_validate_input` or
                :meth:`_validate_output` rejects the input or output.
            AgentError: If the LLM invocation fails, the response cannot be
                parsed, or an unexpected error occurs during execution.
        """
        start = time.perf_counter()
        log = self._log.bind(operation="execute")
        log.info(f"{self._agent_name} execution started")

        try:
            self._validate_input(input)
            request = self._build_prompt(input)
            response = self._invoke(request)
            output = self._parse_response(response, input)
            self._validate_output(output)

        except ApplicationError:
            elapsed = time.perf_counter() - start
            log.bind(elapsed_seconds=round(elapsed, 3)).error(
                f"{self._agent_name} execution failed"
            )
            raise

        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.bind(
                elapsed_seconds=round(elapsed, 3),
                error_type=type(exc).__name__,
            ).error(f"{self._agent_name} execution failed with unexpected error")
            raise AgentError(
                f"{self._agent_name} raised an unexpected error: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start
        log.bind(elapsed_seconds=round(elapsed, 3)).info(
            f"{self._agent_name} execution completed"
        )
        return output

    # -- Abstract interface — subclasses must implement --------------------- #

    @abstractmethod
    def _build_prompt(self, input: InputT) -> GenerationRequest:  # noqa: A002
        """Translate the validated input into a :class:`GenerationRequest`.

        Use :meth:`_make_request` and :meth:`_make_user_message` to construct
        the request rather than building it manually. All system prompt content
        and message structuring belongs here.

        This method must not call any service, perform any I/O, or contain
        business logic beyond prompt assembly.

        Args:
            input: The validated agent input.

        Returns:
            A fully constructed :class:`GenerationRequest` ready for the LLM.
        """

    @abstractmethod
    def _parse_response(self, response: GenerationResponse, input: InputT) -> OutputT:  # noqa: A002
        """Translate the LLM response into the typed output model.

        Implementations receive both the raw :class:`GenerationResponse` and
        the original input so they can correlate context if needed (e.g.
        attaching a requirement id to the generated story collection). Use
        :meth:`_extract_text` and :meth:`_parse_model` to reduce boilerplate.

        This method must not call any service, perform any I/O, or contain
        business logic beyond parsing and mapping the response.

        Args:
            response: The :class:`GenerationResponse` from the LLM invocation.
            input: The original validated input, provided for correlation.

        Returns:
            The typed output produced by this agent.

        Raises:
            AgentError: If the response cannot be parsed into the expected type.
        """

    # -- Validation hooks — no-op base, overrideable by subclasses ---------- #

    def _validate_input(self, input: InputT) -> None:  # noqa: A002
        """Validate the agent input before prompt construction.

        The base implementation rejects ``None`` and is otherwise a no-op.
        Subclasses override to add domain-specific checks, e.g. ensuring a
        requirement string has a minimum length or that a required field is
        populated.

        Structural validation (type, field presence) is Pydantic's
        responsibility and happens at model construction time. This hook is
        intended for *semantic* validation performed by application code.

        Args:
            input: The raw agent input.

        Raises:
            ValidationError: If the input fails validation.
        """
        if input is None:
            raise ValidationError(f"{self._agent_name} received None as input")

    def _validate_output(self, output: OutputT) -> None:
        """Validate the agent output after response parsing.

        The base implementation rejects ``None`` and is otherwise a no-op.
        Subclasses override to add domain-specific checks, e.g. ensuring a
        story collection is non-empty or that required fields were populated
        by the LLM.

        Args:
            output: The parsed agent output.

        Raises:
            ValidationError: If the output fails validation.
        """
        if output is None:
            raise ValidationError(f"{self._agent_name} produced None as output")

    # -- Service invocation ------------------------------------------------- #

    def _invoke(self, request: GenerationRequest) -> GenerationResponse:
        """Send the request to the Anthropic service and return the response.

        This method is the exception-translation boundary between the service
        layer and the agent layer. Every :class:`AnthropicServiceError` is
        caught here and re-raised as :class:`AgentError` so no service-layer
        exception type ever propagates above this class.

        Args:
            request: A validated :class:`GenerationRequest`.

        Returns:
            The :class:`GenerationResponse` from the LLM.

        Raises:
            AgentError: On any Anthropic service failure. The original service
                exception is preserved as ``__cause__`` for full tracebacks.
        """
        log = self._log.bind(
            operation="invoke",
            max_tokens=request.max_tokens,
            message_count=len(request.messages),
        )
        log.debug(f"{self._agent_name} invoking LLM")
        try:
            return self._service.generate(request)
        except _AnthropicServiceError as exc:
            raise AgentError(
                f"{self._agent_name} LLM invocation failed: {exc}"
            ) from exc

    # -- Convenience helpers ------------------------------------------------ #

    def _make_request(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationRequest:
        """Build a :class:`GenerationRequest` using the agent's configured defaults.

        The ``max_tokens`` and ``temperature`` parameters fall back to the
        values set in the agent constructor when omitted. Pass them explicitly
        to override for a specific call.

        Args:
            messages: The ordered list of conversational turns.
            system: Optional system prompt. When omitted, no system turn is
                included in the request.
            max_tokens: Per-call token budget. Overrides :attr:`_max_tokens`
                when provided.
            temperature: Per-call sampling temperature. Overrides
                :attr:`_temperature` when provided.

        Returns:
            A validated :class:`GenerationRequest`.
        """
        return GenerationRequest(
            messages=messages,
            system=system,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
        )

    @staticmethod
    def _make_user_message(content: str) -> Message:
        """Create a single user-role :class:`Message`.

        Args:
            content: The message body. Must be non-empty (enforced by the
                :class:`Message` model).

        Returns:
            A :class:`Message` with ``role="user"``.
        """
        return Message(role="user", content=content)

    @staticmethod
    def _extract_text(response: GenerationResponse) -> str:
        """Extract and return the text content from a generation response.

        Strips leading and trailing whitespace from the concatenated text
        blocks returned by the LLM.

        Args:
            response: The :class:`GenerationResponse` from the LLM.

        Returns:
            The stripped response text.

        Raises:
            AgentError: If the response contains no usable text content.
        """
        text = response.text.strip()
        if not text:
            raise AgentError("Generation response contained no usable text content")
        return text

    def _parse_model(self, text: str, model_class: type[_ModelT]) -> _ModelT:
        """Parse a JSON string into a Pydantic model.

        A convenience method for agents that prompt the LLM to return
        structured JSON output. Catches both JSON parse errors and Pydantic
        validation errors and re-raises them as :class:`AgentError` so callers
        only see a single, typed failure mode.

        Args:
            text: A JSON string, typically produced by :meth:`_extract_text`.
            model_class: The Pydantic model class to validate the parsed data
                against.

        Returns:
            A validated instance of ``model_class``.

        Raises:
            AgentError: If ``text`` is not valid JSON or if the parsed data
                fails ``model_class`` validation.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentError(
                f"{self._agent_name} response is not valid JSON: {exc}"
            ) from exc

        try:
            return model_class.model_validate(data)
        except _PydanticValidationError as exc:
            raise AgentError(
                f"{self._agent_name} response failed {model_class.__name__} validation: {exc}"
            ) from exc
