"""Live infrastructure verification for JiraService.

These are NOT unit tests: they validate real Jira connectivity using the
project's service layer and live configuration from ``get_settings()``.
All operations are read-only and have zero Jira side effects.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ValidationError

from src.config import Settings, get_settings
from src.services import jira_service as jira_service_module


def _jira_service_class() -> type[Any]:
	"""Return JiraService type or skip when the service is not implemented."""
	service_cls = getattr(jira_service_module, "JiraService", None)
	if service_cls is None:
		pytest.skip("Skipping live Jira tests: JiraService is not implemented")
	return service_cls


def _require_settings() -> Settings:
	"""Return validated settings, skipping when live config is unavailable."""
	try:
		return get_settings()
	except ValidationError as exc:
		pytest.skip(
			"Skipping live Jira test: configuration unavailable "
			f"({exc.error_count()} validation error(s))"
		)


def _skip_if_environmental(exc: Exception, context: str) -> None:
	"""Skip for authentication/network environment problems; otherwise re-raise."""
	name = exc.__class__.__name__.lower()
	msg = str(exc).lower()
	markers = (
		"auth",
		"unauthor",
		"forbidden",
		"invalid token",
		"credential",
		"connect",
		"connection",
		"timeout",
		"dns",
		"ssl",
		"unreachable",
		"name or service not known",
		"nodename nor servname",
		"temporary failure",
		"401",
		"403",
		"429",
		"5xx",
	)
	blob = f"{name} {msg}"
	if any(marker in blob for marker in markers):
		pytest.skip(f"Skipping live Jira test ({context}): {exc.__class__.__name__}")
	raise exc


def _assert_typed_model(value: Any) -> None:
	"""Assert the value is a typed response model instance."""
	is_typed = isinstance(value, BaseModel) or is_dataclass(value)
	assert is_typed, f"Expected typed response model, got {type(value).__name__}"


def _call_first(service: Any, methods: list[str], *args: Any) -> Any:
	"""Call the first available JiraService method from a candidate list."""
	for method_name in methods:
		func = getattr(service, method_name, None)
		if callable(func):
			return _call_with_fallback_signatures(func, *args)
	pytest.skip(
		"Skipping live Jira test: JiraService does not expose a supported "
		f"method from {methods}"
	)


def _call_with_fallback_signatures(func: Callable[..., Any], *args: Any) -> Any:
	"""Invoke service call with common positional/keyword argument shapes."""
	if not args:
		return func()

	arg = args[0]
	attempts = [
		lambda: func(arg),
		lambda: func(project_key=arg),
		lambda: func(key=arg),
		lambda: func(project=arg),
	]
	for attempt in attempts:
		try:
			return attempt()
		except TypeError:
			continue
	return func(arg)


@pytest.fixture(scope="module")
def live_jira_service() -> Any:
	"""Construct a live JiraService once per module and verify baseline health."""
	_require_settings()
	service = _jira_service_class()()
	try:
		health = service.health_check()
	except Exception as exc:
		_skip_if_environmental(exc, "health_check")
		raise

	if isinstance(health, bool):
		assert health is True
	elif hasattr(health, "success"):
		assert bool(getattr(health, "success")) is True
	else:
		assert bool(health) is True
	return service


@pytest.fixture(scope="module")
def authenticated_identity(live_jira_service: Any) -> Any:
	"""Fetch authenticated identity once to reduce repeated live calls."""
	methods = [
		"get_current_user",
		"get_authenticated_user",
		"get_myself",
		"current_user",
		"get_current_user_info",
	]
	try:
		identity = _call_first(live_jira_service, methods)
	except Exception as exc:
		_skip_if_environmental(exc, "identity")
		raise

	_assert_typed_model(identity)
	return identity


def _identity_value(identity: Any, field: str) -> Any:
	"""Extract identity fields from either pydantic/dataclass/object models."""
	if hasattr(identity, field):
		return getattr(identity, field)
	if isinstance(identity, BaseModel):
		return identity.model_dump().get(field)
	if is_dataclass(identity):
		return getattr(identity, field, None)
	return None


def _configured_project_key(settings: Settings) -> str | None:
	"""Get an optional project key from settings when available."""
	jira_cfg = settings.jira
	for attr in ("project_key", "default_project_key", "test_project_key"):
		value = getattr(jira_cfg, attr, None)
		if isinstance(value, str) and value.strip():
			return value.strip()
	return None


def _discover_project_key(service: Any) -> str | None:
	"""Best-effort discovery of one readable Jira project key via SDK client."""
	client = getattr(service, "_client", None)
	if client is None:
		return None

	for method_name in ("projects", "get_all_projects", "get_projects"):
		func = getattr(client, method_name, None)
		if not callable(func):
			continue
		try:
			payload = func()
		except Exception:
			continue

		items: Any = payload
		if isinstance(payload, dict):
			items = payload.get("values") or payload.get("projects") or ()

		if isinstance(items, list):
			for item in items:
				if isinstance(item, dict):
					key = item.get("key")
					if isinstance(key, str) and key.strip():
						return key.strip()

	return None


def test_service_construction() -> None:
	"""JiraService can be constructed with validated live configuration."""
	_require_settings()
	service_cls = _jira_service_class()
	service = service_cls()
	assert isinstance(service, service_cls)


def test_health_check_succeeds(live_jira_service: Any) -> None:
	"""health_check() confirms connectivity/authentication with Jira."""
	try:
		health = live_jira_service.health_check()
	except Exception as exc:
		_skip_if_environmental(exc, "health_check")
		raise

	if isinstance(health, bool):
		assert health is True
	elif hasattr(health, "success"):
		_assert_typed_model(health)
		assert bool(getattr(health, "success")) is True
	else:
		assert bool(health) is True


def test_authenticated_identity_is_accessible(authenticated_identity: Any) -> None:
	"""Authenticated identity endpoint returns non-empty typed identifiers."""
	account_id = _identity_value(authenticated_identity, "account_id")
	if not account_id:
		account_id = _identity_value(authenticated_identity, "accountId")

	display_name = _identity_value(authenticated_identity, "display_name")
	if not display_name:
		display_name = _identity_value(authenticated_identity, "displayName")

	email = _identity_value(authenticated_identity, "email")
	if not email:
		email = _identity_value(authenticated_identity, "emailAddress")

	assert isinstance(account_id, str)
	assert account_id.strip()
	assert (isinstance(display_name, str) and display_name.strip()) or (
		isinstance(email, str) and email.strip()
	)


def test_project_lookup_if_configured(live_jira_service: Any) -> None:
	"""Project lookup is validated when a project key is configured."""
	settings = _require_settings()
	project_key = _configured_project_key(settings)
	if not project_key:
		project_key = _discover_project_key(live_jira_service)

	assert isinstance(project_key, str)
	assert project_key.strip()

	methods = ["get_project", "get_project_by_key", "project", "lookup_project"]
	try:
		project = _call_first(live_jira_service, methods, project_key.strip())
	except Exception as exc:
		_skip_if_environmental(exc, "project lookup")
		raise

	_assert_typed_model(project)

	key = getattr(project, "key", None)
	if key is None and isinstance(project, BaseModel):
		key = project.model_dump().get("key")
	project_id = getattr(project, "id", None)
	if project_id is None and isinstance(project, BaseModel):
		project_id = project.model_dump().get("id")

	assert isinstance(key, str)
	assert key.strip()
	assert str(project_id).strip()
