from __future__ import annotations

import pytest

from src.exceptions import (
    AgentError,
    ApplicationError,
    ApprovalError,
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    JiraServiceError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ServiceError,
    ValidationError,
    WorkflowError,
)
from src.main import _http_status_for


# ------------------------------------------------------------------ #
# Exception hierarchy
# ------------------------------------------------------------------ #

def test_configuration_error_is_application_error():
    assert issubclass(ConfigurationError, ApplicationError)


def test_validation_error_is_application_error():
    assert issubclass(ValidationError, ApplicationError)


def test_service_error_is_application_error():
    assert issubclass(ServiceError, ApplicationError)


def test_agent_error_is_application_error():
    assert issubclass(AgentError, ApplicationError)


def test_workflow_error_is_application_error():
    assert issubclass(WorkflowError, ApplicationError)


def test_approval_error_is_application_error():
    assert issubclass(ApprovalError, ApplicationError)


def test_authentication_error_is_service_error():
    assert issubclass(AuthenticationError, ServiceError)


def test_rate_limit_error_is_service_error():
    assert issubclass(RateLimitError, ServiceError)


def test_not_found_error_is_service_error():
    assert issubclass(NotFoundError, ServiceError)


def test_invalid_request_error_is_service_error():
    assert issubclass(InvalidRequestError, ServiceError)


def test_server_error_is_service_error():
    assert issubclass(ServerError, ServiceError)


def test_jira_service_error_is_service_error():
    assert issubclass(JiraServiceError, ServiceError)


# ------------------------------------------------------------------ #
# ApplicationError message contract
# ------------------------------------------------------------------ #

def test_application_error_default_message():
    exc = ApplicationError()
    assert exc.message == ApplicationError.default_message
    assert exc.message


def test_application_error_custom_message():
    exc = ApplicationError("custom message")
    assert exc.message == "custom message"


def test_application_error_str_returns_message():
    exc = ApplicationError("the message")
    assert str(exc) == "the message"


def test_application_error_default_str():
    exc = ApplicationError()
    assert str(exc) == ApplicationError.default_message


# ------------------------------------------------------------------ #
# cause parameter
# ------------------------------------------------------------------ #

def test_cause_is_attached_as_dunder_cause():
    cause = ValueError("root cause")
    exc = AgentError("wrapped", cause=cause)
    assert exc.__cause__ is cause


def test_cause_is_none_by_default():
    exc = AgentError("no cause")
    assert exc.__cause__ is None


def test_cause_does_not_change_message():
    cause = ValueError("root cause")
    exc = AgentError("my message", cause=cause)
    assert exc.message == "my message"


# ------------------------------------------------------------------ #
# Subtype default messages
# ------------------------------------------------------------------ #

def test_validation_error_has_default_message():
    exc = ValidationError()
    assert exc.message == ValidationError.default_message
    assert exc.message


def test_agent_error_has_default_message():
    exc = AgentError()
    assert exc.message == AgentError.default_message
    assert exc.message


def test_workflow_error_has_default_message():
    exc = WorkflowError()
    assert exc.message == WorkflowError.default_message
    assert exc.message


def test_approval_error_has_default_message():
    exc = ApprovalError()
    assert exc.message == ApprovalError.default_message
    assert exc.message


def test_service_error_has_default_message():
    exc = ServiceError()
    assert exc.message == ServiceError.default_message
    assert exc.message


def test_authentication_error_has_default_message():
    exc = AuthenticationError()
    assert exc.message == AuthenticationError.default_message
    assert exc.message


def test_not_found_error_has_default_message():
    exc = NotFoundError()
    assert exc.message == NotFoundError.default_message
    assert exc.message


# ------------------------------------------------------------------ #
# _http_status_for — HTTP status mapping
# ------------------------------------------------------------------ #

def test_http_status_for_validation_error_returns_400():
    assert _http_status_for(ValidationError("invalid")) == 400


def test_http_status_for_invalid_request_error_returns_400():
    assert _http_status_for(InvalidRequestError("bad request")) == 400


def test_http_status_for_authentication_error_returns_401():
    assert _http_status_for(AuthenticationError("unauthorized")) == 401


def test_http_status_for_not_found_error_returns_404():
    assert _http_status_for(NotFoundError("missing")) == 404


def test_http_status_for_rate_limit_error_returns_429():
    assert _http_status_for(RateLimitError("throttled")) == 429


def test_http_status_for_service_error_returns_500():
    assert _http_status_for(ServiceError("generic service")) == 500


def test_http_status_for_agent_error_returns_500():
    assert _http_status_for(AgentError("agent fail")) == 500


def test_http_status_for_workflow_error_returns_500():
    assert _http_status_for(WorkflowError("workflow fail")) == 500


def test_http_status_for_approval_error_returns_500():
    assert _http_status_for(ApprovalError("approval fail")) == 500


def test_http_status_for_server_error_returns_500():
    assert _http_status_for(ServerError("server error")) == 500


def test_http_status_for_configuration_error_returns_500():
    assert _http_status_for(ConfigurationError("bad config")) == 500


def test_http_status_for_application_error_returns_500():
    assert _http_status_for(ApplicationError("base error")) == 500
