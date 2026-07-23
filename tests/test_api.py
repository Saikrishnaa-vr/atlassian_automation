from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.api.routes as _routes
from src.exceptions import ApplicationError
from src.exceptions import ValidationError as AppValidationError
from src.graph.state import ApprovalStatus, WorkflowState, WorkflowStep
from src.main import app


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def _clear_registry():
    _routes._registry.clear()
    _routes._project_keys.clear()
    _routes._get_workflow.cache_clear()
    yield
    _routes._registry.clear()
    _routes._project_keys.clear()
    _routes._get_workflow.cache_clear()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def pending_state() -> WorkflowState:
    return WorkflowState(
        raw_requirement="Build a login system.",
        current_step=WorkflowStep.AWAITING_APPROVAL,
    )


# ------------------------------------------------------------------ #
# GET /
# ------------------------------------------------------------------ #

def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_root_returns_service_identity(client):
    response = client.get("/")
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "atlassian-automation"
    assert data["version"] == "0.1.0"


# ------------------------------------------------------------------ #
# GET /health
# ------------------------------------------------------------------ #

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_healthy(client):
    response = client.get("/health")
    assert response.json() == {"status": "healthy"}


# ------------------------------------------------------------------ #
# POST /requirements
# ------------------------------------------------------------------ #

def test_post_requirements_returns_201(client, pending_state):
    with patch("src.api.routes.run_workflow", return_value=pending_state):
        response = client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    assert response.status_code == 201


def test_post_requirements_returns_workflow_id(client, pending_state):
    with patch("src.api.routes.run_workflow", return_value=pending_state):
        response = client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    data = response.json()
    assert "workflow_id" in data
    assert data["workflow_id"] == pending_state.workflow_id


def test_post_requirements_stores_state_in_registry(client, pending_state):
    with patch("src.api.routes.run_workflow", return_value=pending_state):
        client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    assert pending_state.workflow_id in _routes._registry


def test_post_requirements_stores_project_key(client, pending_state):
    with patch("src.api.routes.run_workflow", return_value=pending_state):
        client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    assert _routes._project_keys.get(pending_state.workflow_id) == "PROJ"


def test_post_requirements_empty_requirement_returns_422(client):
    response = client.post(
        "/requirements",
        json={"requirement": "", "project_key": "PROJ"},
    )
    assert response.status_code == 422


def test_post_requirements_missing_requirement_returns_422(client):
    response = client.post(
        "/requirements",
        json={"project_key": "PROJ"},
    )
    assert response.status_code == 422


def test_post_requirements_missing_project_key_returns_422(client):
    response = client.post(
        "/requirements",
        json={"requirement": "Build something."},
    )
    assert response.status_code == 422


def test_post_requirements_application_error_returns_500(client):
    with patch("src.api.routes.run_workflow", side_effect=ApplicationError("workflow failed")):
        response = client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    assert response.status_code == 500


def test_post_requirements_validation_error_returns_400(client):
    with patch(
        "src.api.routes.run_workflow",
        side_effect=AppValidationError("invalid requirement"),
    ):
        response = client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    assert response.status_code == 400


def test_post_requirements_calls_run_workflow_once(client, pending_state):
    with patch("src.api.routes.run_workflow", return_value=pending_state) as mock_run:
        client.post(
            "/requirements",
            json={"requirement": "Build a login system.", "project_key": "PROJ"},
        )
    assert mock_run.call_count == 1


# ------------------------------------------------------------------ #
# GET /requirements/{workflow_id}
# ------------------------------------------------------------------ #

def test_get_requirement_returns_200(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    response = client.get(f"/requirements/{pending_state.workflow_id}")
    assert response.status_code == 200


def test_get_requirement_returns_correct_state(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    response = client.get(f"/requirements/{pending_state.workflow_id}")
    data = response.json()
    assert data["workflow_id"] == pending_state.workflow_id
    assert data["raw_requirement"] == "Build a login system."


def test_get_requirement_returns_approval_status(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    response = client.get(f"/requirements/{pending_state.workflow_id}")
    data = response.json()
    assert data["approval_status"] == "pending"


def test_get_requirement_unknown_id_returns_404(client):
    response = client.get("/requirements/nonexistent-id")
    assert response.status_code == 404


# ------------------------------------------------------------------ #
# POST /requirements/{workflow_id}/approval
# ------------------------------------------------------------------ #

def test_post_approval_approved_returns_200(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    approved_state = pending_state.model_copy(
        update={
            "approval_status": ApprovalStatus.APPROVED,
            "current_step": WorkflowStep.COMPLETED,
        }
    )
    with patch("src.api.routes.run_workflow", return_value=approved_state):
        response = client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "approved"},
        )
    assert response.status_code == 200


def test_post_approval_rejected_returns_200(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    rejected_state = pending_state.model_copy(
        update={
            "approval_status": ApprovalStatus.REJECTED,
            "current_step": WorkflowStep.FAILED,
        }
    )
    with patch("src.api.routes.run_workflow", return_value=rejected_state):
        response = client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "rejected"},
        )
    assert response.status_code == 200


def test_post_approval_pending_returns_200(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    with patch("src.api.routes.run_workflow", return_value=pending_state):
        response = client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "pending"},
        )
    assert response.status_code == 200


def test_post_approval_unknown_workflow_returns_404(client):
    response = client.post(
        "/requirements/nonexistent-id/approval",
        json={"approval": "approved"},
    )
    assert response.status_code == 404


def test_post_approval_updates_registry(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    approved_state = pending_state.model_copy(
        update={
            "approval_status": ApprovalStatus.APPROVED,
            "current_step": WorkflowStep.COMPLETED,
        }
    )
    with patch("src.api.routes.run_workflow", return_value=approved_state):
        client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "approved"},
        )
    stored = _routes._registry[pending_state.workflow_id]
    assert stored.approval_status == ApprovalStatus.APPROVED


def test_post_approval_response_contains_workflow_id(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    final_state = pending_state.model_copy(
        update={"current_step": WorkflowStep.COMPLETED}
    )
    with patch("src.api.routes.run_workflow", return_value=final_state):
        response = client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "approved"},
        )
    data = response.json()
    assert data["workflow_id"] == pending_state.workflow_id


def test_post_approval_invalid_value_returns_422(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    response = client.post(
        f"/requirements/{pending_state.workflow_id}/approval",
        json={"approval": "yes_please"},
    )
    assert response.status_code == 422


def test_post_approval_boolean_true_returns_422(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    response = client.post(
        f"/requirements/{pending_state.workflow_id}/approval",
        json={"approval": True},
    )
    assert response.status_code == 422


def test_post_approval_application_error_returns_500(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    with patch(
        "src.api.routes.run_workflow",
        side_effect=ApplicationError("workflow failed"),
    ):
        response = client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "approved"},
        )
    assert response.status_code == 500


def test_post_approval_validation_error_returns_400(client, pending_state):
    _routes._registry[pending_state.workflow_id] = pending_state
    _routes._project_keys[pending_state.workflow_id] = "PROJ"
    with patch(
        "src.api.routes.run_workflow",
        side_effect=AppValidationError("invalid"),
    ):
        response = client.post(
            f"/requirements/{pending_state.workflow_id}/approval",
            json={"approval": "approved"},
        )
    assert response.status_code == 400
