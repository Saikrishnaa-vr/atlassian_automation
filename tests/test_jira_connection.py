from __future__ import annotations

import os

import pytest
import requests
from dotenv import load_dotenv


load_dotenv()


def _require_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		pytest.skip(f"Skipping live Jira connection test: missing {name}")
	return value


def test_jira_connection() -> None:
	"""Validate Jira auth by calling the authenticated current-user endpoint."""
	base_url = _require_env("ATLASSIAN_BASE_URL").rstrip("/")
	email = _require_env("ATLASSIAN_EMAIL")
	api_token = _require_env("ATLASSIAN_API_TOKEN")

	response = requests.get(
		f"{base_url}/rest/api/3/myself",
		auth=(email, api_token),
		headers={"Accept": "application/json"},
		timeout=20,
	)

	assert response.status_code == 200, response.text

	data = response.json()
	assert data.get("accountId")
	assert data.get("emailAddress") or data.get("displayName")

