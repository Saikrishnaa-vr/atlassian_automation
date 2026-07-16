from __future__ import annotations

import os

import pytest
from anthropic import Anthropic, NotFoundError
from dotenv import load_dotenv


load_dotenv()


def _require_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		pytest.skip(f"Skipping live LLM connection test: missing {name}")
	return value


def test_anthropic_connection() -> None:
	"""Validate that Anthropic credentials can make a tiny successful request."""
	api_key = _require_env("ANTHROPIC_API_KEY")
	model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip()

	client = Anthropic(api_key=api_key)
	try:
		response = client.messages.create(
			model=model,
			max_tokens=8,
			messages=[{"role": "user", "content": "Reply with: ok"}],
		)
	except NotFoundError:
		pytest.skip(
			"Skipping live LLM connection test: model not found. "
			"Set ANTHROPIC_MODEL in .env to a model enabled for this account."
		)

	assert response.id
	assert response.content
	first_block = response.content[0]
	text = getattr(first_block, "text", "").strip().lower()
	assert text

