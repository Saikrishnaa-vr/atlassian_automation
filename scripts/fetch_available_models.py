from __future__ import annotations

import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv


def _get_api_key() -> str:
	load_dotenv()
	api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
	if not api_key:
		raise RuntimeError("Missing ANTHROPIC_API_KEY in environment or .env")
	return api_key


def main() -> int:
	try:
		client = Anthropic(api_key=_get_api_key())
		page = client.models.list(limit=100)
	except Exception as exc:
		print(f"Failed to fetch models: {exc}", file=sys.stderr)
		return 1

	items = list(page.data)
	if not items:
		print("No models returned for this API key.")
		return 0

	print("Available Anthropic models:\n")
	for model in items:
		model_id = getattr(model, "id", "<unknown-id>")
		display_name = getattr(model, "display_name", "")
		created_at = getattr(model, "created_at", "")
		if display_name:
			print(f"- {model_id} ({display_name})")
		else:
			print(f"- {model_id}")
		if created_at:
			print(f"  created_at: {created_at}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
