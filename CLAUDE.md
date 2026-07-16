# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

This is a **greenfield / scaffold-stage** project. The directory tree and dependencies are in place, but every `.py` file under `src/` and `tests/` is currently empty (0 lines). There is no working entrypoint yet.

Before implementing, read the two planning documents in `.codemie/` — they are the source of truth for what to build and in what order:
- `.codemie/plan.md` — the approved MVP execution plan (build strategy, sequencing, engineering constraints).
- `.codemie/atlassian_audit.md` — the engineering audit (file inventory, dependencies, env setup).

## Tooling & Commands

This project uses **uv** (see `uv.lock`) and targets **Python 3.11** (`.python-version`).

```bash
uv sync                              # install/resolve dependencies into .venv
uv run <cmd>                         # run a command inside the project venv

uv run pytest                        # run all tests
uv run pytest tests/test_llm_connection.py    # run a single test file
uv run pytest tests/test_llm_connection.py::test_name -v   # run one test
uv run pytest -k "jira"              # run tests matching an expression
```

There is no lint/format config committed yet. If adding one, wire it through `uv run`.

Async tests use `pytest-asyncio` (already a dependency); mark coroutine tests with `@pytest.mark.asyncio`.

## Architecture (as planned)

A multi-agent system built on **LangGraph + Claude (Anthropic) + FastAPI + Jira REST API**. The intended layering enforces strict boundaries — respect them when implementing:

- **`src/config.py`** — single source of configuration truth via Pydantic v2 settings (`pydantic-settings`). Loads from `.env`; no hardcoded secrets or values.
- **`src/services/`** — **infrastructure-only** client wrappers (`anthropic_service.py`, `jira_service.py`, `notification_service.py`). These contain connectivity, retries, timeouts, typed request/response contracts, and error mapping only — **no prompts, no business logic, no workflow logic**.
- **`src/schemas/`** — strongly typed Pydantic models used as contracts across every layer (epic, story, task, sprint, jira).
- **`src/agents/`** — reasoning agents. A shared `base_agent.py` provides logging, LLM-invocation abstraction, and validation hooks; concrete agents (requirement, story, etc.) do structured reasoning only. Agents should be pluggable through the `BaseAgent` interface without changing the workflow core.
- **`src/graph/`** — LangGraph orchestration: `state.py` (typed workflow state incl. approval status), `nodes.py`, `workflow.py` (graph wiring).
- **`src/api/routes.py`**, **`src/main.py`** — FastAPI entrypoint (not yet designed in the MVP plan; clarify the human-approval trigger mechanism when implementing).

Dependency direction flows: config → services → schemas → agents → graph → api. Logging is via **Loguru** with structured, non-secret output.

## Critical Constraints (from `.codemie/plan.md`)

- **Human-approval hard-stop before Jira.** The MVP workflow is `Requirement → Story → Human Approval → Jira`. Workflow execution **must hard-stop before any Jira call** when the approval flag is false or missing. Jira service methods must never be invoked in the pre-approval graph path — this is explicitly verified.
- **Vertical slices, one file per turn.** Build incrementally; no big-bang implementation. Implement only the currently requested file.
- **No generic exceptions.** Introduce a shared exceptions module early and reuse it in config/services rather than raising bare `Exception`.
- **Represent approval as an explicit status** (e.g. pending/approved/rejected enum), not an ambiguous boolean.
- **Deferred modules** (placeholders only for now): task, estimation, assignment, reporting, notification agents.
- Follow SOLID, use dependency injection where applicable, and keep models strongly typed.

## Configuration

Runtime config is read from `.env`. Required keys: `ANTHROPIC_API_KEY`, `ATLASSIAN_EMAIL`, `ATLASSIAN_API_TOKEN`, `ATLASSIAN_BASE_URL`, `TAVILY_API_KEY`.

