# AI Sprint Planner & Jira Automation

A multi-agent AI system that turns raw product requirements into structured, review-ready Jira work items — with a human approval gate before anything is written to Jira.

Built with **LangGraph** (orchestration) · **Claude / Anthropic** (reasoning) · **FastAPI** (API) · **Jira REST API** (integration).

> **Status: early / scaffold stage.** The project structure, dependencies, and live connectivity tests are in place, but the application code (`src/`) is not yet implemented. This README describes the intended scope and how to work in the repo today. See [`.codemie/plan.md`](.codemie/plan.md) for the approved build plan and [`.codemie/atlassian_audit.md`](.codemie/atlassian_audit.md) for the current audit.

## What we're trying to achieve

The goal is to compress the manual, repetitive work of sprint planning. A person supplies a requirement in natural language; a chain of specialized agents analyzes it, drafts user stories, and — **only after a human approves** — creates the corresponding tickets in Jira.

Planned capabilities:

- **Requirement analysis** — parse and structure a free-form requirement.
- **User story generation** — derive well-formed stories from the analyzed requirement.
- **Jira ticket creation** — push approved stories into Jira via the REST API.
- **Sprint planning** — (future) organize stories into sprints.
- Deferred for later phases: epic generation, task breakdown, estimation, assignment, and reporting/notifications.

### MVP workflow

```
Requirement  →  Story  →  Human Approval  →  Jira
```

The **human approval step is a hard stop**: no Jira write ever happens while approval is pending, missing, or rejected. This is the core safety property of the system.

## Architecture

Layered, with strict boundaries between infrastructure, contracts, reasoning, and orchestration:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Config | `src/config.py` | Single source of config truth (Pydantic settings from `.env`). |
| Services | `src/services/` | Infrastructure-only client wrappers (Anthropic, Jira, notifications) — connectivity, retries, timeouts, error mapping. No business logic. |
| Schemas | `src/schemas/` | Strongly typed Pydantic models used as contracts across all layers. |
| Agents | `src/agents/` | Claude-backed reasoning agents built on a shared `base_agent`. |
| Graph | `src/graph/` | LangGraph state, nodes, and workflow wiring (including the approval gate). |
| API | `src/api/`, `src/main.py` | FastAPI entrypoint (not yet designed). |

Dependency direction: **config → services → schemas → agents → graph → api**. Logging is via Loguru.

## Getting started

Requires **Python 3.11** and [**uv**](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies into the project venv
uv sync

# 2. Configure credentials (see below), then verify connectivity
uv run pytest
```

### Configuration

Create a `.env` file (it is gitignored) with:

```bash
ANTHROPIC_API_KEY=...        # Claude access
ATLASSIAN_EMAIL=...          # Jira account email
ATLASSIAN_API_TOKEN=...      # Jira API token
ATLASSIAN_BASE_URL=...       # e.g. https://your-org.atlassian.net
TAVILY_API_KEY=...           # web-search access
ANTHROPIC_MODEL=claude-sonnet-5   # optional; defaults to claude-sonnet-5 in tests
```

To discover which Claude model ids are enabled for your key:

```bash
uv run python scripts/fetch_available_models.py
```

## Usage

### Verifying your setup

The repo ships with **live connection tests** that confirm your credentials work against the real Anthropic and Jira APIs:

```bash
uv run pytest                                   # run all tests
uv run pytest tests/test_jira_connection.py     # check Jira auth only
uv run pytest tests/test_llm_connection.py      # check Anthropic auth only
```

These tests **skip** (rather than fail) when their required env vars are missing — so a passing run with no output means the credentials weren't found. Confirm the tests actually ran, not just that they were green.

### Running the application

Not yet available — the workflow, agents, and API entrypoint are still to be implemented per [`.codemie/plan.md`](.codemie/plan.md). Development proceeds in vertical slices, one module at a time.
