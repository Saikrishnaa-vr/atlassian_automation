# AI Sprint Planner & Jira Automation

A multi-agent AI system that turns raw product requirements into structured, review-ready Jira work items — with a human approval gate before anything is written to Jira.

Built with **LangGraph** (orchestration) · **Claude / Anthropic** (reasoning) · **FastAPI** (API) · **Jira REST API** (integration).

> **Status: MVP implemented.** The full pipeline — requirement analysis, story generation, human approval gate, and Jira creation — is wired and running. The API is live. See [`.codemie/plan.md`](.codemie/plan.md) for the approved build plan.

## What it does

A person submits a requirement in natural language. A chain of specialized agents analyzes it, drafts Agile user stories, and — **only after a human explicitly approves** — creates the corresponding tickets in Jira. No Jira write ever happens without an APPROVED decision.

```
Requirement  →  RequirementAgent  →  StoryAgent  →  Human Approval  →  Jira
```

The approval step is a hard stop enforced at two levels: the graph topology routes around the Jira node unless approval is APPROVED, and the Jira node itself refuses to proceed without it.

## Architecture

Layered with strict boundaries. Dependency direction flows top to bottom — no layer imports from a layer above it.

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Config | `src/config.py` | Single source of config truth. Pydantic settings loaded from `.env`. |
| Exceptions | `src/exceptions.py` | Shared exception hierarchy (`ApplicationError` → `ConfigurationError`, `ValidationError`, `ServiceError` (8 subtypes), `AgentError`, `WorkflowError`, `ApprovalError`). |
| Services | `src/services/` | Infrastructure-only client wrappers: Anthropic (retries, backoff, error mapping), Jira (REST API calls), factory singletons. |
| Schemas | `src/schemas/` | Strongly typed Pydantic models used as contracts: `story`, `epic`, `task`, `sprint`, `jira`, `requirement`. |
| Agents | `src/agents/` | Claude-backed reasoning agents on a shared `BaseAgent` (input validation → prompt → LLM invoke → parse → output validation). |
| Infrastructure | `src/infrastructure/` | Cross-cutting utilities: `LangGraphCheckpointSerde` + `make_memory_saver()` for safe Pydantic checkpointing. |
| Graph | `src/graph/` | LangGraph state (`WorkflowState`), nodes, and workflow wiring (interrupt/resume, approval routing). |
| API | `src/api/`, `src/main.py` | FastAPI entrypoint with 5 endpoints and structured error handling. |

## Getting started

Requires **Python 3.11** and [**uv**](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies into the project venv
uv sync

# 2. Configure credentials
cp .env.example .env   # or create .env manually
# edit .env with your keys (see Configuration below)

# 3. Verify connectivity
uv run pytest tests/test_llm_connection.py tests/test_jira_connection.py -v

# 4. Start the API server
uv run uvicorn src.main:app --reload
```

### Configuration

Create a `.env` file (gitignored):

```bash
# Required
ANTHROPIC_API_KEY=...        # Claude API key
JIRA_EMAIL=...               # Jira account email
JIRA_API_TOKEN=...           # Jira API token (create at id.atlassian.com/manage-profile/security/api-tokens)
JIRA_BASE_URL=...            # e.g. https://your-org.atlassian.net

# Optional
ANTHROPIC_MODEL=claude-sonnet-5   # defaults to claude-sonnet-5
APP_NAME=atlassian-automation
ENVIRONMENT=development           # development | testing | production
REQUEST_TIMEOUT=30
LOG_LEVEL=INFO
```

To list which Claude model IDs are enabled for your API key:

```bash
uv run python scripts/fetch_available_models.py
```

## API

The server starts at `http://localhost:8000` by default.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service identity |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ui` | Interactive web UI for submit/approve/refresh workflow |
| `POST` | `/requirements` | Submit a requirement and start the pipeline |
| `GET` | `/requirements/{workflow_id}` | Retrieve current workflow state |
| `POST` | `/requirements/{workflow_id}/approval` | Record a human approval decision |

### Submit a requirement

```http
POST /requirements
Content-Type: application/json

{
  "requirement": "Users should be able to log in with email and password, reset their password via email, and stay logged in for 30 days with a remember-me option.",
  "project_key": "ATLAS"
}
```

Response `201 Created`:
```json
{
  "workflow_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "current_step": "awaiting_approval",
  "approval_status": "pending",
  "generated_stories": { ... },
  ...
}
```

The pipeline pauses after story generation. The workflow is now waiting for a human decision.

### Use the interactive website

After starting FastAPI, open:

`http://127.0.0.1:8000/ui`

The UI lets you:
- submit a requirement,
- auto-capture workflow ID,
- approve/reject/pending from buttons,
- refresh workflow state,
- open created Jira issue links from `jira_results`.

### Record approval

```http
POST /requirements/{workflow_id}/approval
Content-Type: application/json

{
  "approval": "approved"
}
```

`approval` must be one of `"approved"`, `"rejected"`, or `"pending"`.

When `"approved"`, the pipeline resumes and creates Jira issues from the generated stories. When `"rejected"`, the workflow terminates without writing to Jira.

### Retrieve state

```http
GET /requirements/{workflow_id}
```

Returns the current `WorkflowState` for the given workflow ID.

## Workflow: interrupt and resume

The graph is compiled with `interrupt_before=["approval_gate"]` and a `MemorySaver` checkpointer:

1. **First call** (`POST /requirements`) — runs `RequirementAgent` and `StoryAgent`, then pauses before `approval_gate`. State is checkpointed by `workflow_id`.
2. **Approval call** (`POST /requirements/{id}/approval`) — the API records the decision, calls `resume_workflow(compiled, workflow_id, approval)`, which resumes from `approval_gate` without re-running the LLM agents.

```
START → requirement → story_generation → [INTERRUPT]
                                              ↓
                                       approval_gate
                                       ├── APPROVED → jira → END
                                       └── PENDING / REJECTED → END
```

## Implemented components

### Agents

| Agent | Status | Input → Output |
|-------|--------|---------------|
| `RequirementAgent` | Implemented | `str` → `NormalizedRequirement` |
| `StoryAgent` | Implemented | `NormalizedRequirement` → `StoryCollection` |
| `EpicAgent` | Stub (deferred) | — |
| `TaskAgent` | Stub (deferred) | — |
| `EstimationAgent` | Stub (deferred) | — |
| `AssignmentAgent` | Stub (deferred) | — |
| `JiraAgent` | Stub (deferred) | — |
| `ReportingAgent` | Stub (deferred) | — |

### Services

| Service | Status | Description |
|---------|--------|-------------|
| `AnthropicService` | Implemented | Claude API client with retry/backoff and typed request/response |
| `JiraService` | Implemented | Jira REST API client for issue creation (`/rest/api/3/issue`) |
| `NotificationService` | Stub (deferred) | — |

### Infrastructure

| Module | Description |
|--------|-------------|
| `src/infrastructure/checkpoint_serializer.py` | `LangGraphCheckpointSerde` — strips Pydantic `@computed_field` values before msgpack encoding so round-trips through the LangGraph checkpointer always succeed. Use `make_memory_saver()` for every compiled graph. |

## Testing

```bash
uv run pytest                                                  # run all tests (243 total)
uv run pytest -v --ignore=tests/test_llm_connection.py \
              --ignore=tests/test_jira_connection.py          # offline tests only

uv run pytest tests/test_checkpoint_serializer.py -v          # serializer tests (25)
uv run pytest tests/test_workflow.py -v                       # workflow + interrupt/resume
uv run pytest tests/test_api.py -v                            # API endpoint tests
uv run pytest tests/test_llm_connection.py -v                 # live Anthropic test
uv run pytest tests/test_jira_connection.py -v                # live Jira test
```

The two connection tests (`test_llm_connection.py`, `test_jira_connection.py`) are **live integration tests** — they call the real APIs. They `pytest.skip` when credentials are absent, so check for `skipped` rather than just green.

All other tests (241) are offline and run without credentials.

## Known limitations

- **In-memory state only.** The workflow registry (`_registry`) and compiled graphs (`lru_cache`) are held in process memory and are lost on server restart. Resuming an interrupted workflow after a restart is not possible in the current implementation.
- **No persistence layer.** A database-backed checkpoint store (e.g. PostgreSQL via `langgraph-checkpoint-postgres`) is the correct fix; deferred to a later phase.
- **Single-process.** The in-memory `MemorySaver` is not safe across multiple worker processes. Do not run with `--workers N > 1`.
- **Deferred agents.** Epic generation, task breakdown, estimation, assignment, and reporting are stubbed out.

## Project layout

```
src/
├── config.py                  # Pydantic settings (loads from .env)
├── exceptions.py              # Shared exception hierarchy
├── main.py                    # FastAPI app, global error handlers
├── api/
│   └── routes.py              # /requirements endpoints
├── agents/
│   ├── base_agent.py          # Abstract BaseAgent[InputT, OutputT]
│   ├── requirement_agent.py   # RequirementAgent (implemented)
│   ├── story_agent.py         # StoryAgent (implemented)
│   └── *.py                   # epic, task, estimation, … (stubs)
├── graph/
│   ├── state.py               # WorkflowState, ApprovalStatus, WorkflowStep
│   ├── nodes.py               # LangGraph node functions
│   └── workflow.py            # build_workflow / run_workflow / resume_workflow
├── infrastructure/
│   └── checkpoint_serializer.py  # LangGraphCheckpointSerde + make_memory_saver()
├── prompts/
│   ├── requirement_prompt.py  # RequirementAgent system + user prompts
│   └── story_prompt.py        # StoryAgent system + user prompts
├── schemas/
│   ├── base.py                # DomainModel base (frozen, extra=forbid)
│   ├── requirement.py         # NormalizedRequirement
│   ├── story.py               # UserStory, StoryCollection
│   ├── epic.py                # Epic, EpicCollection
│   ├── task.py                # Task, TaskCollection
│   ├── sprint.py              # Sprint, SprintCollection
│   └── jira.py                # JiraIssueCreateRequest, JiraIssueResponse
├── tools/
│   ├── jira_tools.py          # stub (deferred)
│   └── utils.py               # stub (deferred)
└── services/
    ├── anthropic_service.py   # AnthropicService + typed request/response models
    ├── jira_service.py        # JiraService
    ├── notification_service.py  # stub (deferred)
    └── factory.py             # get_anthropic_service() / get_jira_service() singletons

tests/
├── conftest.py                # shared fixtures
├── test_api.py
├── test_agents.py
├── test_checkpoint_serializer.py   # 25 tests: serde round-trips, computed field exclusion
├── test_error_handling.py
├── test_nodes.py
├── test_requirement_agent.py
├── test_serialization.py
├── test_state.py
├── test_story_agent.py
├── test_workflow.py
├── test_llm_connection.py     # live Anthropic integration test
└── test_jira_connection.py    # live Jira integration test

scripts/
└── fetch_available_models.py  # list Claude model IDs for your API key
```
