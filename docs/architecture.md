# Architecture Decisions

## Layered Architecture

The application follows a strict layered dependency direction:

```
API Layer
  ↓ imports
Graph Layer (Workflow & State)
  ↓ imports
Agents Layer (Reasoning)
  ↓ imports
Services Layer (Infrastructure)
  ↓ imports
Schemas Layer (Domain Contracts)
  ↓ imports
Config & Exceptions (Core)
```

Dependencies flow **downward only**. No layer imports from layers above it.

### Core Layer
- `src/config.py` — Typed configuration from environment variables via pydantic-settings
- `src/exceptions.py` — Application exception hierarchy (ApplicationError root + 13 subclasses)

### Schemas Layer
- Domain contracts used across all other layers
- `src/schemas/base.py` — Shared DomainModel base + validators
- `src/schemas/story.py`, `epic.py`, `task.py`, `sprint.py` — Agile domain models
- `src/schemas/jira.py` — Jira I/O contracts
- `src/schemas/requirement.py` — Normalized requirement contract (output from RequirementAgent)

### Services Layer (Infrastructure Only)
- No business logic, no prompts, no workflow orchestration
- `src/services/anthropic_service.py` — Claude API client wrapper (generation, health checks, retries)
- `src/services/jira_service.py` — Jira REST client wrapper (CRUD, search, project lookup)
- `src/services/factory.py` — Cached singleton accessors for dependency wiring

### Agents Layer (Reasoning Only)
- Pure reasoning agents with input validation → prompt building → LLM invocation → response parsing → output validation
- `src/agents/base_agent.py` — BaseAgent abstract class (shared execution lifecycle)
- `src/agents/requirement_agent.py` — Requirement analysis (raw text → NormalizedRequirement)
- `src/agents/story_agent.py` — Story generation (NormalizedRequirement → StoryCollection)
- Deferred: EpicAgent, TaskAgent, EstimationAgent, AssignmentAgent, ReportingAgent

### Graph Layer (Orchestration Only)
- LangGraph state machine + nodes
- `src/graph/state.py` — WorkflowState (single source of truth for execution state)
  - **Critical:** ApprovalStatus enum (PENDING, APPROVED, REJECTED) — never boolean
  - Tracks workflow_id, requirement, stories, approval gate, Jira results, execution metadata, errors
- `src/graph/nodes.py` — Individual node functions (thin orchestration only)
  - requirement_node → RequirementAgent
  - story_generation_node → StoryAgent
  - approval_gate_node → approval status check + routing
  - jira_node (factory) → JiraService (only when APPROVED)
- `src/graph/workflow.py` — LangGraph StateGraph + compiled graph assembly
  - **Approval hard-stop:** Only APPROVED status allows routing to jira_node
  - Jira node is structurally unreachable unless approval status is APPROVED

### API Layer (FastAPI)
- `src/main.py` — FastAPI app (lifespan, exception handlers, global middlewares)
- `src/api/routes.py` — Workflow endpoints (thin routing + validation only)
  - POST /requirements — Submit raw requirement → run full workflow up to approval gate
  - GET /requirements/{workflow_id} — Retrieve current WorkflowState
  - POST /requirements/{workflow_id}/approval — Record approval decision → resume workflow → Jira

## Approval Gate (Critical Design)

**The workflow enforces a hard stop before any Jira operation.**

1. Requirement → story generation runs synchronously, halts at approval gate with `approval_status = PENDING`
2. Human reviews WorkflowState and issues approval decision via POST /requirements/{id}/approval
3. Only when `approval_status == ApprovalStatus.APPROVED` does the workflow proceed to Jira
4. Jira node double-checks approval status before calling any JiraService method

**Guarantee:** No Jira service method is ever invoked in the pre-approval graph path.

## Service Execution Model

- Infrastructure services are currently **synchronous**.
- FastAPI/LangGraph async code should call service methods via thread offloading (e.g., `asyncio.to_thread`).
- Introducing native async service clients is a **future enhancement**, not part of the MVP scope.

## Service Construction

- Prefer service factories in `src/services/factory.py` instead of constructing new service instances throughout the codebase.
- Factory accessors are cached singleton providers:
  - `get_anthropic_service()` — returns cached AnthropicService instance
  - `get_jira_service()` — returns cached JiraService instance
- Direct class construction remains available for dependency injection in tests and explicit wiring scenarios.

## State Management

- **WorkflowState is immutable by convention** (not `frozen=True`; LangGraph pattern)
- Each node receives a state snapshot, derives an updated copy via `model_copy()`, and returns it
- State is the single source of truth: all workflow data flows through it
- Serialization: Pydantic v2 models serialize cleanly to JSON for API responses and storage

## In-Memory MVP Placeholders (Phase 3)

The API layer currently uses:
- `_registry: dict[str, WorkflowState]` — maps workflow_id → state snapshot
- `_project_keys: dict[str, str]` — maps workflow_id → Jira project key

**These are explicit MVP placeholders. They are:**
- Lost on process restart
- Not safe for concurrent access or multi-process deployments
- Without eviction policy (memory leak risk)

**They must be replaced by a persistent store before production use** (Phase 5+).

## Error Handling

- All service exceptions (`AnthropicServiceError`, `JiraServiceError`) are mapped to domain `ApplicationError` subclasses
- No SDK exceptions escape the service layer
- Agents map service errors to `AgentError`
- Graph/workflow errors map to `WorkflowError`
- FastAPI global exception handlers convert `ApplicationError` subclasses to HTTP status codes (400/401/403/404/429/500)

## Logging

- Structured, secret-free logging via Loguru
- Services log timing, retry attempts, and failures (never credentials)
- Agents log execution steps, LLM invocation, response validation
- Nodes log state transitions and routing decisions
- API layer logs HTTP requests, response codes, and error details
