## Plan: MVP Vertical Slice Execution (Approved Constraints)

Build incrementally in strict vertical slices. Implement only one requested file at a time, enforce clean architecture boundaries, and stop workflow execution before Jira until explicit human approval is provided.

**Steps**
1. Phase 1: Infrastructure readiness and isolated service validation.
2. Implement src/config.py first with Pydantic v2 settings, env loading, typed config groups, secret-safe validation, and no hardcoded values.
3. Implement src/services/anthropic_service.py next as infrastructure-only Claude wrapper (no prompts/business logic/workflow logic) with typed request/response contracts, retries, timeouts, and Loguru logging.
4. Implement src/services/jira_service.py next as infrastructure-only Jira wrapper (no prompts/business logic/workflow logic), typed methods for connectivity and safe issue operations, and explicit Jira error mapping.
5. Implement tests/test_llm_connection.py to validate Anthropic service independently.
6. Implement tests/test_jira_connection.py to validate Jira service independently.
7. Gate to phase 2 only after service tests pass and connectivity behavior is verified.
8. Phase 2: Domain contracts and first reasoning agents.
9. Implement schemas in src/schemas as strongly typed Pydantic models used across all layers.
10. Implement src/graph/state.py typed workflow state for MVP path and approval status tracking.
11. Implement src/agents/base_agent.py reusable base agent with shared logging, LLM invocation abstraction, validation hooks, and non-generic error handling.
12. Implement src/agents/requirement_agent.py for requirement analysis and structured output only.
13. Implement src/agents/story_agent.py for story generation from requirement analysis only.
14. Leave non-MVP agents as placeholders only: task, estimation, assignment, reporting.
15. Phase 3: MVP orchestration only.
16. Implement src/graph/workflow.py for strict MVP flow: Requirement -> Story -> Human Approval -> Jira.
17. Ensure workflow hard-stops before Jira when approval flag is false or missing.
18. Implement Jira agent integration point only for approved paths.
19. Confirm no Jira side effects occur pre-approval.

**Relevant files**
- src/config.py — single source of configuration truth; no secret hardcoding.
- src/services/anthropic_service.py — infrastructure-only Claude client wrapper.
- src/services/jira_service.py — infrastructure-only Jira client wrapper.
- tests/test_llm_connection.py — independent Anthropic connectivity/service behavior tests.
- tests/test_jira_connection.py — independent Jira connectivity/service behavior tests.
- src/schemas/epic.py — epic contracts for inter-layer communication.
- src/schemas/story.py — story contracts for inter-layer communication.
- src/schemas/task.py — task placeholder contract for forward compatibility.
- src/schemas/sprint.py — sprint/report placeholder contract for forward compatibility.
- src/schemas/jira.py — Jira request/result contracts for agent-service boundaries.
- src/graph/state.py — typed graph state including approval gate fields.
- src/agents/base_agent.py — reusable shared agent behaviors.
- src/agents/requirement_agent.py — requirement reasoning only.
- src/agents/story_agent.py — story reasoning only.
- src/graph/workflow.py — MVP orchestration and hard approval stop.
- src/agents/task_agent.py — placeholder only for future phase.
- src/agents/estimation_agent.py — placeholder only for future phase.
- src/agents/assignment_agent.py — placeholder only for future phase.
- src/agents/reporting_agent.py — placeholder only for future phase.

**Verification**
1. Phase 1 service verification: run pytest for tests/test_llm_connection.py and tests/test_jira_connection.py independently before any agent/graph integration.
2. Schema validation verification: model validation tests ensure cross-layer compatibility and strict typing.
3. Approval-gate verification: workflow execution with approval=false/missing must stop before any Jira call.
4. Side-effect verification: Jira service methods are never invoked in pre-approval graph path.
5. Logging verification: key steps and failures logged via Loguru with structured, non-secret output.

**Decisions**
- Delivery model: vertical slices only; no big-bang implementation.
- Scope now: MVP path only (Requirement -> Story -> Approval -> Jira).
- Deferred modules: Task/Estimation/Assignment/Reporting/Notification implementation postponed; placeholders only.
- Engineering constraints: SOLID, DI where applicable, typed models, shared exceptions, no generic Exception usage.
- Execution rule: implement only the currently requested file each turn.

**Further Considerations**
1. Shared exceptions module should be introduced early in phase 1 and reused immediately by config/services to prevent generic exceptions.
2. Approval representation in state should include explicit enum/status (e.g., pending/approved/rejected) to avoid boolean ambiguity.
3. Future agents should be pluggable through BaseAgent interface without changing workflow core orchestration.