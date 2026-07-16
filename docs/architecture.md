# Architecture Decisions

## Service Execution Model

- Infrastructure services are currently synchronous.
- Until async service implementations are introduced, async FastAPI/LangGraph code should call service methods via thread offloading (for example, asyncio.to_thread).
- Introducing native async service clients is a future enhancement and is not part of the MVP scope.

## Service Construction

- Prefer service factories in src/services/factory.py instead of constructing new service instances throughout the codebase.
- Factory accessors are cached singleton providers:
  - get_anthropic_service()
  - get_jira_service()
- Direct class construction remains available for dependency injection in tests and explicit wiring scenarios.
