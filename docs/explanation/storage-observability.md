# Storage and Observability

## Storage Providers

Track all agent interactions:
- SQL-based with SQLModel/SQLAlchemy
- Per-agent or shared database
- Analytics via CLI: `wolfharness history stats`

## Observability

Logfire + OpenTelemetry integration:
- Structured logging with context via `logfire.configure()`
- Auto-instrumentation: `logfire.instrument_pydantic_ai()`, `logfire.instrument_mcp()`, `logfire.instrument_fastapi(app)`
- Manual instrumentation: `@logfire.instrument` decorator and `with logfire.span(...)` context manager
- Trace agent execution end-to-end across async task boundaries and subagent delegation
- Export to any OTLP-compatible backend (SigNoz, Jaeger, Honeycomb, etc.) via OTEL env vars
- Disabled in tests via env vars (see conftest.py)
- See [Telemetry & Span Instrumentation](telemetry.md) for mandatory practices
