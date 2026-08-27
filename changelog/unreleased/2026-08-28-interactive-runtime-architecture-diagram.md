# Interactive AgentPool Runtime Architecture Diagram

Added a self-contained interactive architecture diagram at
`docs/explanation/agentpool-runtime-architecture.html`, linked from
Reference → Architecture in the docs navigation.

The diagram maps the AgentPool runtime: the four protocol adapters
(ACP, AG-UI, OpenCode, MCP) entering the SessionPool →
SessionController → RunHandle orchestration chain, the native
PydanticAI agent and capability stack, the session-scoped EventBus,
and the AgentPool service hub. The HTML is a single portable artifact
(no external dependencies) with theme switching, node search, focused
routes, and PNG/SVG/WebM export built in.

Generated with the Archify skill against the live source tree; the same
artifact is deployable as a standalone file or embedded in the docs site.