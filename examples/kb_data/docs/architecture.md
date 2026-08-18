# Architecture Overview

AgentPool bridges multiple protocols with native PydanticAI agents.

## Layers

1. **Configuration** — YAML manifests parsed into Pydantic models
2. **Orchestration** — EventBus, SessionController, RunLoop
3. **Protocols** — ACP, OpenCode, MCP, AG-UI, OpenAI API
4. **Capabilities** — Tools, skills, MCP servers, resources

## Message Flow

```
Client → Protocol Server → SessionController → Agent (PydanticAI)
    → Tool Execution → Event Bus → Protocol Converter → Client
```