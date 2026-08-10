---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/wolfharness/models/agui_agents.py
title: AG-UI Agents
description: AG-UI protocol agent integration
icon: material/monitor
---

AG-UI (Agent User Interface) agents connect to remote HTTP endpoints that implement the AG-UI protocol, enabling integration of any AG-UI compatible server into the AgentPool pool.

## Overview

AG-UI is a protocol for building agent interfaces that provides:

- **HTTP-based communication**: Simple REST endpoints for agent interaction
- **Streaming support**: Real-time response streaming
- **Standardized interface**: Consistent API across different agent implementations

AG-UI agents are useful for:

- Integrating existing AG-UI compatible services
- Building distributed agent architectures
- Connecting to remote agent deployments
- Testing with locally spawned servers

## Configuration Schema

The AG-UI agent accepts the following configuration fields:

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `type` | string | Must be `"agui"` | Required |
| `endpoint` | string | HTTP URL for the AG-UI agent server | Required |
| `timeout` | float | Maximum wait time for agent responses | 30.0 |
| `headers` | dict | Custom HTTP headers for authentication | `{}` |
| `startup_command` | string | Command to spawn the server process | None |
| `startup_delay` | float | Delay before connecting after startup | 0.0 |

## Basic Usage

```yaml
agents:
  remote_assistant:
    type: agui
    endpoint: http://localhost:8000/agent/run
    timeout: 30.0
    headers:
      X-API-Key: ${API_KEY}

  managed_agent:
    endpoint: http://localhost:8765/agent/run
    startup_command: "uv run ag-ui-server config.yml"
    startup_delay: 3.0
```

## Configuration Notes

- The `endpoint` field specifies the HTTP URL for the AG-UI agent server
- Use `headers` for authentication tokens or custom routing headers
- Environment variables can be used in header values: `${VAR_NAME}`
- When `startup_command` is provided, AgentPool will spawn the server process
- The server process is automatically terminated when the agent pool closes
- Use `startup_delay` to give the server time to initialize before connecting
- The `timeout` setting controls the maximum time for agent responses
