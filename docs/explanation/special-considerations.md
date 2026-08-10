# Special Considerations

## Async Context Managers

AgentPool and Agents use async context managers - always use `async with`:
```python
async with AgentPool(manifest) as pool:
    async with pool.get_agent("name") as agent:
        result = await agent.run("prompt")
```

## MCP Server Lifecycle

MCP servers are spawned as subprocesses - pool cleanup handles termination.
Use `ProcessManager` from `anyenv` for external process management.

## UPath for File Operations

Use `UPath` (universal_pathlib) not `Path` - supports remote filesystems (s3://, gs://, etc.)

## Model Configuration

Prefer string shorthand in YAML: `model: "openai:gpt-4o"`
Fallback models: `type: fallback, models: [primary, backup]`

## Entry Points

The project uses entry points for extensibility:
- `wolfharness_toolsets` - Register custom toolsets
- `fsspec.specs` - Filesystem implementations (ACP)
- `universal_pathlib.implementations` - Path implementations
