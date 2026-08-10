# Extending AgentPool

## Tool Implementation

When adding new tools:
1. Create implementation in `wolfharness/tool_impls/<tool_name>/`
2. Define config model in `wolfharness_config/` if complex
3. Register in appropriate toolset (`wolfharness_toolsets/`)
4. Add tests in `tests/tool_impls/`

## Adding Agent Types

New agent types require:
1. Config model in `wolfharness/models/` (inherit from base, set `type` discriminator)
2. Implementation in `wolfharness/agents/`
3. Add to `AnyAgentConfig` union in `manifest.py`
4. Update manifest loading in `pool.py`

## Server Implementation

New protocol servers:
1. Inherit from `BaseServer` in `wolfharness_server/base.py`
2. Implement protocol-specific handlers
3. Use `AggregatingServer` if wrapping multiple agents
4. Add CLI command in `wolfharness_cli/`
