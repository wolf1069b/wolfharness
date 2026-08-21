# Deprecated `agentpool*` shims no longer duplicate modules and classes

Importing a submodule through one of the deprecated `agentpool*` shim
packages (`agentpool`, `agentpool_bot`, `agentpool_cli`,
`agentpool_commands`, `agentpool_config`, `agentpool_prompts`,
`agentpool_server`, `agentpool_storage`, `agentpool_sync`,
`agentpool_toolsets`) re-executed the corresponding `wolfharness*` module
file, creating a second module object with duplicate copies of every class
defined in it. Any process that mixed canonical and shim imports therefore
held two distinct `AgentContext` classes, and identity-based checks such
as `ann is AgentContext` in `wrap_tool_for_pydantic_ai` silently failed.
For MCP tools this bypassed the argument-injection wrapper entirely,
surfacing at runtime as:

```
Error calling tool 'read_resource': ... missing 1 required positional argument: 'agent_ctx'
```

The root cause was in the shim meta-path finder: `find_spec` pre-set
`sys.modules[alias]` and returned a dummy `ModuleSpec(name, loader=None)`.
CPython's `_find_spec` then reused the target module's own spec from
`sys.modules`, and `_load_unlocked` executed that spec — re-running the
module body under the wolfharness name.

The shim now returns a spec with an alias loader whose `create_module`
hands back the already-imported `wolfharness*` module object, so the alias
name is registered against the same module with no re-execution. The
target module's `__spec__` is restored after loading, keeping introspection
and `importlib.reload` consistent.

Adds subprocess-based regression tests
(`tests/compat/test_shim_module_identity.py`) that verify module and class
identity across the shim boundary in both import orders.
