---
rfc_id: RFC-0002
title: Extended Tool Definition and Native PydanticAI Integration
status: IMPLEMENTED
author: Antigravity
reviewers: [Metis]
created: 2026-02-01
last_updated: 2026-02-02
---

## Overview
This RFC details the implementation of extended `Tool` definitions in `wolfharness` to support features from `pydantic-ai`'s `Tool` class—specifically `prepare`, `function_schema`, `name`, and `description`—and the unification of tool conversion logic using `Tool.from_schema` for enhanced validation capabilities.

## Background & Context
AgentPool's `Tool` abstraction serves as a bridge between multiple protocols. Previously, conversion to `pydantic-ai` tools was fragmented:
- `Agent.get_agentlet` manually wrapped tool functions.
- `schema_override` relied on a custom `SchemaWrapper` that lacked full Pydantic validation support (specifically missing `validate_json`).
- `prepare` hooks were not supported for tools with custom schemas.

## Problem Statement
1.  **Validation Gap**: The previous `SchemaWrapper` implementation for schema overrides did not support `validate_json`, causing runtime errors when PydanticAI attempted to validate tool arguments from JSON.
2.  **Divergent Paths**: Tools with custom schemas used a different code path than standard tools, leading to feature disparity (e.g., missing `prepare` support).
3.  **Context Type Hazard**: `pydantic-ai` expects `RunContext`, while `wolfharness` internal tools often depend on `AgentContext`. Using `pydantic_ai.function_schema` on functions with `AgentContext` failed due to type inspection issues with abstract base classes.

## Implementation Details

### 1. Unified Conversion via `Tool.from_schema`
The `Tool.to_pydantic_ai()` method has been refactored to use `pydantic_ai.Tool.from_schema` as the unified mechanism for creating tools with custom definitions.

- **Native Validation**: By using `Tool.from_schema`, we leverage PydanticAI's native validator generation, ensuring `validate_json` is present and functional.
- **Prepare Hook Support**: Since `Tool.from_schema` does not accept a `prepare` argument in its constructor, we explicitly assign the `prepare` hook to the created tool instance immediately after instantiation.

```python
# Pseudo-code of the implementation in Tool.to_pydantic_ai
pydantic_tool = Tool.from_schema(
    function_to_call,
    name=self.name,
    description=self.description,
    json_schema=effective_schema,
    takes_ctx=takes_ctx
)
# Manually attach prepare hook
pydantic_tool.prepare = self._get_effective_prepare()
```

### 2. Robust Schema Generation Fallback
To handle `AgentContext` and other complex types that confuse `pydantic-ai`'s schema generator, we implemented a robust fallback mechanism:

1.  **Primary Path**: Attempt to use `pydantic_ai.function_schema`.
2.  **Fallback Path**: If that fails (e.g., `PydanticUndefinedAnnotation` or `NameError` due to forward refs), catch the exception and use `schemez.create_schema`.
3.  **Schema Cleaning**: The fallback explicitly excludes `AgentContext` and `RunContext` parameters from the generated JSON schema to prevent LLM confusion, while keeping them in the function signature for injection.

### 3. Extended Tool Configuration
The `Tool` class and configuration models have been updated to support:
- **`prepare`**: A `ToolPrepareFunc` that follows the `pydantic-ai` signature: `(ctx: RunContext[TDeps], tool_def: ToolDefinition) -> ToolDefinition | None`.
- **`function_schema`**: Explicit overrides for the function schema (formerly `schema_override`).

### 4. Context Injection
- **`AgentContext`**: Injected via `RunContext.deps` (when `Agent` is initialized with `deps_type=AgentContext`).
- **`RunContext`**: Supported natively by `pydantic-ai`.

## Technical Decisions
- **Removing `SchemaWrapper`**: The custom wrapper class was removed in favor of `Tool.from_schema`, significantly reducing code complexity and maintenance burden.
- **Manual `prepare` Assignment**: A necessary workaround due to `pydantic_ai.Tool.from_schema` API limitations.
- **Consolidated Testing**: Redundant tests were merged into `tests/tools/test_tool_schema.py`, covering validation, fallback logic, and context injection in a single suite.

## Validation
- **Test Coverage**: Added comprehensive tests for:
    - `validate_json` presence on all tool types.
    - Correct schema generation via fallback (excluding context params).
    - `prepare` hook execution on tools with schema overrides.
    - Async and sync tool execution.

## Future Work
- Consider contributing `prepare` argument support to upstream `pydantic-ai.Tool.from_schema`.
- Explore strict `AgentPoolRunContext` type definition.
