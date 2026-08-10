"""pdai Capability implementations for AgentPool.

Phase 6 of the thin-wrapper refactor. Each capability is a composable
agent extension using pydantic-ai's native Capability API.

Capabilities fire hooks only when ``RunExecutor.next(node)`` is called
explicitly (Phase 2 unifies all run paths to use RunExecutor).
"""
