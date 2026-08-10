"""Backward-compatible re-exports from orchestrator/core.py.

All classes have been split into focused modules:
- event_bus.py: EventBus, EventEnvelope, drain_and_merge, merge helpers
- session_controller.py: SessionController, SessionState, exceptions
- session_pool.py: SessionPool

This file re-exports all symbols for backward compatibility.
New code should import from the focused modules directly.
"""

from __future__ import annotations

from wolfharness.orchestrator.event_bus import (
    DEFAULT_QUEUE_MAXSIZE,
    EventBus,
    EventEnvelope,
    _is_immediate,  # noqa: F401  — re-exported for backward compat
    _merge_envelopes,  # noqa: F401
    _merge_key,  # noqa: F401
    _merge_progress_events,  # noqa: F401
    _merge_text_deltas,  # noqa: F401
    _merge_thinking_deltas,  # noqa: F401
    _merge_tool_call_deltas,  # noqa: F401
    _rebind,  # noqa: F401
    drain_and_merge,
)
from wolfharness.orchestrator.run import RunHandle
from wolfharness.orchestrator.session_controller import (
    DEFAULT_SESSION_TTL_SECONDS,
    CheckpointMismatchError,
    SessionBusyError,
    SessionClosedError,
    SessionController,
    SessionLifecyclePolicy,
    SessionNotFoundError,
    SessionState,
)
from wolfharness.orchestrator.session_pool import (
    DEFAULT_MAX_AUTO_RESUME,
    SessionPool,
)


__all__ = [
    "DEFAULT_MAX_AUTO_RESUME",
    "DEFAULT_QUEUE_MAXSIZE",
    "DEFAULT_SESSION_TTL_SECONDS",
    "CheckpointMismatchError",
    "EventBus",
    "EventEnvelope",
    "RunHandle",
    "SessionBusyError",
    "SessionClosedError",
    "SessionController",
    "SessionLifecyclePolicy",
    "SessionNotFoundError",
    "SessionPool",
    "SessionState",
    "drain_and_merge",
]
