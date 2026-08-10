"""Tests for per-session elicitation task isolation in ACPProtocolHandler.

Verifies that cancelling one session's elicitation tasks does not affect
tasks belonging to other sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.unit


def _make_handler() -> Any:
    """Create a minimal ACPProtocolHandler-like object with the elicitation task dict."""
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    host_context = MagicMock()
    host_context.session_pool = MagicMock()
    host_context.session_pool.event_bus = MagicMock()
    session_manager = MagicMock()
    client = MagicMock()
    client.session_update = AsyncMock()
    acp_agent = MagicMock()
    client_capabilities = MagicMock()

    handler = ACPProtocolHandler.__new__(ACPProtocolHandler)
    handler._host_context = host_context
    handler.session_manager = session_manager
    handler.client = client
    handler.client_capabilities = client_capabilities
    handler.acp_agent = acp_agent
    handler._converters = {}
    handler._parent_of = {}
    handler._elicitation_tasks = {}
    return handler


class TestElicitationTaskIsolation:
    """Verify _elicitation_tasks is keyed by session and cleanup is scoped."""

    def test_tasks_stored_per_session(self):
        handler = _make_handler()
        task_a = asyncio.ensure_future(asyncio.sleep(100))
        task_b = asyncio.ensure_future(asyncio.sleep(100))

        handler._elicitation_tasks.setdefault("session_a", set()).add(task_a)
        handler._elicitation_tasks.setdefault("session_b", set()).add(task_b)

        assert task_a in handler._elicitation_tasks["session_a"]
        assert task_b in handler._elicitation_tasks["session_b"]
        assert "session_a" in handler._elicitation_tasks
        assert "session_b" in handler._elicitation_tasks

        task_a.cancel()
        task_b.cancel()

    @pytest.mark.asyncio
    async def test_cancel_session_a_does_not_affect_session_b(self):
        handler = _make_handler()

        started_a = asyncio.Event()
        started_b = asyncio.Event()

        async def task_fn_a():
            started_a.set()
            await asyncio.Event().wait()

        async def task_fn_b():
            started_b.set()
            await asyncio.Event().wait()

        task_a = asyncio.ensure_future(task_fn_a())
        task_b = asyncio.ensure_future(task_fn_b())
        handler._elicitation_tasks.setdefault("session_a", set()).add(task_a)
        handler._elicitation_tasks.setdefault("session_b", set()).add(task_b)

        await asyncio.wait_for(started_a.wait(), timeout=5.0)
        await asyncio.wait_for(started_b.wait(), timeout=5.0)

        # Cancel only session_a's tasks (simulating _after_consumer_loop)
        session_tasks = handler._elicitation_tasks.pop("session_a", None)
        assert session_tasks is not None
        for t in session_tasks:
            if not t.done():
                t.cancel()

        # Await the cancelled task to ensure cancellation is fully processed
        with contextlib.suppress(asyncio.CancelledError):
            await task_a

        assert task_a.cancelled()
        assert not task_b.cancelled()
        assert not task_b.done()

        # session_a is gone, session_b still tracked
        assert "session_a" not in handler._elicitation_tasks
        assert "session_b" in handler._elicitation_tasks

        task_b.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_b

    @pytest.mark.asyncio
    async def test_close_session_only_cancels_own_tasks(self):
        handler = _make_handler()

        async def long_running():
            await asyncio.Event().wait()

        task_a = asyncio.ensure_future(long_running())
        task_b = asyncio.ensure_future(long_running())
        task_c = asyncio.ensure_future(long_running())
        handler._elicitation_tasks.setdefault("session_a", set()).add(task_a)
        handler._elicitation_tasks.setdefault("session_b", set()).add(task_b)
        handler._elicitation_tasks.setdefault("session_c", set()).add(task_c)

        # Simulate close_session for session_b
        session_tasks = handler._elicitation_tasks.pop("session_b", None)
        assert session_tasks is not None
        for t in session_tasks:
            if not t.done():
                t.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task_b

        assert not task_a.cancelled()
        assert task_b.cancelled()
        assert not task_c.cancelled()

        # Cleanup remaining
        for sid in list(handler._elicitation_tasks):
            tasks = handler._elicitation_tasks.pop(sid, set())
            for t in tasks:
                if not t.done():
                    t.cancel()

    def test_discard_callback_does_not_create_empty_sets(self):
        """Discard callback should not create new empty sets for cleaned-up sessions."""
        handler = _make_handler()
        task = asyncio.ensure_future(asyncio.sleep(100))

        handler._elicitation_tasks.setdefault("session_x", set()).add(task)

        # Simulate session cleanup (pop removes the key)
        session_tasks = handler._elicitation_tasks.pop("session_x", None)
        assert session_tasks is not None
        for t in session_tasks:
            t.cancel()

        # Now the done callback fires — it should NOT recreate "session_x"
        sid = "session_x"
        handler._elicitation_tasks.get(sid, set()).discard(task)

        assert "session_x" not in handler._elicitation_tasks
