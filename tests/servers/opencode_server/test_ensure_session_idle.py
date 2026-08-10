"""Unit tests for ``_ensure_session_idle`` helper.

Covers four scenarios:
- Session already idle (no active run) → no-op.
- Session busy, cancel succeeds and completes within timeout → idle event broadcast.
- Session busy, cancel times out → warning logged, ``current_run_id`` force-cleared.
- Session busy, ``cancel_run`` raises → warning logged, ``current_run_id`` force-cleared.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest

from wolfharness_server.opencode_server.routes.session_routes import _ensure_session_idle


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.state import ServerState

pytestmark = pytest.mark.unit


def _make_session_state(run_id: str | None) -> Mock:
    """Create a mock ``SessionState`` with a settable ``current_run_id``."""
    session_state = Mock()
    session_state.current_run_id = run_id
    return session_state


def _make_state(
    session_state: Mock | None,
    session_pool: Mock | None = None,
) -> ServerState:
    """Build a minimal ``ServerState`` mock suitable for ``_ensure_session_idle``.

    The helper only touches:
    - ``state.session_controller`` (may be ``None``)
    - ``state.session_controller.get_session(session_id)``
    - ``state.pool.session_pool``
    - ``state.pool.session_pool.cancel_run(run_id)``
    - ``state.pool.session_pool.wait_for_completion(session_id)``
    - ``state.broadcast_event(event)``
    """
    state = Mock()
    state.broadcast_event = AsyncMock()

    if session_state is not None:
        state.session_controller = Mock()
        state.session_controller.get_session.return_value = session_state
    else:
        state.session_controller = None

    pool = Mock()
    pool.session_pool = session_pool
    state.pool = pool
    state.pool_or_none = pool

    return state  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_ensure_session_idle_already_idle() -> None:
    """Given: session has no active run (``current_run_id`` is ``None``).

    When: ``_ensure_session_idle`` is called.

    Then: returns immediately — no cancel, no broadcast, no state mutation.
    """
    session_state = _make_session_state(run_id=None)
    state = _make_state(session_state=session_state, session_pool=Mock())

    await _ensure_session_idle(state, "sess-1")

    state.broadcast_event.assert_not_awaited()
    assert session_state.current_run_id is None


@pytest.mark.asyncio
async def test_ensure_session_idle_cancel_succeeds_within_timeout() -> None:
    """Given: session is busy with an active run.

    When: ``cancel_run`` succeeds and ``wait_for_completion`` returns within timeout.

    Then: ``cancel_run`` is called, idle ``SessionStatusEvent`` is broadcast,
    and ``current_run_id`` is cleared.
    """
    session_state = _make_session_state(run_id="run-abc")
    session_pool = Mock()
    session_pool.cancel_run = Mock()
    session_pool.wait_for_completion = AsyncMock(return_value="sess-1")
    state = _make_state(session_state=session_state, session_pool=session_pool)

    await _ensure_session_idle(state, "sess-1")

    session_pool.cancel_run.assert_called_once_with("run-abc")
    session_pool.wait_for_completion.assert_awaited_once_with("sess-1")
    state.broadcast_event.assert_awaited_once()
    broadcasted_event = state.broadcast_event.await_args.args[0]
    assert broadcasted_event.type == "session.status"
    assert broadcasted_event.properties.status.type == "idle"
    assert session_state.current_run_id is None


@pytest.mark.asyncio
async def test_ensure_session_idle_cancel_times_out() -> None:
    """Given: session is busy, ``wait_for_completion`` does not finish in time.

    When: ``asyncio.wait_for`` raises ``TimeoutError``.

    Then: warning is logged, ``current_run_id`` is force-cleared, and the
    helper proceeds without raising.
    """
    session_state = _make_session_state(run_id="run-slow")
    session_pool = Mock()
    session_pool.cancel_run = Mock()

    async def _hang_forever(_session_id: str) -> str:
        await asyncio.sleep(999)  # will be cancelled by wait_for
        return _session_id

    session_pool.wait_for_completion = AsyncMock(side_effect=_hang_forever)
    state = _make_state(session_state=session_state, session_pool=session_pool)

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes._IDLE_WAIT_TIMEOUT",
        0.05,
    ):
        await _ensure_session_idle(state, "sess-1")

    session_pool.cancel_run.assert_called_once_with("run-slow")
    assert session_state.current_run_id is None
    state.broadcast_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_session_idle_cancel_run_raises_value_error() -> None:
    """Given: ``current_run_id`` is set but the RunHandle is gone (stale reference).

    When: ``cancel_run`` raises ``ValueError`` (run not found).

    Then: exception is caught, warning logged, ``current_run_id`` is
    force-cleared without waiting, and the helper proceeds without raising.
    """
    session_state = _make_session_state(run_id="run-stale")
    session_pool = Mock()
    session_pool.cancel_run = Mock(side_effect=ValueError("No active run found with ID: run-stale"))
    session_pool.wait_for_completion = AsyncMock()
    state = _make_state(session_state=session_state, session_pool=session_pool)

    await _ensure_session_idle(state, "sess-1")

    session_pool.cancel_run.assert_called_once_with("run-stale")
    session_pool.wait_for_completion.assert_not_awaited()
    assert session_state.current_run_id is None
    state.broadcast_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_session_idle_cancel_run_raises_runtime_error() -> None:
    """Given: ``cancel_run`` raises ``RuntimeError`` (unexpected failure).

    When: the exception propagates inside ``_ensure_session_idle``.

    Then: exception is caught, warning logged, ``current_run_id`` is
    force-cleared, and the helper proceeds without raising.
    """
    session_state = _make_session_state(run_id="run-broken")
    session_pool = Mock()
    session_pool.cancel_run = Mock(side_effect=RuntimeError("unexpected failure"))
    session_pool.wait_for_completion = AsyncMock()
    state = _make_state(session_state=session_state, session_pool=session_pool)

    await _ensure_session_idle(state, "sess-1")

    session_pool.cancel_run.assert_called_once_with("run-broken")
    session_pool.wait_for_completion.assert_not_awaited()
    assert session_state.current_run_id is None
    state.broadcast_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_session_idle_no_session_controller() -> None:
    """Given: ``state.session_controller`` is ``None``.

    When: ``_ensure_session_idle`` is called.

    Then: returns immediately — no side effects.
    """
    state = _make_state(session_state=None, session_pool=Mock())

    await _ensure_session_idle(state, "sess-1")  # type: ignore[arg-type]

    state.broadcast_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_session_idle_session_not_found() -> None:
    """Given: ``session_controller.get_session`` returns ``None``.

    When: ``_ensure_session_idle`` is called.

    Then: returns immediately — no cancel, no broadcast.
    """
    state = Mock()
    state.broadcast_event = AsyncMock()
    state.session_controller = Mock()
    state.session_controller.get_session.return_value = None
    pool = Mock()
    pool.session_pool = Mock()
    state.pool = pool
    state.pool_or_none = pool

    await _ensure_session_idle(state, "sess-missing")  # type: ignore[arg-type]

    state.broadcast_event.assert_not_awaited()
