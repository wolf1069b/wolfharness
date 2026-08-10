"""Tests for list_sessions store-first restore behavior.

Verifies that list_sessions queries the session store for persisted sessions
( source of truth), overlays in-memory active sessions, and handles edge cases
like store=None, store failures, and cwd filtering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from wolfharness.sessions.models import SessionData
from wolfharness_server.opencode_server.models import Session
from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated
from wolfharness_server.opencode_server.routes.session_routes import list_sessions


pytestmark = pytest.mark.integration


def _make_session_data(
    session_id: str,
    cwd: str = "/test",
    title: str = "Test",
    updated: datetime | None = None,
) -> SessionData:
    """Create a SessionData for testing."""
    now = updated or datetime.now(UTC)
    return SessionData(
        session_id=session_id,
        agent_name="engineer",
        cwd=cwd,
        created_at=now,
        last_active=now,
        metadata={"title": title},
    )


def _make_session(
    session_id: str,
    directory: str = "/test",
    title: str = "Test",
    updated_ms: int = 1000,
) -> Session:
    """Create an OpenCode Session for testing."""
    return Session(
        id=session_id,
        project_id="test-project",
        title=title,
        directory=directory,
        time=TimeCreatedUpdated(created=updated_ms, updated=updated_ms),
    )


def _make_state(
    *,
    store: Any = None,
    session_controller_sessions: list[Any] | None = None,
    cached_sessions: dict[str, Session] | None = None,
    base_path: str = "/test",
    agent_list_sessions: Any = None,
) -> Mock:
    """Create a mock ServerState for testing list_sessions."""
    state = Mock()
    state.base_path = base_path
    state.sessions = cached_sessions or {}

    # Session pool
    session_pool = Mock()
    session_pool.sessions = Mock()
    session_pool.sessions.store = store
    state.pool = Mock()
    state.pool.session_pool = session_pool
    state.pool_or_none = state.pool

    # Session controller
    session_controller = Mock()
    session_controller.list_sessions = Mock(return_value=session_controller_sessions or [])
    state.session_controller = session_controller

    # Agent (legacy path)
    agent = Mock()
    if agent_list_sessions is not None:
        agent.list_sessions = agent_list_sessions
    else:
        agent.list_sessions = AsyncMock(return_value=[])
    state.agent = agent

    return state


# -- Task 2.1: Sessions restored after restart --


async def test_list_sessions_returns_persisted_sessions_after_restart():
    """After restart (empty SessionController), persisted sessions from store are returned."""
    store_data = [
        _make_session_data("ses_1", cwd="/test", title="Session 1"),
        _make_session_data("ses_2", cwd="/test", title="Session 2"),
    ]
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=["ses_1", "ses_2"])
    store.load_sessions_batch = AsyncMock(return_value=store_data)

    state = _make_state(store=store, session_controller_sessions=[])

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.side_effect = lambda data: _make_session(
            data.session_id, directory=data.cwd or "/test", title=data.metadata.get("title", "")
        )
        result = await list_sessions(state, directory="/test")

    assert len(result) == 2
    assert {s.id for s in result} == {"ses_1", "ses_2"}


# -- Task 2.2: In-memory session overrides store version --


async def test_in_memory_session_overrides_store_version():
    """In-memory cached session takes precedence over store version (fresher status)."""
    store_data = [_make_session_data("ses_1", cwd="/test", title="From Store")]
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=["ses_1"])
    store.load_sessions_batch = AsyncMock(return_value=store_data)

    # In-memory cached version with different title
    in_memory_session = _make_session(
        "ses_1", directory="/test", title="From Memory", updated_ms=2000
    )
    controller_info = Mock()
    controller_info.session_id = "ses_1"

    state = _make_state(
        store=store,
        session_controller_sessions=[controller_info],
        cached_sessions={"ses_1": in_memory_session},
    )

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.return_value = _make_session("ses_1", directory="/test", title="From Store")
        result = await list_sessions(state, directory="/test")

    assert len(result) == 1
    # In-memory version should win
    assert result[0].title == "From Memory"


# -- Task 2.3: Newly created in-memory session appears in results --


async def test_newly_created_in_memory_session_appears():
    """Session in SessionController but not in store should appear in results."""
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=[])
    store.load_sessions_batch = AsyncMock(return_value=[])

    new_session = _make_session("ses_new", directory="/test", title="New Session")
    controller_info = Mock()
    controller_info.session_id = "ses_new"

    state = _make_state(
        store=store,
        session_controller_sessions=[controller_info],
        cached_sessions={"ses_new": new_session},
    )

    result = await list_sessions(state, directory="/test")
    assert len(result) == 1
    assert result[0].id == "ses_new"


# -- Task 2.4: Python-level cwd filter catches sessions from other directories --


async def test_python_cwd_filter_catches_other_directories():
    """When store returns sessions from all cwds, Python filter excludes others."""
    # Simulate a store that doesn't filter by cwd
    store_data = [
        _make_session_data("ses_1", cwd="/test", title="Correct"),
        _make_session_data("ses_2", cwd="/other", title="Wrong"),
    ]
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=["ses_1", "ses_2"])  # ignores cwd
    store.load_sessions_batch = AsyncMock(return_value=store_data)

    state = _make_state(store=store, session_controller_sessions=[])

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.side_effect = lambda data: _make_session(
            data.session_id, directory=data.cwd or "", title=data.metadata.get("title", "")
        )
        result = await list_sessions(state, directory="/test")

    assert len(result) == 1
    assert result[0].id == "ses_1"


# -- Task 2.5: Legacy path unchanged --


async def test_legacy_path_when_no_session_controller():
    """When session_controller is None, legacy agent.list_sessions path is used."""
    store_data = [_make_session_data("ses_1", cwd="/test")]
    agent_list = AsyncMock(return_value=store_data)

    state = _make_state(store=None, agent_list_sessions=agent_list)
    state.session_controller = None

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.return_value = _make_session("ses_1", directory="/test")
        result = await list_sessions(state, directory="/test")

    assert len(result) == 1
    agent_list.assert_called_once()


# -- Task 2.6: cwd filter on store-first path --


async def test_cwd_filter_on_store_first_path():
    """SQL-level cwd filter: store.list_session_ids(cwd=...) is called with cwd."""
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=["ses_1"])
    store.load_sessions_batch = AsyncMock(return_value=[_make_session_data("ses_1", cwd="/test")])

    state = _make_state(store=store, session_controller_sessions=[])

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.return_value = _make_session("ses_1", directory="/test")
        await list_sessions(state, directory="/test")

    store.list_session_ids.assert_called_once_with(cwd="/test")


# -- Task 2.7: store is None returns in-memory only --


async def test_store_is_none_returns_in_memory_only():
    """When store is None, only in-memory sessions from SessionController are returned."""
    in_memory = _make_session("ses_1", directory="/test", title="In Memory")
    controller_info = Mock()
    controller_info.session_id = "ses_1"

    state = _make_state(
        store=None,
        session_controller_sessions=[controller_info],
        cached_sessions={"ses_1": in_memory},
    )

    result = await list_sessions(state, directory="/test")
    assert len(result) == 1
    assert result[0].id == "ses_1"


# -- Task 2.8: Store query failure degrades gracefully --


async def test_store_query_failure_degrades_gracefully():
    """When store query raises, endpoint falls back to in-memory sessions only."""
    store = Mock()
    store.list_session_ids = AsyncMock(side_effect=RuntimeError("DB locked"))

    in_memory = _make_session("ses_1", directory="/test", title="In Memory")
    controller_info = Mock()
    controller_info.session_id = "ses_1"

    state = _make_state(
        store=store,
        session_controller_sessions=[controller_info],
        cached_sessions={"ses_1": in_memory},
    )

    result = await list_sessions(state, directory="/test")
    assert len(result) == 1
    assert result[0].id == "ses_1"


# -- Task 2.9: Merged list sorted by recency --


async def test_merged_list_sorted_by_recency():
    """After merge, sessions are sorted by time.updated descending."""
    store_data = [
        _make_session_data(
            "ses_old",
            cwd="/test",
            title="Old",
            updated=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        _make_session_data(
            "ses_new",
            cwd="/test",
            title="New",
            updated=datetime(2024, 6, 1, tzinfo=UTC),
        ),
    ]
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=["ses_old", "ses_new"])
    store.load_sessions_batch = AsyncMock(return_value=store_data)

    state = _make_state(store=store, session_controller_sessions=[])

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.side_effect = lambda data: _make_session(
            data.session_id,
            directory=data.cwd or "/test",
            title=data.metadata.get("title", ""),
            updated_ms=int(data.last_active.timestamp() * 1000),
        )
        result = await list_sessions(state, directory="/test")

    assert len(result) == 2
    # Newer session should come first
    assert result[0].id == "ses_new"
    assert result[1].id == "ses_old"


# -- Task 2.10: In-memory-only sessions with different cwd not appended --


async def test_in_memory_only_different_cwd_not_appended():
    """In-memory-only session with different cwd is NOT appended."""
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=[])
    store.load_sessions_batch = AsyncMock(return_value=[])

    other_session = _make_session("ses_other", directory="/other", title="Other")
    controller_info = Mock()
    controller_info.session_id = "ses_other"

    state = _make_state(
        store=store,
        session_controller_sessions=[controller_info],
        cached_sessions={"ses_other": other_session},
    )

    result = await list_sessions(state, directory="/test")
    assert len(result) == 0


# -- Task 2.11: Empty store returns empty list --


async def test_empty_store_returns_empty():
    """When store has 0 sessions, returns empty list (or only in-memory if any)."""
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=[])
    store.load_sessions_batch = AsyncMock(return_value=[])

    state = _make_state(store=store, session_controller_sessions=[])

    result = await list_sessions(state, directory="/test")
    assert len(result) == 0


# -- Task 2.12: state.sessions cache populated --


async def test_state_sessions_cache_populated():
    """Store-loaded sessions are cached in state.sessions for subsequent get_or_load_session."""
    store_data = [_make_session_data("ses_1", cwd="/test", title="Cached")]
    store = Mock()
    store.list_session_ids = AsyncMock(return_value=["ses_1"])
    store.load_sessions_batch = AsyncMock(return_value=store_data)

    state = _make_state(store=store, session_controller_sessions=[])

    with patch(
        "wolfharness_server.opencode_server.routes.session_routes.session_data_to_opencode"
    ) as mock_convert:
        mock_convert.return_value = _make_session("ses_1", directory="/test")
        await list_sessions(state, directory="/test")

    # Session should be cached in state.sessions
    assert "ses_1" in state.sessions
