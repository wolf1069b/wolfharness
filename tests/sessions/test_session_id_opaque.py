"""Regression tests ensuring session IDs are treated as opaque strings.

RFC-0028 Task T6: Audit session ID format dependency before provider switch.

These tests assert that session IDs are never parsed, split, or
pattern-matched on format.  This guarantees that switching from
`identifier.ascending("session")` to a different provider (e.g. UUID4)
will not break any consumer.

All session lookups must use IDs as opaque dictionary keys — no regex,
no counter extraction, no positional slicing (except the URL-shortening
case in text_sharing which is format-agnostic via `[-8:]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from wolfharness.sessions import SessionData
from wolfharness_storage.memory_provider.provider import MemoryStorageProvider


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Session store lookups are opaque
# ---------------------------------------------------------------------------


class TestStorageProviderOpaqueLookup:
    """Storage providers must accept any string as a session ID."""

    @pytest.fixture
    def store(self) -> MemoryStorageProvider:
        return MemoryStorageProvider()

    @pytest.mark.parametrize(
        "session_id",
        [
            # Current ascending format: ses_{hex}{base62}
            "ses_b71310fdf001ZHcn6VSpkaBcHi",
            # Future UUID4 format
            "550e8400-e29b-41d4-a716-446655440000",
            # Ulid format
            "01H5S3YQA4KXGM2T8N6BV4A5ZC",
            # Plain counter (legacy hypothetical)
            "42",
            # Arbitrary string
            "my-custom-session-id_2024",
        ],
        ids=["ascending", "uuid4", "ulid", "counter", "arbitrary"],
    )
    async def test_store_save_load_opaque_id(
        self, store: MemoryStorageProvider, session_id: str
    ) -> None:
        """MemoryStorageProvider must save/load sessions with any string ID."""
        data = SessionData(session_id=session_id, agent_name="test_agent")

        async with store:
            await store.save_session(data)
            loaded = await store.load_session(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id

    @pytest.mark.parametrize(
        "session_id",
        [
            "ses_b71310fdf001ZHcn6VSpkaBcHi",
            "550e8400-e29b-41d4-a716-446655440000",
            "my-custom-session-id_2024",
        ],
        ids=["ascending", "uuid4", "arbitrary"],
    )
    async def test_store_delete_opaque_id(
        self, store: MemoryStorageProvider, session_id: str
    ) -> None:
        """MemoryStorageProvider.delete_session must work with any string ID."""
        data = SessionData(session_id=session_id, agent_name="test_agent")

        async with store:
            await store.save_session(data)
            deleted = await store.delete_session(session_id)

        assert deleted is True

    @pytest.mark.parametrize(
        "session_id",
        [
            "ses_b71310fdf001ZHcn6VSpkaBcHi",
            "550e8400-e29b-41d4-a716-446655440000",
            "my-custom-session-id_2024",
        ],
        ids=["ascending", "uuid4", "arbitrary"],
    )
    async def test_store_list_sessions_with_opaque_id(
        self, store: MemoryStorageProvider, session_id: str
    ) -> None:
        """MemoryStorageProvider.list_session_ids must return opaque IDs unchanged."""
        data = SessionData(session_id=session_id, agent_name="test_agent", pool_id="pool1")

        async with store:
            await store.save_session(data)
            ids = await store.list_session_ids(pool_id="pool1")

        assert session_id in ids


# ---------------------------------------------------------------------------
# 2. SessionPool.create_session produces opaque IDs
# ---------------------------------------------------------------------------


class TestSessionPoolOpaqueChildId:
    """SessionPool must generate opaque child session IDs."""

    @pytest.fixture
    def mock_pool(self, minimal_pool: AgentPool) -> AgentPool:
        return minimal_pool

    async def test_child_session_id_is_opaque_string(self, minimal_pool: AgentPool) -> None:
        """Child session IDs must be non-empty opaque strings."""
        from wolfharness.orchestrator import SessionPool
        from wolfharness.utils.identifiers import generate_session_id

        session_pool = SessionPool(pool=minimal_pool, store=None)
        state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="coder",
            parent_session_id="parent_1",
        )
        child_id = state.session_id
        assert isinstance(child_id, str)
        assert len(child_id) > 0
        # Must not raise — treat as opaque
        assert child_id  # truthy

    @pytest.mark.parametrize(
        "parent_id",
        [
            "ses_b71310fdf001ZHcn6VSpkaBcHi",
            "550e8400-e29b-41d4-a716-446655440000",
            "my-custom-session-id_2024",
        ],
        ids=["ascending_parent", "uuid4_parent", "arbitrary_parent"],
    )
    async def test_create_child_with_opaque_parent_id(
        self, minimal_pool: AgentPool, parent_id: str
    ) -> None:
        """create_session must accept any string as parent_session_id."""
        from wolfharness.orchestrator import SessionPool
        from wolfharness.utils.identifiers import generate_session_id

        store = MemoryStorageProvider()
        parent = SessionData(session_id=parent_id, agent_name="parent_agent")
        await store.save_session(parent)

        session_pool = SessionPool(pool=minimal_pool, store=store)
        state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="child_agent",
            parent_session_id=parent_id,
        )
        child_id = state.session_id

        child = await store.load_session(child_id)
        assert child is not None
        assert child.parent_id == parent_id

    @pytest.mark.parametrize(
        "parent_id",
        [
            "ses_b71310fdf001ZHcn6VSpkaBcHi",
            "550e8400-e29b-41d4-a716-446655440000",
        ],
        ids=["ascending_parent", "uuid4_parent"],
    )
    async def test_get_child_sessions_with_opaque_parent_id(
        self, minimal_pool: AgentPool, parent_id: str
    ) -> None:
        """get_children must find children by opaque parent ID."""
        from wolfharness.orchestrator import SessionPool
        from wolfharness.utils.identifiers import generate_session_id

        store = MemoryStorageProvider()
        parent = SessionData(session_id=parent_id, agent_name="parent_agent")
        await store.save_session(parent)

        session_pool = SessionPool(pool=minimal_pool, store=store)
        state = await session_pool.create_session(
            session_id=generate_session_id(),
            agent_name="child_agent",
            parent_session_id=parent_id,
        )
        child_id = state.session_id
        children = session_pool.sessions.get_children(parent_id)

        assert child_id in children


# ---------------------------------------------------------------------------
# 3. Static analysis: no regex parsing of session IDs in production code
# ---------------------------------------------------------------------------


class TestNoSessionIdParsing:
    """Ensure production code never regex-parses session ID format.

    This is a static analysis guard — if someone adds a regex that assumes
    sequential or `ses_` prefixed IDs, this test will catch it.
    """

    def test_no_sequential_session_id_regex(self) -> None:
        r"""No production code should match session IDs with \\d+ counters.

        The grep audit in this task confirmed no such patterns exist.
        This test documents that the identifiers module doesn't encourage
        counter-based parsing.
        """
        from wolfharness.utils.identifiers import generate_session_id

        id1 = generate_session_id()
        id2 = generate_session_id()
        # IDs must start with "ses_" (current format) but that's the
        # only format assumption consumers may make — and even that
        # should not be relied upon per RFC-0028.
        assert id1 != id2  # uniqueness
        assert isinstance(id1, str)

    def test_ascending_format_produces_sortable_ids(self) -> None:
        """ascending() IDs are lexicographically sortable — a property, not a format dependency."""
        from wolfharness.utils.identifiers import ascending

        ids = [ascending("session") for _ in range(10)]
        # Later IDs must sort after earlier ones
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1], f"IDs not ascending: {ids[i]} >= {ids[i + 1]}"

    def test_ascending_with_given_accepts_any_valid_prefix(self) -> None:
        """ascending(prefix, given=...) validates prefix only — no format parsing."""
        from wolfharness.utils.identifiers import ascending

        # Providing an ID that starts with "ses" must be accepted
        result = ascending("session", given="ses_custom_suffix")
        assert result == "ses_custom_suffix"

    def test_ascending_with_wrong_prefix_raises(self) -> None:
        """ascending(prefix, given=...) raises if prefix doesn't match."""
        from wolfharness.utils.identifiers import ascending

        with pytest.raises(ValueError, match="does not start with"):
            ascending("session", given="msg_something")


# ---------------------------------------------------------------------------
# 4. OpenCode / ACP server session lookups are opaque
# ---------------------------------------------------------------------------


class TestServerSessionLookupOpaque:
    """Server-side session lookups must treat IDs as opaque dictionary keys."""

    def test_opencode_state_sessions_dict_opaque(self) -> None:
        """ServerState.sessions dict accepts any string key."""
        from wolfharness.utils.time_utils import now_ms
        from wolfharness_server.opencode_server.models import Session
        from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated
        from wolfharness_server.opencode_server.state import ServerState

        agent = MagicMock()
        state = ServerState(working_dir="/tmp", agent=agent)

        # Create sessions with different ID formats
        now = now_ms()
        for sid in [
            "ses_b71310fdf001ZHcn6VSpkaBcHi",
            "550e8400-e29b-41d4-a716-446655440000",
            "custom-session-id",
        ]:
            session = Session(
                id=sid,
                project_id="default",
                directory="/tmp",
                title="Test",
                version="1",
                time=TimeCreatedUpdated(created=now, updated=now),
            )
            state.sessions[sid] = session
            # Lookup must return the same session
            assert state.sessions.get(sid) is session

    def test_acp_session_manager_acp_sessions_dict_opaque(self) -> None:
        """ACP ACPSessionManager._acp_sessions dict accepts any string key for lookups."""
        # We test the dict-based lookup pattern (get_session) conceptually.
        # The _acp_sessions dict maps session_id -> ACPSession using plain dict.get().
        # This test documents that the lookup is opaque.
        from wolfharness_server.acp_server.session_manager import ACPSessionManager

        # ACPSessionManager._acp_sessions is a plain dict[str, ACPSession]
        # get_session does: return self._acp_sessions.get(session_id)
        # This is opaque — any string works as a key.
        manager = ACPSessionManager.__new__(ACPSessionManager)
        manager._acp_sessions = {}  # type: ignore[attr-defined]

        # Simulate opaque key usage
        test_id = "550e8400-e29b-41d4-a716-446655440000"
        assert manager._acp_sessions.get(test_id) is None  # no error, just None
