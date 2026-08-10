"""Unit tests for McpConfigEntry and McpConfigSnapshot."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

from pydantic import HttpUrl
import pytest

from wolfharness.mcp_server.config_snapshot import McpConfigEntry, McpConfigSnapshot
from wolfharness_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stdio_config() -> StdioMCPServerConfig:
    return StdioMCPServerConfig(name="stdio-srv", command="python", args=["-m", "srv"])


@pytest.fixture
def sse_config() -> SSEMCPServerConfig:
    return SSEMCPServerConfig(name="sse-srv", url=HttpUrl("http://localhost:8080/sse"))


@pytest.fixture
def http_config() -> StreamableHTTPMCPServerConfig:
    return StreamableHTTPMCPServerConfig(
        name="http-srv", url=HttpUrl("https://api.example.com/mcp")
    )


@pytest.fixture
def pool_entry(stdio_config: StdioMCPServerConfig) -> McpConfigEntry:
    return McpConfigEntry(server_config=stdio_config, source="pool")


@pytest.fixture
def agent_entry(sse_config: SSEMCPServerConfig) -> McpConfigEntry:
    return McpConfigEntry(server_config=sse_config, source="agent")


@pytest.fixture
def session_entry(http_config: StreamableHTTPMCPServerConfig) -> McpConfigEntry:
    return McpConfigEntry(server_config=http_config, source="session")


@pytest.fixture
def skill_entry(stdio_config: StdioMCPServerConfig) -> McpConfigEntry:
    return McpConfigEntry(server_config=stdio_config, source="skill", skill_name="my-skill")


# ---------------------------------------------------------------------------
# McpConfigEntry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_entry_creation_with_defaults(stdio_config: StdioMCPServerConfig) -> None:
    """McpConfigEntry can be created with server_config and source only."""
    entry = McpConfigEntry(server_config=stdio_config, source="pool")
    assert entry.server_config is stdio_config
    assert entry.source == "pool"
    assert entry.skill_name is None


@pytest.mark.unit
def test_entry_creation_with_skill_name(stdio_config: StdioMCPServerConfig) -> None:
    """McpConfigEntry stores skill_name when provided."""
    entry = McpConfigEntry(server_config=stdio_config, source="skill", skill_name="my-skill")
    assert entry.skill_name == "my-skill"


@pytest.mark.unit
def test_entry_is_frozen(stdio_config: StdioMCPServerConfig) -> None:
    """McpConfigEntry is frozen — attribute assignment raises FrozenInstanceError."""
    entry = McpConfigEntry(server_config=stdio_config, source="pool")
    with pytest.raises(FrozenInstanceError):
        entry.source = "agent"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# McpConfigSnapshot — creation and properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_snapshot() -> None:
    """Default snapshot has all empty tuples."""
    snapshot = McpConfigSnapshot()
    assert snapshot.pool_configs == ()
    assert snapshot.agent_configs == ()
    assert snapshot.session_configs == ()
    assert snapshot.skill_configs == ()
    assert snapshot.all_configs == ()
    assert snapshot.global_configs == ()
    assert snapshot.session_scoped_configs == ()


@pytest.mark.unit
def test_all_configs(
    pool_entry: McpConfigEntry,
    agent_entry: McpConfigEntry,
    session_entry: McpConfigEntry,
    skill_entry: McpConfigEntry,
) -> None:
    """all_configs returns entries in canonical order: pool, agent, session, skill."""
    snapshot = McpConfigSnapshot(
        pool_configs=(pool_entry,),
        agent_configs=(agent_entry,),
        session_configs=(session_entry,),
        skill_configs=(skill_entry,),
    )
    assert snapshot.all_configs == (pool_entry, agent_entry, session_entry, skill_entry)


@pytest.mark.unit
def test_global_configs(
    pool_entry: McpConfigEntry,
    agent_entry: McpConfigEntry,
    session_entry: McpConfigEntry,
    skill_entry: McpConfigEntry,
) -> None:
    """global_configs returns pool + agent entries only."""
    snapshot = McpConfigSnapshot(
        pool_configs=(pool_entry,),
        agent_configs=(agent_entry,),
        session_configs=(session_entry,),
        skill_configs=(skill_entry,),
    )
    assert snapshot.global_configs == (pool_entry, agent_entry)


@pytest.mark.unit
def test_session_scoped_configs(
    pool_entry: McpConfigEntry,
    agent_entry: McpConfigEntry,
    session_entry: McpConfigEntry,
    skill_entry: McpConfigEntry,
) -> None:
    """session_scoped_configs returns session + skill entries only."""
    snapshot = McpConfigSnapshot(
        pool_configs=(pool_entry,),
        agent_configs=(agent_entry,),
        session_configs=(session_entry,),
        skill_configs=(skill_entry,),
    )
    assert snapshot.session_scoped_configs == (session_entry, skill_entry)


@pytest.mark.unit
def test_snapshot_is_frozen(pool_entry: McpConfigEntry) -> None:
    """McpConfigSnapshot is frozen."""
    snapshot = McpConfigSnapshot(pool_configs=(pool_entry,))
    with pytest.raises(FrozenInstanceError):
        snapshot.pool_configs = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# McpConfigSnapshot — with_skill_configs / with_session_configs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_with_skill_configs_returns_new_snapshot(
    pool_entry: McpConfigEntry,
    skill_entry: McpConfigEntry,
) -> None:
    """with_skill_configs returns a new snapshot with updated skill_configs."""
    original = McpConfigSnapshot(pool_configs=(pool_entry,))
    new_skills = (skill_entry,)

    updated = original.with_skill_configs(new_skills)

    assert updated is not original
    assert updated.skill_configs == new_skills
    # Original is unchanged (immutable)
    assert original.skill_configs == ()
    # Non-skill configs are preserved
    assert updated.pool_configs == (pool_entry,)
    assert updated.agent_configs == ()
    assert updated.session_configs == ()


@pytest.mark.unit
def test_with_session_configs_returns_new_snapshot(
    pool_entry: McpConfigEntry,
    agent_entry: McpConfigEntry,
    session_entry: McpConfigEntry,
    skill_entry: McpConfigEntry,
) -> None:
    """with_session_configs returns a new snapshot with updated session_configs."""
    original = McpConfigSnapshot(
        pool_configs=(pool_entry,),
        agent_configs=(agent_entry,),
        skill_configs=(skill_entry,),
    )
    new_sessions = (session_entry,)

    updated = original.with_session_configs(new_sessions)

    assert updated is not original
    assert updated.session_configs == new_sessions
    # Original is unchanged
    assert original.session_configs == ()
    # Other configs are preserved
    assert updated.pool_configs == (pool_entry,)
    assert updated.agent_configs == (agent_entry,)
    assert updated.skill_configs == (skill_entry,)


@pytest.mark.unit
def test_with_skill_configs_replaces_existing(
    pool_entry: McpConfigEntry,
    skill_entry: McpConfigEntry,
) -> None:
    """with_skill_configs replaces existing skill configs, not appends."""
    original = McpConfigSnapshot(skill_configs=(skill_entry,))
    new_skills: tuple[McpConfigEntry, ...] = ()

    updated = original.with_skill_configs(new_skills)

    assert updated.skill_configs == ()
    assert original.skill_configs == (skill_entry,)


@pytest.mark.unit
def test_with_session_configs_replaces_existing(
    session_entry: McpConfigEntry,
) -> None:
    """with_session_configs replaces existing session configs, not appends."""
    original = McpConfigSnapshot(session_configs=(session_entry,))
    new_sessions: tuple[McpConfigEntry, ...] = ()

    updated = original.with_session_configs(new_sessions)

    assert updated.session_configs == ()
    assert original.session_configs == (session_entry,)
