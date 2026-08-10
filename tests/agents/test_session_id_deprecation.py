"""Tests for AgentRunContext.session_id as a first-class field."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from wolfharness.agents.context import AgentRunContext


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# First-class field behaviour
# ---------------------------------------------------------------------------


def test_session_id_get_no_warning() -> None:
    """Accessing session_id on an instance does not emit any warning."""
    ctx = AgentRunContext()
    sid = ctx.session_id
    assert isinstance(sid, str)


def test_session_id_set_no_warning() -> None:
    """Setting session_id on an instance does not emit any warning."""
    ctx = AgentRunContext()
    ctx.session_id = "custom-id"
    assert ctx.session_id == "custom-id"


def test_session_id_get_returns_uuid_by_default() -> None:
    """Default session_id is a UUID hex string."""
    ctx = AgentRunContext()
    sid = ctx.session_id
    assert isinstance(sid, str)
    assert len(sid) == 32  # uuid4.hex is 32 chars


def test_session_id_set_and_get_roundtrip() -> None:
    """Set value persists and is returned on subsequent get."""
    ctx = AgentRunContext()
    ctx.session_id = "my-session"
    assert ctx.session_id == "my-session"


def test_session_id_independent_per_instance() -> None:
    """Each AgentRunContext instance gets its own session_id."""
    ctx_a = AgentRunContext()
    ctx_b = AgentRunContext()
    assert ctx_a.session_id != ctx_b.session_id


def test_class_level_access_returns_str_type() -> None:
    """Accessing session_id on the class returns the field descriptor (str type)."""
    from dataclasses import Field

    field_info = AgentRunContext.__dataclass_fields__["session_id"]
    assert isinstance(field_info, Field)


# ---------------------------------------------------------------------------
# asdict() compatibility
# ---------------------------------------------------------------------------


def test_asdict_includes_session_id() -> None:
    """dataclasses.asdict() includes session_id in the result."""
    ctx = AgentRunContext()
    d = asdict(ctx)
    assert "session_id" in d
    assert isinstance(d["session_id"], str)


def test_asdict_session_id_value_matches_direct_access() -> None:
    """Value from asdict() matches direct getattr access."""
    ctx = AgentRunContext()
    ctx.session_id = "asdict-test"
    d = asdict(ctx)
    assert d["session_id"] == "asdict-test"


# ---------------------------------------------------------------------------
# Other fields unaffected
# ---------------------------------------------------------------------------


def test_other_fields_unaffected() -> None:
    """Non-session_id fields work normally."""
    ctx = AgentRunContext(depth=3, deps={"key": "val"})
    assert ctx.depth == 3
    assert ctx.deps == {"key": "val"}
    assert ctx.cancelled is False


# ---------------------------------------------------------------------------
# session_id in __init__ signature
# ---------------------------------------------------------------------------


def test_session_id_in_init() -> None:
    """session_id is in __init__ and can be passed explicitly."""
    import inspect

    sig = inspect.signature(AgentRunContext)
    assert "session_id" in sig.parameters
    ctx = AgentRunContext(session_id="explicit-id")
    assert ctx.session_id == "explicit-id"


# ---------------------------------------------------------------------------
# event_bus field
# ---------------------------------------------------------------------------


def test_event_bus_defaults_to_none() -> None:
    """event_bus defaults to None."""
    ctx = AgentRunContext()
    assert ctx.event_bus is None


def test_event_bus_can_be_set() -> None:
    """event_bus can be set to a value."""
    ctx = AgentRunContext(event_bus="fake-bus")  # type: ignore[arg-type]
    assert ctx.event_bus == "fake-bus"


def test_event_bus_in_init() -> None:
    """event_bus is in __init__ signature."""
    import inspect

    sig = inspect.signature(AgentRunContext)
    assert "event_bus" in sig.parameters
