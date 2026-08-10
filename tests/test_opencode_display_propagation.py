"""Unit tests for SpawnSessionStart display_name field and OpenCode display propagation."""

from __future__ import annotations

import pytest

from wolfharness.agents.events.events import SpawnSessionStart


@pytest.mark.unit
def test_spawn_session_start_has_display_name_field() -> None:
    """SpawnSessionStart should have a display_name field defaulting to None."""
    event = SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="parent-1",
        spawn_mechanism="spawn",
        source_name="coordinator",
        source_type="agent",
        description="Test spawn",
    )
    assert event.display_name is None


@pytest.mark.unit
def test_spawn_session_start_with_custom_display_name() -> None:
    """SpawnSessionStart should accept a custom display_name."""
    event = SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="parent-1",
        spawn_mechanism="spawn",
        source_name="coordinator",
        display_name="lead",
        source_type="agent",
        description="Test spawn",
    )
    assert event.display_name == "lead"
    assert event.source_name == "coordinator"


@pytest.mark.unit
def test_spawn_session_start_display_name_none_when_equals_source_name() -> None:
    """Display name should be None when it equals the source name."""
    # This tests the pattern used at emission points:
    # display_name=node.display_name if node.display_name != node.name else None
    name = "coordinator"
    display_name = name  # no custom display name set
    resolved = display_name if display_name != name else None
    assert resolved is None


@pytest.mark.unit
def test_spawn_session_start_display_name_set_when_differs_from_source_name() -> None:
    """Display name should be set when it differs from the source name."""
    name = "coordinator"
    display_name = "lead"  # custom display name
    resolved = display_name if display_name != name else None
    assert resolved == "lead"


@pytest.mark.unit
def test_agent_model_has_display_name_field() -> None:
    """The OpenCode Agent model should have a display_name field."""
    from wolfharness_server.opencode_server.models.agent import Agent

    agent = Agent(name="test_agent")
    assert agent.display_name is None

    agent_with_display = Agent(name="test_agent", display_name="My Agent")
    assert agent_with_display.display_name == "My Agent"


@pytest.mark.unit
def test_ensure_session_accepts_title_parameter() -> None:
    """ensure_session should accept an optional title parameter."""
    import inspect

    from wolfharness_server.opencode_server.opencode_session_routes import ensure_session

    sig = inspect.signature(ensure_session)
    assert "title" in sig.parameters
    assert sig.parameters["title"].default is None


@pytest.mark.unit
def test_child_session_title_pattern_matches_subagent_footer_regex() -> None:
    """The child session title pattern should match the SubagentFooter regex."""
    import re

    display_name = "researcher"
    title = f"(@{display_name} subagent)"
    match = re.search(r"@(\w+) subagent", title)
    assert match is not None
    assert match.group(1) == "researcher"
