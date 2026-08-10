"""Unit tests for team-mode display enrichment.

Tests SpawnSessionStart metadata, ToolPart differentiation, and session title.
"""

from __future__ import annotations

import re

import pytest

from wolfharness.agents.events.events import SpawnSessionStart


def _make_team_spawn_event(
    display_name: str = "researcher",
    team_name: str = "sales",
    team_role: str = "member",
) -> SpawnSessionStart:
    """Create a SpawnSessionStart with team context metadata."""
    return SpawnSessionStart(
        child_session_id="child-1",
        parent_session_id="parent-1",
        spawn_mechanism="spawn",
        source_name="coordinator",
        display_name=display_name,
        source_type="agent",
        depth=1,
        description="Team member: researcher",
        metadata={
            "team_id": "team_abc123",
            "team_name": team_name,
            "team_role": team_role,
            "team_member_name": display_name,
        },
    )


def _make_regular_spawn_event() -> SpawnSessionStart:
    """Create a SpawnSessionStart without team context."""
    return SpawnSessionStart(
        child_session_id="child-2",
        parent_session_id="parent-2",
        spawn_mechanism="spawn",
        source_name="analyst",
        display_name=None,
        source_type="agent",
        depth=1,
        description="Subagent task",
        metadata={},
    )


@pytest.mark.unit
def test_team_spawn_event_includes_team_context_metadata() -> None:
    """Team member SpawnSessionStart should include team context metadata."""
    event = _make_team_spawn_event()
    assert event.metadata.get("team_id") == "team_abc123"
    assert event.metadata.get("team_name") == "sales"
    assert event.metadata.get("team_role") == "member"
    assert event.metadata.get("team_member_name") == "researcher"


@pytest.mark.unit
def test_team_member_toolpart_has_team_prefix() -> None:
    """Team member ToolPart subagent_type should have 'Team ·' prefix."""
    event = _make_team_spawn_event()
    display_name = event.display_name or event.source_name
    team_id = event.metadata.get("team_id")
    subagent_type = f"Team · {display_name}" if team_id is not None else display_name
    assert subagent_type == "Team · researcher"


@pytest.mark.unit
def test_regular_subagent_toolpart_no_team_prefix() -> None:
    """Regular subagent ToolPart subagent_type should NOT have 'Team ·' prefix."""
    event = _make_regular_spawn_event()
    display_name = event.display_name or event.source_name
    team_id = event.metadata.get("team_id")
    subagent_type = f"Team · {display_name}" if team_id is not None else display_name
    assert subagent_type == "analyst"
    assert not subagent_type.startswith("Team ·")


@pytest.mark.unit
def test_team_member_toolpart_description_includes_role_and_team() -> None:
    """Team member ToolPart description should include role and team name."""
    event = _make_team_spawn_event(team_role="member", team_name="sales")
    team_name = event.metadata.get("team_name", "")
    team_role = event.metadata.get("team_role", "member")
    role_label = "Lead" if team_role == "lead" else "Member"
    team_context = f"{role_label} in team '{team_name}' — " if team_name else ""
    description = f"{team_context}{event.description or event.display_name}"
    assert "Member" in description
    assert "sales" in description
    assert "Team member: researcher" in description


@pytest.mark.unit
def test_lead_toolpart_description_includes_lead_role() -> None:
    """Lead member ToolPart description should include 'Lead' role."""
    event = _make_team_spawn_event(team_role="lead")
    team_role = event.metadata.get("team_role", "member")
    role_label = "Lead" if team_role == "lead" else "Member"
    assert role_label == "Lead"


@pytest.mark.unit
def test_team_member_session_title_includes_team_and_role() -> None:
    """Team member child session title should include team name and role."""
    event = _make_team_spawn_event(display_name="researcher", team_name="sales", team_role="member")
    display_name = event.display_name or event.source_name
    team_name = event.metadata.get("team_name", "")
    team_role = event.metadata.get("team_role", "member")
    role_label = "Lead" if team_role == "lead" else "Member"
    team_prefix = f"Team '{team_name}' · {role_label} " if team_name else ""
    title = f"{team_prefix}(@{display_name} subagent)"
    assert "Team 'sales'" in title
    assert "Member" in title
    assert "@researcher" in title


@pytest.mark.unit
def test_regular_subagent_session_title_no_team_context() -> None:
    """Regular subagent session title should not include team context."""
    event = _make_regular_spawn_event()
    display_name = event.display_name or event.source_name
    team_id = event.metadata.get("team_id")
    if team_id is not None:
        title = f"Team prefix (@{display_name} subagent)"
    else:
        title = f"(@{display_name} subagent)"
    assert title == "(@analyst subagent)"
    assert "Team" not in title


@pytest.mark.unit
def test_subagent_footer_regex_extracts_display_name_from_team_title() -> None:
    r"""SubagentFooter regex @(\w+) subagent should extract display_name from team session title."""
    event = _make_team_spawn_event(display_name="researcher", team_name="sales", team_role="member")
    display_name = event.display_name or event.source_name
    team_name = event.metadata.get("team_name", "")
    team_role = event.metadata.get("team_role", "member")
    role_label = "Lead" if team_role == "lead" else "Member"
    team_prefix = f"Team '{team_name}' · {role_label} " if team_name else ""
    title = f"{team_prefix}(@{display_name} subagent)"
    match = re.search(r"@(\w+) subagent", title)
    assert match is not None
    assert match.group(1) == "researcher"
