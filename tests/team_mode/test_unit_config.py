"""Unit tests for TeamModeConfig and resolve_team_mode."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from wolfharness_config.team_mode import (
    BlackboardConfig,
    MemberSpec,
    TeamBounds,
    TeamDefaultsConfig,
    TeamModeConfig,
    resolve_team_mode,
)


@pytest.mark.unit
def test_construct_with_all_defaults() -> None:
    """Given no arguments, TeamModeConfig uses all default values."""
    config = TeamModeConfig()

    assert config.enabled is False
    assert config.member_eligible == []
    assert config.lead_eligible == []
    assert config.base_dir is None
    assert config.ttl_hours == 72
    assert config.bounds == TeamBounds()
    assert config.blackboard == BlackboardConfig()
    assert config.message_max_bytes == 65536
    assert config.inbox_max_bytes == 1048576
    assert config.notice_delivery_mode == "steer"
    assert config.defaults is None


@pytest.mark.unit
def test_construct_with_custom_values() -> None:
    """Given custom values, TeamModeConfig stores them correctly."""
    bounds = TeamBounds(
        max_members=3,
        max_parallel_members=2,
        max_wall_clock_minutes=15,
        max_member_turns=10,
    )
    blackboard = BlackboardConfig(write_policy="lead_only", max_size_mb=50)
    defaults = TeamDefaultsConfig(
        team_name="squad_alpha",
        members=[MemberSpec(name="coder", agent="coder")],
    )

    config = TeamModeConfig(
        enabled=True,
        member_eligible=["coder", "reviewer"],
        lead_eligible=["coordinator"],
        base_dir="/tmp/teams",
        ttl_hours=24,
        bounds=bounds,
        blackboard=blackboard,
        message_max_bytes=32768,
        inbox_max_bytes=524288,
        protocol_template="Custom {team_name} {role} {member_name}",
        notice_delivery_mode="queue",
        defaults=defaults,
    )

    assert config.enabled is True
    assert config.member_eligible == ["coder", "reviewer"]
    assert config.lead_eligible == ["coordinator"]
    assert config.base_dir == "/tmp/teams"
    assert config.ttl_hours == 24
    assert config.bounds.max_members == 3
    assert config.bounds.max_parallel_members == 2
    assert config.bounds.max_wall_clock_minutes == 15
    assert config.bounds.max_member_turns == 10
    assert config.blackboard.write_policy == "lead_only"
    assert config.blackboard.max_size_mb == 50
    assert config.message_max_bytes == 32768
    assert config.inbox_max_bytes == 524288
    assert config.protocol_template == "Custom {team_name} {role} {member_name}"
    assert config.notice_delivery_mode == "queue"
    assert config.defaults is not None
    assert config.defaults.team_name == "squad_alpha"
    assert config.defaults.members[0].name == "coder"
    assert config.defaults.members[0].agent == "coder"


@pytest.mark.unit
def test_construct_with_negative_bounds_raises_validation_error() -> None:
    """Given negative bounds, ValidationError is raised."""
    with pytest.raises(ValidationError) as exc_info:
        TeamBounds(max_members=-1)

    errors = exc_info.value.errors()
    assert len(errors) >= 1
    assert any("max_members" in str(e.get("loc", "")) for e in errors)


@pytest.mark.unit
def test_defaults_with_member_in_eligible_succeeds() -> None:
    """Given defaults members in member_eligible, construction succeeds."""
    defaults = TeamDefaultsConfig(
        team_name="translation_team",
        members=[
            MemberSpec(name="translator", agent="translator"),
            MemberSpec(name="reviewer", agent="reviewer"),
        ],
    )

    config = TeamModeConfig(
        enabled=True,
        member_eligible=["translator", "reviewer"],
        defaults=defaults,
    )

    assert config.defaults is not None
    assert len(config.defaults.members) == 2


@pytest.mark.unit
def test_defaults_with_member_not_in_eligible_raises_validation_error() -> None:
    """Given defaults member not in member_eligible, ValidationError is raised."""
    defaults = TeamDefaultsConfig(
        team_name="bad_team",
        members=[MemberSpec(name="outsider", agent="outsider_agent")],
    )

    with pytest.raises(ValidationError) as exc_info:
        TeamModeConfig(
            enabled=True,
            member_eligible=["translator", "reviewer"],
            defaults=defaults,
        )

    assert "outsider" in str(exc_info.value)


@pytest.mark.unit
def test_resolve_team_mode_both_none_returns_none() -> None:
    """Given both configs None, resolve_team_mode returns None."""
    result = resolve_team_mode(None, None)
    assert result is None


@pytest.mark.unit
def test_resolve_team_mode_global_only_returns_global() -> None:
    """Given only global config, resolve_team_mode returns it."""
    global_config = TeamModeConfig(enabled=True, member_eligible=["agent_a"])

    result = resolve_team_mode(global_config, None)

    assert result is global_config


@pytest.mark.unit
def test_resolve_team_mode_both_merges_agent_overrides() -> None:
    """Given both configs, agent-specific fields override global."""
    global_config = TeamModeConfig(
        enabled=True,
        member_eligible=["translator", "reviewer"],
        lead_eligible=["coordinator"],
        base_dir="/tmp/global",
        ttl_hours=48,
        message_max_bytes=32768,
    )

    agent_config = TeamModeConfig(
        enabled=True,
        member_eligible=["translator", "reviewer"],
        base_dir="/tmp/agent",
        ttl_hours=24,
    )

    result = resolve_team_mode(global_config, agent_config)

    assert result is not None
    # Agent overrides
    assert result.base_dir == "/tmp/agent"
    assert result.ttl_hours == 24
    # Inherited from global
    assert result.lead_eligible == ["coordinator"]
    assert result.message_max_bytes == 32768


@pytest.mark.unit
def test_protocol_template_renders_with_placeholders() -> None:
    """Given the default protocol_template, str.format renders all placeholders."""
    config = TeamModeConfig()

    rendered = config.protocol_template.format(
        team_name="alpha_squad",
        role="lead",
        member_name="coordinator",
    )

    assert "alpha_squad" in rendered
    assert "lead" in rendered
    assert "coordinator" in rendered


@pytest.mark.unit
def test_effective_base_dir_defaults_to_tempdir() -> None:
    """Given base_dir=None, effective_base_dir returns system temp dir."""
    import tempfile

    config = TeamModeConfig()

    assert config.effective_base_dir == tempfile.gettempdir()


# ------------------------------------------------------------------
# 1.3 MemberSpec.instructions validation
# ------------------------------------------------------------------


@pytest.mark.unit
def test_member_spec_instructions_default_empty_string() -> None:
    """Given no instructions, MemberSpec.instructions defaults to empty string."""
    member = MemberSpec(name="translator", agent="translator")

    assert member.instructions == ""


@pytest.mark.unit
def test_member_spec_instructions_non_empty_string() -> None:
    """Given a non-empty instructions string, MemberSpec stores it."""
    member = MemberSpec(
        name="translator",
        agent="translator",
        instructions="Translate all API docs to French.",
    )

    assert member.instructions == "Translate all API docs to French."


@pytest.mark.unit
def test_member_spec_instructions_multiline_string() -> None:
    """Given a multiline instructions string, MemberSpec stores it verbatim."""
    instructions = (
        "You are responsible for:\n"
        "- Translating API docs\n"
        "- Reviewing translations\n"
        "- Reporting progress daily"
    )
    member = MemberSpec(name="translator", agent="translator", instructions=instructions)

    assert member.instructions == instructions
    assert "\n" in member.instructions
    assert "- Translating API docs" in member.instructions


# ------------------------------------------------------------------
# 2.6 Protocol template rendering
# ------------------------------------------------------------------


@pytest.mark.unit
def test_protocol_template_includes_tasks_subsection() -> None:
    """Given the default protocol template, it includes a ### Tasks subsection."""
    config = TeamModeConfig()

    rendered = config.protocol_template.format(
        team_name="alpha",
        role="lead",
        member_name="coordinator",
    )

    assert "### Tasks" in rendered


@pytest.mark.unit
def test_protocol_template_includes_blackboard_subsection() -> None:
    """Given the default protocol template, it includes a ### Blackboard subsection."""
    config = TeamModeConfig()

    rendered = config.protocol_template.format(
        team_name="alpha",
        role="lead",
        member_name="coordinator",
    )

    assert "### Blackboard" in rendered


@pytest.mark.unit
def test_protocol_template_includes_messages_subsection() -> None:
    """Given the default protocol template, it includes a ### Messages subsection."""
    config = TeamModeConfig()

    rendered = config.protocol_template.format(
        team_name="alpha",
        role="lead",
        member_name="coordinator",
    )

    assert "### Messages" in rendered


@pytest.mark.unit
def test_protocol_template_has_no_guidelines_section() -> None:
    """Given the default protocol template, it does NOT contain a ## Guidelines section."""
    config = TeamModeConfig()

    rendered = config.protocol_template.format(
        team_name="alpha",
        role="lead",
        member_name="coordinator",
    )

    assert "## Guidelines" not in rendered


@pytest.mark.unit
def test_member_spec_skills_serializes_and_accepts_uri() -> None:
    """MemberSpec.skills validates as list[str], supports skill:// URIs."""
    spec = MemberSpec(
        name="translator", agent="translator", skills=["lodestone", "skill://foo/refs/g.md"]
    )
    assert spec.skills == ["lodestone", "skill://foo/refs/g.md"]


@pytest.mark.unit
def test_member_spec_skills_defaults_to_empty() -> None:
    """MemberSpec.skills defaults to [] when omitted."""
    spec = MemberSpec(name="translator", agent="translator")
    assert spec.skills == []


@pytest.mark.unit
def test_resolve_team_mode_keeps_defaults_member_skills() -> None:
    """Agent-level defaults.members[].skills survives the merge."""
    global_config = TeamModeConfig(
        enabled=True,
        member_eligible=["a"],
        defaults=TeamDefaultsConfig(
            team_name="g",
            members=[MemberSpec(name="a", agent="a")],
        ),
    )
    agent_config = TeamModeConfig(
        enabled=True,
        member_eligible=["a"],
        defaults=TeamDefaultsConfig(
            team_name="g",
            members=[
                MemberSpec(name="a", agent="a", skills=["lodestone"]),
            ],
        ),
    )

    merged = resolve_team_mode(global_config, agent_config)

    assert merged is not None
    assert merged.defaults is not None
    assert merged.defaults.members[0].skills == ["lodestone"]
