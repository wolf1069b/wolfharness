"""Tests for the teams → graph translation layer."""

from __future__ import annotations

import pytest

from wolfharness_config import (
    GraphConfig,
    GraphEdgeConfig,
    GraphStepConfig,
    TeamConfig,
    TeamMemberConfig,
    translate_config_to_graph,
    translate_connections_to_edges,
    translate_team_to_graph,
    translate_teams_to_graphs,
)
from wolfharness_config.forward_targets import (
    CallableConnectionConfig,
    FileConnectionConfig,
    NodeConnectionConfig,
)
from wolfharness_config.nodes import NodeConfig


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


def make_sequential_team(
    name: str = "review_pipeline",
    members: list[str] | None = None,
    shared_prompt: str | None = None,
    member_timeout: float | None = None,
) -> TeamConfig:
    """Create a sequential TeamConfig for testing."""
    return TeamConfig(
        name=name,
        mode="sequential",
        members=members or ["analyzer", "reviewer", "formatter"],
        shared_prompt=shared_prompt,
        member_timeout=member_timeout,
    )


def make_parallel_team(
    name: str = "parallel_coders",
    members: list[str] | None = None,
    shared_prompt: str | None = None,
) -> TeamConfig:
    """Create a parallel TeamConfig for testing."""
    return TeamConfig(
        name=name,
        mode="parallel",
        members=members or ["claude", "goose"],
        shared_prompt=shared_prompt,
    )


def make_team_with_member_configs(
    name: str = "mixed_team",
    mode: str = "sequential",
) -> TeamConfig:
    """Create a TeamConfig with TeamMemberConfig objects (prompt_template)."""
    return TeamConfig(
        name=name,
        mode=mode,
        members=[
            "agent_a",
            TeamMemberConfig(name="agent_b", prompt_template="Review: {{ prompt }}"),
            TeamMemberConfig(name="agent_c", prompt_template=None),
        ],
        shared_prompt="Work together",
        member_timeout=60.0,
    )


# =============================================================================
# Sequential team translation tests
# =============================================================================


class TestSequentialTranslation:
    """Test translation of sequential teams to graph config."""

    def test_sequential_team_basic(self) -> None:
        """Sequential team produces chained steps with linear edges."""
        team = make_sequential_team()
        graph = translate_team_to_graph("review_pipeline", team)

        assert graph.name == "review_pipeline"
        assert len(graph.steps) == 3
        assert [s.id for s in graph.steps] == ["analyzer", "reviewer", "formatter"]
        assert [s.agent for s in graph.steps] == ["analyzer", "reviewer", "formatter"]

    def test_sequential_team_edges(self) -> None:
        """Sequential team has start→s1→s2→s3→end edges (4 edges for 3 members)."""
        team = make_sequential_team()
        graph = translate_team_to_graph("review_pipeline", team)

        assert len(graph.edges) == 4
        # start → analyzer
        assert graph.edges[0].from_ == "start"
        assert graph.edges[0].to == "analyzer"
        # analyzer → reviewer
        assert graph.edges[1].from_ == "analyzer"
        assert graph.edges[1].to == "reviewer"
        # reviewer → formatter
        assert graph.edges[2].from_ == "reviewer"
        assert graph.edges[2].to == "formatter"
        # formatter → end
        assert graph.edges[3].from_ == "formatter"
        assert graph.edges[3].to == "end"

    def test_sequential_team_single_member(self) -> None:
        """Sequential team with one member has start→s1→end (2 edges)."""
        team = make_sequential_team(members=["solo"])
        graph = translate_team_to_graph("single", team)

        assert len(graph.steps) == 1
        assert len(graph.edges) == 2
        assert graph.edges[0].from_ == "start"
        assert graph.edges[0].to == "solo"
        assert graph.edges[1].from_ == "solo"
        assert graph.edges[1].to == "end"

    def test_sequential_team_shared_prompt(self) -> None:
        """Shared prompt is mapped to all steps."""
        team = make_sequential_team(shared_prompt="Be thorough")
        graph = translate_team_to_graph("team", team)

        for step in graph.steps:
            assert step.shared_prompt == "Be thorough"

    def test_sequential_team_member_timeout(self) -> None:
        """Member timeout is mapped to all steps."""
        team = make_sequential_team(member_timeout=120.0)
        graph = translate_team_to_graph("team", team)

        for step in graph.steps:
            assert step.member_timeout == 120.0

    def test_sequential_team_empty_members(self) -> None:
        """Sequential team with no members produces no steps or edges."""
        team = TeamConfig(name="empty", mode="sequential", members=[])
        graph = translate_team_to_graph("empty", team)

        assert len(graph.steps) == 0
        assert len(graph.edges) == 0


# =============================================================================
# Parallel team translation tests
# =============================================================================


class TestParallelTranslation:
    """Test translation of parallel teams to graph config."""

    def test_parallel_team_basic(self) -> None:
        """Parallel team produces Fork + Join edges."""
        team = make_parallel_team()
        graph = translate_team_to_graph("parallel_coders", team)

        assert graph.name == "parallel_coders"
        assert len(graph.steps) == 2
        assert [s.id for s in graph.steps] == ["claude", "goose"]

    def test_parallel_team_edges(self) -> None:
        """Parallel team has Fork (start→[all]) and Join ([all]→end)."""
        team = make_parallel_team()
        graph = translate_team_to_graph("parallel_coders", team)

        assert len(graph.edges) == 2
        # Fork: start → [claude, goose]
        assert graph.edges[0].from_ == "start"
        assert graph.edges[0].to == ["claude", "goose"]
        # Join: [claude, goose] → end
        assert graph.edges[1].from_ == ["claude", "goose"]
        assert graph.edges[1].to == "end"

    def test_parallel_team_single_member(self) -> None:
        """Parallel team with one member still gets Fork+Join."""
        team = make_parallel_team(members=["solo"])
        graph = translate_team_to_graph("single", team)

        assert len(graph.steps) == 1
        assert len(graph.edges) == 2

    def test_parallel_team_three_members(self) -> None:
        """Parallel team with 3 members has Fork to 3 and Join from 3."""
        team = make_parallel_team(members=["a", "b", "c"])
        graph = translate_team_to_graph("triple", team)

        assert len(graph.steps) == 3
        assert graph.edges[0].to == ["a", "b", "c"]
        assert graph.edges[1].from_ == ["a", "b", "c"]

    def test_parallel_team_empty_members(self) -> None:
        """Parallel team with no members produces no steps or edges."""
        team = TeamConfig(name="empty", mode="parallel", members=[])
        graph = translate_team_to_graph("empty", team)

        assert len(graph.steps) == 0
        assert len(graph.edges) == 0


# =============================================================================
# Member config translation tests
# =============================================================================


class TestMemberConfigTranslation:
    """Test translation of TeamMemberConfig (prompt_template)."""

    def test_prompt_template_mapped_to_step(self) -> None:
        """TeamMemberConfig.prompt_template is mapped to GraphStepConfig."""
        team = make_team_with_member_configs()
        graph = translate_team_to_graph("mixed_team", team)

        assert len(graph.steps) == 3
        # agent_a: no prompt_template (plain string member)
        assert graph.steps[0].prompt_template is None
        # agent_b: has prompt_template
        assert graph.steps[1].prompt_template == "Review: {{ prompt }}"
        # agent_c: TeamMemberConfig but prompt_template is None
        assert graph.steps[2].prompt_template is None

    def test_shared_prompt_and_template_coexist(self) -> None:
        """Both shared_prompt and per-member prompt_template are preserved."""
        team = make_team_with_member_configs()
        graph = translate_team_to_graph("mixed_team", team)

        for step in graph.steps:
            assert step.shared_prompt == "Work together"
        assert graph.steps[1].prompt_template == "Review: {{ prompt }}"

    def test_member_timeout_mapped_to_all_steps(self) -> None:
        """member_timeout is mapped to all steps."""
        team = make_team_with_member_configs()
        graph = translate_team_to_graph("mixed_team", team)

        for step in graph.steps:
            assert step.member_timeout == 60.0


# =============================================================================
# Batch translation tests
# =============================================================================


class TestBatchTranslation:
    """Test translate_teams_to_graphs batch function."""

    def test_multiple_teams(self) -> None:
        """Multiple teams produce multiple graph configs."""
        teams = {
            "seq_team": make_sequential_team(name="seq_team"),
            "par_team": make_parallel_team(name="par_team"),
        }
        graphs = translate_teams_to_graphs(teams)

        assert len(graphs) == 2
        assert graphs[0].name == "seq_team"
        assert graphs[1].name == "par_team"

    def test_empty_teams(self) -> None:
        """Empty teams dict produces empty list."""
        graphs = translate_teams_to_graphs({})
        assert len(graphs) == 0


# =============================================================================
# Full config translation tests
# =============================================================================


class TestConfigTranslation:
    """Test translate_config_to_graph full manifest translation."""

    def test_no_teams_no_connections_returns_none(self) -> None:
        """No teams or connections returns None."""
        from wolfharness_config.nodes import NodeConfig

        agents = {
            "solo": NodeConfig(name="solo"),
        }
        result = translate_config_to_graph(agents, None, None)
        assert result is None

    def test_with_teams_produces_graph(self) -> None:
        """Teams produce a graph config."""
        from wolfharness_config.nodes import NodeConfig

        agents = {
            "analyzer": NodeConfig(name="analyzer"),
            "reviewer": NodeConfig(name="reviewer"),
        }
        teams = {
            "pipeline": make_sequential_team(name="pipeline", members=["analyzer", "reviewer"]),
        }
        result = translate_config_to_graph(agents, teams, None)

        assert result is not None
        assert len(result.steps) == 2
        assert len(result.edges) == 3  # start→a, a→b, b→end

    def test_existing_graph_preserved(self) -> None:
        """Existing graph: section is preserved when no teams/connections."""
        existing = GraphConfig(
            name="existing",
            steps=[GraphStepConfig(id="a", agent="a")],
            edges=[GraphEdgeConfig(**{"from": "start", "to": "a"})],
        )
        result = translate_config_to_graph({}, None, existing)

        assert result is not None
        assert result.name == "existing"
        assert len(result.steps) == 1


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_mode_raises(self) -> None:
        """Unknown team mode raises ValueError."""
        team = TeamConfig.model_construct(name="bad", mode="invalid", members=["a"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown team mode"):
            translate_team_to_graph("bad", team)

    def test_mcp_servers_mapped_to_steps(self) -> None:
        """MCP servers from team config are mapped to each step."""
        team = TeamConfig(
            name="team",
            mode="sequential",
            members=["a", "b"],
            mcp_servers=["uvx mcp-server-filesystem"],
        )
        graph = translate_team_to_graph("team", team)

        for step in graph.steps:
            assert len(step.mcp_servers) == 1

    def test_graph_step_config_defaults(self) -> None:
        """GraphStepConfig has correct defaults for new fields."""
        step = GraphStepConfig(id="test", agent="test")
        assert step.shared_prompt is None
        assert step.prompt_template is None
        assert step.member_timeout is None
        assert step.member_retry_attempts == 0
        assert step.member_retry_delay == 0.0


# =============================================================================
# Connection translation tests
# =============================================================================


class TestConnectionTranslation:
    """Test translation of agent connections to graph edges."""

    def test_node_connection_translated_to_edge(self) -> None:
        """NodeConnectionConfig is translated to a GraphEdgeConfig."""
        agents = {
            "analyzer": NodeConfig(
                name="analyzer",
                connections=[
                    NodeConnectionConfig(name="reviewer", connection_type="run"),
                ],
            ),
            "reviewer": NodeConfig(name="reviewer"),
        }
        edges = translate_connections_to_edges(agents)

        assert len(edges) == 1
        assert edges[0].from_ == "analyzer"
        assert edges[0].to == "reviewer"
        assert edges[0].mode == "run"

    def test_file_connection_skipped(self) -> None:
        """FileConnectionConfig is skipped (no 'name' key, no edge)."""
        agents = {
            "analyzer": NodeConfig(
                name="analyzer",
                connections=[
                    FileConnectionConfig(path="logs/messages.txt"),
                ],
            ),
        }
        edges = translate_connections_to_edges(agents)

        assert len(edges) == 0

    def test_callable_connection_skipped(self) -> None:
        """CallableConnectionConfig is skipped (no 'name' key, no edge)."""
        agents = {
            "analyzer": NodeConfig(
                name="analyzer",
                connections=[
                    CallableConnectionConfig(callable="builtins:print"),
                ],
            ),
        }
        edges = translate_connections_to_edges(agents)

        assert len(edges) == 0

    def test_mixed_connections_only_node_translated(self) -> None:
        """A mix of node, file, and callable connections: only node connections become edges."""
        agents = {
            "analyzer": NodeConfig(
                name="analyzer",
                connections=[
                    FileConnectionConfig(path="logs/messages.txt"),
                    NodeConnectionConfig(name="reviewer"),
                    CallableConnectionConfig(callable="builtins:print"),
                    NodeConnectionConfig(name="formatter", connection_type="forward"),
                ],
            ),
            "reviewer": NodeConfig(name="reviewer"),
            "formatter": NodeConfig(name="formatter"),
        }
        edges = translate_connections_to_edges(agents)

        assert len(edges) == 2
        assert edges[0].to == "reviewer"
        assert edges[1].to == "formatter"
        assert edges[1].mode == "forward"

    def test_config_translation_with_connections(self) -> None:
        """translate_config_to_graph produces edges from agent connections."""
        agents = {
            "analyzer": NodeConfig(
                name="analyzer",
                connections=[NodeConnectionConfig(name="reviewer")],
            ),
            "reviewer": NodeConfig(
                name="reviewer",
                connections=[FileConnectionConfig(path="out.txt")],
            ),
        }
        result = translate_config_to_graph(agents, None, None)

        assert result is not None
        # Both agents appear as steps (analyzer has connections, reviewer has connections)
        step_ids = {s.id for s in result.steps}
        assert "analyzer" in step_ids
        assert "reviewer" in step_ids
        # Only the node→node connection becomes an edge; file connection is skipped
        assert len(result.edges) == 1
        assert result.edges[0].from_ == "analyzer"
        assert result.edges[0].to == "reviewer"

    def test_empty_teams_dict_still_translates_connections(self) -> None:
        """An empty (but not None) teams dict does not break connection translation."""
        agents = {
            "a": NodeConfig(
                name="a",
                connections=[NodeConnectionConfig(name="b")],
            ),
            "b": NodeConfig(name="b"),
        }
        result = translate_config_to_graph(agents, {}, None)

        assert result is not None
        assert len(result.edges) == 1
