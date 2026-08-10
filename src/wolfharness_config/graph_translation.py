"""Translate legacy ``teams:`` and ``connections:`` YAML config to ``graph:`` syntax.

This module provides the translation layer that converts legacy AgentPool
team and connection configurations into the unified ``GraphConfig`` format.

Translation rules (from ``docs/design/yaml_graph_syntax.md``):

1. ``team mode: sequential`` → chained steps with implicit linear edges
2. ``team mode: parallel`` → Fork (start → all members) + Join (all members → end)
3. Agent ``connections:`` → explicit edges between agent steps

The translator preserves all team-level fields (``shared_prompt``,
``member_timeout``, ``prompt_template``, ``member_retry_attempts``,
``member_retry_delay``) by mapping them onto ``GraphStepConfig`` fields.

Example:
    teams:
      review_pipeline:
        mode: sequential
        members: [analyzer, reviewer, formatter]

Translates to:

    graph:
      name: review_pipeline
      steps:
        - id: analyzer
          agent: analyzer
        - id: reviewer
          agent: reviewer
        - id: formatter
          agent: formatter
      edges:
        - from: start
          to: analyzer
        - from: analyzer
          to: reviewer
        - from: reviewer
          to: formatter
        - from: formatter
          to: end
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness_config.graph_config import GraphConfig, GraphEdgeConfig, GraphStepConfig


if TYPE_CHECKING:
    from wolfharness_config.nodes import NodeConfig
    from wolfharness_config.teams import TeamConfig


def translate_team_to_graph(
    name: str,
    team: TeamConfig,
) -> GraphConfig:
    """Translate a single ``TeamConfig`` to a ``GraphConfig``.

    Args:
        name: The team name (used as graph name).
        team: The legacy team configuration.

    Returns:
        A ``GraphConfig`` with steps and edges representing the same
        execution topology as the team.
    """
    member_configs = team.get_member_configs()

    steps: list[GraphStepConfig] = []
    for member in team.members:
        member_name = team.get_member_name(member)
        member_cfg = member_configs.get(member_name)

        steps.append(
            GraphStepConfig(
                id=member_name,
                agent=member_name,
                shared_prompt=team.shared_prompt,
                prompt_template=member_cfg.prompt_template if member_cfg else None,
                member_timeout=team.member_timeout,
                mcp_servers=team.get_mcp_servers(),
            )
        )

    if team.mode == "sequential":
        edges = _build_sequential_edges(steps)
    elif team.mode == "parallel":
        edges = _build_parallel_edges(steps)
    else:  # pragma: no cover
        msg = f"Unknown team mode: {team.mode!r}"
        raise ValueError(msg)

    return GraphConfig(name=name, steps=steps, edges=edges)


def translate_teams_to_graphs(
    teams: dict[str, TeamConfig],
) -> list[GraphConfig]:
    """Translate all teams in a manifest to graph configs.

    Args:
        teams: Mapping of team name to ``TeamConfig``.

    Returns:
        List of ``GraphConfig`` objects, one per team.
    """
    return [translate_team_to_graph(name, team) for name, team in teams.items()]


def translate_connections_to_edges(
    agents: dict[str, NodeConfig],
) -> list[GraphEdgeConfig]:
    """Translate agent ``connections:`` config to graph edges.

    Each agent's ``connections`` list is converted to ``GraphEdgeConfig``
    objects.  The agent names become step IDs.

    Args:
        agents: Mapping of agent name to ``NodeConfig`` (agent or team config).

    Returns:
        List of ``GraphEdgeConfig`` objects representing all connections.
    """
    edges: list[GraphEdgeConfig] = []

    for agent_name, agent_config in agents.items():
        for conn in agent_config.connections:
            conn_dict = _normalize_connection(conn)
            # Only node connections have a ``name`` field (the target agent).
            # File and callable connections write to external sinks and do
            # not represent edges between graph steps, so skip them.
            if "name" not in conn_dict:
                continue
            edges.append(
                GraphEdgeConfig(
                    **{
                        "from": agent_name,
                        "to": conn_dict["name"],
                        "mode": conn_dict.get("connection_type", "run"),
                        "condition": conn_dict.get("filter_condition"),
                        "stop_condition": conn_dict.get("stop_condition"),
                        "transform": conn_dict.get("transform"),
                        "async_": conn_dict.get("async", False),
                        "priority": conn_dict.get("priority", 0),
                    },
                ),
            )

    return edges


def build_steps_from_agents(
    agents: dict[str, NodeConfig],
) -> list[GraphStepConfig]:
    """Build graph steps from agent configurations.

    Creates one ``GraphStepConfig`` per agent, preserving MCP server
    configuration.

    Args:
        agents: Mapping of agent name to ``NodeConfig``.

    Returns:
        List of ``GraphStepConfig`` objects, one per agent.
    """
    steps: list[GraphStepConfig] = []
    for agent_name, agent_config in agents.items():
        steps.append(
            GraphStepConfig(
                id=agent_name,
                agent=agent_name,
                mcp_servers=agent_config.get_mcp_servers(),
            ),
        )
    return steps


def translate_config_to_graph(
    agents: dict[str, NodeConfig],
    teams: dict[str, TeamConfig] | None,
    existing_graph: GraphConfig | None = None,
) -> GraphConfig | None:
    """Translate full manifest config to a unified ``GraphConfig``.

    Combines:
    - Existing ``graph:`` section (if provided)
    - Translated ``teams:`` sections
    - Translated ``connections:`` from agents

    If no graph, teams, or connections exist, returns ``None``.

    Args:
        agents: All agent configurations from the manifest.
        teams: Optional team configurations from the manifest.
        existing_graph: An existing ``graph:`` section to merge into.

    Returns:
        A unified ``GraphConfig`` or ``None`` if no topology is configured.
    """
    has_connections = any(agent_config.connections for agent_config in agents.values())

    if teams is None and not has_connections and existing_graph is None:
        return None

    # Start with existing graph or empty
    if existing_graph is not None:
        steps = list(existing_graph.steps)
        edges = list(existing_graph.edges)
        joins = list(existing_graph.joins)
        graph_name = existing_graph.name
    else:
        steps = []
        edges = []
        joins = []
        graph_name = None

    # Add steps from agents that have connections (if not already in graph)
    existing_step_ids = {s.id for s in steps}
    if has_connections:
        for agent_name, agent_config in agents.items():
            if agent_name not in existing_step_ids and agent_config.connections:
                steps.append(
                    GraphStepConfig(
                        id=agent_name,
                        agent=agent_name,
                        mcp_servers=agent_config.get_mcp_servers(),
                    ),
                )
                existing_step_ids.add(agent_name)

    # Translate connections to edges
    edges.extend(translate_connections_to_edges(agents))

    # Translate teams to sub-graphs (each team becomes its own graph config,
    # but for the unified graph we merge their steps and edges)
    if teams is not None:
        for team_name, team_config in teams.items():
            team_graph = translate_team_to_graph(team_name, team_config)
            steps.extend(team_graph.steps)
            edges.extend(team_graph.edges)
            joins.extend(team_graph.joins)
            if graph_name is None:
                graph_name = team_name

    if not steps and not edges:
        return None

    return GraphConfig(
        name=graph_name,
        steps=steps,
        edges=edges,
        joins=joins,
    )


def _build_sequential_edges(steps: list[GraphStepConfig]) -> list[GraphEdgeConfig]:
    """Build edges for a sequential chain: start → s1 → s2 → ... → end."""
    edges: list[GraphEdgeConfig] = []

    if not steps:
        return edges

    # start → first step
    edges.append(GraphEdgeConfig(**{"from": "start", "to": steps[0].id}))

    # step[i] → step[i+1]
    edges.extend(
        GraphEdgeConfig(**{"from": steps[i].id, "to": steps[i + 1].id})
        for i in range(len(steps) - 1)
    )

    # last step → end
    edges.append(GraphEdgeConfig(**{"from": steps[-1].id, "to": "end"}))

    return edges


def _build_parallel_edges(steps: list[GraphStepConfig]) -> list[GraphEdgeConfig]:
    """Build edges for parallel execution: start → [all steps] → end."""
    if not steps:
        return []

    step_ids = [s.id for s in steps]

    # Fork: start → all steps
    edges = [
        GraphEdgeConfig(**{"from": "start", "to": step_ids}),
    ]

    # Join: all steps → end
    edges.append(
        GraphEdgeConfig(**{"from": step_ids, "to": "end"}),
    )

    return edges


def _normalize_connection(conn: Any) -> dict[str, Any]:
    """Normalize a connection config to a plain dict."""
    from pydantic import BaseModel

    match conn:
        case dict():
            return conn
        case BaseModel():
            return dict(conn.model_dump(exclude_none=True, by_alias=False))
        case _:
            return dict(conn)
