"""Teams mixin for SessionPool.

Extracted from session_pool.py as part of the session-debt-cleanup file split.
Contains team creation from config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from wolfharness.delegation.base_team import BaseTeam
    from wolfharness.messaging.messagenode import MessageNode
    from wolfharness_config.teams import TeamConfig


logger = get_logger(__name__)


def _build_team_from_config(
    name: str,
    team_config: TeamConfig,
    nodes: Sequence[MessageNode[Any, Any]],
) -> BaseTeam[Any, Any]:
    """Construct a ``BaseTeam`` from a ``TeamConfig`` and pre-resolved nodes.

    This replaces the deprecated ``TeamConfig.get_team()`` method, keeping
    team instantiation in the core layer instead of the config layer.
    """
    from wolfharness.delegation.base_team import BaseTeam

    member_configs = team_config.get_member_configs()

    return BaseTeam(
        nodes,
        mode=team_config.mode,
        name=name,
        display_name=team_config.display_name,
        shared_prompt=team_config.shared_prompt,
        mcp_servers=team_config.get_mcp_servers(),
        member_prompt_templates=member_configs or None,
        member_timeout=team_config.member_timeout,
    )


class SessionPoolTeamsMixin:
    """Mixin providing team creation methods for SessionPool.

    Attributes:
        pool: AgentPool instance (provided by SessionPool).
    """

    pool: Any

    async def create_team_from_config(
        self,
        team_name: str,
        team_config: TeamConfig,
    ) -> BaseTeam[Any, Any]:
        """Create a team from config using session-level agent resolution.

        For each member in the team config, resolves the agent via
        :meth:`SessionController.get_or_create_session_agent`, then
        constructs a :class:`BaseTeam` with ``mode`` set to the
        configured execution mode.

        Member names are stored on the resulting team nodes; actual
        session agents are created per-execution by
        :meth:`BaseTeam._resolve_scoped_team_nodes`.

        Args:
            team_name: Name for the created team.
            team_config: Team configuration from the manifest.

        Returns:
            A ``BaseTeam`` instance with the configured execution mode.

        Raises:
            ValueError: If a member name is not found in the manifest
                agents or teams section.
        """
        from wolfharness_config.context import ConfigContextManager

        member_names = [team_config.get_member_name(m) for m in team_config.members]

        nodes: list[MessageNode[Any, Any]] = []
        for member_name in member_names:
            cfg = self.pool.manifest.agents.get(member_name)
            if cfg is not None:
                # Create a stateless agent without entering its async context.
                # This avoids spawning MCP subprocesses for temporary template
                # agents — actual per-session agents are created later by
                # BaseTeam._resolve_scoped_team_nodes() during execution.
                if cfg.name is None:
                    cfg = cfg.model_copy(update={"name": member_name})
                with ConfigContextManager(self.pool._config_file_path):
                    agent: MessageNode[Any, Any] = cfg.get_agent(pool=self.pool)
                nodes.append(agent)
            elif member_name in self.pool.manifest.teams:
                nested_config = self.pool.manifest.teams[member_name]
                nested_team = await self.create_team_from_config(member_name, nested_config)
                nodes.append(nested_team)
            else:
                msg = f"Team member {member_name!r} not found in manifest agents or teams"
                raise ValueError(msg)

        return _build_team_from_config(team_name, team_config, nodes)
