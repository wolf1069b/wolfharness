"""Team configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wolfharness_config.nodes import NodeConfig


ExecutionMode = Literal["parallel", "sequential"]


class TeamMemberConfig(BaseModel):
    """Configuration for a single team member with optional prompt template.

    When ``prompt_template`` is set, the member receives a Jinja2-rendered
    prompt instead of the team's shared prompt.  Available template variables:

    - ``prompt``: the caller-supplied prompt (stringified)
    - ``shared_prompt``: the team-level ``shared_prompt`` value
    - ``extra``: a dict of additional variables passed via ``template_vars`` kwarg
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(title="Member name")
    """Name of the agent or team to include."""

    prompt_template: str | None = Field(
        default=None,
        title="Per-member Jinja2 prompt template",
    )
    """Optional Jinja2 template.  ``None`` means use the default shared prompt."""


class TeamConfig(NodeConfig):
    """Configuration for a team or chain of message nodes.

    Teams can be either parallel execution groups or sequential chains.
    They can contain both agents and other teams as members.

    Docs: https://phil65.github.io/wolfharness/YAML%20Configuration/team_configuration/
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "fluent:people-team-24-regular",
            "x-doc-title": "Team Configuration",
        }
    )

    mode: ExecutionMode = Field(examples=["parallel", "sequential"], title="Execution mode")
    """Execution mode for team members."""

    members: list[str | TeamMemberConfig] = Field(
        examples=[
            ["agent1", "agent2"],
            [{"name": "reviewer_a"}, {"name": "reviewer_b", "prompt_template": "{{ prompt }}"}],
        ],
        title="Team members",
    )
    """Names of agents/teams, or ``TeamMemberConfig`` objects with per-member prompt templates."""

    shared_prompt: str | None = Field(
        default=None,
        examples=["Work together to solve this problem", "Follow the team guidelines"],
        title="Shared prompt",
    )
    """Optional shared prompt for this team."""

    member_timeout: float | None = Field(
        default=None,
        examples=[60.0, 120.0, 300.0],
        title="Per-member timeout (seconds)",
    )
    """Maximum seconds each member may run before being cancelled.

    When set, members that exceed this deadline are cancelled and recorded
    in ``TeamResponse.errors`` as ``TimeoutError``.  Members that finish
    in time are **not** affected — their results are preserved even when
    siblings time out.  ``None`` (default) means no timeout.
    """

    def get_member_name(self, member: str | TeamMemberConfig) -> str:
        """Extract the member name from a plain string or config object."""
        if isinstance(member, str):
            return member
        return member.name

    def get_member_configs(self) -> dict[str, TeamMemberConfig]:
        """Build a name → config mapping for members that have prompt templates."""
        return {
            m.name: m
            for m in self.members
            if isinstance(m, TeamMemberConfig) and m.prompt_template is not None
        }
