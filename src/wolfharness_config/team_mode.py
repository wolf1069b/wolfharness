"""Team mode configuration model.

Defines the :class:`TeamModeConfig` Pydantic model that controls dynamic
team creation, inter-agent messaging, blackboard state, and bounds
enforcement for collaborative multi-agent teams.

The model supports per-agent overlay via :func:`resolve_team_mode`,
which merges a global config with agent-specific overrides.

Example YAML::

    team_mode:
      enabled: true
      member_eligible: [translator, reviewer]
      lead_eligible: [coordinator]
      bounds:
        max_members: 5
        max_wall_clock_minutes: 30
      defaults:
        team_name: translation_team
        members:
          - name: translator
            agent: translator
"""

from __future__ import annotations

import tempfile
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, field_validator, model_validator
from schemez import Schema


_DEFAULT_PROTOCOL_TEMPLATE = """\
You are {member_name}, a member of team "{team_name}".
Your role is: {role}.

## Team Communication Protocol

1. Use `send_message` to communicate with other team members.
2. Use `task_create` and `task_create_batch` to track work items.
3. Use `read_blackboard` and `write_blackboard` to share context.
4. Use `team_status` to check the status of all members.
5. If you are the lead, you may use `team_create` and `team_delete`.

## Communication Channels

### Tasks (structured work tracking)

- Use `task_create` and `task_create_batch` to define work items with clear \
deliverables.
- Use `task_update(status=...)` to change task lifecycle state only \
(pending → in_progress → completed/failed).
- Use `task_update(technical_note=...)` ONLY for technical notes that future \
readers of the task need. Do NOT use task notes for progress chatter.
- Use `task_update(progress_current=..., progress_total=...)` to report \
quantitative progress on long-running tasks.
- Task dependencies (`blocked_by`) are declarative — the system tracks them \
and notifies you when they resolve. Do NOT send_message to ask if a blocking \
task is done; use `team_status` or `task_get` to check.

### Blackboard (shared persistent state)

- Use `write_blackboard` for structured deliverables: research findings, \
analysis results, documents, code snippets.
- Use `write_blackboard(mode="append")` for accumulating findings that grow \
over time.
- Blackboard is NOT for communication. Do NOT write messages, status updates, \
or questions to the blackboard.
- Before starting work, `read_blackboard` for any keys relevant to your role \
to avoid duplicating effort.

### Messages (interpersonal communication)

- Use `send_message` for: asking questions, requesting help, reporting \
blockers, coordinating handoffs, notifying completion.
- Messages are ephemeral — they arrive as notifications but are NOT persisted \
in team state. If the information needs to be durable, write it to the \
blackboard instead (or use `persist_to_blackboard` parameter).
- Use `message_type` to categorize: "question", "escalation", "handoff", \
"notification".
"""


class TeamBounds(Schema):
    """Bounds limiting team size and execution duration.

    Attributes:
        max_members: Maximum number of members per team.
        max_parallel_members: Maximum members running concurrently.
        max_wall_clock_minutes: Maximum wall-clock time per team.
        max_member_turns: Maximum turns per member session.
    """

    model_config = ConfigDict(frozen=True)

    max_members: int = Field(default=10, ge=1, title="Max members per team")
    max_parallel_members: int = Field(default=5, ge=1, title="Max parallel members")
    max_wall_clock_minutes: int = Field(default=60, ge=1, title="Max wall-clock minutes")
    max_member_turns: int = Field(default=20, ge=1, title="Max member turns")

    @field_validator(
        "max_members",
        "max_parallel_members",
        "max_wall_clock_minutes",
        "max_member_turns",
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        """Ensure all bounds are strictly positive."""
        if v <= 0:
            msg = f"Bound must be > 0, got {v}"
            raise ValueError(msg)
        return v


class BlackboardConfig(Schema):
    """Configuration for the team blackboard.

    Attributes:
        write_policy: Who can write to the blackboard.
        max_size_mb: Maximum blackboard size in megabytes.
    """

    model_config = ConfigDict(frozen=True)

    write_policy: Literal["open", "lead_only"] = Field(
        default="open",
        title="Blackboard write policy",
    )
    max_size_mb: int = Field(default=100, ge=1, title="Max blackboard size (MB)")

    @field_validator("max_size_mb")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        """Ensure max_size_mb is strictly positive."""
        if v <= 0:
            msg = f"max_size_mb must be > 0, got {v}"
            raise ValueError(msg)
        return v


class MemberSpec(Schema):
    """Specification of a single team member in defaults.

    Attributes:
        name: Display name of the member within the team.
        agent: Agent name to map to in the registry.
        instructions: Per-member instruction text injected into the member's
            system prompt as a ``## Your Assignment`` section.
        skills: Optional skill names or ``skill://`` URIs (including reference
            paths) injected as instruction text into the member's system
            prompt when the team is created from these defaults.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(title="Member name")
    agent: str = Field(title="Agent name in registry")
    instructions: str = Field(
        default="",
        title="Per-member instructions",
        description="Per-member instruction text injected into the member's "
        "system prompt as a '## Your Assignment' section",
    )
    skills: list[str] = Field(
        default_factory=list,
        title="Per-member skills",
        description="Optional skill names or skill:// URIs (including "
        "reference paths) injected as instruction text into the member's "
        "system prompt",
    )


class TeamDefaultsConfig(Schema):
    """Default team members for team_create.

    When present, team_create uses these members when the LLM calls it
    without explicit members.

    Attributes:
        team_name: Name of the team to create.
        members: List of default members for the team.
    """

    model_config = ConfigDict(frozen=True)

    team_name: str = Field(title="Team name")
    members: list[MemberSpec] = Field(default_factory=list, title="Default members")


class TeamModeConfig(Schema):
    """Configuration for dynamic team mode.

    Controls whether agents can form ad-hoc teams, message each other,
    and share state via a blackboard. Supports per-agent overlay via
    :func:`resolve_team_mode`.

    Attributes:
        enabled: Whether team mode is active.
        member_eligible: Agent names eligible to be team members.
        lead_eligible: Agent names eligible to be team leads.
        base_dir: Base directory for team state files.
        ttl_hours: Hours before stale team state is cleaned up.
        bounds: Limits on team size and execution.
        blackboard: Blackboard configuration.
        message_max_bytes: Maximum message size in bytes.
        inbox_max_bytes: Maximum inbox size in bytes.
        protocol_template: Template for team protocol instructions.
        notice_delivery_mode: Delivery mode for team notifications
            (broadcast_on_create, team_add_member notices, send_message).
            ``"steer"`` (default) injects mid-turn; ``"queue"`` waits
            for next turn.
        defaults: Optional default team members for team_create.
        broadcast_on_create: Whether to auto-broadcast a notification to
            all team members (excluding the lead) when a new member is
            added via ``team_add_member``. The broadcast includes the new
            member's name, agent, and the updated member roster.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, title="Enable team mode")
    member_eligible: list[str] = Field(default_factory=list, title="Member-eligible agents")
    lead_eligible: list[str] = Field(default_factory=list, title="Lead-eligible agents")
    base_dir: str | None = Field(default=None, title="Base directory for team state")
    ttl_hours: int = Field(default=72, ge=1, title="TTL in hours")
    bounds: TeamBounds = Field(default_factory=TeamBounds, title="Team bounds")
    blackboard: BlackboardConfig = Field(
        default_factory=BlackboardConfig,
        title="Blackboard config",
    )
    message_max_bytes: int = Field(default=65536, ge=1, title="Max message size (bytes)")
    inbox_max_bytes: int = Field(default=1048576, ge=1, title="Max inbox size (bytes)")
    protocol_template: str = Field(
        default=_DEFAULT_PROTOCOL_TEMPLATE,
        title="Protocol template",
    )
    notice_delivery_mode: Literal["steer", "queue"] = Field(
        default="steer",
        title="Notice delivery mode",
    )
    defaults: TeamDefaultsConfig | None = Field(
        default=None,
        title="Default team members for team_create",
    )
    broadcast_on_create: bool = Field(
        default=True,
        title="Auto-broadcast on member creation",
    )
    max_watch_timeout: int = Field(
        default=120,
        ge=1,
        title="Max watch timeout (seconds)",
        description="Maximum seconds a watch call (list_blackboard, team_status) "
        "will block waiting for changes. Acts as a safety cap when the caller "
        "specifies timeout<=0 (no limit) or a timeout exceeding this value.",
    )
    idle_timeout: float = Field(
        default=600.0,
        gt=0,
        title="Idle timeout (seconds)",
        description="Seconds of lead inactivity before auto-closing team member "
        "sessions. The cleanup polls every ``poll_interval`` seconds and closes "
        "members when the lead has been inactive (no send_message) for longer "
        "than this value, provided no session has an active run.",
    )
    poll_interval: float = Field(
        default=30.0,
        gt=0,
        title="Cleanup poll interval (seconds)",
        description="Interval at which the idle-cleanup loop checks lead "
        "activity. Lower values mean faster detection but more CPU overhead.",
    )

    @field_validator("ttl_hours")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        """Ensure ttl_hours is strictly positive."""
        if v <= 0:
            msg = f"ttl_hours must be > 0, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_defaults_members(self) -> TeamModeConfig:
        """Ensure all defaults member agents are in member_eligible."""
        if self.defaults is None:
            return self
        eligible_set = set(self.member_eligible)
        for member in self.defaults.members:
            if member.agent not in eligible_set:
                msg = (
                    f"defaults member agent '{member.agent}' is not in "
                    f"member_eligible list: {self.member_eligible}"
                )
                raise ValueError(msg)
        return self

    @property
    def effective_base_dir(self) -> str:
        """Return the base directory, defaulting to the system temp dir."""
        if self.base_dir is not None:
            return self.base_dir
        return tempfile.gettempdir()


def resolve_team_mode(
    global_config: TeamModeConfig | None,
    agent_config: TeamModeConfig | None,
) -> TeamModeConfig | None:
    """Resolve team mode config by merging global and per-agent settings.

    If both are None, returns None. If only one is provided, returns it.
    If both are provided, agent-specific non-default fields override
    the global config via model_copy(update=...).

    Args:
        global_config: The global team_mode config from the manifest.
        agent_config: The per-agent team_mode overlay.

    Returns:
        Merged TeamModeConfig, or None if both inputs are None.
    """
    match global_config, agent_config:
        case None, None:
            return None
        case None, agent:
            return agent
        case global_cfg, None:
            return global_cfg
        case _:
            pass

    # Both non-None: merge agent overrides onto global config.
    assert global_config is not None
    assert agent_config is not None
    global_dict = global_config.model_dump()
    agent_dict = agent_config.model_dump()

    # Collect fields where the agent config differs from its own defaults.
    # We compare against agent defaults to detect which fields were
    # explicitly set by the user.
    agent_defaults = TeamModeConfig().model_dump()
    overrides: dict[str, object] = {}
    for field_name in global_dict:
        if agent_dict[field_name] != agent_defaults[field_name]:
            overrides[field_name] = agent_dict[field_name]

    if not overrides:
        return global_config

    # ``model_copy(update=...)`` skips validation, so nested models that were
    # overridden arrive as raw dicts. Re-hydrate them back into model
    # instances so the merged config keeps typed nested fields
    # (``defaults.members[].skills``, bounds, blackboard).
    nested_models: dict[str, type[Schema]] = {
        "defaults": TeamDefaultsConfig,
        "bounds": TeamBounds,
        "blackboard": BlackboardConfig,
    }
    for field_name, model_type in nested_models.items():
        raw = overrides.get(field_name)
        if isinstance(raw, dict):
            overrides[field_name] = model_type(**cast(dict[str, Any], raw))

    return global_config.model_copy(update=overrides)


__all__ = [
    "BlackboardConfig",
    "MemberSpec",
    "TeamBounds",
    "TeamDefaultsConfig",
    "TeamModeConfig",
    "resolve_team_mode",
]
