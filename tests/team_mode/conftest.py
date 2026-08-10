"""Shared fixtures and helpers for team-mode tests.

Provides:
- ``team_mode_pool`` / ``team_mode_pool_with_defaults`` fixtures (re-exported)
- Message inspection helpers (ported from pydantic-ai-harness patterns)
- ``build_agent_context`` helper for direct capability testing
- ``FunctionModel`` factory helpers for scripted multi-turn flows
- Team-mode test helpers: config builders, metadata factories, mock builders

See ``tests/AGENTS.md`` for the L1-L4 testing guide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

# Re-export fixtures so tests in this directory can use them directly.
from tests.fixtures.team_mode_pool import (  # noqa: F401
    team_mode_pool,
    team_mode_pool_with_defaults,
)


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolReturnPart

    from wolfharness import AgentPool
    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness_config.team_mode import TeamModeConfig


# ---------------------------------------------------------------------------
# Message inspection helpers (ported from pydantic-ai-harness patterns)
# ---------------------------------------------------------------------------


def _tool_returns_by_name(messages: list[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    """Extract ``ToolReturnPart`` entries matching ``tool_name`` from messages."""
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    return [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


def _tool_call_names(messages: list[ModelMessage]) -> list[str]:
    """Extract ordered tool call names from ``ModelResponse`` entries."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    return [
        part.tool_name
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]


def _user_prompt_text(messages: list[ModelMessage]) -> str:
    """Extract the user prompt text from the first ``ModelRequest``."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ""


# ---------------------------------------------------------------------------
# AgentContextDeps builder for direct capability testing
# ---------------------------------------------------------------------------


def build_agent_context(
    pool: AgentPool[Any],
    session_id: str,
    team_mode_config: TeamModeConfig,
) -> AgentContextDeps:
    """Construct a real ``AgentContextDeps`` for calling team tools directly.

    This mirrors what the RunLoop creates per-turn, allowing tests to
    invoke ``TeamCommCapability`` methods without going through
    ``Agent.run()``.
    """
    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.capabilities.runloop_delegation import RunLoopDelegationService
    from wolfharness.host.context import RunScope
    from wolfharness.host.registry import AgentRegistry

    session_pool = pool.session_pool
    assert session_pool is not None
    session = session_pool.sessions.get_session(session_id)
    assert session is not None

    host_ctx = pool.get_context()
    registry = AgentRegistry(dict.fromkeys(pool.manifest.agents))
    delegation = RunLoopDelegationService(
        registry=registry,
        host=host_ctx,
        session_id=session_id,
    )
    scope = RunScope(
        config_id=None,
        tenant_id=None,
        user_id=None,
        session_id=session_id,
    )
    return AgentContextDeps(
        agent_registry=registry,
        delegation=delegation,
        session=session,
        scope=scope,
        host=host_ctx,
        team_mode_config=team_mode_config,
    )


def make_mock_run_context(agent_ctx: AgentContextDeps) -> MagicMock:
    """Create a mock pydantic-ai ``RunContext`` with ``AgentContextDeps`` as deps.

    ``_resolve_agent_context`` checks ``isinstance(deps, AgentContextDeps)``
    from ``capabilities.agent_context`` — our ``AgentContextDeps`` matches that
    check and is returned directly.
    """
    ctx: Any = MagicMock()
    ctx.deps = agent_ctx
    return ctx


# ---------------------------------------------------------------------------
# FunctionModel factory helpers (ported from pydantic-ai-harness patterns)
# ---------------------------------------------------------------------------


def make_lifecycle_model(steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """Create a ``FunctionModel`` that issues tool calls in sequence.

    Args:
        steps: Ordered list of ``(tool_name, args)`` tuples.  After all
            tool calls are issued, the model returns a final text response.

    Returns:
        A ``FunctionModel`` instance with both ``function`` and
        ``stream_function`` set, so it works with both ``Agent.run()``
        and ``Agent.run_stream()``.
    """
    import json

    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    calls: dict[str, int] = {"n": 0}

    def model_fn(messages: list[Any], info: Any) -> ModelResponse:
        calls["n"] += 1
        idx = calls["n"] - 1
        if idx < len(steps):
            tool_name, args = steps[idx]
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id=f"call_{idx}")],
            )
        return ModelResponse(parts=[TextPart(content="done")])

    async def stream_fn(messages: list[Any], info: Any) -> Any:
        """Stream function yielding DeltaToolCalls then text."""
        calls["n"] += 1
        idx = calls["n"] - 1
        if idx < len(steps):
            tool_name, args = steps[idx]
            yield {
                0: DeltaToolCall(
                    name=tool_name,
                    json_args=json.dumps(args),
                    tool_call_id=f"call_{idx}",
                ),
            }
        else:
            yield "done"

    return FunctionModel(function=model_fn, stream_function=stream_fn)


# ---------------------------------------------------------------------------
# Team-mode test helpers (shared across unit + integration test files)
# ---------------------------------------------------------------------------


def make_enabled_config(
    *,
    member_eligible: list[str] | None = None,
    lead_eligible: list[str] | None = None,
    base_dir: str | None = None,
    notice_delivery_mode: str = "steer",
) -> TeamModeConfig:
    """Create an enabled TeamModeConfig for testing."""
    from wolfharness_config.team_mode import TeamModeConfig

    return TeamModeConfig(
        enabled=True,
        member_eligible=member_eligible or ["worker", "reviewer"],
        lead_eligible=lead_eligible or ["coordinator"],
        base_dir=base_dir,
        notice_delivery_mode=notice_delivery_mode,
    )


def make_lead_metadata(team_id: str = "team_123") -> dict[str, Any]:
    """Create session metadata for a lead agent."""
    meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    if team_id:
        meta["team_id"] = team_id
        meta["team_name"] = "alpha_team"
    return meta


def make_member_metadata(
    team_id: str = "team_123",
    member_name: str = "translator_agent",
) -> dict[str, Any]:
    """Create session metadata for a team member."""
    return {
        "team_id": team_id,
        "team_name": "alpha_team",
        "team_role": "member",
        "team_member_name": member_name,
    }


def make_run_context(
    metadata: dict[str, Any] | None = None,
    session_pool: MagicMock | None = None,
    config: TeamModeConfig | None = None,
    base_dir: str | None = None,
    agent_registry: MagicMock | None = None,
    session_id: str = "lead_session_001",
    delegation: MagicMock | None = None,
) -> MagicMock:
    """Create a mock RunContext with AgentContextDeps deps.

    If ``session_pool`` is provided, ensures its async methods and
    sub-attributes are properly mocked for team tool operations.
    """
    from wolfharness.capabilities.agent_context import AgentContextDeps

    cfg = config or make_enabled_config(base_dir=base_dir)

    if session_pool is not None:
        if not isinstance(session_pool.create_child_session, AsyncMock):
            child_state: Any = MagicMock()
            child_state.session_id = "child_session"
            session_pool.create_child_session = AsyncMock(return_value=child_state)
        if not isinstance(
            getattr(session_pool.sessions, "get_or_create_session_agent", None),
            AsyncMock,
        ):
            session_pool.sessions = MagicMock()
            session_pool.sessions.get_or_create_session_agent = AsyncMock()
        eb: Any = session_pool.event_bus
        if not (isinstance(eb, MagicMock) and isinstance(getattr(eb, "publish", None), AsyncMock)):
            session_pool.event_bus = None

    agent_ctx: Any = MagicMock(spec=AgentContextDeps)
    agent_ctx.session.metadata = metadata if metadata is not None else make_lead_metadata()
    agent_ctx.host.session_pool = session_pool
    agent_ctx.team_mode_config = cfg
    agent_ctx.agent_registry = agent_registry or MagicMock()
    agent_ctx.session.session_id = session_id
    agent_ctx.delegation = delegation or MagicMock()

    ctx: Any = MagicMock()
    ctx.deps = agent_ctx
    return ctx


def init_team(
    base_dir: str,
    team_id: str = "team_123",
    team_name: str = "alpha_team",
    members: list[dict[str, str]] | None = None,
) -> None:
    """Initialize a real FileTeamState with a team and registered members."""
    from wolfharness.capabilities.file_team_state import FileTeamState

    if members is None:
        members = [
            {"name": "translator_agent", "agent": "worker"},
            {"name": "reviewer_agent", "agent": "reviewer"},
        ]
    state = FileTeamState(base_dir)
    state.init(team_id, team_name, members)
    for m in members:
        state.register_member(team_id, m["name"], f"sess_{m['name']}")


def make_mock_pool() -> MagicMock:
    """Create a mock SessionPool with async send_message and close_session."""
    pool = MagicMock()
    pool.send_message = AsyncMock(return_value="msg_id_001")
    pool.close_session = AsyncMock()
    mock_child_state: Any = MagicMock()
    mock_child_state.session_id = "child_session_001"
    pool.create_child_session = AsyncMock(return_value=mock_child_state)
    pool.sessions = MagicMock()
    pool.sessions.get_or_create_session_agent = AsyncMock()
    pool.event_bus = None
    return pool


def make_mock_registry() -> MagicMock:
    """Create a mock AgentRegistry."""
    registry = MagicMock()
    registry.exists = MagicMock(return_value=True)
    return registry
