"""L2 flow tests for team mode using FunctionModel-scripted multi-turn sequences.

These tests verify the full team lifecycle as a deterministic state machine.
Each test scripts a sequence of LLM tool calls via ``FunctionModel`` and
dispatches them to ``TeamCommCapability`` methods, asserting on each return
value to verify the state transitions.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from wolfharness.capabilities.team_comm_capability import TeamCommCapability
from wolfharness_config.team_mode import TeamModeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_enabled_config(
    *,
    member_eligible: list[str] | None = None,
    lead_eligible: list[str] | None = None,
    base_dir: str | None = None,
) -> TeamModeConfig:
    """Create an enabled TeamModeConfig for flow testing."""
    return TeamModeConfig(
        enabled=True,
        member_eligible=member_eligible or ["worker", "reviewer"],
        lead_eligible=lead_eligible or ["coordinator"],
        base_dir=base_dir,
    )


def _make_lead_metadata(team_id: str | None = None) -> dict[str, Any]:
    """Create session metadata for a lead agent."""
    meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }
    if team_id is not None:
        meta["team_id"] = team_id
        meta["team_name"] = "alpha_team"
    return meta


def _make_member_metadata(team_id: str = "team_123") -> dict[str, Any]:
    """Create session metadata for a team member."""
    return {
        "team_id": team_id,
        "team_name": "alpha_team",
        "team_role": "member",
        "team_member_name": "translator_agent",
    }


def _make_run_context(
    metadata: dict[str, Any] | None = None,
    session_pool: MagicMock | None = None,
    config: TeamModeConfig | None = None,
    base_dir: str | None = None,
    agent_registry: MagicMock | None = None,
    session_id: str = "lead_session_001",
    delegation: MagicMock | None = None,
) -> MagicMock:
    """Create a mock RunContext with AgentContextDeps deps.

    Args:
        metadata: Session metadata dict (defaults to lead metadata).
        session_pool: Mock SessionPool (or None to test missing pool).
        config: TeamModeConfig (defaults to enabled config).
        base_dir: Optional base_dir override for TeamModeConfig.
        agent_registry: Mock AgentRegistry (defaults to a permissive mock).
        session_id: Session ID string for the mock SessionState.
        delegation: Mock DelegationService (defaults to a generic MagicMock).

    Returns:
        A MagicMock whose .deps is a mock AgentContextDeps.
    """
    from wolfharness.capabilities.agent_context import AgentContextDeps

    cfg = config or _make_enabled_config(base_dir=base_dir)

    agent_ctx = MagicMock(spec=AgentContextDeps)
    agent_ctx.session.metadata = metadata if metadata is not None else _make_lead_metadata()
    agent_ctx.host.session_pool = session_pool
    agent_ctx.team_mode_config = cfg
    agent_ctx.agent_registry = agent_registry or MagicMock()
    agent_ctx.session.session_id = session_id
    agent_ctx.delegation = delegation or MagicMock()

    ctx = MagicMock()
    ctx.deps = agent_ctx
    return ctx


def _init_team(
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


def _make_mock_pool() -> MagicMock:
    """Create a mock SessionPool with async send_message and close_session."""
    pool = MagicMock()
    pool.send_message = AsyncMock(return_value="msg_id_001")
    pool.close_session = AsyncMock()
    # Mock create_child_session for _create_member_session path
    mock_child_state = MagicMock()
    mock_child_state.session_id = "child_session_001"
    pool.create_child_session = AsyncMock(return_value=mock_child_state)
    pool.sessions = MagicMock()
    pool.sessions.get_or_create_session_agent = AsyncMock()
    pool.event_bus = None  # Skip SpawnSessionStart emission in unit tests
    return pool


def _make_mock_registry(exists: bool = True) -> MagicMock:
    """Create a mock AgentRegistry."""
    registry = MagicMock()
    registry.exists = MagicMock(return_value=exists)
    return registry


def _make_mock_delegation() -> MagicMock:
    """Create a mock DelegationService with async create_child_session."""
    delegation = MagicMock()
    delegation.create_child_session = AsyncMock(return_value="child_session_001")
    return delegation


def _make_agent_info() -> AgentInfo:
    """Create a minimal AgentInfo for FunctionModel calls."""
    return AgentInfo(
        function_tools=[],
        allow_text_output=True,
        output_tools=[],
        model_settings=None,
        model_request_parameters=None,
        instructions=None,
    )


def _make_flow_model(
    scripted_calls: list[dict[str, Any]],
) -> tuple[FunctionModel, Any]:
    """Create a FunctionModel that returns scripted tool calls in sequence.

    Args:
        scripted_calls: List of dicts with ``tool`` (str) and optional
            ``args`` (dict) keys. Each entry corresponds to one turn.

    Returns:
        A tuple of (FunctionModel, model_function) where model_function
        is the underlying async callable for direct invocation.
    """
    call_count = 0

    async def model_function(
        messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        del messages, agent_info  # unused — deterministic script
        nonlocal call_count
        idx = call_count
        call_count += 1
        if idx < len(scripted_calls):
            call = scripted_calls[idx]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=call["tool"],
                        args=call.get("args", {}),
                    ),
                ],
            )
        return ModelResponse(parts=[TextPart(content="Done")])

    return FunctionModel(function=model_function), model_function


async def _run_flow(
    model_function: Any,
    cap: TeamCommCapability,
    ctx_factory: Any,
    *,
    max_turns: int = 15,
) -> list[str]:
    """Run a scripted flow by dispatching FunctionModel tool calls to capability methods.

    Args:
        model_function: The async model function from ``_make_flow_model``.
        cap: The TeamCommCapability instance with registered tools.
        ctx_factory: A callable that returns a fresh mock RunContext per turn.
        max_turns: Safety limit to prevent infinite loops.

    Returns:
        List of tool result strings, one per turn.
    """
    results: list[str] = []
    messages: list[ModelMessage] = []
    agent_info = _make_agent_info()

    for _ in range(max_turns):
        response = await model_function(messages, agent_info)

        tool_parts = [p for p in response.parts if isinstance(p, ToolCallPart)]
        if not tool_parts:
            break  # TextPart("Done") — flow complete

        part = tool_parts[0]
        tool_name = part.tool_name
        raw_args = part.args
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        ctx = ctx_factory()
        result = await _dispatch_tool(cap, tool_name, ctx, args)
        results.append(result)
        messages.append(response)

    return results


async def _dispatch_tool(  # noqa: PLR0911
    cap: TeamCommCapability,
    tool_name: str,
    ctx: MagicMock,
    args: dict[str, Any],
) -> str:
    """Dispatch a tool call to the corresponding TeamCommCapability method.

    Args:
        cap: The capability instance.
        tool_name: Name of the team tool to call.
        ctx: The mock RunContext.
        args: Keyword arguments for the tool method.

    Returns:
        The string result from the tool method.

    Raises:
        ValueError: If the tool name is not recognized.
    """
    match tool_name:
        case "team_create":
            return await cap.team_create(ctx, args["name"], args["members"])
        case "team_delete":
            return await cap.team_delete(ctx)
        case "team_status":
            return await cap.team_status(ctx)
        case "send_message":
            return await cap.send_message(
                ctx,
                args["to"],
                args["body"],
                message_type=args.get("message_type", ""),
            )
        case "task_create":
            return await cap.task_create(
                ctx,
                args["subject"],
                owner=args.get("owner", "translator_agent"),
                description=args.get("description", ""),
                blocked_by=args.get("blocked_by"),
            )
        case "task_list":
            return await cap.task_list(ctx)
        case "task_update":
            return await cap.task_update(
                ctx,
                args["task_id"],
                status=args.get("status", ""),
                owner=args.get("owner", ""),
            )
        case "read_blackboard":
            return await cap.read_blackboard(ctx, args["key"])
        case "write_blackboard":
            return await cap.write_blackboard(
                ctx,
                args["key"],
                args["value"],
                expected_version=args.get("expected_version"),
            )
        case "list_blackboard":
            return await cap.list_blackboard(ctx)
        case "delete_blackboard":
            return await cap.delete_blackboard(ctx, args["key"])
        case "shutdown_request":
            return await cap.shutdown_request(ctx, args["member_name"])
        case _:
            msg = f"Unknown tool: {tool_name}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Test 1: Full lifecycle — 7-turn conversation
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_flow_full_lifecycle(tmp_path: Any) -> None:
    """Script a 7-turn conversation covering the full team lifecycle.

    Turn 1: team_create with 2 members → assert team_id returned
    Turn 2: team_status → assert team info returned
    Turn 3: task_create with subject "Review PR" → assert task_id returned
    Turn 4: task_list → assert task appears in list
    Turn 5: write_blackboard key="review_status" value="in_progress" → version=1
    Turn 6: read_blackboard key="review_status" → value matches
    Turn 7: team_delete → assert "Team deleted" returned
    """
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    mock_pool = _make_mock_pool()
    mock_registry = _make_mock_registry()
    mock_delegation = _make_mock_delegation()

    lead_meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }

    def ctx_factory() -> MagicMock:
        return _make_run_context(
            metadata=lead_meta,
            session_pool=mock_pool,
            config=config,
            base_dir=str(tmp_path),
            agent_registry=mock_registry,
            delegation=mock_delegation,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    _model, model_fn = _make_flow_model([
        {
            "tool": "team_create",
            "args": {
                "name": "test_team",
                "members": [
                    {"name": "analyst", "agent": "worker"},
                    {"name": "reviewer", "agent": "reviewer"},
                ],
            },
        },
        {"tool": "team_status", "args": {}},
        {
            "tool": "task_create",
            "args": {
                "subject": "Review PR",
                "description": "Review PR #42",
            },
        },
        {"tool": "task_list", "args": {}},
        {
            "tool": "write_blackboard",
            "args": {
                "key": "review_status",
                "value": "in_progress",
            },
        },
        {"tool": "read_blackboard", "args": {"key": "review_status"}},
        {"tool": "team_delete", "args": {}},
    ])

    results = await _run_flow(model_fn, cap, ctx_factory)

    assert len(results) == 7

    # Turn 1: team_create
    assert "Team 'test_team' created with 2 members" in results[0].return_value
    assert "team_id=" in results[0].return_value
    team_id = results[0].return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "test_team"

    # Turn 2: team_status
    assert "test_team" in results[1].return_value
    assert "analyst" in results[1].return_value
    assert "reviewer" in results[1].return_value

    # Turn 3: task_create
    assert results[2].return_value.startswith("Task created: ")
    task_id = results[2].return_value.replace("Task created: ", "")

    # Turn 4: task_list
    assert "<task_list>" in results[3].return_value
    assert "Review PR" in results[3].return_value
    assert task_id in results[3].return_value

    # Turn 5: write_blackboard
    assert results[4].return_value == "Written, version=1"

    # Turn 6: read_blackboard
    rv5 = (
        "\n".join(results[5].return_value)
        if isinstance(results[5].return_value, list)
        else results[5].return_value
    )
    assert "<blackboard" in rv5
    assert "in_progress" in rv5
    assert 'version="1"' in rv5

    # Turn 7: team_delete
    assert results[6].return_value == "Team deleted"
    assert mock_pool.close_session.await_count == 2


# ---------------------------------------------------------------------------
# Test 2: Send message and task update
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_flow_send_message_and_task_update(tmp_path: Any) -> None:
    """Multi-turn flow with messaging and task lifecycle.

    Turn 1: team_create with 1 member
    Turn 2: send_message to member
    Turn 3: task_create with blocked_by=[]
    Turn 4: task_update to mark task completed
    Turn 5: team_delete
    """
    config = _make_enabled_config(
        member_eligible=["worker"],
        base_dir=str(tmp_path),
    )
    mock_pool = _make_mock_pool()
    mock_registry = _make_mock_registry()
    mock_delegation = _make_mock_delegation()

    lead_meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }

    def ctx_factory() -> MagicMock:
        return _make_run_context(
            metadata=lead_meta,
            session_pool=mock_pool,
            config=config,
            base_dir=str(tmp_path),
            agent_registry=mock_registry,
            delegation=mock_delegation,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    _model, model_fn = _make_flow_model([
        {
            "tool": "team_create",
            "args": {
                "name": "solo_team",
                "members": [
                    {"name": "worker_agent", "agent": "worker"},
                ],
            },
        },
        {
            "tool": "send_message",
            "args": {
                "to": "worker_agent",
                "body": "Please start working on the task",
            },
        },
        {
            "tool": "task_create",
            "args": {
                "subject": "Implement feature X",
                "description": "Build the new feature",
                "blocked_by": [],
            },
        },
        {
            "tool": "task_update",
            "args": {
                "task_id": "__PLACEHOLDER__",
                "status": "completed",
            },
        },
        {"tool": "team_delete", "args": {}},
    ])

    # We need the task_id from turn 3 for turn 4, so run turn-by-turn.
    agent_info = _make_agent_info()
    messages: list[ModelMessage] = []
    all_results: list[str] = []

    # Turn 1: team_create
    resp = await model_fn(messages, agent_info)
    ctx = ctx_factory()
    r1_args = resp.parts[0].args
    r1 = await _dispatch_tool(
        cap,
        "team_create",
        ctx,
        r1_args if isinstance(r1_args, dict) else json.loads(r1_args),
    )
    all_results.append(r1)
    messages.append(resp)
    assert "Team 'solo_team' created with 1 members" in r1.return_value
    team_id = r1.return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "solo_team"

    # Turn 2: send_message
    resp = await model_fn(messages, agent_info)
    ctx = ctx_factory()
    r2 = await _dispatch_tool(
        cap,
        "send_message",
        ctx,
        {"to": "worker_agent", "body": "Please start working on the task"},
    )
    all_results.append(r2)
    messages.append(resp)
    assert r2.return_value == "Message sent to worker_agent"

    # Turn 3: task_create
    resp = await model_fn(messages, agent_info)
    ctx = ctx_factory()
    r3 = await _dispatch_tool(
        cap,
        "task_create",
        ctx,
        {
            "subject": "Implement feature X",
            "description": "Build the new feature",
            "blocked_by": [],
        },
    )
    all_results.append(r3)
    messages.append(resp)
    assert r3.return_value.startswith("Task created: ")
    task_id = r3.return_value.replace("Task created: ", "")

    # Turn 4: task_update with the actual task_id
    resp = await model_fn(messages, agent_info)
    ctx = ctx_factory()
    r4 = await _dispatch_tool(
        cap,
        "task_update",
        ctx,
        {
            "task_id": task_id,
            "status": "completed",
        },
    )
    all_results.append(r4)
    messages.append(resp)
    updated = r4
    assert 'status="completed"' in updated.return_value
    assert "<task" in updated.return_value

    # Turn 5: team_delete
    resp = await model_fn(messages, agent_info)
    ctx = ctx_factory()
    r5 = await _dispatch_tool(cap, "team_delete", ctx, {})
    all_results.append(r5)
    messages.append(resp)
    assert r5.return_value == "Team deleted"
    assert mock_pool.close_session.await_count == 1


# ---------------------------------------------------------------------------
# Test 3: Broadcast message
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_flow_broadcast_message(tmp_path: Any) -> None:
    """Lead broadcasts to all members.

    Turn 1: team_create with 2 members
    Turn 2: send_message with to="*" (broadcast)
    Turn 3: team_delete
    """
    config = _make_enabled_config(
        member_eligible=["worker", "reviewer"],
        base_dir=str(tmp_path),
    )
    mock_pool = _make_mock_pool()
    mock_registry = _make_mock_registry()
    mock_delegation = _make_mock_delegation()

    lead_meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }

    def ctx_factory() -> MagicMock:
        return _make_run_context(
            metadata=lead_meta,
            session_pool=mock_pool,
            config=config,
            base_dir=str(tmp_path),
            agent_registry=mock_registry,
            delegation=mock_delegation,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    _model, model_fn = _make_flow_model([
        {
            "tool": "team_create",
            "args": {
                "name": "broadcast_team",
                "members": [
                    {"name": "analyst", "agent": "worker"},
                    {"name": "reviewer", "agent": "reviewer"},
                ],
            },
        },
        {
            "tool": "send_message",
            "args": {
                "to": "*",
                "body": "Team standup at 9am",
            },
        },
        {"tool": "team_delete", "args": {}},
    ])

    results = await _run_flow(model_fn, cap, ctx_factory)

    assert len(results) == 3

    # Turn 1: team_create
    assert "Team 'broadcast_team' created with 2 members" in results[0].return_value
    team_id = results[0].return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "broadcast_team"

    # Turn 2: broadcast
    assert "Broadcast sent to 2 members" in results[1].return_value
    # 2 send_message calls for broadcast (one per member) + 2 from team_create = 4 total
    assert mock_pool.send_message.await_count >= 4

    # Turn 3: team_delete
    assert results[2].return_value == "Team deleted"
    assert mock_pool.close_session.await_count == 2


# ---------------------------------------------------------------------------
# Test 4: Blackboard versioning with optimistic locking
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_flow_blackboard_versioning(tmp_path: Any) -> None:
    """Optimistic locking on blackboard writes.

    Turn 1: team_create
    Turn 2: write_blackboard key="config" value="v1" → version=1
    Turn 3: write_blackboard key="config" value="v2" expected_version=1 → version=2
    Turn 4: write_blackboard key="config" value="v3" expected_version=1 → conflict error
    Turn 5: team_delete
    """
    config = _make_enabled_config(
        member_eligible=["worker"],
        base_dir=str(tmp_path),
    )
    mock_pool = _make_mock_pool()
    mock_registry = _make_mock_registry()
    mock_delegation = _make_mock_delegation()

    lead_meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }

    def ctx_factory() -> MagicMock:
        return _make_run_context(
            metadata=lead_meta,
            session_pool=mock_pool,
            config=config,
            base_dir=str(tmp_path),
            agent_registry=mock_registry,
            delegation=mock_delegation,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    _model, model_fn = _make_flow_model([
        {
            "tool": "team_create",
            "args": {
                "name": "bb_team",
                "members": [
                    {"name": "worker_agent", "agent": "worker"},
                ],
            },
        },
        {
            "tool": "write_blackboard",
            "args": {
                "key": "config",
                "value": "v1",
            },
        },
        {
            "tool": "write_blackboard",
            "args": {
                "key": "config",
                "value": "v2",
                "expected_version": 1,
            },
        },
        {
            "tool": "write_blackboard",
            "args": {
                "key": "config",
                "value": "v3",
                "expected_version": 1,
            },
        },
        {"tool": "team_delete", "args": {}},
    ])

    results = await _run_flow(model_fn, cap, ctx_factory)

    assert len(results) == 5

    # Turn 1: team_create
    assert "Team 'bb_team' created with 1 members" in results[0].return_value
    team_id = results[0].return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "bb_team"

    # Turn 2: first write — version=1
    assert results[1].return_value == "Written, version=1"

    # Turn 3: second write with correct expected_version=1 → version=2
    assert results[2].return_value == "Written, version=2"

    # Turn 4: third write with stale expected_version=1 → conflict
    assert results[3].return_value == "Conflict: current version is 2"

    # Turn 5: team_delete
    assert results[4].return_value == "Team deleted"


# ---------------------------------------------------------------------------
# Test 5: Error recovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_flow_error_recovery(tmp_path: Any) -> None:
    """Error handling and recovery within a flow.

    Turn 1: task_create without team → "Not in a team session" error
    Turn 2: team_create → success
    Turn 3: task_create → success (recovery after error)
    Turn 4: team_delete
    """
    config = _make_enabled_config(
        member_eligible=["worker"],
        base_dir=str(tmp_path),
    )
    mock_pool = _make_mock_pool()
    mock_registry = _make_mock_registry()
    mock_delegation = _make_mock_delegation()

    # Start WITHOUT team_id — turn 1 should fail
    lead_meta: dict[str, Any] = {
        "team_role": "lead",
        "team_member_name": "coordinator",
    }

    def ctx_factory() -> MagicMock:
        return _make_run_context(
            metadata=lead_meta,
            session_pool=mock_pool,
            config=config,
            base_dir=str(tmp_path),
            agent_registry=mock_registry,
            delegation=mock_delegation,
        )

    cap = TeamCommCapability(config, "coordinator", lead_meta)

    # Run turns 1-3 first (without team_delete) so we can verify task state
    # before the team is cleaned up.
    _model, model_fn = _make_flow_model([
        {
            "tool": "task_create",
            "args": {
                "subject": "Premature task",
                "description": "This should fail — no team yet",
            },
        },
        {
            "tool": "team_create",
            "args": {
                "name": "recovery_team",
                "members": [
                    {"name": "worker_agent", "agent": "worker"},
                ],
            },
        },
        {
            "tool": "task_create",
            "args": {
                "subject": "Recovered task",
                "description": "This should succeed after team is created",
            },
        },
    ])

    results = await _run_flow(model_fn, cap, ctx_factory)

    assert len(results) == 3

    # Turn 1: task_create without team → error
    assert results[0].return_value == "Not in a team session"

    # Turn 2: team_create → success
    assert "Team 'recovery_team' created with 1 members" in results[1].return_value
    team_id = results[1].return_value.split("team_id=")[1].strip()
    lead_meta["team_id"] = team_id
    lead_meta["team_name"] = "recovery_team"

    # Turn 3: task_create → success (recovery)
    assert results[2].return_value.startswith("Task created: ")
    task_id = results[2].return_value.replace("Task created: ", "")

    # Verify the task was actually created (before team_delete cleans up)
    ctx_verify = ctx_factory()
    list_result = await cap.task_list(ctx_verify)
    assert "<task_list>" in list_result.return_value
    assert "Recovered task" in list_result.return_value
    assert task_id in list_result.return_value

    # Turn 4: team_delete (run separately to preserve verification above)
    delete_result = await cap.team_delete(ctx_factory())
    assert delete_result.return_value == "Team deleted"
    assert mock_pool.close_session.await_count == 1
