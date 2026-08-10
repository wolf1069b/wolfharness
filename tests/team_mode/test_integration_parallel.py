"""L2 integration tests for parallel delegation via multiple ToolCallParts.

These tests verify that when a ``FunctionModel`` issues multiple
``ToolCallPart`` entries in a single ``ModelResponse``, all tool calls
are dispatched and executed successfully.  This simulates the pattern
where an LLM assigns tasks to multiple team members in parallel within
a single turn.

The tests use the real ``team_mode_pool`` fixture and custom
``FunctionModel`` factories that produce ``ModelResponse`` objects with
multiple ``ToolCallPart`` entries.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from tests.team_mode.conftest import build_agent_context, make_mock_run_context
from wolfharness.capabilities.team_comm_capability import TeamCommCapability


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness_config.team_mode import TeamModeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LEAD_METADATA: dict[str, Any] = {
    "team_role": "lead",
    "team_member_name": "coordinator",
}


def _make_agent_info() -> AgentInfo:
    """Create a minimal ``AgentInfo`` for ``FunctionModel`` calls."""
    return AgentInfo(
        function_tools=[],
        allow_text_output=True,
        output_tools=[],
        model_settings=None,
        model_request_parameters=None,
        instructions=None,
    )


def _make_parallel_model(
    turns: list[list[tuple[str, dict[str, Any]]]],
) -> tuple[FunctionModel, Any]:
    """Create a ``FunctionModel`` that issues multiple tool calls per turn.

    Each "turn" is a list of ``(tool_name, args)`` tuples.  All tool
    calls within a turn are issued as separate ``ToolCallPart`` entries
    in a single ``ModelResponse``.  After all turns are consumed, the
    model returns a final ``TextPart("done")`` response.

    Args:
        turns: List of turns, where each turn is a list of
            ``(tool_name, args)`` tuples.

    Returns:
        A tuple of ``(FunctionModel, model_function)`` where
        ``model_function`` is the underlying async callable.
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
        if idx < len(turns):
            parts = [
                ToolCallPart(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id=f"call_{idx}_{i}",
                )
                for i, (tool_name, args) in enumerate(turns[idx])
            ]
            return ModelResponse(parts=parts)
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(function=model_function), model_function


async def _setup_lead_session(
    pool: AgentPool[Any],
    session_id: str,
) -> tuple[str, TeamModeConfig, Any]:
    """Create a lead session and return (session_id, config, agent_ctx).

    Args:
        pool: Real AgentPool instance.
        session_id: Unique session identifier for the lead.

    Returns:
        A tuple of (session_id, team_mode_config, agent_ctx).
    """
    manifest = pool.manifest
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    session_pool = pool.session_pool
    assert session_pool is not None
    await session_pool.create_session(
        session_id,
        agent_name="coordinator",
        team_role="lead",
        team_member_name="coordinator",
    )

    agent_ctx = build_agent_context(pool, session_id, team_mode_config)
    return session_id, team_mode_config, agent_ctx


async def _create_team(
    cap: TeamCommCapability,
    agent_ctx: Any,
    team_name: str,
    members: list[dict[str, str]],
) -> str:
    """Call ``team_create`` and return the extracted team_id.

    Args:
        cap: TeamCommCapability instance with lead metadata.
        agent_ctx: Real AgentContext from build_agent_context.
        team_name: Human-readable team name.
        members: List of member dicts with ``agent`` and ``name`` keys.

    Returns:
        The team_id string.
    """
    ctx = make_mock_run_context(agent_ctx)
    result = await cap.team_create(ctx, team_name, members)
    assert "team_id=" in result.return_value
    team_id = result.return_value.split("team_id=")[1].strip()
    agent_ctx.session.metadata["team_id"] = team_id
    agent_ctx.session.metadata["team_name"] = team_name
    cap._session_metadata["team_id"] = team_id
    cap._session_metadata["team_name"] = team_name
    return team_id


async def _dispatch_tool(
    cap: TeamCommCapability,
    tool_name: str,
    ctx: Any,
    args: dict[str, Any],
) -> str:
    """Dispatch a single tool call to the corresponding capability method.

    Args:
        cap: The TeamCommCapability instance.
        tool_name: Name of the team tool to call.
        ctx: The mock RunContext.
        args: Keyword arguments for the tool method.

    Returns:
        The string result from the tool method.

    Raises:
        ValueError: If the tool name is not recognized.
    """
    match tool_name:
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
        case "team_create":
            return await cap.team_create(ctx, args["name"], args["members"])
        case "team_delete":
            return await cap.team_delete(ctx)
        case _:
            msg = f"Unknown tool: {tool_name}"
            raise ValueError(msg)


async def _dispatch_parallel_turn(
    cap: TeamCommCapability,
    response: ModelResponse,
    ctx: Any,
) -> list[str]:
    """Dispatch all ToolCallParts in a single ModelResponse.

    Args:
        cap: The TeamCommCapability instance.
        response: The ModelResponse containing one or more ToolCallParts.
        ctx: The mock RunContext.

    Returns:
        List of tool result strings, one per ToolCallPart.
    """
    results: list[str] = []
    for part in response.parts:
        if not isinstance(part, ToolCallPart):
            continue
        raw_args = part.args
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        result = await _dispatch_tool(cap, part.tool_name, ctx, args)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_multiple_tool_calls_in_one_response(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Multiple tool calls in a single ModelResponse are all dispatched.

    Given: a team with 2 members and a FunctionModel that issues 2
        ``send_message`` calls in a single ``ModelResponse``.

    When: the parallel tool calls are dispatched.

    Then: both messages are delivered successfully.  Each
        ``send_message`` returns ``"Message sent to {member}"``.  This
        verifies that multiple tool calls within a single model
        response are handled independently and both succeed.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        team_mode_pool,
        "parallel-msg-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    await _create_team(
        cap,
        agent_ctx,
        "parallel_msg_team",
        [
            {"agent": "worker", "name": "worker_1"},
            {"agent": "reviewer", "name": "reviewer_1"},
        ],
    )

    # Build a FunctionModel that issues 2 send_message calls in one turn.
    _model, model_fn = _make_parallel_model([
        [
            (
                "send_message",
                {"to": "worker_1", "body": "Task A: analyze the data"},
            ),
            (
                "send_message",
                {"to": "reviewer_1", "body": "Task B: review the report"},
            ),
        ],
    ])

    agent_info = _make_agent_info()
    messages: list[ModelMessage] = []
    response = await model_fn(messages, agent_info)

    # Verify the response contains 2 ToolCallParts.
    tool_parts = [p for p in response.parts if isinstance(p, ToolCallPart)]
    assert len(tool_parts) == 2

    ctx = make_mock_run_context(agent_ctx)
    results = await _dispatch_parallel_turn(cap, response, ctx)

    assert len(results) == 2
    assert results[0].return_value == "Message sent to worker_1"
    assert results[1].return_value == "Message sent to reviewer_1"

    # Cleanup.
    session_pool = team_mode_pool.session_pool
    assert session_pool is not None
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)


@pytest.mark.integration
async def test_parallel_task_assignment(
    team_mode_pool: AgentPool[Any],
) -> None:
    """Parallel task assignment creates all tasks in one turn.

    Given: a team with 2 members and a FunctionModel that issues 2
        ``task_create`` calls in a single ``ModelResponse``.

    When: the parallel tool calls are dispatched.

    Then: both tasks are created successfully.  Each ``task_create``
        returns ``"Task created: {task_id}"``.  ``task_list`` confirms
        2 tasks exist on the shared task board.  This verifies that
        parallel task assignment within a single turn creates all
        tasks without interference.
    """
    session_id, config, agent_ctx = await _setup_lead_session(
        team_mode_pool,
        "parallel-task-lead-001",
    )

    cap = TeamCommCapability(
        config,
        "coordinator",
        session_metadata={**_LEAD_METADATA},
    )

    await _create_team(
        cap,
        agent_ctx,
        "parallel_task_team",
        [
            {"agent": "worker", "name": "worker_1"},
            {"agent": "reviewer", "name": "reviewer_1"},
        ],
    )

    # Build a FunctionModel that issues 2 task_create calls in one turn.
    _model, model_fn = _make_parallel_model([
        [
            (
                "task_create",
                {
                    "subject": "Analyze dataset",
                    "description": "Worker: analyze the customer dataset",
                },
            ),
            (
                "task_create",
                {
                    "subject": "Review findings",
                    "description": "Reviewer: review the analysis findings",
                },
            ),
        ],
    ])

    agent_info = _make_agent_info()
    messages: list[ModelMessage] = []
    response = await model_fn(messages, agent_info)

    # Verify the response contains 2 ToolCallParts.
    tool_parts = [p for p in response.parts if isinstance(p, ToolCallPart)]
    assert len(tool_parts) == 2

    ctx = make_mock_run_context(agent_ctx)
    results = await _dispatch_parallel_turn(cap, response, ctx)

    assert len(results) == 2
    assert results[0].return_value.startswith("Task created: ")
    assert results[1].return_value.startswith("Task created: ")

    # Extract task IDs and verify they are distinct.
    task_id_1 = results[0].return_value.replace("Task created: ", "")
    task_id_2 = results[1].return_value.replace("Task created: ", "")
    assert task_id_1 != task_id_2

    # Verify both tasks appear in task_list.
    list_result = await cap.task_list(ctx)
    assert "<task_list>" in list_result.return_value
    assert "Analyze dataset" in list_result.return_value
    assert "Review findings" in list_result.return_value
    assert task_id_1 in list_result.return_value
    assert task_id_2 in list_result.return_value

    # Cleanup.
    session_pool = team_mode_pool.session_pool
    assert session_pool is not None
    await cap.team_delete(ctx)
    await session_pool.close_session(session_id)
