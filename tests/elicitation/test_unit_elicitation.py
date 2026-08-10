"""Unit tests for the Durable Elicitation Bridge implementation.

Covers Tasks 9.1-9.7, 9.14-9.16, 9.18-9.19, 9.21-9.22 from the
durable-elicitation-bridge plan.

Refs: https://github.com/Leoyzen/wolfharness/issues/107


# TODO: L2 migration — test uses complex inline mock_pool + mock_session_pool
# patterns that require significant rework for real pool migration.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import ElicitRequestFormParams, ElicitResult
from pydantic import TypeAdapter
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, RunContext
import pytest

from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events.events import ElicitationDeferredEvent
from wolfharness.agents.native_agent.elicitation_bridge import (
    ElicitationFutureRegistry,
    create_elicitation_bridge_capability,
)
from wolfharness.agents.native_agent.elicitation_strategy import (
    CheckpointResolutionStrategy,
    ElicitationResolutionStrategy,
    ProtocolResolutionStrategy,
)
from wolfharness.sessions.models import ElicitationResumePayload, PendingDeferredCall
from wolfharness.tools import CallDeferred
from wolfharness.ui.base import InputProvider
from wolfharness.ui.elicitation import normalize_elicit_content


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def agent_ctx() -> AgentContext:
    """Create a minimal AgentContext for testing."""
    node = MagicMock()
    node.name = "test-agent"
    run_ctx = AgentRunContext(session_id="test-session")
    return AgentContext(node=node, run_ctx=run_ctx)


@pytest.fixture
def run_ctx(agent_ctx: AgentContext) -> RunContext[Any]:
    """Create a mock RunContext with AgentContext deps."""
    model = MagicMock()
    model.system = "test"
    model.model_name = "test-model"
    return RunContext(
        deps=agent_ctx,
        model=model,
        usage=MagicMock(),
    )


@pytest.fixture
def form_params() -> ElicitRequestFormParams:
    """Create form elicitation params for testing."""
    return ElicitRequestFormParams(
        message="Please enter your name",
        requestedSchema={"type": "object", "properties": {"name": {"type": "string"}}},
        mode="form",
    )


# ============================================================================
# Task 9.1: Serialization roundtrip
# ============================================================================


@pytest.mark.unit
def test_pending_deferred_call_serialization_roundtrip() -> None:
    """PendingDeferredCall with elicitation fields survives serialization roundtrip."""
    original = PendingDeferredCall(
        tool_call_id="tc-001",
        tool_name="mcp_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
        elicitation_message="Enter your API key",
        elicitation_schema={"type": "object", "properties": {"key": {"type": "string"}}},
        elicitation_mode="form",
        mcp_server_id="server-1",
    )
    adapter: TypeAdapter[list[PendingDeferredCall]] = TypeAdapter(list[PendingDeferredCall])
    serialized = adapter.dump_python([original])
    deserialized_list = adapter.validate_python(serialized)
    assert len(deserialized_list) == 1
    result = deserialized_list[0]
    assert result.tool_call_id == "tc-001"
    assert result.tool_name == "mcp_tool"
    assert result.deferred_kind == "elicitation"
    assert result.deferred_strategy == "block"
    assert result.elicitation_message == "Enter your API key"
    assert result.elicitation_schema == {
        "type": "object",
        "properties": {"key": {"type": "string"}},
    }
    assert result.elicitation_mode == "form"
    assert result.mcp_server_id == "server-1"


@pytest.mark.unit
def test_pending_deferred_call_serialization_none_fields() -> None:
    """PendingDeferredCall with None elicitation fields serializes correctly."""
    original = PendingDeferredCall(
        tool_call_id="tc-002",
        tool_name="bash",
        deferred_kind="external",
        deferred_strategy="block",
    )
    adapter: TypeAdapter[list[PendingDeferredCall]] = TypeAdapter(list[PendingDeferredCall])
    serialized = adapter.dump_python([original])
    deserialized_list = adapter.validate_python(serialized)
    assert len(deserialized_list) == 1
    result = deserialized_list[0]
    assert result.tool_call_id == "tc-002"
    assert result.elicitation_message is None
    assert result.elicitation_schema is None
    assert result.elicitation_mode is None
    assert result.mcp_server_id is None


# ============================================================================
# Task 9.2: handle_elicitation durable=True
# ============================================================================


@pytest.mark.unit
async def test_handle_elicitation_durable_true_mcp(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation raises CallDeferred when durable and in MCP callback."""
    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = True
    with pytest.raises(CallDeferred) as exc_info:
        await agent_ctx.handle_elicitation(form_params)
    assert exc_info.value.metadata is not None
    assert exc_info.value.metadata["deferred_kind"] == "elicitation"
    elicitation = exc_info.value.metadata["elicitation"]
    assert elicitation["message"] == "Please enter your name"
    assert elicitation["requestedSchema"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    assert elicitation["mode"] == "form"
    # Side-channel should NOT be set by handle_elicitation directly.
    assert agent_ctx._pending_elicitation_deferral is None


@pytest.mark.unit
async def test_handle_elicitation_durable_true_local(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation awaits future when durable and NOT in MCP callback.

    For local tools, the agent run suspends on `await future` instead of
    raising CallDeferred. This is the "pause" pattern — no re-execution.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False  # Local tool path
    agent_ctx.tool_call_id = "tc-local-1"

    # Set up registry on run_ctx
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    # Schedule future resolution after a short delay
    payload = ElicitationResumePayload(
        deferred_handle="tc-local-1",
        action="accept",
        content={"name": "Alice"},
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-local-1", payload)

    task = asyncio.create_task(resolve_later())

    result = await agent_ctx.handle_elicitation(form_params)
    await task

    assert result.action == "accept"
    assert result.content == {"name": "Alice"}
    # No CallDeferred, no side-channel — agent run continued naturally


# ============================================================================
# Task 9.3: handle_elicitation durable=False
# ============================================================================


@pytest.mark.unit
async def test_handle_elicitation_durable_false(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation calls get_elicitation when not durable."""
    expected_result = ElicitResult(action="accept", content={"name": "Alice"})
    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = False
    provider.get_elicitation = AsyncMock(return_value=expected_result)
    agent_ctx.input_provider = provider
    result = await agent_ctx.handle_elicitation(form_params)
    provider.get_elicitation.assert_awaited_once_with(form_params)
    assert result == expected_result
    assert agent_ctx._pending_elicitation_deferral is None


# ============================================================================
# Task 9.4: call_tool raises CallDeferred
# ============================================================================


@pytest.mark.unit
async def test_call_tool_raises_call_deferred(agent_ctx: AgentContext) -> None:
    """MCPClient.call_tool raises CallDeferred when side-channel is set."""
    from wolfharness.mcp_server.client import MCPClient
    from wolfharness_config.mcp_server import StdioMCPServerConfig

    agent_ctx._pending_elicitation_deferral = {
        "message": "Enter credentials",
        "requestedSchema": {"type": "object"},
        "mode": "form",
    }
    config = StdioMCPServerConfig(command="echo", args=["test"])
    client = MCPClient(config=config)
    mock_inner_client = MagicMock()
    mock_call_result = MagicMock()
    mock_call_result.is_error = False
    mock_call_result.content = []
    mock_call_result.data = "test-data"
    mock_inner_client.call_tool = AsyncMock(return_value=mock_call_result)
    mock_inner_client.is_connected.return_value = True
    mock_inner_client.session = MagicMock()
    mock_inner_client.session.get_server_capabilities.return_value = None
    client._client = mock_inner_client

    with (
        patch(
            "wolfharness.mcp_server.conversions.from_mcp_content",
            new=AsyncMock(return_value=[]),
        ),
        pytest.raises(CallDeferred) as exc_info,
    ):
        await client.call_tool("test_tool", MagicMock(), {"arg": "val"}, agent_ctx)
    assert exc_info.value.metadata is not None
    assert exc_info.value.metadata["deferred_kind"] == "elicitation"
    elicitation_params = exc_info.value.metadata["elicitation"]
    assert elicitation_params["message"] == "Enter credentials"
    assert agent_ctx._pending_elicitation_deferral is None


# ============================================================================
# Task 9.5: call_tool normal path (no deferral)
# ============================================================================


@pytest.mark.unit
async def test_call_tool_normal_path(agent_ctx: AgentContext) -> None:
    """MCPClient.call_tool returns normally when no deferral is pending."""
    from wolfharness.mcp_server.client import MCPClient
    from wolfharness_config.mcp_server import StdioMCPServerConfig

    assert agent_ctx._pending_elicitation_deferral is None
    config = StdioMCPServerConfig(command="echo", args=["test"])
    client = MCPClient(config=config)
    mock_inner_client = MagicMock()
    mock_call_result = MagicMock()
    mock_call_result.is_error = False
    mock_call_result.content = []
    mock_call_result.data = "normal-result"
    mock_inner_client.call_tool = AsyncMock(return_value=mock_call_result)
    mock_inner_client.is_connected.return_value = True
    mock_inner_client.session = MagicMock()
    mock_inner_client.session.get_server_capabilities.return_value = None
    client._client = mock_inner_client

    with patch(
        "wolfharness.mcp_server.conversions.from_mcp_content",
        new=AsyncMock(return_value=[]),
    ):
        result = await client.call_tool("test_tool", MagicMock(), {"arg": "val"}, agent_ctx)
    assert result == "normal-result"


# ============================================================================
# Task 9.6: Bridge handles elicitation + passthrough
# ============================================================================


@pytest.mark.unit
async def test_bridge_handles_elicitation_and_passthrough(run_ctx: RunContext[Any]) -> None:
    """Elicitation bridge handles elicitation calls and passes through non-elicitation."""
    registry = ElicitationFutureRegistry()
    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    cap = create_elicitation_bridge_capability(
        registry=registry,
        checkpoint_manager=mock_checkpoint,
        agent_config_hash="abc123",
    )
    elicitation_call = ToolCallPart(
        tool_name="mcp_tool",
        args={"query": "data"},
        tool_call_id="tc-elicit-1",
    )
    normal_call = ToolCallPart(
        tool_name="bash",
        args={"cmd": "ls"},
        tool_call_id="tc-normal-1",
    )
    requests = DeferredToolRequests(
        calls=[elicitation_call, normal_call],
        metadata={
            "tc-elicit-1": {
                "deferred_kind": "elicitation",
                "elicitation": {
                    "message": "Enter key",
                    "requestedSchema": {"type": "object"},
                    "mode": "form",
                },
            },
            "tc-normal-1": {
                "deferred_kind": "external",
            },
        },
    )
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    run_ctx.deps.run_ctx.event_bus = mock_bus

    with patch(
        "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
        new_callable=AsyncMock,
    ) as mock_emit:
        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

    mock_checkpoint.checkpoint.assert_awaited_once()
    checkpoint_kwargs = mock_checkpoint.checkpoint.call_args.kwargs
    assert checkpoint_kwargs["session_id"] == "test-session"
    assert checkpoint_kwargs["agent_config_hash"] == "abc123"
    pending_calls = checkpoint_kwargs["pending_calls"]
    assert len(pending_calls) == 1
    assert pending_calls[0].deferred_kind == "elicitation"
    assert pending_calls[0].tool_call_id == "tc-elicit-1"
    mock_emit.assert_awaited_once()
    emit_event = mock_emit.call_args[0][1]
    assert isinstance(emit_event, ElicitationDeferredEvent)
    assert emit_event.deferred_handle == "tc-elicit-1"
    assert emit_event.message == "Enter key"
    assert "tc-elicit-1" in registry
    assert run_ctx.deps.run_ctx.checkpointed is True
    assert result is None


# ============================================================================
# Task 9.7: FutureRegistry lifecycle
# ============================================================================


@pytest.mark.unit
async def test_future_registry_lifecycle() -> None:
    """ElicitationFutureRegistry register → resolve → reject_all lifecycle."""
    registry = ElicitationFutureRegistry()
    future1 = registry.register("handle1")
    assert not future1.done()
    payload = ElicitationResumePayload(
        deferred_handle="handle1",
        action="accept",
        content={"value": "test"},
    )
    registry.resolve("handle1", payload)
    assert future1.done()
    assert future1.result() == payload
    assert "handle1" not in registry
    future2 = registry.register("handle2")
    assert not future2.done()
    test_exception = Exception("session closed")
    registry.reject_all(test_exception)
    assert future2.done()
    assert future2.exception() is test_exception
    assert "handle2" not in registry


# ============================================================================
# Task 9.14: supports_durable_elicitation dynamic property
# ============================================================================


@pytest.mark.unit
def test_acp_input_provider_supports_durable_elicitation() -> None:
    """ACPInputProvider.supports_durable_elicitation reflects session.checkpoint_enabled."""
    from wolfharness_server.acp_server.input_provider import ACPInputProvider

    mock_session = MagicMock()
    mock_session.checkpoint_enabled = True
    provider = ACPInputProvider(session=mock_session)
    assert provider.supports_durable_elicitation is True
    mock_session.checkpoint_enabled = False
    assert provider.supports_durable_elicitation is False


@pytest.mark.unit
def test_opencode_input_provider_supports_durable_elicitation() -> None:
    """OpenCodeInputProvider.supports_durable_elicitation checks session state."""
    from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider

    mock_state = MagicMock()
    mock_session = MagicMock()
    mock_session.checkpoint_enabled = True
    mock_controller = MagicMock()
    mock_controller.get_session.return_value = mock_session
    mock_state.session_controller = mock_controller
    provider = OpenCodeInputProvider(state=mock_state, session_id="sess-1")
    assert provider.supports_durable_elicitation is True
    mock_session.checkpoint_enabled = False
    assert provider.supports_durable_elicitation is False
    mock_controller.get_session.return_value = None
    assert provider.supports_durable_elicitation is False
    mock_state.session_controller = None
    assert provider.supports_durable_elicitation is False


# ============================================================================
# Task 9.15: Side-channel cleanup on error
# ============================================================================


@pytest.mark.unit
async def test_side_channel_cleanup_on_error(agent_ctx: AgentContext) -> None:
    """Per-call elicitation handler is cleaned up when call_tool raises an error.

    The finally block in call_tool() clears _current_elicitation_handler
    regardless of success or failure, ensuring no stale handler leaks
    into the next call.
    """
    from wolfharness.mcp_server.client import MCPClient
    from wolfharness_config.mcp_server import StdioMCPServerConfig

    agent_ctx._pending_elicitation_deferral = {
        "message": "Enter key",
        "requestedSchema": {"type": "object"},
        "mode": "form",
    }
    config = StdioMCPServerConfig(command="echo", args=["test"])
    client = MCPClient(config=config)
    mock_inner_client = MagicMock()
    mock_inner_client.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))
    mock_inner_client.is_connected.return_value = True
    mock_inner_client.session = MagicMock()
    mock_inner_client.session.get_server_capabilities.return_value = None
    client._client = mock_inner_client

    with pytest.raises(RuntimeError, match="MCP tool call failed"):
        await client.call_tool("test_tool", MagicMock(), {"arg": "val"}, agent_ctx)
    # The finally block must clear the per-call elicitation handler
    assert client._current_elicitation_handler is None


# ============================================================================
# Task 9.16: Bridge positioning in capability chain
# ============================================================================


@pytest.mark.unit
async def test_bridge_positioning_elicitation_before_approval(run_ctx: RunContext[Any]) -> None:
    """Elicitation bridge handles elicitation calls before approval bridge sees them."""
    registry = ElicitationFutureRegistry()
    cap = create_elicitation_bridge_capability(registry=registry)
    elicitation_call = ToolCallPart(
        tool_name="mcp_elicitation_tool",
        args={"prompt": "Enter credentials"},
        tool_call_id="tc-both-1",
    )
    requests = DeferredToolRequests(
        calls=[elicitation_call],
        metadata={
            "tc-both-1": {
                "deferred_kind": "elicitation",
                "elicitation": {
                    "message": "Enter credentials",
                    "requestedSchema": {"type": "object"},
                    "mode": "form",
                },
            },
        },
    )
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    run_ctx.deps.run_ctx.event_bus = mock_bus

    with patch(
        "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
        new_callable=AsyncMock,
    ):
        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

    assert result is None
    assert "tc-both-1" in registry
    assert run_ctx.deps.run_ctx.checkpointed is True


# ============================================================================
# Task 9.18: Backward compatibility serialization
# ============================================================================


@pytest.mark.unit
def test_backward_compatibility_old_format_serialization() -> None:
    """Old-format PendingDeferredCall (without elicitation fields) deserializes correctly."""
    old_format: dict[str, Any] = {
        "tool_call_id": "tc-old-1",
        "tool_name": "external_tool",
        "deferred_kind": "external",
        "deferred_strategy": "block",
    }
    adapter: TypeAdapter[list[PendingDeferredCall]] = TypeAdapter(list[PendingDeferredCall])
    deserialized_list = adapter.validate_python([old_format])
    assert len(deserialized_list) == 1
    result = deserialized_list[0]
    assert result.tool_call_id == "tc-old-1"
    assert result.tool_name == "external_tool"
    assert result.deferred_kind == "external"
    assert result.deferred_strategy == "block"
    assert result.elicitation_message is None
    assert result.elicitation_schema is None
    assert result.elicitation_mode is None
    assert result.mcp_server_id is None


# ============================================================================
# Task 9.19: ElicitationResumePayload decline/cancel actions
# ============================================================================


@pytest.mark.unit
def test_elicitation_resume_payload_decline_and_cancel() -> None:
    """ElicitationResumePayload constructs correctly for decline and cancel actions."""
    decline_payload = ElicitationResumePayload(
        deferred_handle="tc-001",
        action="decline",
    )
    assert decline_payload.deferred_handle == "tc-001"
    assert decline_payload.action == "decline"
    assert decline_payload.content is None

    cancel_payload = ElicitationResumePayload(
        deferred_handle="tc-002",
        action="cancel",
    )
    assert cancel_payload.deferred_handle == "tc-002"
    assert cancel_payload.action == "cancel"
    assert cancel_payload.content is None

    accept_payload = ElicitationResumePayload(
        deferred_handle="tc-003",
        action="accept",
        content={"name": "Alice"},
    )
    assert accept_payload.action == "accept"
    assert accept_payload.content == {"name": "Alice"}


@pytest.mark.unit
async def test_handle_elicitation_cached_response(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation returns cached response when available (crash recovery)."""
    cached_result = ElicitResult(action="accept", content={"name": "Bob"})
    agent_ctx.tool_call_id = "tc-cached-1"
    assert agent_ctx.run_ctx is not None
    agent_ctx.run_ctx.cached_elicitation_responses["tc-cached-1"] = cached_result
    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    provider.get_elicitation = AsyncMock()
    agent_ctx.input_provider = provider
    result = await agent_ctx.handle_elicitation(form_params)
    assert result == cached_result
    provider.get_elicitation.assert_not_awaited()
    assert agent_ctx._pending_elicitation_deferral is None


# ============================================================================
# Task 9.21: Strategy classes
# ============================================================================


@pytest.mark.unit
async def test_checkpoint_resolution_strategy_delegates() -> None:
    """CheckpointResolutionStrategy.resolve() calls CheckpointManager.checkpoint()."""
    mock_manager = MagicMock()
    mock_manager.checkpoint = AsyncMock()
    strategy = CheckpointResolutionStrategy(
        checkpoint_manager=mock_manager,
        session_id="sess-strategy",
        message_history=[],
        agent_config_hash="hash123",
    )
    pending_call = PendingDeferredCall(
        tool_call_id="tc-strategy-1",
        tool_name="mcp_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
    )
    response = ElicitationResumePayload(
        deferred_handle="tc-strategy-1",
        action="accept",
        content={"value": "yes"},
    )
    result = await strategy.resolve(pending_call, response)
    mock_manager.checkpoint.assert_awaited_once()
    checkpoint_kwargs = mock_manager.checkpoint.call_args.kwargs
    assert checkpoint_kwargs["session_id"] == "sess-strategy"
    assert checkpoint_kwargs["agent_config_hash"] == "hash123"
    assert checkpoint_kwargs["pending_calls"] == [pending_call]
    assert result == response


@pytest.mark.unit
async def test_protocol_resolution_strategy_raises() -> None:
    """ProtocolResolutionStrategy.resolve() raises NotImplementedError."""
    strategy = ProtocolResolutionStrategy()
    pending_call = PendingDeferredCall(
        tool_call_id="tc-proto-1",
        tool_name="mcp_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
    )
    response = ElicitationResumePayload(
        deferred_handle="tc-proto-1",
        action="decline",
    )
    with pytest.raises(NotImplementedError, match="MRTR support not yet available"):
        await strategy.resolve(pending_call, response)


@pytest.mark.unit
def test_elicitation_resolution_strategy_runtime_checkable() -> None:
    """ElicitationResolutionStrategy is runtime_checkable and works with isinstance()."""
    checkpoint_strategy = CheckpointResolutionStrategy(
        checkpoint_manager=MagicMock(),
        session_id="sess-1",
        message_history=[],
    )
    assert isinstance(checkpoint_strategy, ElicitationResolutionStrategy)
    protocol_strategy = ProtocolResolutionStrategy()
    assert isinstance(protocol_strategy, ElicitationResolutionStrategy)


# ============================================================================
# Task 9.22: Elicitation timeout
# ============================================================================


@pytest.mark.unit
def test_elicitation_timeout_serialization() -> None:
    """PendingDeferredCall with timeout field serializes and deserializes correctly."""
    original = PendingDeferredCall(
        tool_call_id="tc-timeout-1",
        tool_name="mcp_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
        timeout=timedelta(seconds=300),
        elicitation_message="Enter OTP",
        elicitation_schema={"type": "object"},
        elicitation_mode="form",
    )
    assert original.timeout == timedelta(seconds=300)
    adapter: TypeAdapter[list[PendingDeferredCall]] = TypeAdapter(list[PendingDeferredCall])
    serialized = adapter.dump_python([original])
    deserialized_list = adapter.validate_python(serialized)
    assert len(deserialized_list) == 1
    result = deserialized_list[0]
    assert result.timeout is not None
    assert result.timeout == timedelta(seconds=300)
    assert result.deferred_kind == "elicitation"
    assert result.elicitation_message == "Enter OTP"


# ============================================================================
# PR Review Coverage: P0 — message_history passed to checkpoint
# ============================================================================


@pytest.mark.unit
async def test_handle_elicitation_passes_current_messages_to_checkpoint(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation passes run_ctx.current_messages to checkpoint, not [].

    Verifies that the P0 fix (message_history=[] bug) is in effect:
    real message history from the pydantic-ai context is used for
    checkpoint, preventing crash recovery from re-executing all prior
    tool calls.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-msg-1"

    assert agent_ctx.run_ctx is not None
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    # Set up active_agent_run with messages — this is what
    # handle_elicitation() uses for checkpoint message history.
    fake_messages = [
        MagicMock(),
        MagicMock(),
    ]
    mock_agent_run = MagicMock()
    mock_agent_run.all_messages = MagicMock(return_value=fake_messages)
    mock_run_handle = MagicMock()
    mock_run_handle.active_agent_run = mock_agent_run
    agent_ctx.run_ctx._run_handle = mock_run_handle

    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    agent_ctx.run_ctx.event_bus = mock_bus

    payload = ElicitationResumePayload(
        deferred_handle="tc-msg-1",
        action="accept",
        content={"name": "test"},
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-msg-1", payload)

    task = asyncio.create_task(resolve_later())
    await agent_ctx.handle_elicitation(form_params)
    await task

    mock_checkpoint.checkpoint.assert_awaited_once()
    checkpoint_kwargs = mock_checkpoint.checkpoint.call_args.kwargs
    # P0 fix: message_history should be the real messages, not []
    assert checkpoint_kwargs["message_history"] is fake_messages
    assert len(checkpoint_kwargs["message_history"]) == 2


# ============================================================================
# PR Review Coverage: P1a — checkpoint failure doesn't set checkpointed=True
# ============================================================================


@pytest.mark.unit
async def test_handle_elicitation_checkpoint_failure_doesnt_set_checkpointed(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """When checkpoint fails, run_ctx.checkpointed stays False.

    Verifies the P1a fix: checkpoint failure is logged but doesn't
    set checkpointed=True. The in-process future await still works,
    but crash recovery is unavailable.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-fail-1"

    assert agent_ctx.run_ctx is not None
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    # Checkpoint manager that always fails
    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock(side_effect=RuntimeError("disk full"))
    agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    agent_ctx.run_ctx.event_bus = mock_bus

    payload = ElicitationResumePayload(
        deferred_handle="tc-fail-1",
        action="decline",
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-fail-1", payload)

    task = asyncio.create_task(resolve_later())
    result = await agent_ctx.handle_elicitation(form_params)
    await task

    # Checkpoint was attempted but failed
    mock_checkpoint.checkpoint.assert_awaited_once()
    # P1a fix: checkpointed should NOT be True on failure
    assert agent_ctx.run_ctx.checkpointed is False
    # The future await still works — result returned
    assert isinstance(result, ElicitResult)
    assert result.action == "decline"


# ============================================================================
# PR Review Coverage: P2 — session store status updated to "checkpointed"
# ============================================================================


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
@pytest.mark.unit
async def test_handle_elicitation_updates_session_status_to_checkpointed(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation updates session store status to 'checkpointed'.

    Verifies the P2 fix: after saving checkpoint, the session store
    status is updated from 'active' to 'checkpointed' so resume_session()
    can find it without the allow_active_run workaround.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload, SessionData

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-status-1"

    assert agent_ctx.run_ctx is not None
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    agent_ctx.run_ctx.event_bus = mock_bus

    # Set up session store with "active" status
    session_data = SessionData(
        session_id="test-session",
        agent_name="test-agent",
        status="active",
        agent_type="native",
    )
    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=session_data)
    mock_store.save_session = AsyncMock()

    # Wire up the store through the mock chain
    mock_pool = MagicMock()
    mock_session_pool = MagicMock()
    mock_sessions = MagicMock()
    mock_sessions.store = mock_store
    mock_session_pool.sessions = mock_sessions
    mock_pool.session_pool = mock_session_pool
    mock_host_ctx = MagicMock()
    mock_host_ctx.session_pool = mock_session_pool
    mock_pool.get_context.return_value = mock_host_ctx
    agent_ctx.node.agent_pool = mock_pool
    agent_ctx.node.host_context = mock_host_ctx

    payload = ElicitationResumePayload(
        deferred_handle="tc-status-1",
        action="accept",
        content={"value": "yes"},
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-status-1", payload)

    task = asyncio.create_task(resolve_later())
    await agent_ctx.handle_elicitation(form_params)
    await task

    # P2 fix: session store status should be updated to "checkpointed"
    mock_store.load_session.assert_awaited()
    mock_store.save_session.assert_awaited_once()
    saved_data = mock_store.save_session.call_args[0][0]
    assert saved_data.status == "checkpointed"


@pytest.mark.skip(reason="L2 migration: requires mock internals — remains L1 unit test")
@pytest.mark.unit
async def test_handle_elicitation_skips_status_update_if_not_active(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """Session status update is skipped if status is not 'active'.

    If the status is already 'checkpointed' (e.g., from a prior
    elicitation), the update is a no-op — no unnecessary writes.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload, SessionData

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-skip-1"

    assert agent_ctx.run_ctx is not None
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    agent_ctx.run_ctx.event_bus = mock_bus

    # Session already checkpointed — no update needed
    session_data = SessionData(
        session_id="test-session",
        agent_name="test-agent",
        status="checkpointed",
        agent_type="native",
    )
    mock_store = MagicMock()
    mock_store.load_session = AsyncMock(return_value=session_data)
    mock_store.save_session = AsyncMock()

    mock_pool = MagicMock()
    mock_session_pool = MagicMock()
    mock_sessions = MagicMock()
    mock_sessions.store = mock_store
    mock_session_pool.sessions = mock_sessions
    mock_pool.session_pool = mock_session_pool
    mock_host_ctx = MagicMock()
    mock_host_ctx.session_pool = mock_session_pool
    mock_pool.get_context.return_value = mock_host_ctx
    agent_ctx.node.agent_pool = mock_pool
    agent_ctx.node.host_context = mock_host_ctx

    payload = ElicitationResumePayload(
        deferred_handle="tc-skip-1",
        action="decline",
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-skip-1", payload)

    task = asyncio.create_task(resolve_later())
    await agent_ctx.handle_elicitation(form_params)
    await task

    # Store was loaded but NOT saved (status was already "checkpointed")
    mock_store.load_session.assert_awaited()
    mock_store.save_session.assert_not_awaited()


# ============================================================================
# Bug 15: checkpoint message history from active_agent_run
# ============================================================================


@pytest.mark.unit
async def test_handle_elicitation_uses_active_agent_run_messages_for_checkpoint(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """handle_elicitation always uses active_agent_run.all_messages() for checkpoint.

    Bug 15: Tools without AgentContext param (e.g. question_for_user) don't
    trigger tool_wrapping.py's current_messages update. The fix removes
    current_messages entirely and always reads from active_agent_run —
    the authoritative pydantic-ai AgentRun object.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-bug15-1"

    assert agent_ctx.run_ctx is not None
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    agent_ctx.run_ctx.event_bus = mock_bus

    # Simulate active_agent_run with messages
    fake_messages = [MagicMock(), MagicMock(), MagicMock()]
    mock_agent_run = MagicMock()
    mock_agent_run.all_messages = MagicMock(return_value=fake_messages)
    mock_run_handle = MagicMock()
    mock_run_handle.active_agent_run = mock_agent_run
    agent_ctx.run_ctx._run_handle = mock_run_handle

    payload = ElicitationResumePayload(
        deferred_handle="tc-bug15-1",
        action="accept",
        content={"name": "test"},
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-bug15-1", payload)

    task = asyncio.create_task(resolve_later())
    await agent_ctx.handle_elicitation(form_params)
    await task

    mock_checkpoint.checkpoint.assert_awaited_once()
    checkpoint_kwargs = mock_checkpoint.checkpoint.call_args.kwargs
    assert checkpoint_kwargs["message_history"] is fake_messages
    assert len(checkpoint_kwargs["message_history"]) == 3
    mock_agent_run.all_messages.assert_called_once()


@pytest.mark.unit
async def test_handle_elicitation_empty_checkpoint_when_no_active_run(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """When no active_agent_run is available, checkpoint gets empty messages.

    This is a safety net — active_agent_run should always be available
    during tool execution, but if it's not, the checkpoint still saves
    with empty messages rather than crashing.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-no-run-1"

    assert agent_ctx.run_ctx is not None
    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    agent_ctx.run_ctx.event_bus = mock_bus

    # No active_agent_run — _run_handle is None
    agent_ctx.run_ctx._run_handle = None

    payload = ElicitationResumePayload(
        deferred_handle="tc-no-run-1",
        action="accept",
        content={"name": "test"},
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-no-run-1", payload)

    task = asyncio.create_task(resolve_later())
    await agent_ctx.handle_elicitation(form_params)
    await task

    mock_checkpoint.checkpoint.assert_awaited_once()
    checkpoint_kwargs = mock_checkpoint.checkpoint.call_args.kwargs
    assert checkpoint_kwargs["message_history"] == []


@pytest.mark.unit
def test_normalize_elicit_content_flattens_nested_values() -> None:
    """Nested dicts and non-string lists are serialized to JSON strings."""
    content = {
        "plain": "text",
        "count": 42,
        "enabled": True,
        "ratio": 3.14,
        "tags": ["a", "b"],
        "nested": {"annotations": []},
        "mixed_list": [1, 2, 3],
    }
    normalized = normalize_elicit_content(content)
    assert normalized is not None
    assert normalized["plain"] == "text"
    assert normalized["count"] == 42
    assert normalized["enabled"] is True
    assert normalized["ratio"] == 3.14
    assert normalized["tags"] == ["a", "b"]
    assert normalized["nested"] == '{"annotations": []}'
    assert normalized["mixed_list"] == "[1, 2, 3]"
    result = ElicitResult(action="accept", content=normalized)
    assert result.action == "accept"


@pytest.mark.unit
async def test_handle_elicitation_resume_payload_normalizes_nested_content(
    agent_ctx: AgentContext, form_params: ElicitRequestFormParams
) -> None:
    """Resume path must not crash when payload.content contains nested dicts.

    Regression for: request_comment URL-mode annotation returns content like
    {"content": {...}} and the resume path constructs mcp.types.ElicitResult
    directly from ElicitationResumePayload.content.
    """
    import asyncio

    from wolfharness.agents.native_agent.elicitation_bridge import (
        ElicitationFutureRegistry,
    )
    from wolfharness.sessions.models import ElicitationResumePayload

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    agent_ctx.input_provider = provider
    agent_ctx.in_mcp_callback = False
    agent_ctx.tool_call_id = "tc-nested-1"

    registry = ElicitationFutureRegistry()
    agent_ctx.run_ctx.elicitation_registry = registry

    payload = ElicitationResumePayload(
        deferred_handle="tc-nested-1",
        action="accept",
        content={"content": {"annotations": ["note"]}, "score": 0.9},
    )

    async def resolve_later() -> None:
        await asyncio.sleep(0.05)
        registry.resolve("tc-nested-1", payload)

    task = asyncio.create_task(resolve_later())
    result = await agent_ctx.handle_elicitation(form_params)
    await task

    assert result.action == "accept"
    assert result.content is not None
    assert result.content["content"] == '{"annotations": ["note"]}'
    assert result.content["score"] == 0.9


@pytest.mark.unit
def test_normalize_elicit_content_none_returns_none() -> None:
    """normalize_elicit_content(None) returns None."""
    assert normalize_elicit_content(None) is None


@pytest.mark.unit
def test_normalize_elicit_content_empty_dict() -> None:
    """normalize_elicit_content({}) returns {} and validates."""
    normalized = normalize_elicit_content({})
    assert normalized == {}
    result = ElicitResult(action="accept", content=normalized)
    assert result.content == {}


@pytest.mark.unit
def test_normalize_elicit_content_none_values_become_empty_string() -> None:
    """None values in content are converted to empty strings.

    The MCP wire protocol (Zod schema) rejects ``null`` even though the
    Python SDK type annotation permits it.  This is a regression test for
    the ``content._meta: null`` / ``content.content: null`` validation
    error from the ACP client.
    """
    content = {
        "_meta": None,
        "content": None,
        "text": "hello",
    }
    normalized = normalize_elicit_content(content)
    assert normalized is not None
    assert normalized["_meta"] == ""
    assert normalized["content"] == ""
    assert normalized["text"] == "hello"
    result = ElicitResult(action="accept", content=normalized)
    assert result.content is not None
    assert result.content["_meta"] == ""
    assert result.content["content"] == ""
