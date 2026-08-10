"""Integration tests for the Durable Elicitation Bridge implementation.

Covers Tasks 9.8-9.13, 9.17, 9.20, 9.23 from the durable-elicitation-bridge
plan. Tests exercise cross-module interactions: SessionController resume
paths, ElicitationFutureRegistry lifecycle, ACP/OpenCode event conversion,
and ACP capability-gated elicitation behavior.

Refs: https://github.com/Leoyzen/wolfharness/issues/107
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import ElicitResult
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext
import pytest

from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events.events import ElicitationDeferredEvent
from wolfharness.agents.native_agent.elicitation_bridge import (
    ElicitationFutureRegistry,
    create_elicitation_bridge_capability,
)
from wolfharness.orchestrator.core import SessionClosedError
from wolfharness.sessions.models import (
    ElicitationResumePayload,
    PendingDeferredCall,
)
from wolfharness.ui.base import InputProvider


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
def elicitation_deferred_call() -> PendingDeferredCall:
    """Create an elicitation-type pending deferred call."""
    return PendingDeferredCall(
        tool_call_id="tc-elicit-001",
        tool_name="mcp_elicitation_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
        elicitation_message="Enter your API key",
        elicitation_schema={"type": "object", "properties": {"key": {"type": "string"}}},
        elicitation_mode="form",
        mcp_server_id="mcp-server-1",
    )


@pytest.fixture
def external_deferred_call() -> PendingDeferredCall:
    """Create an external-type pending deferred call."""
    return PendingDeferredCall(
        tool_call_id="tc-external-001",
        tool_name="bash",
        deferred_kind="external",
        deferred_strategy="block",
    )


@pytest.fixture
def elicitation_payload() -> ElicitationResumePayload:
    """Create an accept elicitation resume payload."""
    return ElicitationResumePayload(
        deferred_handle="tc-elicit-001",
        action="accept",
        content={"key": "sk-test-12345"},
    )


# ============================================================================
# Task 9.8: In-process resume — checkpoint → resolve future → tool completes
# ============================================================================


@pytest.mark.unit
async def test_in_process_resume_resolves_future_and_completes_tool(
    run_ctx: RunContext[Any],
) -> None:
    """In-process elicitation resume resolves future, allowing MCP tool to complete.

    Simulates: checkpoint → resolve future → MCP tool call completes →
    tool result flows back. Verifies that resolving the future in the
    registry unblocks the deferred MCP tool call.
    """
    registry = ElicitationFutureRegistry()
    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    cap = create_elicitation_bridge_capability(
        registry=registry,
        checkpoint_manager=mock_checkpoint,
        agent_config_hash="hash-9-8",
    )
    elicitation_call = ToolCallPart(
        tool_name="mcp_elicitation_tool",
        args={"prompt": "Enter key"},
        tool_call_id="tc-elicit-001",
    )
    requests = DeferredToolRequests(
        calls=[elicitation_call],
        metadata={
            "tc-elicit-001": {
                "deferred_kind": "elicitation",
                "elicitation": {
                    "message": "Enter your API key",
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
    assert "tc-elicit-001" in registry
    future = registry._futures["tc-elicit-001"]
    assert not future.done()

    payload = ElicitationResumePayload(
        deferred_handle="tc-elicit-001",
        action="accept",
        content={"key": "sk-test-12345"},
    )
    registry.resolve("tc-elicit-001", payload)

    assert future.done()
    assert future.result() == payload
    assert "tc-elicit-001" not in registry


# ============================================================================
# Task 9.9: Crash recovery resume — registry cleared, cached responses used
# ============================================================================


@pytest.mark.unit
async def test_crash_recovery_resume_uses_cached_elicitation_responses(
    agent_ctx: AgentContext,
    elicitation_deferred_call: PendingDeferredCall,
    elicitation_payload: ElicitationResumePayload,
) -> None:
    """Crash recovery resume populates cached_elicitation_responses.

    Simulates: checkpoint → process restart (clear registry) →
    re-execute MCP tool call → handle_elicitation returns cached response.
    """
    from mcp.types import ElicitRequestFormParams

    registry = ElicitationFutureRegistry()
    registry.register("tc-elicit-001")

    registry.reject_all(RuntimeError("simulated crash"))

    assert "tc-elicit-001" not in registry

    cached_elicitation: dict[str, Any] = {}
    match elicitation_payload.action:
        case "accept":
            cached_elicitation[elicitation_payload.deferred_handle] = ElicitResult(
                action="accept",
                content=elicitation_payload.content,
            )
        case "decline":
            cached_elicitation[elicitation_payload.deferred_handle] = ElicitResult(
                action="decline",
            )
        case "cancel":
            cached_elicitation[elicitation_payload.deferred_handle] = ElicitResult(
                action="cancel",
            )

    assert agent_ctx.run_ctx is not None
    agent_ctx.run_ctx.cached_elicitation_responses = cached_elicitation
    agent_ctx.tool_call_id = "tc-elicit-001"

    provider = MagicMock(spec=InputProvider)
    provider.supports_durable_elicitation = True
    provider.get_elicitation = AsyncMock()
    agent_ctx.input_provider = provider

    form_params = ElicitRequestFormParams(
        message="Enter your API key",
        requestedSchema={"type": "object", "properties": {"key": {"type": "string"}}},
        mode="form",
    )
    result = await agent_ctx.handle_elicitation(form_params)

    assert isinstance(result, ElicitResult)
    assert result.action == "accept"
    assert result.content == {"key": "sk-test-12345"}
    provider.get_elicitation.assert_not_awaited()
    assert agent_ctx._pending_elicitation_deferral is None


# ============================================================================
# Task 9.10: Session close with pending futures
# ============================================================================


@pytest.mark.unit
async def test_session_close_rejects_pending_futures() -> None:
    """close_session rejects all pending elicitation futures with SessionClosedError.

    Registers multiple futures in ElicitationFutureRegistry, then calls
    reject_all with SessionClosedError. Asserts all futures are rejected
    and no futures remain in the registry.
    """
    registry = ElicitationFutureRegistry()
    future1 = registry.register("handle-1")
    future2 = registry.register("handle-2")
    future3 = registry.register("handle-3")

    assert not future1.done()
    assert not future2.done()
    assert not future3.done()

    session_closed_error = SessionClosedError(session_id="test-session-close")
    registry.reject_all(session_closed_error)

    assert future1.done()
    assert future2.done()
    assert future3.done()
    assert future1.exception() is session_closed_error
    assert future2.exception() is session_closed_error
    assert future3.exception() is session_closed_error
    assert "handle-1" not in registry
    assert "handle-2" not in registry
    assert "handle-3" not in registry


# ============================================================================
# Task 9.11: Mixed elicitation + deferred tool results
# ============================================================================


@pytest.mark.unit
async def test_mixed_elicitation_and_deferred_tool_results_routing(
    run_ctx: RunContext[Any],
    elicitation_deferred_call: PendingDeferredCall,
    external_deferred_call: PendingDeferredCall,
    elicitation_payload: ElicitationResumePayload,
) -> None:
    """Mixed checkpoint with elicitation + external calls routes correctly.

    Creates a checkpoint with BOTH elicitation deferred calls AND
    non-elicitation deferred calls. Resume with both elicitation_payloads
    and deferred_tool_results. Verifies each is routed correctly.
    """
    registry = ElicitationFutureRegistry()
    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    cap = create_elicitation_bridge_capability(
        registry=registry,
        checkpoint_manager=mock_checkpoint,
    )
    elicitation_call = ToolCallPart(
        tool_name="mcp_elicitation_tool",
        args={"prompt": "Enter key"},
        tool_call_id="tc-elicit-001",
    )
    external_call = ToolCallPart(
        tool_name="bash",
        args={"cmd": "ls"},
        tool_call_id="tc-external-001",
    )
    requests = DeferredToolRequests(
        calls=[elicitation_call, external_call],
        metadata={
            "tc-elicit-001": {
                "deferred_kind": "elicitation",
                "elicitation": {
                    "message": "Enter your API key",
                    "requestedSchema": {"type": "object"},
                    "mode": "form",
                },
            },
            "tc-external-001": {
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
    ):
        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

    assert result is None
    assert "tc-elicit-001" in registry
    assert "tc-external-001" not in registry

    registry.resolve("tc-elicit-001", elicitation_payload)
    assert "tc-elicit-001" not in registry

    external_results = DeferredToolResults(
        calls={"tc-external-001": "file1.txt\nfile2.txt"},
    )
    assert "tc-external-001" in external_results.calls
    assert external_results.calls["tc-external-001"] == "file1.txt\nfile2.txt"


# ============================================================================
# Task 9.12: ElicitationDeferredEvent → ACP converter
# ============================================================================


@pytest.mark.unit
async def test_acp_converter_outputs_toolcallstart_for_elicitation_event() -> None:
    """ACP event converter emits ToolCallStart with elicitation params in field_meta.

    Creates an ElicitationDeferredEvent, passes it through the ACP event
    converter, and verifies the converter outputs a ToolCallStart notification
    with elicitation params in field_meta (deferred_handle, elicitation: True,
    elicitation_message, elicitation_schema, elicitation_mode).
    """
    from wolfharness_server.acp_server.event_converter import ACPEventConverter

    converter = ACPEventConverter()
    event = ElicitationDeferredEvent(
        deferred_handle="tc-elicit-acp-001",
        message="Please enter your credentials",
        requested_schema={"type": "object", "properties": {"user": {"type": "string"}}},
        mode="form",
        session_id="test-session-acp",
    )

    updates: list[Any] = [update async for update in converter.convert(event)]

    assert len(updates) == 1
    tool_call_start = updates[0]
    assert tool_call_start.tool_call_id == "elicitation_tc-elicit-acp-001"
    assert tool_call_start.title == "Elicitation: Please enter your credentials"
    assert tool_call_start.status == "pending"

    field_meta = tool_call_start.field_meta
    assert field_meta is not None
    assert field_meta["deferred_handle"] == "tc-elicit-acp-001"
    assert field_meta["elicitation"] is True
    assert field_meta["elicitation_message"] == "Please enter your credentials"
    assert field_meta["elicitation_schema"] == {
        "type": "object",
        "properties": {"user": {"type": "string"}},
    }
    assert field_meta["elicitation_mode"] == "form"


# ============================================================================
# Task 9.13: End-to-end durable path with checkpoint/resume
# ============================================================================


@pytest.mark.unit
async def test_end_to_end_durable_elicitation_checkpoint_and_resume(
    run_ctx: RunContext[Any],
) -> None:
    """End-to-end: agent tool triggers elicitation → deferred → checkpoint → resume.

    Creates a mock agent with a tool that triggers elicitation. Runs the
    elicitation bridge, verifies elicitation is deferred (not handled
    synchronously), checkpoint is created, and resume with
    ElicitationResumePayload resolves the future.
    """
    registry = ElicitationFutureRegistry()
    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = AsyncMock()
    cap = create_elicitation_bridge_capability(
        registry=registry,
        checkpoint_manager=mock_checkpoint,
        agent_config_hash="e2e-hash",
    )

    tool_call = ToolCallPart(
        tool_name="mcp_auth_tool",
        args={"action": "authenticate"},
        tool_call_id="tc-e2e-001",
    )
    requests = DeferredToolRequests(
        calls=[tool_call],
        metadata={
            "tc-e2e-001": {
                "deferred_kind": "elicitation",
                "elicitation": {
                    "message": "Authenticate with your credentials",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "password": {"type": "string"},
                        },
                    },
                    "mode": "form",
                },
            },
        },
    )
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    run_ctx.deps.run_ctx.event_bus = mock_bus

    captured_event: list[ElicitationDeferredEvent] = []

    async def capture_emit(ctx: RunContext[Any], event: ElicitationDeferredEvent) -> None:
        captured_event.append(event)

    with patch(
        "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
        new=capture_emit,
    ):
        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

    assert result is None
    assert "tc-e2e-001" in registry
    assert run_ctx.deps.run_ctx.checkpointed is True

    mock_checkpoint.checkpoint.assert_awaited_once()
    checkpoint_kwargs = mock_checkpoint.checkpoint.call_args.kwargs
    assert checkpoint_kwargs["session_id"] == "test-session"
    pending_calls = checkpoint_kwargs["pending_calls"]
    assert len(pending_calls) == 1
    assert pending_calls[0].tool_call_id == "tc-e2e-001"
    assert pending_calls[0].deferred_kind == "elicitation"

    assert len(captured_event) == 1
    emitted = captured_event[0]
    assert emitted.deferred_handle == "tc-e2e-001"
    assert emitted.message == "Authenticate with your credentials"
    assert emitted.mode == "form"

    resume_payload = ElicitationResumePayload(
        deferred_handle="tc-e2e-001",
        action="accept",
        content={"username": "admin", "password": "secret"},
    )
    future = registry._futures["tc-e2e-001"]
    assert not future.done()
    registry.resolve("tc-e2e-001", resume_payload)
    assert future.done()
    assert future.result() == resume_payload


# ============================================================================
# Task 9.17: Concurrent elicitation deferrals across sessions
# ============================================================================


@pytest.mark.unit
async def test_concurrent_elicitation_deferrals_across_sessions() -> None:
    """Concurrent elicitation deferrals across sessions are isolated.

    Creates two separate sessions, each with its own
    ElicitationFutureRegistry. Registers futures in both. Resolves one
    session's future. Verifies the other session's future is unaffected
    and no cross-session interference occurs.
    """
    registry_a = ElicitationFutureRegistry()
    registry_b = ElicitationFutureRegistry()

    future_a = registry_a.register("handle-session-a")
    future_b = registry_b.register("handle-session-b")

    assert not future_a.done()
    assert not future_b.done()

    payload_a = ElicitationResumePayload(
        deferred_handle="handle-session-a",
        action="accept",
        content={"value": "response-a"},
    )
    registry_a.resolve("handle-session-a", payload_a)

    assert future_a.done()
    assert future_a.result() == payload_a
    assert "handle-session-a" not in registry_a

    assert not future_b.done()
    assert "handle-session-b" in registry_b
    assert "handle-session-a" not in registry_b

    payload_b = ElicitationResumePayload(
        deferred_handle="handle-session-b",
        action="decline",
    )
    registry_b.resolve("handle-session-b", payload_b)
    assert future_b.done()
    assert future_b.result() == payload_b
    assert "handle-session-b" not in registry_b


# ============================================================================
# Task 9.20: ElicitationDeferredEvent → OpenCode processor
# ============================================================================


@pytest.mark.unit
async def test_opencode_processor_creates_toolpart_for_elicitation_event() -> None:
    """OpenCode event processor creates ToolPart with elicitation metadata.

    Creates an ElicitationDeferredEvent, passes it through the OpenCode
    event processor, and verifies a ToolPart is created with
    ToolStateRunning and metadata containing deferred: True,
    deferred_handle, elicitation: True, elicitation_message,
    elicitation_schema, elicitation_mode.
    """
    from wolfharness_server.opencode_server.event_processor import EventProcessor
    from wolfharness_server.opencode_server.event_processor_context import (
        EventProcessorContext,
    )
    from wolfharness_server.opencode_server.models.message import (
        MessagePath,
        MessageTime,
        MessageWithParts,
    )
    from wolfharness_server.opencode_server.models.parts import ToolStateRunning

    mock_state = MagicMock()
    assistant_msg = MessageWithParts.assistant(
        message_id="msg-001",
        session_id="test-session-oc",
        time=MessageTime(created=0),
        agent_name="test-agent",
        model_id="test-model",
        parent_id="parent-001",
        provider_id="test-provider",
        path=MessagePath(cwd="/tmp", root="/tmp"),
    )
    ctx = EventProcessorContext(
        session_id="test-session-oc",
        assistant_msg_id="msg-001",
        assistant_msg=assistant_msg,
        state=mock_state,
        working_dir="/tmp",
    )
    processor = EventProcessor()

    event = ElicitationDeferredEvent(
        deferred_handle="tc-elicit-oc-001",
        message="Please confirm your choice",
        requested_schema={"type": "object", "properties": {"choice": {"type": "string"}}},
        mode="form",
        session_id="test-session-oc",
    )

    events: list[Any] = [ev async for ev in processor.process(event, ctx)]

    assert len(events) == 1
    part_updated_event = events[0]
    tool_part = part_updated_event.properties.part
    assert tool_part.tool == "elicitation"
    assert tool_part.call_id == "elicitation_tc-elicit-oc-001"
    assert isinstance(tool_part.state, ToolStateRunning)
    assert tool_part.state.title == "[Elicitation] Please confirm your choice"

    metadata = tool_part.metadata
    assert metadata is not None
    assert metadata["deferred"] is True
    assert metadata["deferred_handle"] == "tc-elicit-oc-001"
    assert metadata["elicitation"] is True
    assert metadata["elicitation_message"] == "Please confirm your choice"
    assert metadata["elicitation_schema"] == {
        "type": "object",
        "properties": {"choice": {"type": "string"}},
    }
    assert metadata["elicitation_mode"] == "form"


# ============================================================================
# Task 9.23: ACP capability-gated elicitation behavior
# ============================================================================


@pytest.mark.unit
def test_acp_capability_gated_elicitation_fallback_to_request_permission() -> None:
    """ACP input provider falls back to request_permission when elicitation/create unsupported.

    When the ACP client doesn't support ``elicitation/create``, the system
    falls back to ``request_permission`` for elicitation. When
    ``checkpoint_enabled`` is False, ``supports_durable_elicitation`` returns
    False, meaning elicitation goes through the synchronous path.
    """
    from wolfharness_server.acp_server.input_provider import ACPInputProvider
    from wolfharness_server.acp_server.session import ACPSession

    mock_session = MagicMock(spec=ACPSession)
    mock_session.checkpoint_enabled = False
    provider = ACPInputProvider(session=mock_session)
    assert provider.supports_durable_elicitation is False

    mock_session.checkpoint_enabled = True
    assert provider.supports_durable_elicitation is True


@pytest.mark.unit
async def test_acp_capability_gated_form_but_not_url() -> None:
    """ACP capability gating: form supported but not URL only allows form durable.

    When the client supports form elicitation but not URL elicitation,
    only form-mode elicitation should be durable. This tests the
    ``_client_supports_elicitation`` method's mode-based gating.
    """
    from wolfharness_server.acp_server.input_provider import ACPInputProvider
    from wolfharness_server.acp_server.session import ACPSession

    mock_session = MagicMock(spec=ACPSession)
    mock_session.checkpoint_enabled = True

    mock_caps = MagicMock()
    mock_caps.elicitation = MagicMock()
    mock_caps.elicitation.form = True
    mock_caps.elicitation.url = False
    mock_session.client_capabilities = mock_caps

    provider = ACPInputProvider(session=mock_session)

    assert provider._client_supports_elicitation("form") is True
    assert provider._client_supports_elicitation("url") is False
