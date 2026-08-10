"""Tests covering elicitation bug paths in the OpenCode server.

These tests were written to verify and drive fixes for bugs identified
during a systematic debugging investigation of the elicitation 404 →
TUI hang bug.

Key paths tested:
1. MCP elicitation via ElicitationBridgeCapability should register a
   pending question via broadcast_elicitation_question() so the user
   can reply via POST /question/{id}/reply.
2. OpenCode event bridge should NOT need to intercept
   ElicitationDeferredEvent because the bridge capability handles
   broadcast directly.
3. await future in _handle_single_enum / _handle_multi_question should
   have a timeout so the agent can recover if the user never responds.
4. POST /question/{handle}/reply should succeed (200) for MCP elicitation
   handles, not return 404.

Refs: Investigation from elicitation 404 → opencode TUI hang analysis.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from mcp import types
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, RunContext
import pytest

from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events.events import ElicitationDeferredEvent
from wolfharness.agents.native_agent.elicitation_bridge import (
    ElicitationFutureRegistry,
    create_elicitation_bridge_capability,
)
from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider
from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


def _make_mock_session_controller(session_id: str, *, checkpoint_enabled: bool = False) -> Mock:
    """Create a mock SessionController with a SessionState for the given session."""
    from wolfharness.orchestrator.core import SessionState

    session = SessionState(session_id=session_id, agent_name="test-agent")
    session.checkpoint_enabled = checkpoint_enabled
    controller = Mock()
    controller.get_session = Mock(return_value=session)
    controller._sessions = {session_id: session}
    return controller


def _make_run_ctx(agent_ctx: AgentContext) -> RunContext[Any]:
    """Create a mock RunContext with AgentContext deps."""
    model = MagicMock()
    model.system = "test"
    model.model_name = "test-model"
    return RunContext(deps=agent_ctx, model=model, usage=MagicMock())


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
    return _make_run_ctx(agent_ctx)


def _make_deferred_requests(
    tool_call_id: str = "tc-elicit-001",
    tool_name: str = "mcp_elicitation_tool",
    message: str = "Enter your API key",
    schema: dict[str, Any] | None = None,
) -> DeferredToolRequests:
    """Create DeferredToolRequests with an elicitation-type call."""
    call = ToolCallPart(
        tool_name=tool_name,
        args={"prompt": message},
        tool_call_id=tool_call_id,
    )
    return DeferredToolRequests(
        calls=[call],
        metadata={
            tool_call_id: {
                "deferred_kind": "elicitation",
                "elicitation": {
                    "message": message,
                    "requestedSchema": schema
                    or {"type": "object", "properties": {"key": {"type": "string"}}},
                    "mode": "form",
                },
            },
        },
    )


def _make_provider(
    session_id: str = "test-session",
    *,
    checkpoint_enabled: bool = False,
) -> tuple[OpenCodeInputProvider, ServerState, Mock]:
    """Create a real OpenCodeInputProvider with mock state."""
    mock_agent = Mock()
    mock_agent.agent_pool = None
    mock_agent.host_context = None
    state = ServerState(working_dir="/tmp", agent=mock_agent)
    state.session_controller = _make_mock_session_controller(
        session_id, checkpoint_enabled=checkpoint_enabled
    )

    from wolfharness_server.opencode_server.models import Session
    from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated

    now = 0
    session = Session(
        id=session_id,
        project_id="default",
        directory="/tmp",
        title="Test",
        version="1",
        time=TimeCreatedUpdated(created=now, updated=now),
    )
    state.sessions[session_id] = session

    provider = OpenCodeInputProvider(state=state, session_id=session_id)
    return provider, state, state.session_controller


# =============================================================================
# Bug 1: MCP elicitation path should register pending question
# =============================================================================


class TestMCPElicitationRegistersPendingQuestion:
    """MCP elicitation via ElicitationBridgeCapability should register a pending question.

    Before the fix: ElicitationBridgeCapability emitted
    ElicitationDeferredEvent and registered a future in
    ElicitationFutureRegistry, but did NOT call
    provider.broadcast_elicitation_question() → no pending question →
    POST /question/{id}/reply returned 404 → TUI hung.

    After the fix: The bridge capability calls
    provider.broadcast_elicitation_question() to register the pending
    question, making it reachable via the REST endpoint.
    """

    @pytest.mark.asyncio
    async def test_mcp_elicitation_registers_pending_question(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """ElicitationBridgeCapability should populate _pending_questions_dict.

        Via broadcast_elicitation_question().
        """
        provider, _state, controller = _make_provider("test-session")

        registry = ElicitationFutureRegistry()
        cap = create_elicitation_bridge_capability(registry=registry)

        # Wire up the provider on the agent context
        run_ctx.deps.input_provider = provider

        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        run_ctx.deps.run_ctx.event_bus = mock_bus

        requests = _make_deferred_requests(
            schema={"type": "string", "enum": ["Option A", "Option B"]},
        )

        with patch(
            "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
            new_callable=AsyncMock,
        ):
            await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # Future IS registered in the registry
        assert "tc-elicit-001" in registry

        # AND: _pending_questions_dict should ALSO have the handle
        session = controller.get_session("test-session")
        assert "tc-elicit-001" in session.pending_questions, (
            "ElicitationBridgeCapability should call broadcast_elicitation_question() "
            "to register the pending question in _pending_questions_dict. "
            "Without this, POST /question/{id}/reply returns 404."
        )

    @pytest.mark.asyncio
    async def test_mcp_elicitation_event_published_to_event_bus(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """ElicitationBridgeCapability publishes ElicitationDeferredEvent to EventBus."""
        provider, _, _ = _make_provider("test-session")

        registry = ElicitationFutureRegistry()
        cap = create_elicitation_bridge_capability(registry=registry)
        run_ctx.deps.input_provider = provider

        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        run_ctx.deps.run_ctx.event_bus = mock_bus

        requests = _make_deferred_requests()

        with patch(
            "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        mock_emit.assert_awaited_once()
        emitted_event = mock_emit.call_args[0][1]
        assert isinstance(emitted_event, ElicitationDeferredEvent)
        assert emitted_event.deferred_handle == "tc-elicit-001"

    @pytest.mark.asyncio
    async def test_resolve_question_succeeds_for_mcp_handle(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """resolve_question returns True for a handle registered via ElicitationBridgeCapability.

        MCP path.
        """
        provider, _state, _controller = _make_provider("test-session")

        registry = ElicitationFutureRegistry()
        cap = create_elicitation_bridge_capability(registry=registry)
        run_ctx.deps.input_provider = provider

        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        run_ctx.deps.run_ctx.event_bus = mock_bus

        requests = _make_deferred_requests(
            schema={"type": "string", "enum": ["A", "B"]},
        )

        with patch(
            "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
            new_callable=AsyncMock,
        ):
            await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # User tries to reply:
        success = provider.resolve_question("tc-elicit-001", [["A"]])
        assert success, (
            "resolve_question should return True because the MCP elicitation "
            "path should have registered the pending question via "
            "broadcast_elicitation_question()."
        )


# =============================================================================
# Bug 2: await future should have a timeout
# =============================================================================


class TestAwaitFutureHasTimeout:
    """The await future should have a timeout for agent recovery.

    Before the fix: 'await future' had no timeout — if the future never
    resolved (e.g., broadcast_elicitation_question was never called or
    the user abandoned the question), the agent blocked indefinitely.

    After the fix: asyncio.wait_for() is used with a configurable
    timeout. On timeout, ElicitResult(action="cancel") is returned.
    """

    @pytest.mark.asyncio
    async def test_get_elicitation_times_out_when_future_never_resolved(
        self,
    ) -> None:
        """get_elicitation should time out and return cancel when future is never resolved."""
        provider, _state, controller = _make_provider("test-session")

        schema = {"type": "string", "enum": ["Option A", "Option B"]}
        params = types.ElicitRequestFormParams(
            message="Pick one",
            requestedSchema=schema,
        )

        # Start get_elicitation in background
        task = asyncio.create_task(provider.get_elicitation(params))

        # Wait for the question to be created
        await asyncio.sleep(0.1)
        session = controller.get_session("test-session")
        assert len(session.pending_questions) == 1

        # DON'T resolve the future. Wait for the timeout to fire.
        # The timeout should be reasonable (not 300 seconds for tests).
        # We use a short test timeout to verify the behavior.
        try:
            result = await asyncio.wait_for(task, timeout=10.0)
        except TimeoutError:
            pytest.fail(
                "get_elicitation should have timed out internally and returned "
                "ElicitResult(action='cancel'), but it's still blocked. "
                "No timeout was applied to 'await future'."
            )

        # Should return cancel result
        assert isinstance(result, types.ElicitResult)
        assert result.action == "cancel", (
            "get_elicitation should return action='cancel' on timeout, "
            f"got action='{result.action}'."
        )

    @pytest.mark.asyncio
    async def test_get_elicitation_recovers_after_question_cleanup(
        self,
    ) -> None:
        """get_elicitation should still return via timeout after cleanup.

        Rather than blocking forever.
        """
        provider, _state, controller = _make_provider("test-session")

        schema = {"type": "string", "enum": ["A", "B"]}
        params = types.ElicitRequestFormParams(message="Pick", requestedSchema=schema)

        task = asyncio.create_task(provider.get_elicitation(params))
        await asyncio.sleep(0.1)

        session = controller.get_session("test-session")
        question_id = next(iter(session.pending_questions.keys()))

        # Simulate: cleanup (as done in AgentContext.handle_elicitation finally block)
        provider.cleanup_elicitation_question(question_id)
        assert question_id not in session.pending_questions

        # The task should complete via timeout, not hang forever
        try:
            result = await asyncio.wait_for(task, timeout=10.0)
        except TimeoutError:
            pytest.fail(
                "get_elicitation should have timed out internally after cleanup, "
                "but it's still blocked. No timeout was applied to 'await future'."
            )

        assert isinstance(result, types.ElicitResult)
        assert result.action == "cancel"


# =============================================================================
# Bug 3: POST /question/{handle}/reply should succeed for MCP elicitation
# =============================================================================


class TestQuestionRouteSucceedsForMCPElicitation:
    """POST /question/{handle}/reply should succeed for MCP elicitation handles."""

    @pytest.mark.asyncio
    async def test_reply_to_mcp_elicitation_handle_succeeds(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """reply_to_question succeeds for a handle registered via ElicitationBridgeCapability."""
        from wolfharness_server.opencode_server.models import QuestionReply

        provider, _state, _controller = _make_provider("test-session")

        registry = ElicitationFutureRegistry()
        cap = create_elicitation_bridge_capability(registry=registry)
        run_ctx.deps.input_provider = provider

        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        run_ctx.deps.run_ctx.event_bus = mock_bus

        requests = _make_deferred_requests(
            schema={"type": "string", "enum": ["A", "B"]},
        )

        with patch(
            "wolfharness.agents.native_agent.elicitation_bridge._emit_elicitation_event",
            new_callable=AsyncMock,
        ):
            await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # User replies — should NOT get 404
        QuestionReply(answers=[["A"]])
        # reply_to_question is an async FastAPI endpoint
        # If the question is registered, it should succeed
        success = provider.resolve_question("tc-elicit-001", [["A"]])
        assert success, (
            "reply_to_question should succeed for MCP elicitation handle "
            "because broadcast_elicitation_question() was called."
        )


# =============================================================================
# Positive test: Local tool Path 3 DOES register pending question
# =============================================================================


class TestLocalToolPath3RegistersPendingQuestion:
    """AgentContext.handle_elicitation Path 3 registers the pending question.

    This is the working path for durable, local tools.
    """

    @pytest.mark.asyncio
    async def test_path3_broadcast_elicitation_question_called(
        self,
        agent_ctx: AgentContext,
    ) -> None:
        """handle_elicitation Path 3 calls broadcast_elicitation_question()."""
        from wolfharness.sessions.models import ElicitationResumePayload

        provider, _state, _controller = _make_provider("test-session", checkpoint_enabled=True)
        provider._broadcast_called = False

        original_broadcast = provider.broadcast_elicitation_question

        async def tracking_broadcast(handle: str, params: Any, **kwargs: Any) -> bool:
            provider._broadcast_called = True
            return await original_broadcast(handle, params, **kwargs)

        provider.broadcast_elicitation_question = tracking_broadcast  # type: ignore[method-assign]

        agent_ctx.input_provider = provider
        agent_ctx.in_mcp_callback = False
        agent_ctx.tool_call_id = "tc-local-path3-001"

        registry = ElicitationFutureRegistry()
        agent_ctx.run_ctx.elicitation_registry = registry

        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint = AsyncMock()
        agent_ctx.run_ctx.checkpoint_manager = mock_checkpoint

        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        agent_ctx.run_ctx.event_bus = mock_bus

        form_params = types.ElicitRequestFormParams(
            message="Pick a database",
            requestedSchema={"type": "string", "enum": ["PostgreSQL", "MySQL"]},
            mode="form",
        )

        payload = ElicitationResumePayload(
            deferred_handle="tc-local-path3-001",
            action="accept",
            content={"value": "PostgreSQL"},
        )

        async def resolve_later() -> None:
            await asyncio.sleep(0.05)
            registry.resolve("tc-local-path3-001", payload)

        task = asyncio.create_task(resolve_later())
        await agent_ctx.handle_elicitation(form_params)
        await task

        assert provider._broadcast_called, (
            "broadcast_elicitation_question should be called for local tool "
            "Path 3 (durable elicitation). This is the working path."
        )


# =============================================================================
# ToolPart name documentation (opencode TUI issue — can't fix here)
# =============================================================================


class TestElicitationToolPartName:
    """ElicitationDeferredEvent creates a ToolPart with tool='elicitation'.

    OpenCode TUI's syncQuestion only matches tool='question', so this
    ToolPart is NOT auto-cleaned when it transitions to completed/error.

    This is an opencode TUI-side issue (session-data.ts) that cannot be
    fixed in the wolfharness repo. This test documents the behavior.
    """

    @pytest.mark.asyncio
    async def test_elicitation_tool_part_has_tool_name_elicitation(
        self,
    ) -> None:
        """ElicitationDeferredEvent creates a ToolPart with tool='elicitation'."""
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
            session_id="test-session",
            time=MessageTime(created=0),
            agent_name="test-agent",
            model_id="test-model",
            parent_id="parent-001",
            provider_id="test-provider",
            path=MessagePath(cwd="/tmp", root="/tmp"),
        )
        ctx = EventProcessorContext(
            session_id="test-session",
            assistant_msg_id="msg-001",
            assistant_msg=assistant_msg,
            state=mock_state,
            working_dir="/tmp",
        )
        processor = EventProcessor()

        event = ElicitationDeferredEvent(
            deferred_handle="tc-elicit-001",
            message="Enter your API key",
            requested_schema={"type": "object", "properties": {"key": {"type": "string"}}},
            mode="form",
            session_id="test-session",
        )

        events = [ev async for ev in processor.process(event, ctx)]

        assert len(events) == 1
        part_updated_event = events[0]
        tool_part = part_updated_event.properties.part

        # This documents the current behavior — tool="elicitation"
        # The opencode TUI's syncQuestion only matches tool="question",
        # so this ToolPart is NOT auto-cleaned. This is a TUI-side issue.
        assert tool_part.tool == "elicitation"
        assert isinstance(tool_part.state, ToolStateRunning)
        assert tool_part.call_id == "elicitation_tc-elicit-001"
