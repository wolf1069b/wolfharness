"""Tests for DeferredToolBridge — pydantic-ai HandleDeferredToolCalls capability.

Verifies that tools with deferred=True are intercepted before approval_bridge:
- block strategy → ToolCallDeferredEvent emitted, excluded from returned results
- continue strategy → resolved inline with placeholder, included in results
- non-deferred tools → return None to pass through to next capability
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tools import (
    DeferredToolRequests,
    RunContext,
)
import pytest

from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.events.events import ToolCallDeferredEvent


pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def agent_ctx() -> AgentContext:
    """Create a minimal AgentContext for bridge testing."""
    node = MagicMock()
    node.name = "test-agent"
    agent_run_ctx = AgentRunContext(session_id="test-session")
    return AgentContext(node=node, run_ctx=agent_run_ctx)


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
def deferred_tools() -> dict[str, str]:
    """Tool name → deferred_strategy mapping for deferred tools."""
    return {
        "bash_exec": "block",
        "background_task": "continue",
    }


@pytest.fixture
def block_tool_call() -> ToolCallPart:
    """A tool call with block deferred strategy."""
    return ToolCallPart(
        tool_name="bash_exec",
        args={"command": "sleep 60"},
        tool_call_id="tc-block-1",
    )


@pytest.fixture
def continue_tool_call() -> ToolCallPart:
    """A tool call with continue deferred strategy."""
    return ToolCallPart(
        tool_name="background_task",
        args={"task": "index_docs"},
        tool_call_id="tc-continue-1",
    )


@pytest.fixture
def non_deferred_tool_call() -> ToolCallPart:
    """A normal tool call (not in deferred_tools mapping)."""
    return ToolCallPart(
        tool_name="read_file",
        args={"path": "/tmp/test.txt"},
        tool_call_id="tc-normal-1",
    )


# ============================================================================
# Tests: create_deferred_bridge_capability
# ============================================================================


class TestCreateDeferredBridgeCapability:
    """Tests for the factory function."""

    def test_returns_handle_deferred_tool_calls(self) -> None:
        """Factory returns a HandleDeferredToolCalls capability."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        cap = create_deferred_bridge_capability(deferred_tools={"bash": "block"})
        assert isinstance(cap, HandleDeferredToolCalls)

    def test_handler_is_callable(self) -> None:
        """Handler function is properly set on the capability."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        cap = create_deferred_bridge_capability(deferred_tools={})
        assert callable(cap.handler)

    @pytest.mark.anyio
    async def test_empty_deferred_tools_returns_none(
        self,
        run_ctx: RunContext[Any],
        block_tool_call: ToolCallPart,
    ) -> None:
        """When deferred_tools is empty, all calls pass through (return None)."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[block_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools={})

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)
        assert result is None


# ============================================================================
# Tests: block strategy
# ============================================================================


class TestBlockStrategy:
    """Tests for tools with deferred_strategy='block'."""

    @pytest.mark.anyio
    async def test_block_tool_emits_deferred_event(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        block_tool_call: ToolCallPart,
    ) -> None:
        """Block-strategy tool emits ToolCallDeferredEvent."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[block_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        # We need to check that the event is emitted via the event_bus.
        # Patch the _emit_deferred_event to verify it's called correctly.
        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

            mock_emit.assert_awaited_once()
            call_args = mock_emit.call_args
            # Verify the event has correct fields
            # _emit_deferred_event(ctx, event) — positional args
            event = call_args[0][1]
            assert isinstance(event, ToolCallDeferredEvent)
            assert event.tool_call_id == "tc-block-1"
            assert event.tool_name == "bash_exec"
            assert event.deferred_strategy == "block"
            assert event.status == "pending"
            assert event.session_id == "test-session"

    @pytest.mark.anyio
    async def test_block_tool_excluded_from_results(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        block_tool_call: ToolCallPart,
    ) -> None:
        """Block-strategy tool is EXCLUDED from returned DeferredToolResults."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[block_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ):
            result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # Block calls should be excluded from results
        # When only block calls exist, results should have empty calls/approvals
        # (not None, because we DID handle them — by deciding they stay unresolved)
        assert result is not None
        assert "tc-block-1" not in result.calls
        assert "tc-block-1" not in result.approvals

    @pytest.mark.anyio
    async def test_block_tool_remaining_in_requests(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        block_tool_call: ToolCallPart,
    ) -> None:
        """After results applied, block tool remains in remaining requests."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[block_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ):
            result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # The block call should still be unresolved
        assert result is not None
        remaining = requests.remaining(result)
        assert remaining is not None
        assert len(remaining.calls) == 1
        assert remaining.calls[0].tool_call_id == "tc-block-1"

    @pytest.mark.anyio
    async def test_multiple_block_tools_all_excluded(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
    ) -> None:
        """All block-strategy tools are excluded from results."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        calls = [
            ToolCallPart(tool_name="bash_exec", args={}, tool_call_id="tc-1"),
            ToolCallPart(tool_name="bash_exec", args={}, tool_call_id="tc-2"),
        ]
        requests = DeferredToolRequests(calls=calls)
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        assert mock_emit.await_count == 2
        assert result is not None
        assert "tc-1" not in result.calls
        assert "tc-2" not in result.calls
        remaining = requests.remaining(result)
        assert remaining is not None
        assert len(remaining.calls) == 2


# ============================================================================
# Tests: continue strategy
# ============================================================================


class TestContinueStrategy:
    """Tests for tools with deferred_strategy='continue'."""

    @pytest.mark.anyio
    async def test_continue_tool_resolved_with_placeholder(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        continue_tool_call: ToolCallPart,
    ) -> None:
        """Continue-strategy tool is resolved with a ToolReturn placeholder."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[continue_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        assert result is not None
        assert "tc-continue-1" in result.calls
        call_result = result.calls["tc-continue-1"]
        assert isinstance(call_result, ToolReturn)
        assert "processing in the background" in str(call_result.return_value)

    @pytest.mark.anyio
    async def test_continue_tool_not_remaining(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        continue_tool_call: ToolCallPart,
    ) -> None:
        """Continue-strategy tool is fully resolved — no remaining requests."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[continue_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        assert result is not None
        remaining = requests.remaining(result)
        assert remaining is None  # All resolved

    @pytest.mark.anyio
    async def test_continue_tool_does_not_emit_deferred_event(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        continue_tool_call: ToolCallPart,
    ) -> None:
        """Continue-strategy tools do NOT emit ToolCallDeferredEvent."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[continue_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        mock_emit.assert_not_called()


# ============================================================================
# Tests: mixed scenarios
# ============================================================================


class TestMixedStrategies:
    """Tests for mixed block + continue + non-deferred calls."""

    @pytest.mark.anyio
    async def test_mixed_block_and_continue(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        block_tool_call: ToolCallPart,
        continue_tool_call: ToolCallPart,
    ) -> None:
        """Block excluded, continue resolved with placeholder."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[block_tool_call, continue_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        assert result is not None

        # Continue call resolved
        assert "tc-continue-1" in result.calls
        assert isinstance(result.calls["tc-continue-1"], ToolReturn)

        # Block call excluded from results
        assert "tc-block-1" not in result.calls
        assert "tc-block-1" not in result.approvals

        # Block call event emitted
        mock_emit.assert_awaited_once()

        # Block call remains unresolved
        remaining = requests.remaining(result)
        assert remaining is not None
        assert len(remaining.calls) == 1
        assert remaining.calls[0].tool_call_id == "tc-block-1"

    @pytest.mark.anyio
    async def test_mixed_deferred_and_non_deferred(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        block_tool_call: ToolCallPart,
        non_deferred_tool_call: ToolCallPart,
    ) -> None:
        """Non-deferred tools excluded from results → pass through to next capability."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[block_tool_call, non_deferred_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ):
            result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # Block call excluded from results
        assert result is not None
        assert "tc-block-1" not in result.calls

        # Non-deferred call also excluded from results (not our concern)
        assert "tc-normal-1" not in result.calls

        # Both remain unresolved → next capability (approval_bridge) gets them
        remaining = requests.remaining(result)
        assert remaining is not None
        assert len(remaining.calls) == 2


# ============================================================================
# Tests: non-deferred tools (pass through)
# ============================================================================


class TestNonDeferredPassThrough:
    """Tests for tools NOT in the deferred_tools mapping."""

    @pytest.mark.anyio
    async def test_non_deferred_tool_returns_none(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        non_deferred_tool_call: ToolCallPart,
    ) -> None:
        """Non-deferred tools cause the bridge to return None."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[non_deferred_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)
        assert result is None

    @pytest.mark.anyio
    async def test_empty_requests_returns_none(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
    ) -> None:
        """Empty requests return None."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(calls=[], approvals=[])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)
        assert result is None

    @pytest.mark.anyio
    async def test_only_approvals_no_deferred_tools_returns_none(
        self,
        run_ctx: RunContext[Any],
        deferred_tools: dict[str, str],
        non_deferred_tool_call: ToolCallPart,
    ) -> None:
        """Approval-only requests (not in deferred_tools) return None."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        requests = DeferredToolRequests(approvals=[non_deferred_tool_call])
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools)

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)
        assert result is None


# ============================================================================
# Tests: approval list handling
# ============================================================================


class TestApprovalListHandling:
    """Tests for deferred tools that appear in requests.approvals (unapproved kind).

    The DeferredToolBridge only handles ``requests.calls`` (external execution).
    ``requests.approvals`` pass through to the next capability (approval_bridge).
    """

    @pytest.mark.anyio
    async def test_unapproved_block_tool_in_approvals_passes_through(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """Deferred unapproved tool with block strategy in approvals → pass through."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        # Tool with deferred_kind="unapproved" appears in approvals list
        approval_call = ToolCallPart(
            tool_name="dangerous_op",
            args={"target": "/etc/passwd"},
            tool_call_id="tc-unapproved-1",
        )
        requests = DeferredToolRequests(approvals=[approval_call])

        deferred_tools_map = {"dangerous_op": "block"}
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools_map)

        # Approvals are NOT handled by deferred bridge — pass through to approval_bridge
        with patch(
            "wolfharness.agents.native_agent.deferred_bridge._emit_deferred_event",
            new_callable=AsyncMock,
        ) as mock_emit:
            result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # Bridge returns None for approval-only requests
        assert result is None
        mock_emit.assert_not_called()

    @pytest.mark.anyio
    async def test_continue_unapproved_tool_in_approvals_passes_through(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """Deferred unapproved tool with continue strategy in approvals → pass through."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            create_deferred_bridge_capability,
        )

        approval_call = ToolCallPart(
            tool_name="safe_background_task",
            args={"job": "cleanup"},
            tool_call_id="tc-cont-approval-1",
        )
        requests = DeferredToolRequests(approvals=[approval_call])

        deferred_tools_map = {"safe_background_task": "continue"}
        cap = create_deferred_bridge_capability(deferred_tools=deferred_tools_map)

        result = await cap.handle_deferred_tool_calls(run_ctx, requests=requests)

        # Bridge returns None for approval-only requests
        assert result is None


# ============================================================================
# Tests: _emit_deferred_event helper
# ============================================================================


class TestEmitDeferredEvent:
    """Tests for the _emit_deferred_event helper function."""

    @pytest.mark.anyio
    async def test_emit_uses_event_bus_when_available(
        self,
        run_ctx: RunContext[Any],
        block_tool_call: ToolCallPart,
    ) -> None:
        """Event is published via event_bus when available."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            _emit_deferred_event,
        )

        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        run_ctx.deps.run_ctx.event_bus = mock_bus

        event = ToolCallDeferredEvent(
            tool_call_id="tc-1",
            tool_name="bash",
            deferred_strategy="block",
            status="pending",
            session_id="test-session",
        )

        await _emit_deferred_event(run_ctx, event)

        mock_bus.publish.assert_called_once_with("test-session", event)

    @pytest.mark.anyio
    async def test_emit_graceful_when_no_event_bus(
        self,
        run_ctx: RunContext[Any],
    ) -> None:
        """No crash when event_bus is unavailable."""
        from wolfharness.agents.native_agent.deferred_bridge import (
            _emit_deferred_event,
        )

        run_ctx.deps.run_ctx.event_bus = None

        event = ToolCallDeferredEvent(
            tool_call_id="tc-1",
            tool_name="bash",
            deferred_strategy="block",
            status="pending",
            session_id="test-session",
        )

        # Should not raise
        await _emit_deferred_event(run_ctx, event)
