"""Tests for the ApprovalRequiredToolset -> InputProvider bridge.

Verifies that pydantic-ai deferred tool approval requests are correctly
routed through AgentPool's InputProvider and mapped back to pydantic-ai's
expected ToolApproved/ToolDenied format.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import (
    DeferredToolRequests,
    RunContext,
    ToolApproved,
    ToolDenied,
)
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentContext, AgentRunContext
from wolfharness.agents.native_agent.approval_bridge import (
    _map_confirmation_result,
    _resolve_deferred_approvals,
    create_approval_bridge_capability,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_agent() -> Agent[Any]:
    """Create an agent with mocked internals for approval bridge testing."""
    model = TestModel(custom_output_text="test")
    return Agent(name="approval-test-agent", model=model)


@pytest.fixture
def mock_run_context() -> RunContext[Any]:
    """Create a mock RunContext with AgentContext deps."""
    node = MagicMock()
    node.name = "test-agent"
    node.tool_confirmation_mode = "per_tool"

    agent_run_ctx = AgentRunContext(session_id="test-session")
    agent_ctx = AgentContext(node=node, run_ctx=agent_run_ctx)

    model = MagicMock()
    model.system = "test"
    model.model_name = "test-model"

    return RunContext(
        deps=agent_ctx,
        model=model,
        usage=MagicMock(),
    )


@pytest.fixture
def sample_deferred_requests() -> DeferredToolRequests:
    """Create sample DeferredToolRequests with approval requests."""
    return DeferredToolRequests(
        approvals=[
            ToolCallPart(
                tool_name="dangerous_tool",
                args={"target": "/etc/passwd"},
                tool_call_id="tc-123",
            ),
            ToolCallPart(
                tool_name="safe_tool",
                args={"query": "hello"},
                tool_call_id="tc-456",
            ),
        ]
    )


class TestMapConfirmationResult:
    """Test suite for _map_confirmation_result helper."""

    def test_allow_maps_to_tool_approved(self) -> None:
        """Allow result maps to ToolApproved."""
        result = _map_confirmation_result("allow", "test_tool")
        assert isinstance(result, ToolApproved)

    def test_skip_maps_to_tool_denied(self) -> None:
        """Skip result maps to ToolDenied with skip message."""
        result = _map_confirmation_result("skip", "test_tool")
        assert isinstance(result, ToolDenied)
        assert "skipped" in result.message
        assert "test_tool" in result.message

    def test_abort_run_maps_to_tool_denied(self) -> None:
        """abort_run result maps to ToolDenied with abort message."""
        result = _map_confirmation_result("abort_run", "test_tool")
        assert isinstance(result, ToolDenied)
        assert "run aborted" in result.message
        assert "test_tool" in result.message

    def test_abort_chain_maps_to_tool_denied(self) -> None:
        """abort_chain result maps to ToolDenied with abort message."""
        result = _map_confirmation_result("abort_chain", "test_tool")
        assert isinstance(result, ToolDenied)
        assert "chain aborted" in result.message
        assert "test_tool" in result.message


class TestResolveDeferredApprovals:
    """Test suite for _resolve_deferred_approvals."""

    @pytest.mark.anyio
    async def test_approval_routed_to_input_provider(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """Deferred approval requests are routed to InputProvider."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")

        mock_run_context.deps.input_provider = mock_provider

        result = await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        assert result is not None
        assert mock_provider.get_tool_confirmation.call_count == 2

    @pytest.mark.anyio
    async def test_allow_result_maps_to_tool_approved(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """InputProvider 'allow' maps to ToolApproved."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")

        mock_run_context.deps.input_provider = mock_provider

        result = await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        assert result is not None
        assert isinstance(result.approvals["tc-123"], ToolApproved)
        assert isinstance(result.approvals["tc-456"], ToolApproved)

    @pytest.mark.anyio
    async def test_skip_result_maps_to_tool_denied(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """InputProvider 'skip' maps to ToolDenied."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(return_value="skip")

        mock_run_context.deps.input_provider = mock_provider

        result = await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        assert result is not None
        assert isinstance(result.approvals["tc-123"], ToolDenied)
        assert "skipped" in result.approvals["tc-123"].message

    @pytest.mark.anyio
    async def test_mixed_approvals_and_denials(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """Mixed allow/skip results handled correctly."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(side_effect=["allow", "skip"])

        mock_run_context.deps.input_provider = mock_provider

        result = await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        assert result is not None
        assert isinstance(result.approvals["tc-123"], ToolApproved)
        assert isinstance(result.approvals["tc-456"], ToolDenied)

    @pytest.mark.anyio
    async def test_never_mode_auto_approves(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """Never mode: bridge routes to provider when called directly.

        In never mode, ApprovalRequiredToolset is not applied, so deferred
        requests never reach the bridge. This test verifies the bridge
        routes to the provider when called directly (no mode check).
        """
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")

        mock_run_context.deps.input_provider = mock_provider
        mock_run_context.deps.node.tool_confirmation_mode = "never"

        result = await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        assert result is not None
        assert mock_provider.get_tool_confirmation.call_count == 2

    @pytest.mark.anyio
    async def test_provider_error_defaults_to_denial(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """InputProvider error defaults to ToolDenied."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(side_effect=RuntimeError("Provider failed"))

        mock_run_context.deps.input_provider = mock_provider

        result = await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        assert result is not None
        assert isinstance(result.approvals["tc-123"], ToolDenied)
        assert "skipped" in result.approvals["tc-123"].message

    @pytest.mark.anyio
    async def test_empty_approvals_returns_none(
        self,
        mock_run_context: RunContext[Any],
    ) -> None:
        """Empty approval requests returns None."""
        requests = DeferredToolRequests(approvals=[])

        result = await _resolve_deferred_approvals(mock_run_context, requests)

        assert result is None

    @pytest.mark.anyio
    async def test_confirmation_context_has_tool_details(
        self,
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """InputProvider receives context with correct tool details."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")

        mock_run_context.deps.input_provider = mock_provider

        await _resolve_deferred_approvals(mock_run_context, sample_deferred_requests)

        # Check first call
        call_args = mock_provider.get_tool_confirmation.call_args_list[0]
        ctx = call_args.args[0]
        assert ctx.tool_name == "dangerous_tool"
        assert ctx.tool_call_id == "tc-123"
        assert ctx.tool_input == {"target": "/etc/passwd"}


class TestCreateApprovalBridgeCapability:
    """Test suite for create_approval_bridge_capability."""

    def test_returns_handle_deferred_tool_calls(self, mock_agent: Agent[Any]) -> None:
        """Returns a HandleDeferredToolCalls capability."""
        from pydantic_ai.capabilities import HandleDeferredToolCalls

        cap = create_approval_bridge_capability(mock_agent)
        assert isinstance(cap, HandleDeferredToolCalls)

    @pytest.mark.anyio
    async def test_handler_resolves_approvals(
        self,
        mock_agent: Agent[Any],
        mock_run_context: RunContext[Any],
        sample_deferred_requests: DeferredToolRequests,
    ) -> None:
        """Capability handler resolves approval requests."""
        mock_provider = MagicMock()
        mock_provider.get_tool_confirmation = AsyncMock(return_value="allow")

        mock_run_context.deps.input_provider = mock_provider

        cap = create_approval_bridge_capability(mock_agent)
        result = await cap.handle_deferred_tool_calls(
            mock_run_context, requests=sample_deferred_requests
        )

        assert result is not None
        assert isinstance(result.approvals["tc-123"], ToolApproved)

    @pytest.mark.anyio
    async def test_handler_returns_none_for_no_approvals(
        self,
        mock_agent: Agent[Any],
        mock_run_context: RunContext[Any],
    ) -> None:
        """Capability handler returns None when no approval requests."""
        requests = DeferredToolRequests(approvals=[])

        cap = create_approval_bridge_capability(mock_agent)
        result = await cap.handle_deferred_tool_calls(mock_run_context, requests=requests)

        assert result is None


class TestGetAgentletIntegration:
    """Test suite for get_agentlet() integration."""

    @pytest.mark.anyio
    async def test_get_agentlet_includes_approval_bridge_capability(
        self,
        mock_agent: Agent[Any],
    ) -> None:
        """get_agentlet() includes the approval bridge capability."""
        from pydantic_ai.capabilities import HandleDeferredToolCalls

        with patch("wolfharness.agents.native_agent.agent.PydanticAgent") as mock_pydantic_agent:
            mock_pydantic_agent.return_value = MagicMock()
            await mock_agent.get_agentlet(None, None, None)

            call_kwargs = mock_pydantic_agent.call_args.kwargs
            capabilities = call_kwargs.get("capabilities", []) or []

            bridge_caps = [cap for cap in capabilities if isinstance(cap, HandleDeferredToolCalls)]
            assert len(bridge_caps) == 3, (
                "Expected three HandleDeferredToolCalls capabilities "
                "(DeferredToolBridge + ElicitationBridge + ApprovalBridge)"
            )
