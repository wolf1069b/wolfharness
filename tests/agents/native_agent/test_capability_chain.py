"""Tests for capability chain ordering in get_agentlet().

Verifies that DeferredToolBridge is registered BEFORE ApprovalBridge
in the pydantic-ai capability chain, as required by Decision 9.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_agent() -> Agent[Any]:
    """Create an agent with TestModel for capability chain testing."""
    model = TestModel(custom_output_text="test")
    return Agent(name="chain-test-agent", model=model)


# ---------------------------------------------------------------------------
# Test: DeferredToolBridge is registered BEFORE ApprovalBridge
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deferred_bridge_before_approval_bridge(
    mock_agent: Agent[Any],
) -> None:
    """DeferredToolBridge capability appears before ElicitationBridge before ApprovalBridge."""
    # Patch at the DEFINITION module (deferred_bridge, elicitation_bridge,
    # approval_bridge), not the importing module (agent), because get_agentlet()
    # uses local imports.
    with (
        patch(
            "wolfharness.agents.native_agent.deferred_bridge.create_deferred_bridge_capability",
        ) as mock_create_deferred,
        patch(
            "wolfharness.agents.native_agent.elicitation_bridge.create_elicitation_bridge_capability",
        ) as mock_create_elicitation,
        patch(
            "wolfharness.agents.native_agent.approval_bridge.create_approval_bridge_capability",
        ) as mock_create_approval,
        patch(
            "wolfharness.agents.native_agent.agent.PydanticAgent",
        ) as mock_pydantic_agent,
    ):
        # Create distinguishable mock capabilities
        deferred_cap = MagicMock(spec=HandleDeferredToolCalls)
        elicitation_cap = MagicMock(spec=HandleDeferredToolCalls)
        approval_cap = MagicMock(spec=HandleDeferredToolCalls)
        mock_create_deferred.return_value = deferred_cap
        mock_create_elicitation.return_value = elicitation_cap
        mock_create_approval.return_value = approval_cap
        mock_pydantic_agent.return_value = MagicMock()

        await mock_agent.get_agentlet(None, None, None)

        call_kwargs = mock_pydantic_agent.call_args.kwargs
        capabilities = call_kwargs.get("capabilities", []) or []

        # All three capabilities must be in the list
        assert deferred_cap in capabilities, (
            "DeferredToolBridge capability must be present in capability chain"
        )
        assert elicitation_cap in capabilities, (
            "ElicitationBridge capability must be present in capability chain"
        )
        assert approval_cap in capabilities, (
            "ApprovalBridge capability must be present in capability chain"
        )

        # Deferred bridge → Elicitation bridge → Approval bridge (order matters)
        deferred_idx = capabilities.index(deferred_cap)
        elicitation_idx = capabilities.index(elicitation_cap)
        approval_idx = capabilities.index(approval_cap)
        assert deferred_idx < elicitation_idx < approval_idx, (
            f"DeferredToolBridge at {deferred_idx} must be BEFORE "
            f"ElicitationBridge at {elicitation_idx} BEFORE "
            f"ApprovalBridge at {approval_idx} in capability chain"
        )


# ---------------------------------------------------------------------------
# Test: DeferredBridge receives deferred tools mapping
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deferred_bridge_receives_deferred_tools_mapping(
    mock_agent: Agent[Any],
) -> None:
    """create_deferred_bridge_capability gets deferred_tools with correct mapping."""
    # Create mock tools
    mock_deferred_tool = MagicMock()
    mock_deferred_tool.name = "bash_exec"
    mock_deferred_tool.deferred = True
    mock_deferred_tool.deferred_strategy = "block"

    mock_continue_tool = MagicMock()
    mock_continue_tool.name = "background_task"
    mock_continue_tool.deferred = True
    mock_continue_tool.deferred_strategy = "continue"

    mock_normal_tool = MagicMock()
    mock_normal_tool.name = "read_file"
    mock_normal_tool.deferred = False

    # Mock get_tools to return our test tools
    mock_agent._get_all_tools = AsyncMock(
        return_value=[mock_deferred_tool, mock_continue_tool, mock_normal_tool]
    )

    with (
        patch(
            "wolfharness.agents.native_agent.deferred_bridge.create_deferred_bridge_capability",
        ) as mock_create_deferred,
        patch(
            "wolfharness.agents.native_agent.approval_bridge.create_approval_bridge_capability",
        ) as mock_create_approval,
        patch(
            "wolfharness.agents.native_agent.agent.PydanticAgent",
        ) as mock_pydantic_agent,
    ):
        deferred_cap = MagicMock(spec=HandleDeferredToolCalls)
        approval_cap = MagicMock(spec=HandleDeferredToolCalls)
        mock_create_deferred.return_value = deferred_cap
        mock_create_approval.return_value = approval_cap
        mock_pydantic_agent.return_value = MagicMock()

        await mock_agent.get_agentlet(None, None, None)

        # Verify create_deferred_bridge_capability was called with deferred_tools mapping
        mock_create_deferred.assert_called_once()
        deferred_tools_arg = mock_create_deferred.call_args[0][0]

        assert "bash_exec" in deferred_tools_arg, (
            "Deferred tool 'bash_exec' must be in deferred_tools mapping"
        )
        assert "background_task" in deferred_tools_arg, (
            "Deferred tool 'background_task' must be in deferred_tools mapping"
        )
        assert deferred_tools_arg["bash_exec"] == "block"
        assert deferred_tools_arg["background_task"] == "continue"

        # Non-deferred tools should NOT be in deferred_tools
        assert "read_file" not in deferred_tools_arg, (
            "Non-deferred tool 'read_file' must NOT be in deferred_tools mapping"
        )


# ---------------------------------------------------------------------------
# Test: Empty deferred_tools when no deferred tools are configured
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deferred_bridge_empty_when_no_deferred_tools(
    mock_agent: Agent[Any],
) -> None:
    """create_deferred_bridge_capability gets empty dict when no deferred tools."""
    # Mock get_tools to return only non-deferred tools
    mock_normal_tool = MagicMock()
    mock_normal_tool.name = "read_file"
    mock_normal_tool.deferred = False

    mock_agent._get_all_tools = AsyncMock(return_value=[mock_normal_tool])

    with (
        patch(
            "wolfharness.agents.native_agent.deferred_bridge.create_deferred_bridge_capability",
        ) as mock_create_deferred,
        patch(
            "wolfharness.agents.native_agent.approval_bridge.create_approval_bridge_capability",
        ) as mock_create_approval,
        patch(
            "wolfharness.agents.native_agent.agent.PydanticAgent",
        ) as mock_pydantic_agent,
    ):
        deferred_cap = MagicMock(spec=HandleDeferredToolCalls)
        approval_cap = MagicMock(spec=HandleDeferredToolCalls)
        mock_create_deferred.return_value = deferred_cap
        mock_create_approval.return_value = approval_cap
        mock_pydantic_agent.return_value = MagicMock()

        await mock_agent.get_agentlet(None, None, None)

        # Verify create_deferred_bridge_capability was called with empty dict
        mock_create_deferred.assert_called_once()
        deferred_tools_arg = mock_create_deferred.call_args[0][0]
        assert deferred_tools_arg == {}, (
            "deferred_tools should be empty when no deferred tools are configured"
        )


# ---------------------------------------------------------------------------
# Test: get_tools failure handled gracefully
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deferred_bridge_handles_get_tools_failure(
    mock_agent: Agent[Any],
) -> None:
    """DeferredToolBridge handles get_tools() failure gracefully (empty dict)."""
    mock_agent._get_all_tools = AsyncMock(side_effect=RuntimeError("Provider error"))

    with (
        patch(
            "wolfharness.agents.native_agent.deferred_bridge.create_deferred_bridge_capability",
        ) as mock_create_deferred,
        patch(
            "wolfharness.agents.native_agent.approval_bridge.create_approval_bridge_capability",
        ) as mock_create_approval,
        patch(
            "wolfharness.agents.native_agent.agent.PydanticAgent",
        ) as mock_pydantic_agent,
    ):
        deferred_cap = MagicMock(spec=HandleDeferredToolCalls)
        approval_cap = MagicMock(spec=HandleDeferredToolCalls)
        mock_create_deferred.return_value = deferred_cap
        mock_create_approval.return_value = approval_cap
        mock_pydantic_agent.return_value = MagicMock()

        # Should not raise despite get_tools() failure
        await mock_agent.get_agentlet(None, None, None)

        # Verify deferred bridge was still created (with empty dict fallback)
        mock_create_deferred.assert_called_once()
        deferred_tools_arg = mock_create_deferred.call_args[0][0]
        assert deferred_tools_arg == {}, (
            "deferred_tools should fall back to empty dict on get_tools() failure"
        )

        # Verify approval bridge still created
        mock_create_approval.assert_called_once()
