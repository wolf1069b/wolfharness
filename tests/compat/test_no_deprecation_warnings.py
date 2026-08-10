"""TDD RED phase: Assert NO DeprecationWarning from deprecated APIs.

These tests currently FAIL because the deprecated APIs still emit DeprecationWarnings.
They will PASS after tasks 1.9-1.11 remove the warnings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import pytest

from wolfharness.hooks.agent_hooks import AgentHooks
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.messaging import ChatMessage
from wolfharness.messaging.connection_manager import ConnectionManager
from wolfharness.messaging.messagenode import MessageNode
from wolfharness.utils.context_wrapping import wrap_instruction


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _get_deprecation_warnings(
    captured: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Filter captured warnings to only DeprecationWarning."""
    return [w for w in captured if issubclass(w.category, DeprecationWarning)]


# ── Minimal concrete MessageNode for testing connect_to / create_connection ──


class _FakeMessageNode(MessageNode[Any, Any]):
    """Minimal non-abstract MessageNode for testing deprecated connect methods."""

    async def get_stats(self) -> Any:
        return {}

    async def run_iter(self, *prompts: Any, **kwargs: Any) -> AsyncIterator[ChatMessage[Any]]:
        if False:  # pragma: no cover — never yields in tests
            yield ChatMessage[Any]()  # type: ignore[abstract]


# ── Tests ──


def test_mcpmanager_no_deprecation_warning() -> None:
    """MCPManager instantiation must not emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MCPManager()

    deprecation = _get_deprecation_warnings(w)
    assert len(deprecation) == 0, (
        f"MCPManager() emitted {len(deprecation)} DeprecationWarning(s): "
        f"{[str(d.message) for d in deprecation]}"
    )


def test_agenthooks_no_deprecation_warning() -> None:
    """AgentHooks instantiation must not emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AgentHooks()

    deprecation = _get_deprecation_warnings(w)
    assert len(deprecation) == 0, (
        f"AgentHooks() emitted {len(deprecation)} DeprecationWarning(s): "
        f"{[str(d.message) for d in deprecation]}"
    )


def test_wrap_instruction_no_deprecation_warning() -> None:
    """wrap_instruction() must not emit DeprecationWarning."""

    def dummy_instruction(ctx: Any) -> str:
        return "dummy"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wrap_instruction(dummy_instruction)

    deprecation = _get_deprecation_warnings(w)
    assert len(deprecation) == 0, (
        f"wrap_instruction() emitted {len(deprecation)} DeprecationWarning(s): "
        f"{[str(d.message) for d in deprecation]}"
    )


def test_messagenode_connect_to_no_deprecation_warning() -> None:
    """MessageNode.connect_to() must not emit DeprecationWarning."""
    node = _FakeMessageNode(name="test_node")
    target = _FakeMessageNode(name="target_node")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        node.connect_to(target)

    deprecation = _get_deprecation_warnings(w)
    assert len(deprecation) == 0, (
        f"MessageNode.connect_to() emitted {len(deprecation)} DeprecationWarning(s): "
        f"{[str(d.message) for d in deprecation]}"
    )


def test_connectionmanager_create_connection_no_deprecation_warning() -> None:
    """ConnectionManager.create_connection() must not emit DeprecationWarning."""
    node = _FakeMessageNode(name="test_node")
    target = _FakeMessageNode(name="target_node")
    cm = ConnectionManager(node)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cm.create_connection(node, target)

    deprecation = _get_deprecation_warnings(w)
    assert len(deprecation) == 0, (
        f"ConnectionManager.create_connection() emitted {len(deprecation)} "
        f"DeprecationWarning(s): {[str(d.message) for d in deprecation]}"
    )
