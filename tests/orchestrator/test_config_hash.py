"""Tests for compute_agent_config_hash drift detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from wolfharness.agents.native_agent.checkpoint import compute_agent_config_hash


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from wolfharness.tools.base import Tool


def _make_tool(
    name: str = "test_tool",
    deferred: bool = False,
    deferred_kind: Literal["external", "unapproved"] = "external",
    deferred_strategy: Literal["block", "continue", "stream"] = "block",
    description: str = "A test tool",
) -> Tool:
    """Helper to create a FunctionTool for testing."""
    from wolfharness.tools.base import FunctionTool

    return FunctionTool(
        name=name,
        description=description,
        deferred=deferred,
        deferred_kind=deferred_kind,
        deferred_strategy=deferred_strategy,
        callable=lambda: None,
    )


class TestConfigHashDeterministic:
    """Same tools → same hash."""

    def test_config_hash_deterministic(self) -> None:
        """Two identical tool lists must produce the same hash."""
        tool_a = _make_tool(name="alpha", deferred=True)
        tool_b = _make_tool(name="beta", deferred=False)

        hash_1 = compute_agent_config_hash([tool_a, tool_b])
        hash_2 = compute_agent_config_hash([tool_a, tool_b])

        assert hash_1 == hash_2, "Identical tool configs must produce the same hash"

    def test_config_hash_order_independent(self) -> None:
        """Hash must be independent of tool list order."""
        tool_a = _make_tool(name="alpha", deferred=True)
        tool_b = _make_tool(name="beta", deferred=False)

        hash_1 = compute_agent_config_hash([tool_a, tool_b])
        hash_2 = compute_agent_config_hash([tool_b, tool_a])

        assert hash_1 == hash_2, "Hash must be order-independent"


class TestConfigHashDetectsChanges:
    """Different deferred configs → different hash."""

    def test_config_hash_detects_deferred_flag_change(self) -> None:
        """Changing deferred flag must produce different hash."""
        tool_default = _make_tool(name="my_tool", deferred=False)
        tool_deferred = _make_tool(name="my_tool", deferred=True)

        hash_default = compute_agent_config_hash([tool_default])
        hash_deferred = compute_agent_config_hash([tool_deferred])

        assert hash_default != hash_deferred, (
            "Different deferred settings must produce different hashes"
        )

    def test_config_hash_detects_deferred_kind_change(self) -> None:
        """Changing deferred_kind must produce different hash."""
        tool_external = _make_tool(name="my_tool", deferred=True, deferred_kind="external")
        tool_unapproved = _make_tool(name="my_tool", deferred=True, deferred_kind="unapproved")

        hash_external = compute_agent_config_hash([tool_external])
        hash_unapproved = compute_agent_config_hash([tool_unapproved])

        assert hash_external != hash_unapproved, (
            "Different deferred_kind must produce different hashes"
        )

    def test_config_hash_detects_strategy_change(self) -> None:
        """Changing deferred_strategy must produce different hash."""
        tool_block = _make_tool(name="my_tool", deferred=True, deferred_strategy="block")
        tool_continue = _make_tool(name="my_tool", deferred=True, deferred_strategy="continue")

        hash_block = compute_agent_config_hash([tool_block])
        hash_continue = compute_agent_config_hash([tool_continue])

        assert hash_block != hash_continue, (
            "Different deferred_strategy must produce different hashes"
        )


class TestConfigHashIgnoresNonDeterministic:
    """Non-deferred fields must not affect hash."""

    def test_config_hash_ignores_description_change(self) -> None:
        """Different descriptions must not change the hash."""
        tool_1 = _make_tool(name="my_tool", description="Original description")
        tool_2 = _make_tool(name="my_tool", description="Changed description")

        hash_1 = compute_agent_config_hash([tool_1])
        hash_2 = compute_agent_config_hash([tool_2])

        assert hash_1 == hash_2, "Description changes must not affect the config hash"

    def test_config_hash_ignores_enabled_change(self) -> None:
        """Different enabled flags must not change the hash."""
        tool_1 = _make_tool(name="my_tool", deferred=True)
        tool_2 = _make_tool(name="my_tool", deferred=True)

        # Set enabled directly on the dataclass after creation
        from dataclasses import replace

        tool_1b = replace(tool_1, enabled=True)
        tool_2b = replace(tool_2, enabled=False)

        hash_1 = compute_agent_config_hash([tool_1b])
        hash_2 = compute_agent_config_hash([tool_2b])

        assert hash_1 == hash_2, "Enabled flag changes must not affect the config hash"
