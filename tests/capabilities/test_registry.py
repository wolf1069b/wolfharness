"""Tests for entry-point capability discovery and registry.

Covers:
1. ``discover_entry_point_capabilities()`` returns correct mapping
2. ``CapabilityNotFoundError`` lists all available types in error message
3. ``AgentFactory.compile()`` populates ``entry_point_capabilities``
4. ``AgentFactory.resolve_capability_type()`` resolves known types
5. ``AgentFactory.resolve_capability_type()`` raises for unknown types
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from wolfharness.capabilities.registry import (
    CapabilityNotFoundError,
    discover_entry_point_capabilities,
    resolve_capability_type,
)


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pydantic_ai.capabilities import AbstractCapability

    from wolfharness import AgentPool


# ---- Helpers ----


class _FakeEntryPoint:
    """Minimal fake entry point for testing discovery."""

    def __init__(self, name: str, loaded_value: Any) -> None:
        self._name = name
        self._loaded_value = loaded_value

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> Any:
        return self._loaded_value


def _make_fake_entry_points(
    mapping: dict[str, type[AbstractCapability[object]]],
) -> list[_FakeEntryPoint]:
    """Create fake entry points from a name→class mapping."""
    return [_FakeEntryPoint(name, cls) for name, cls in mapping.items()]


# ---- discover_entry_point_capabilities tests ----


def test_discover_returns_mapping_from_entry_points() -> None:
    """discover_entry_point_capabilities returns a name→class mapping."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
    from wolfharness.capabilities.subagent_capability import SubagentCapability

    fake_eps = _make_fake_entry_points(
        {"function_tools": FunctionToolsetCapability, "subagent": SubagentCapability},
    )
    with patch(
        "wolfharness.capabilities.registry.entry_points",
        return_value=fake_eps,
    ):
        result = discover_entry_point_capabilities()
    assert result == {"function_tools": FunctionToolsetCapability, "subagent": SubagentCapability}


def test_discover_returns_empty_when_no_entry_points() -> None:
    """discover_entry_point_capabilities returns empty dict when no entry points."""
    with patch(
        "wolfharness.capabilities.registry.entry_points",
        return_value=[],
    ):
        result = discover_entry_point_capabilities()
    assert result == {}


def test_discover_first_wins_on_duplicate_names() -> None:
    """discover_entry_point_capabilities keeps the first entry on duplicate names."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
    from wolfharness.capabilities.subagent_capability import SubagentCapability

    fake_eps = [
        _FakeEntryPoint("dup", FunctionToolsetCapability),
        _FakeEntryPoint("dup", SubagentCapability),
    ]
    with patch(
        "wolfharness.capabilities.registry.entry_points",
        return_value=fake_eps,
    ):
        result = discover_entry_point_capabilities()
    assert result == {"dup": FunctionToolsetCapability}


# ---- CapabilityNotFoundError tests ----


def test_capability_not_found_error_lists_available_types() -> None:
    """CapabilityNotFoundError lists all available types in the message."""
    available = ["function_tools", "mcp", "subagent"]
    err = CapabilityNotFoundError("nonexistent", available)
    assert err.requested_type == "nonexistent"
    assert err.available_types == ["function_tools", "mcp", "subagent"]
    msg = str(err)
    assert "nonexistent" in msg
    assert "function_tools" in msg
    assert "mcp" in msg
    assert "subagent" in msg


def test_capability_not_found_error_with_empty_available() -> None:
    """CapabilityNotFoundError works with empty available list."""
    err = CapabilityNotFoundError("anything", [])
    assert err.requested_type == "anything"
    assert err.available_types == []
    msg = str(err)
    assert "anything" in msg
    assert "(none)" in msg


# ---- resolve_capability_type tests ----


def test_resolve_capability_type_found() -> None:
    """resolve_capability_type returns the class when type is registered."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability

    registry: dict[str, type[AbstractCapability[object]]] = {
        "function_tools": FunctionToolsetCapability,
    }
    result = resolve_capability_type("function_tools", registry)
    assert result is FunctionToolsetCapability


def test_resolve_capability_type_not_found_raises() -> None:
    """resolve_capability_type raises CapabilityNotFoundError for unknown type."""
    registry: dict[str, type[AbstractCapability[object]]] = {
        "function_tools": type("FakeCap", (), {}),
    }
    with pytest.raises(CapabilityNotFoundError) as exc_info:
        resolve_capability_type("unknown_type", registry)
    assert exc_info.value.requested_type == "unknown_type"
    assert "function_tools" in exc_info.value.available_types


# ---- AgentFactory integration tests ----


def test_factory_compile_discovers_entry_point_capabilities(minimal_pool: AgentPool) -> None:
    """AgentFactory.compile() populates entry_point_capabilities from discovery."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
    from wolfharness.host.factory import AgentFactory

    fake_eps = _make_fake_entry_points({"function_tools": FunctionToolsetCapability})

    factory = AgentFactory(minimal_pool)

    with (
        patch(
            "wolfharness.capabilities.registry.entry_points",
            return_value=fake_eps,
        ),
        patch("wolfharness.host.registry.AgentRegistry"),
    ):
        # compile() iterates manifest.agents — use empty manifest.
        mock_manifest = MagicMock()
        mock_manifest.agents = {}
        mock_host_context = MagicMock()
        factory.compile(mock_manifest, mock_host_context)

    assert factory.entry_point_capabilities == {"function_tools": FunctionToolsetCapability}


def test_factory_resolve_capability_type_known(minimal_pool: AgentPool) -> None:
    """AgentFactory.resolve_capability_type resolves a known type."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
    from wolfharness.capabilities.subagent_capability import SubagentCapability
    from wolfharness.host.factory import AgentFactory

    fake_eps = _make_fake_entry_points(
        {"function_tools": FunctionToolsetCapability, "subagent": SubagentCapability},
    )
    factory = AgentFactory(minimal_pool)

    with (
        patch(
            "wolfharness.capabilities.registry.entry_points",
            return_value=fake_eps,
        ),
        patch("wolfharness.host.registry.AgentRegistry"),
    ):
        mock_manifest = MagicMock()
        mock_manifest.agents = {}
        mock_host_context = MagicMock()
        factory.compile(mock_manifest, mock_host_context)

    result = factory.resolve_capability_type("subagent")
    assert result is SubagentCapability


def test_factory_resolve_capability_type_unknown_raises(minimal_pool: AgentPool) -> None:
    """AgentFactory.resolve_capability_type raises for unknown type."""
    from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
    from wolfharness.host.factory import AgentFactory

    fake_eps = _make_fake_entry_points({"function_tools": FunctionToolsetCapability})
    factory = AgentFactory(minimal_pool)

    with (
        patch(
            "wolfharness.capabilities.registry.entry_points",
            return_value=fake_eps,
        ),
        patch("wolfharness.host.registry.AgentRegistry"),
    ):
        mock_manifest = MagicMock()
        mock_manifest.agents = {}
        mock_host_context = MagicMock()
        factory.compile(mock_manifest, mock_host_context)

    with pytest.raises(CapabilityNotFoundError) as exc_info:
        factory.resolve_capability_type("nonexistent")
    assert exc_info.value.requested_type == "nonexistent"
    assert "function_tools" in exc_info.value.available_types
