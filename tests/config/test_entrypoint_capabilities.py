"""Tests for entry-point-based capability resolution from YAML config.

Tests that ``type: <entrypoint_name>`` in YAML resolves to the correct
:class:`EntryPointCapabilityConfig` and that the config's ``build()``
method instantiates the capability class registered via the
``wolfharness.capabilities`` entry-point group.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from wolfharness import AgentsManifest, NativeAgentConfig
from wolfharness_config.capabilities import (
    EntryPointCapabilityConfig,
    GenericCapabilityConfig,
    build_capability,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake capability classes used for testing entrypoint resolution
# ---------------------------------------------------------------------------


class _FakeMermaidLintCapability:
    """Fake capability class simulating an entry-point-registered capability."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeCustomCapability:
    """Another fake capability for multi-entrypoint tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


_FAKE_REGISTRY: dict[str, type] = {
    "mermaid_lint": _FakeMermaidLintCapability,
    "custom_cap": _FakeCustomCapability,
}

# Patch paths
_DISCOVER_PATH = "wolfharness.capabilities.registry.discover_entry_point_capabilities"
_ENTRY_POINTS_PATH = "importlib.metadata.entry_points"


def _make_fake_entry_points(registry: dict[str, type]) -> list[MagicMock]:
    """Create fake EntryPoint objects that mimic importlib.metadata.entry_points."""
    eps: list[MagicMock] = []
    for name, cls in registry.items():
        ep = MagicMock()
        ep.name = name
        ep.load = MagicMock(return_value=cls)
        eps.append(ep)
    return eps


# ---------------------------------------------------------------------------
# YAML fixtures
# ---------------------------------------------------------------------------


YAML_WITH_ENTRYPOINT_CAP = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities:
      - type: mermaid_lint
        args:
          strict: true
"""

YAML_WITH_ENTRYPOINT_AND_IMPORT = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities:
      - type: mermaid_lint
        args:
          strict: true
      - type: pydantic_ai.capabilities.Instrumentation
        args:
          service_name: test
"""

YAML_WITH_UNKNOWN_SHORT_NAME = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities:
      - type: some_unknown_name
        args:
          foo: bar
"""


# ---------------------------------------------------------------------------
# Tests: YAML parsing → EntryPointCapabilityConfig
# ---------------------------------------------------------------------------


def test_entrypoint_capability_parsed_from_yaml() -> None:
    """Test that a known entry-point name produces EntryPointCapabilityConfig."""
    with patch(_DISCOVER_PATH, return_value=_FAKE_REGISTRY):
        manifest = AgentsManifest.from_yaml(YAML_WITH_ENTRYPOINT_CAP)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert len(agent.capabilities) == 1
    cap = agent.capabilities[0]
    assert isinstance(cap, EntryPointCapabilityConfig)
    assert cap.type == "mermaid_lint"
    assert cap.args == {"strict": True}


def test_entrypoint_and_import_path_coexist() -> None:
    """Test that entry-point names and import paths coexist in the same config."""
    with patch(_DISCOVER_PATH, return_value=_FAKE_REGISTRY):
        manifest = AgentsManifest.from_yaml(YAML_WITH_ENTRYPOINT_AND_IMPORT)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert len(agent.capabilities) == 2

    cap0 = agent.capabilities[0]
    assert isinstance(cap0, EntryPointCapabilityConfig)
    assert cap0.type == "mermaid_lint"

    cap1 = agent.capabilities[1]
    assert isinstance(cap1, GenericCapabilityConfig)
    assert cap1.type == "pydantic_ai.capabilities.Instrumentation"


def test_unknown_name_falls_back_to_generic() -> None:
    """Test that names not in entry-point registry fall back to GenericCapabilityConfig."""
    with patch(_DISCOVER_PATH, return_value=_FAKE_REGISTRY):
        manifest = AgentsManifest.from_yaml(YAML_WITH_UNKNOWN_SHORT_NAME)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert len(agent.capabilities) == 1
    cap = agent.capabilities[0]
    assert isinstance(cap, GenericCapabilityConfig)
    assert cap.type == "some_unknown_name"


def test_empty_registry_falls_back_to_generic() -> None:
    """Test that an empty entry-point registry falls back to GenericCapabilityConfig."""
    with patch(_DISCOVER_PATH, return_value={}):
        manifest = AgentsManifest.from_yaml(YAML_WITH_ENTRYPOINT_CAP)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert len(agent.capabilities) == 1
    cap = agent.capabilities[0]
    assert isinstance(cap, GenericCapabilityConfig)


# ---------------------------------------------------------------------------
# Tests: EntryPointCapabilityConfig.build()
# ---------------------------------------------------------------------------


def test_entrypoint_config_builds_capability() -> None:
    """Test that EntryPointCapabilityConfig.build() instantiates the correct class."""
    fake_eps = _make_fake_entry_points(_FAKE_REGISTRY)
    with patch(_ENTRY_POINTS_PATH, return_value=fake_eps):
        config = EntryPointCapabilityConfig(type="mermaid_lint", args={"strict": True})
        result = config.build()
    assert isinstance(result, _FakeMermaidLintCapability)
    assert result.kwargs == {"strict": True}


def test_entrypoint_config_build_with_empty_args() -> None:
    """Test that EntryPointCapabilityConfig.build() works with no args."""
    fake_eps = _make_fake_entry_points(_FAKE_REGISTRY)
    with patch(_ENTRY_POINTS_PATH, return_value=fake_eps):
        config = EntryPointCapabilityConfig(type="custom_cap")
        result = config.build()
    assert isinstance(result, _FakeCustomCapability)
    assert result.kwargs == {}


def test_entrypoint_config_build_unknown_raises() -> None:
    """Test that build() raises ValueError for an unknown entry-point name."""
    fake_eps = _make_fake_entry_points(_FAKE_REGISTRY)
    with patch(_ENTRY_POINTS_PATH, return_value=fake_eps):
        config = EntryPointCapabilityConfig(type="nonexistent_cap")
        with pytest.raises(ValueError, match="Unknown entry-point capability"):
            config.build()


# ---------------------------------------------------------------------------
# Tests: build_capability() dispatch
# ---------------------------------------------------------------------------


def test_build_capability_dispatches_entrypoint() -> None:
    """Test that build_capability() handles EntryPointCapabilityConfig."""
    fake_eps = _make_fake_entry_points(_FAKE_REGISTRY)
    with patch(_ENTRY_POINTS_PATH, return_value=fake_eps):
        config = EntryPointCapabilityConfig(type="mermaid_lint", args={"strict": False})
        result = build_capability(config)
    assert isinstance(result, _FakeMermaidLintCapability)
    assert result.kwargs == {"strict": False}


if __name__ == "__main__":
    pytest.main(["-v", __file__])
