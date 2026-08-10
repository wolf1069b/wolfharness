"""Tests for ServerState.resolve_default_model_info() and dynamic model default propagation.

Replaces hardcoded "sonnet"/"claude-code"/"default"/"wolfharness" fallbacks with
config-derived defaults resolved from the configured agent's model_name.
"""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock

import pytest

from wolfharness_server.opencode_server.state import ServerState


pytestmark = pytest.mark.unit


def _make_server_state(model_name: str | None) -> ServerState:
    """Create a ServerState with a mock agent whose model_name returns the given string."""
    agent = Mock()
    # model_name is a property on BaseAgent, so we need to set it on the class mock
    type(agent).model_name = PropertyMock(return_value=model_name)
    # model_variants must be a dict (not a Mock) so _find_variant_name can iterate
    agent._agent_pool.manifest.model_variants = {}
    return ServerState(working_dir="/tmp", agent=agent)


class TestResolveDefaultModelInfo:
    """resolve_default_model_info() splits agent.model_name into (model_id, provider_id)."""

    def test_resolves_openai_model(self) -> None:
        """When agent.model_name is 'openai:gpt-4o', returns ('gpt-4o', 'openai')."""
        state = _make_server_state("openai:gpt-4o")
        model_id, provider_id = state.resolve_default_model_info()
        assert model_id == "gpt-4o"
        assert provider_id == "openai"

    def test_resolves_anthropic_model(self) -> None:
        """When agent.model_name is 'anthropic:claude-sonnet-4-0'."""
        state = _make_server_state("anthropic:claude-sonnet-4-0")
        model_id, provider_id = state.resolve_default_model_info()
        assert model_id == "claude-sonnet-4-0"
        assert provider_id == "anthropic"

    def test_resolves_model_with_colon_in_name(self) -> None:
        """Splits on first colon only (e.g. 'openai:o3-mini:2025-01-31')."""
        state = _make_server_state("openai:o3-mini:2025-01-31")
        model_id, provider_id = state.resolve_default_model_info()
        assert model_id == "o3-mini:2025-01-31"
        assert provider_id == "openai"

    def test_falls_back_when_model_name_is_none(self) -> None:
        """When agent.model_name is None, returns ('default', 'wolfharness')."""
        state = _make_server_state(None)
        model_id, provider_id = state.resolve_default_model_info()
        assert model_id == "default"
        assert provider_id == "wolfharness"

    def test_falls_back_when_model_name_has_no_colon(self) -> None:
        """When agent.model_name has no colon (e.g. 'gpt-4o'), returns ('default', 'wolfharness').

        This is a graceful fallback — a well-formed model_name should always
        include a provider prefix, but we handle the edge case safely.
        """
        state = _make_server_state("gpt-4o")
        model_id, provider_id = state.resolve_default_model_info()
        assert model_id == "default"
        assert provider_id == "wolfharness"
