"""Tests for model state configured-first logic (Phase 1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from acp.schema import SessionModelState
from wolfharness_server.acp_server.provider_router import ProviderRouter
from wolfharness_server.shared.model_utils import build_model_state_for_acp


pytestmark = pytest.mark.unit


class MockManifest:
    """Mock manifest for testing."""

    def __init__(self, model_variants=None):
        self.model_variants = model_variants or {}
        self.acp = None


class MockPool:
    """Mock agent pool for testing."""

    def __init__(self, manifest=None):
        self.manifest = manifest


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, model_name="gpt-4o", pool=None, toko_models=None):
        self.model_name = model_name
        self.host_context = pool
        self._toko_models = toko_models or []

    async def get_available_models(self):
        return self._toko_models


def create_toko_model(model_id, name, description="", provider=""):
    """Create a mock tokonomics model."""
    m = MagicMock()
    m.id = model_id
    m.id_override = None
    m.name = name
    m.description = description
    m.provider = provider
    return m


@pytest.fixture
def manifest_with_variants():
    """Manifest with configured variants."""
    from wolfharness.models.model_configs import StringModelConfig

    variants = {
        "fast_gpt": StringModelConfig(identifier="openai:gpt-4o-mini"),
        "smart": StringModelConfig(identifier="anthropic:claude-sonnet-4-5"),
    }
    return MockManifest(model_variants=variants)


@pytest.fixture
def empty_manifest():
    """Empty manifest."""
    return MockManifest()


class TestBuildModelStateForAcp:
    """Test build_model_state_for_acp configured-first logic."""

    async def test_configured_first_priority(self, manifest_with_variants):
        """Configured variants take priority over tokonomics."""
        pool = MockPool(manifest=manifest_with_variants)
        toko_models = [create_toko_model("openai:gpt-4o", "GPT-4o")]
        agent = MockAgent(model_name="openai:gpt-4o-mini", pool=pool, toko_models=toko_models)
        router = ProviderRouter(manifest_with_variants)  # type: ignore[arg-type]

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is not None
        assert isinstance(state, SessionModelState)
        model_ids = {m.model_id for m in state.available_models}
        # model_id should be variant names (aliases), not resolved identifiers
        assert "fast_gpt" in model_ids
        assert "smart" in model_ids
        # name should be the variant name (alias)
        names = {m.name for m in state.available_models}
        assert "fast_gpt" in names
        assert "smart" in names
        # Tokonomics models should NOT appear when configured variants exist
        assert "openai:gpt-4o" not in model_ids

    async def test_tokonomics_fallback(self, empty_manifest):
        """When no configured variants, tokonomics is used."""
        pool = MockPool(manifest=empty_manifest)
        toko_models = [
            create_toko_model("openai:gpt-4o", "GPT-4o", provider="openai"),
            create_toko_model("anthropic:claude-sonnet", "Claude Sonnet", provider="anthropic"),
        ]
        agent = MockAgent(model_name="openai:gpt-4o", pool=pool, toko_models=toko_models)
        router = ProviderRouter(empty_manifest)  # type: ignore[arg-type]

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is not None
        model_ids = {m.model_id for m in state.available_models}
        assert "openai:gpt-4o" in model_ids
        assert "anthropic:claude-sonnet" in model_ids

    async def test_provider_filtering(self, empty_manifest):
        """Disabled providers are filtered out from tokonomics fallback."""
        pool = MockPool(manifest=empty_manifest)
        toko_models = [
            create_toko_model("openai:gpt-4o", "GPT-4o", provider="openai"),
            create_toko_model("anthropic:claude-sonnet", "Claude Sonnet", provider="anthropic"),
        ]
        agent = MockAgent(model_name="openai:gpt-4o", pool=pool, toko_models=toko_models)
        router = ProviderRouter(empty_manifest)  # type: ignore[arg-type]
        await router.disable_provider("openai")

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is not None
        model_ids = {m.model_id for m in state.available_models}
        # openai provider models should be filtered out
        assert "openai:gpt-4o" not in model_ids
        # anthropic provider models should remain
        assert "anthropic:claude-sonnet" in model_ids

    async def test_empty_state(self):
        """No configured variants and no tokonomics returns None."""
        pool = MockPool(manifest=MockManifest())
        agent = MockAgent(model_name="gpt-4o", pool=pool, toko_models=[])
        router = ProviderRouter(MockManifest())  # type: ignore[arg-type]

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is None

    async def test_error_handling(self):
        """get_available_models() raising returns None gracefully."""
        pool = MockPool(manifest=MockManifest())
        agent = MockAgent(model_name="gpt-4o", pool=pool)
        agent.get_available_models = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        router = ProviderRouter(MockManifest())  # type: ignore[arg-type]

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is None

    async def test_current_model_in_configured(self, manifest_with_variants):
        """Current model is set correctly when in configured list."""
        pool = MockPool(manifest=manifest_with_variants)
        # model_name should be the resolved identifier to match model_id
        agent = MockAgent(model_name="anthropic:claude-sonnet-4-5", pool=pool)
        router = ProviderRouter(manifest_with_variants)  # type: ignore[arg-type]

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is not None
        assert state.current_model_id == "smart"

    async def test_current_model_not_in_list(self, manifest_with_variants):
        """Current model is inserted when not in configured variants."""
        pool = MockPool(manifest=manifest_with_variants)
        agent = MockAgent(model_name="unknown-model", pool=pool)
        router = ProviderRouter(manifest_with_variants)  # type: ignore[arg-type]

        state = await build_model_state_for_acp(agent, router)  # type: ignore[arg-type]

        assert state is not None
        # Current model should be inserted at the front for IDE visibility
        assert state.current_model_id == "unknown-model"
        assert state.available_models[0].model_id == "unknown-model"
        # Configured variants should still be present (resolved identifiers may
        # differ from raw identifiers depending on environment)
        model_ids = {m.model_id for m in state.available_models}
        assert "unknown-model" in model_ids
        # At least the 2 configured variants plus the unknown model
        assert len(model_ids) >= 3
