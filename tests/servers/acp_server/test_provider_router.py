"""Tests for ProviderRouter."""

from __future__ import annotations

import asyncio

import pytest

from wolfharness.models.model_configs import StringModelConfig
from wolfharness_server.acp_server.provider_router import ProviderRouter


pytestmark = pytest.mark.integration


class MockManifest:
    """Mock manifest for testing."""

    def __init__(self, model_variants=None):
        self.model_variants = model_variants or {}
        self.acp = None


@pytest.fixture
def manifest_with_variants():
    """Create a mock manifest with model variants using real config types."""
    variants = {
        "fast_gpt": StringModelConfig(identifier="openai:gpt-4o-mini"),
        "thinking": StringModelConfig(identifier="anthropic:claude-sonnet-4-5"),
    }
    return MockManifest(model_variants=variants)


@pytest.fixture
def empty_manifest():
    """Create an empty mock manifest."""
    return MockManifest()


class TestProviderRouter:
    """Test ProviderRouter functionality."""

    def test_derive_providers_from_manifest(self, manifest_with_variants):
        """ProviderRouter should derive providers from manifest variants."""
        router = ProviderRouter(manifest_with_variants)
        providers = router.get_providers()

        assert len(providers) >= 1
        provider_ids = {p.id for p in providers}
        assert "openai" in provider_ids
        assert "anthropic" in provider_ids

        for p in providers:
            assert p.id
            assert p.current is not None

    def test_empty_manifest_no_providers(self, empty_manifest):
        """ProviderRouter with empty manifest should return no providers."""
        router = ProviderRouter(empty_manifest)
        providers = router.get_providers()
        assert providers == []

    def test_none_manifest_no_providers(self):
        """ProviderRouter with None manifest should return no providers."""
        router = ProviderRouter(None)
        providers = router.get_providers()
        assert providers == []

    def test_set_provider_override(self, manifest_with_variants):
        """ProviderRouter should support overriding provider base_url."""
        router = ProviderRouter(manifest_with_variants)

        asyncio.run(router.set_provider_override("openai", base_url="https://custom.openai.com"))

        provider = router.get_provider("openai")
        assert provider is not None
        assert provider.current is not None
        assert provider.current.base_url == "https://custom.openai.com"

    def test_set_provider_override_api_key_id(self, manifest_with_variants):
        """ProviderRouter should support overriding provider api_key_id."""
        router = ProviderRouter(manifest_with_variants)

        asyncio.run(router.set_provider_override("openai", api_key_id="my_key"))

        provider = router.get_provider("openai")
        assert provider is not None
        # api_key_id override is stored internally
        assert router._overrides.get("openai", {}).get("api_key_id") == "my_key"

    def test_set_provider_override_unknown_raises(self, manifest_with_variants):
        """Setting override for unknown provider should raise ValueError."""
        router = ProviderRouter(manifest_with_variants)

        with pytest.raises(ValueError, match="Unknown provider"):
            asyncio.run(router.set_provider_override("nonexistent", base_url="http://x"))

    def test_disable_provider(self, manifest_with_variants):
        """ProviderRouter should support disabling providers."""
        router = ProviderRouter(manifest_with_variants)

        asyncio.run(router.disable_provider("openai"))

        assert router.is_provider_disabled("openai") is True
        provider = router.get_provider("openai")
        assert provider is not None
        assert provider.current is None  # disabled provider has no current config

    def test_disable_provider_unknown_silently(self, manifest_with_variants):
        """Disabling unknown provider should silently succeed (for tokonomics providers)."""
        router = ProviderRouter(manifest_with_variants)

        # Should not raise for unknown providers (e.g., tokonomics-discovered)
        asyncio.run(router.disable_provider("nonexistent"))
        assert router.is_provider_disabled("nonexistent") is True

    def test_enable_provider(self, manifest_with_variants):
        """ProviderRouter should support re-enabling disabled providers."""
        router = ProviderRouter(manifest_with_variants)

        asyncio.run(router.disable_provider("openai"))
        assert router.is_provider_disabled("openai") is True

        asyncio.run(router.enable_provider("openai"))
        assert router.is_provider_disabled("openai") is False

    def test_enable_provider_unknown_raises(self, manifest_with_variants):
        """Enabling unknown provider should raise ValueError."""
        router = ProviderRouter(manifest_with_variants)

        with pytest.raises(ValueError, match="Unknown provider"):
            asyncio.run(router.enable_provider("nonexistent"))

    def test_get_provider_returns_none_for_unknown(self, manifest_with_variants):
        """get_provider should return None for unknown provider."""
        router = ProviderRouter(manifest_with_variants)
        assert router.get_provider("nonexistent") is None

    def test_disabled_providers_filtered_from_list(self, manifest_with_variants):
        """Disabled providers should have disabled status in get_providers."""
        router = ProviderRouter(manifest_with_variants)

        asyncio.run(router.disable_provider("openai"))
        providers = router.get_providers()

        openai_provider = next((p for p in providers if p.id == "openai"), None)
        assert openai_provider is not None
        assert openai_provider.current is None  # disabled provider has no current config

    async def test_concurrent_override(self, manifest_with_variants):
        """Concurrent override calls should not corrupt state."""
        router = ProviderRouter(manifest_with_variants)

        async def override_task(i: int):
            await router.set_provider_override("openai", base_url=f"https://url{i}.com")

        await asyncio.gather(*[override_task(i) for i in range(10)])

        provider = router.get_provider("openai")
        assert provider is not None
        assert provider.current is not None
        # One of the overrides should have won
        assert provider.current.base_url is not None
        assert provider.current.base_url.startswith("https://url")
