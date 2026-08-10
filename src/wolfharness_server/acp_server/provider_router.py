"""Provider router for ACP protocol.

Manages LLM provider metadata, override tracking, and capability state
for ACP `providers/*` protocol methods.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from acp.schema.providers import ProviderCurrentConfig, ProviderInfo
from wolfharness.log import get_logger
from wolfharness_server.shared.model_utils import _extract_provider


if TYPE_CHECKING:
    from wolfharness.models.manifest import AgentsManifest

logger = get_logger(__name__)


class ProviderRouter:
    """Routes and manages LLM provider metadata for ACP protocol.

    Derives available providers from manifest model_variants, tracks
    override state (base_url, api_key), and handles provider disable/enable.
    """

    def __init__(self, manifest: AgentsManifest | None) -> None:
        """Initialize ProviderRouter with manifest-derived providers.

        Args:
            manifest: Agent manifest containing model_variants configuration.
        """
        self._manifest = manifest
        self._providers: dict[str, ProviderInfo] = {}
        self._overrides: dict[str, dict[str, str | None]] = {}
        self._disabled: set[str] = set()
        self._lock = asyncio.Lock()
        self._derive_providers_from_manifest()

    def _derive_providers_from_manifest(self) -> None:
        """Extract ProviderInfo list from manifest model_variants."""
        if not self._manifest or not self._manifest.model_variants:
            return

        for config in self._manifest.model_variants.values():
            provider_name = _extract_provider(config)
            base_url = self._get_default_base_url(provider_name)

            # Use provider_name as the provider id, grouping variants by provider
            if provider_name not in self._providers:
                self._providers[provider_name] = ProviderInfo(
                    id=provider_name,
                    supported=[provider_name],
                    required=False,
                    current=ProviderCurrentConfig(
                        api_type=provider_name,
                        base_url=base_url or "",
                    )
                    if base_url
                    else None,
                )

    def _get_default_base_url(self, provider: str) -> str:
        """Return best-effort default base URL for known providers.

        Args:
            provider: Provider name.

        Returns:
            Default base URL or empty string if unknown.
        """
        urls: dict[str, str] = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "google": "https://generativelanguage.googleapis.com/v1beta",
            "mistral": "https://api.mistral.ai/v1",
            "cohere": "https://api.cohere.ai/v1",
            "azure_openai": "",
            "bedrock": "",
        }
        return urls.get(provider, "")

    def get_providers(self) -> list[ProviderInfo]:
        """Return all providers with current override/disable state applied.

        Returns:
            List of ProviderInfo with current state.
        """
        result: list[ProviderInfo] = []
        for pid, info in self._providers.items():
            is_disabled = pid in self._disabled
            override = self._overrides.get(pid, {})

            # Build current config
            current = info.current
            if current is not None:
                base_url = override.get("base_url", current.base_url)
                if isinstance(base_url, str):
                    current = ProviderCurrentConfig(
                        api_type=current.api_type,
                        base_url=base_url,
                        headers=current.headers,
                    )

            result.append(
                ProviderInfo(
                    id=info.id,
                    supported=info.supported,
                    required=info.required,
                    current=None if is_disabled else current,
                )
            )
        return result

    def get_provider(self, provider_id: str) -> ProviderInfo | None:
        """Get a specific provider by ID.

        Args:
            provider_id: Provider identifier.

        Returns:
            ProviderInfo if found, None otherwise.
        """
        if provider_id not in self._providers:
            return None
        providers = self.get_providers()
        for p in providers:
            if p.id == provider_id:
                return p
        return None

    async def set_provider_override(
        self,
        provider_id: str,
        base_url: str | None = None,
        api_key_id: str | None = None,
    ) -> None:
        """Set override values for a provider.

        If the provider is not already registered but is a well-known
        provider (present in the default base URL table), it is registered
        on-the-fly. This allows ``providers/set`` to accept well-known
        provider IDs (e.g. ``"openai"``, ``"anthropic"``) even when no
        ``model_variants`` are configured in the manifest.

        Args:
            provider_id: Provider to override.
            base_url: Optional custom base URL.
            api_key_id: Optional API key identifier.

        Raises:
            ValueError: If provider_id is unknown (not registered and not
                a well-known provider).
        """
        async with self._lock:
            if provider_id not in self._providers:
                # Register well-known providers on-the-fly.
                default_base_url = self._get_default_base_url(provider_id)
                if not default_base_url:
                    msg = f"Unknown provider: {provider_id}"
                    raise ValueError(msg)
                self._providers[provider_id] = ProviderInfo(
                    id=provider_id,
                    supported=[provider_id],
                    required=False,
                    current=ProviderCurrentConfig(
                        api_type=provider_id,
                        base_url=default_base_url,
                    ),
                )
                logger.info(
                    "Registered provider on-the-fly",
                    provider_id=provider_id,
                    base_url=default_base_url,
                )
            if provider_id not in self._overrides:
                self._overrides[provider_id] = {}
            if base_url is not None:
                self._overrides[provider_id]["base_url"] = base_url
            if api_key_id is not None:
                self._overrides[provider_id]["api_key_id"] = api_key_id
            logger.info("Provider override set", provider_id=provider_id)

    async def disable_provider(self, provider_id: str) -> None:
        """Disable a provider.

        Args:
            provider_id: Provider to disable. Unknown providers are silently
                added to the disabled set (they won't appear in listings).
        """
        async with self._lock:
            self._disabled.add(provider_id)
            logger.info("Provider disabled", provider_id=provider_id)

    async def enable_provider(self, provider_id: str) -> None:
        """Enable a previously disabled provider.

        Args:
            provider_id: Provider to enable.

        Raises:
            ValueError: If provider_id is unknown.
        """
        async with self._lock:
            if provider_id not in self._providers:
                msg = f"Unknown provider: {provider_id}"
                raise ValueError(msg)
            self._disabled.discard(provider_id)
            logger.info("Provider enabled", provider_id=provider_id)

    def is_provider_disabled(self, provider_id: str) -> bool:
        """Check if a provider is disabled.

        Args:
            provider_id: Provider identifier.

        Returns:
            True if provider is disabled, False otherwise.
        """
        return provider_id in self._disabled
