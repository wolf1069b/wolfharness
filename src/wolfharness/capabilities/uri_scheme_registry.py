"""UriSchemeRegistry — central authority for URI scheme ownership.

Maps each URI scheme to exactly one resource provider, enabling
deterministic scheme-based routing and conflict detection at
registration time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

from wolfharness.capabilities.resource_protocols import (
    UriSchemeConflictError,
)


if TYPE_CHECKING:
    from wolfharness.capabilities.resource_protocols import ResourceAccess


class UriSchemeRegistry:
    """Maps URI schemes to authorized resource providers.

    Every scheme is owned by at most one provider. Registration
    detects conflicting claims and raises ``UriSchemeConflictError``.
    Lookup returns the registered provider for a scheme, or ``None``
    for unregistered schemes.

    Providers with no owned schemes (opaque passthrough) are not
    registered here — they are collected separately and consulted
    when no scheme owner is found.
    """

    def __init__(self) -> None:
        self._scheme_to_provider: dict[str, ResourceAccess] = {}
        self._scheme_to_name: dict[str, str] = {}

    def register(
        self,
        provider_name: str,
        schemes: frozenset[str],
        provider: ResourceAccess,
    ) -> None:
        """Register a provider as the authoritative owner of URI schemes.

        Args:
            provider_name: Human-readable name of the provider.
            schemes: Set of URI scheme strings this provider owns.
            provider: The ``ResourceAccess`` instance to route to.

        Raises:
            UriSchemeConflictError: If any scheme is already claimed.
        """
        for scheme in schemes:
            existing = self._scheme_to_name.get(scheme)
            if existing is not None:
                raise UriSchemeConflictError(
                    scheme=scheme,
                    existing_provider=existing,
                    conflicting_provider=provider_name,
                )
        for scheme in schemes:
            self._scheme_to_provider[scheme] = provider
            self._scheme_to_name[scheme] = provider_name

    def lookup(self, scheme: str) -> ResourceAccess | None:
        """Return the provider authorized for a URI scheme.

        Args:
            scheme: The URI scheme to look up (e.g. ``"viking"``).

        Returns:
            The ``ResourceAccess`` provider for that scheme, or
            ``None`` if no provider is registered for the scheme.
        """
        return self._scheme_to_provider.get(scheme)

    def registered_schemes(self) -> frozenset[str]:
        """Return all registered URI schemes.

        Returns:
            ``frozenset`` of scheme strings.
        """
        return frozenset(self._scheme_to_provider)

    def owner_of(self, scheme: str) -> str | None:
        """Return the provider name registered for a URI scheme.

        Args:
            scheme: The URI scheme to query.

        Returns:
            The provider name, or ``None`` if unregistered.
        """
        return self._scheme_to_name.get(scheme)

    def unregister(self, provider: ResourceAccess) -> None:
        """Remove all schemes owned by a provider.

        Args:
            provider: The provider to unregister.
        """
        schemes_to_remove = [
            scheme for scheme, p in self._scheme_to_provider.items() if p is provider
        ]
        for scheme in schemes_to_remove:
            del self._scheme_to_provider[scheme]
            del self._scheme_to_name[scheme]

    def registered_providers(self) -> Sequence[ResourceAccess]:
        """Return all registered providers (deduplicated).

        Returns:
            Sequence of unique ``ResourceAccess`` instances.
        """
        return list(dict.fromkeys(self._scheme_to_provider.values()))
