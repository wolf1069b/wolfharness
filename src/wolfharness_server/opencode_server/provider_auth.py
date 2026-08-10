"""Provider authentication service.

Composable auth backend system matching the opencode plugin auth pattern.
Each provider registers a backend that handles its specific auth flow
(OAuth PKCE, device code, API key, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from wolfharness.auth.anthropic_auth import (
    OAUTH_MANUAL_REDIRECT_URI,
    AnthropicOAuthToken,
    AnthropicTokenStore,
    build_authorization_url,
    exchange_code_for_token,
    generate_pkce,
)
from wolfharness_server.opencode_server.models.agent import (
    ProviderAuthAuthorization,
    ProviderAuthMethod,
)


if TYPE_CHECKING:
    from wolfharness_server.opencode_server.models.agent import AuthInfo


class ProviderAuthBackend(ABC):
    """Protocol for a provider-specific auth backend."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        ...

    @abstractmethod
    def methods(self) -> list[ProviderAuthMethod]:
        """Return available auth methods for this provider."""
        ...

    @abstractmethod
    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        """Start an authorization flow.

        Args:
            method: Index into the methods list.

        Returns:
            Authorization info with URL and instructions.
        """
        ...

    @abstractmethod
    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        """Handle the auth callback / code exchange.

        Returns:
            True if auth succeeded.

        Raises:
            ValueError: If required parameters are missing or exchange fails.
        """
        ...

    async def set_credentials(self, info: AuthInfo) -> bool:
        """Store credentials for this provider.

        Default implementation is a no-op. Override for providers that
        support direct credential setting (e.g. API key or token import).
        """
        return False

    async def remove_credentials(self) -> bool:
        """Remove stored credentials for this provider.

        Default implementation is a no-op.
        """
        return False


class AnthropicAuthBackend(ProviderAuthBackend):
    """Anthropic OAuth (PKCE) auth backend."""

    def __init__(self) -> None:
        self._pending_verifiers: dict[str, str] = {}

    @property
    def provider_id(self) -> str:
        return "anthropic"

    def methods(self) -> list[ProviderAuthMethod]:
        return [ProviderAuthMethod(type="oauth", label="Connect Claude Max/Pro")]

    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        verifier, challenge = generate_pkce()
        auth_url = build_authorization_url(
            verifier, challenge, redirect_uri=OAUTH_MANUAL_REDIRECT_URI
        )
        self._pending_verifiers[verifier] = verifier
        return ProviderAuthAuthorization(
            url=auth_url,
            instructions="Sign in with your Anthropic account and copy the authorization code",
            method="code",
        )

    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        if not code or not verifier:
            raise ValueError("Missing code or verifier for Anthropic OAuth")
        token = exchange_code_for_token(
            code, state=verifier, verifier=verifier, redirect_uri=OAUTH_MANUAL_REDIRECT_URI
        )
        store = AnthropicTokenStore()
        store.save(token)
        self._pending_verifiers.pop(verifier, None)
        return True

    async def set_credentials(self, info: AuthInfo) -> bool:
        if not info.token:
            return False
        store = AnthropicTokenStore()
        token = AnthropicOAuthToken(
            access_token=info.token,
            refresh_token=info.refresh or "",
            expires_at=info.expires or 0,
        )
        store.save(token)
        return True

    async def remove_credentials(self) -> bool:
        store = AnthropicTokenStore()
        store.clear()
        return True


COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_HEADERS = {
    "accept": "application/json",
    "editor-version": "Neovim/0.6.1",
    "editor-plugin-version": "copilot.vim/1.16.0",
    "content-type": "application/json",
    "user-agent": "GithubCopilot/1.155.0",
}


class CopilotAuthBackend(ProviderAuthBackend):
    """GitHub Copilot device-code auth backend."""

    def __init__(self) -> None:
        self._pending_device_codes: dict[str, str] = {}

    @property
    def provider_id(self) -> str:
        return "copilot"

    def methods(self) -> list[ProviderAuthMethod]:
        return [ProviderAuthMethod(type="oauth", label="Connect GitHub Copilot")]

    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://github.com/login/device/code",
                headers=COPILOT_HEADERS,
                json={"client_id": COPILOT_CLIENT_ID, "scope": "read:user"},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data["verification_uri"]
        self._pending_device_codes[device_code] = device_code

        return ProviderAuthAuthorization(
            url=verification_uri,
            instructions=f"Enter code: {user_code}",
            method="auto",
        )

    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        if not device_code:
            msg = "Missing device_code for Copilot OAuth"
            raise ValueError(msg)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://github.com/login/oauth/access_token",
                headers=COPILOT_HEADERS,
                json={
                    "client_id": COPILOT_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            data: dict[str, Any] = resp.json()

        if "error" in data:
            detail = data.get("error_description", data["error"])
            raise ValueError(detail)

        if data.get("access_token"):
            self._pending_device_codes.pop(device_code, None)
            # TODO: Store copilot token via Auth.set equivalent
            return True

        raise ValueError("No token received")


@dataclass
class ProviderAuthService:
    """Registry of provider auth backends.

    Mirrors opencode's ProviderAuth namespace — routes call service methods
    instead of containing provider-specific logic.
    """

    _backends: dict[str, ProviderAuthBackend] = field(default_factory=dict)

    def register(self, backend: ProviderAuthBackend) -> None:
        """Register an auth backend."""
        self._backends[backend.provider_id] = backend

    def get_backend(self, provider_id: str) -> ProviderAuthBackend:
        """Get backend by provider ID.

        Raises:
            KeyError: If provider_id is not registered.
        """
        try:
            return self._backends[provider_id]
        except KeyError:
            msg = f"Unknown provider: {provider_id}"
            raise KeyError(msg) from None

    def methods(self) -> dict[str, list[ProviderAuthMethod]]:
        """Return auth methods for all registered providers."""
        return {pid: backend.methods() for pid, backend in self._backends.items()}

    async def authorize(self, provider_id: str, method: int = 0) -> ProviderAuthAuthorization:
        """Start auth flow for a provider."""
        return await self.get_backend(provider_id).authorize(method)

    async def callback(
        self,
        provider_id: str,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        """Handle auth callback for a provider."""
        return await self.get_backend(provider_id).callback(
            code=code, device_code=device_code, verifier=verifier
        )

    async def set_credentials(self, provider_id: str, info: AuthInfo) -> bool:
        """Set credentials for a provider."""
        return await self.get_backend(provider_id).set_credentials(info)

    async def remove_credentials(self, provider_id: str) -> bool:
        """Remove credentials for a provider."""
        return await self.get_backend(provider_id).remove_credentials()


def create_default_auth_service() -> ProviderAuthService:
    """Create auth service with built-in providers."""
    service = ProviderAuthService()
    service.register(AnthropicAuthBackend())
    service.register(CopilotAuthBackend())
    return service
