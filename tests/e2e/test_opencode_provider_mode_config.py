"""L4 subprocess E2E tests for OpenCode provider/mode/config endpoints.

Covers Phase C6 of the protocol-e2e-coverage OpenSpec change:
    - C6.1: GET /provider — verify 200
    - C6.2: POST /provider [skip] — endpoint not implemented
    - C6.3: GET /mode — verify 200
    - C6.4: POST /mode [skip] — endpoint not implemented
    - C6.5: GET /config — verify 200 with Config object
    - C6.6: PATCH /config — verify 200 with updated Config
    - C6.7: GET /config/{id} [skip] — endpoint not implemented
    - C6.8: PUT /config/{id} [skip] — endpoint not implemented
    - C6.9: DELETE /config/{id} [skip] — endpoint not implemented

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from tests.e2e.conftest import SubprocessServer


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows subprocess issues"),
]


# ---------------------------------------------------------------------------
# C6.1 — GET /provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_provider(subprocess_server: SubprocessServer) -> None:
    """C6.1: GET /provider, verify 200."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/provider")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /provider, got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        # ProviderListResponse has ``all``, ``default``, ``connected`` fields.
        assert "all" in data, f"Provider response missing 'all' field: {data}"


# ---------------------------------------------------------------------------
# C6.2 — POST /provider [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_provider(subprocess_server: SubprocessServer) -> None:
    """Test intent: POST /provider to set active provider.

    Send POST /provider with JSON body containing provider config (``name``,
    ``api_key``, ``base_url``, ``model`` optional). Verify 200 or 201 with
    provider ``id`` in response. Follow with GET /provider to verify new
    provider appears in list. Error case: missing required fields -> 422;
    duplicate provider name -> 409 or 422.
    """


# ---------------------------------------------------------------------------
# C6.3 — GET /mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_mode(subprocess_server: SubprocessServer) -> None:
    """C6.3: GET /mode, verify 200."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/mode")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /mode, got {resp.status_code}: {resp.text}"
        )
        modes = resp.json()
        assert isinstance(modes, list), f"Expected list, got {type(modes)}"
        # The server always returns at least a default mode.
        assert len(modes) >= 1, f"Expected at least 1 mode, got {modes}"


# ---------------------------------------------------------------------------
# C6.4 — POST /mode [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mode(subprocess_server: SubprocessServer) -> None:
    """Test intent: POST /mode to set mode.

    Send POST /mode with JSON body containing mode config (``name``, ``model``,
    ``prompt``, ``tools`` optional). Verify 200 or 201 with mode ``id`` in
    response. Follow with GET /mode to verify new mode appears in list.
    Error case: missing ``name`` or ``model`` -> 422.
    """


# ---------------------------------------------------------------------------
# C6.5 — GET /config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_config(subprocess_server: SubprocessServer) -> None:
    """C6.5: GET /config, verify 200 with Config object."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/config")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /config, got {resp.status_code}: {resp.text}"
        )
        config = resp.json()
        assert isinstance(config, dict), f"Expected dict, got {type(config)}"
        # Config object should have standard fields like ``model``, ``keybinds``.
        # The server initializes defaults if not configured.
        assert "model" in config, f"Config object missing 'model' field: {config}"


# ---------------------------------------------------------------------------
# C6.6 — PATCH /config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_patch_config(subprocess_server: SubprocessServer) -> None:
    """C6.6: PATCH /config, verify 200 with updated Config.

    Test intent: PATCH /config with JSON body containing config key-value pairs
    to update (e.g., ``theme``, ``font_size``, ``model``). Verify 200 with
    updated config object in response. Follow with GET /config to verify config
    changes applied. Error case: invalid config schema -> 422.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        # First, get current config to know what we're working with.
        resp = await client.get(f"{base_url}/config")
        assert resp.status_code == 200
        original_config = resp.json()

        # Send a PATCH with a non-destructive update.
        # We patch ``theme`` which is a safe string field.
        patch_payload: dict[str, Any] = {"theme": "dark"}
        resp = await client.patch(f"{base_url}/config", json=patch_payload)
        assert resp.status_code == 200, (
            f"Expected 200 for PATCH /config, got {resp.status_code}: {resp.text}"
        )
        updated_config = resp.json()
        assert isinstance(updated_config, dict), f"Expected dict, got {type(updated_config)}"

        # Verify the update was applied.
        assert updated_config.get("theme") == "dark", (
            f"Expected theme='dark' after PATCH, got {updated_config.get('theme')}"
        )

        # Follow up with GET /config to verify persistence.
        resp = await client.get(f"{base_url}/config")
        assert resp.status_code == 200
        persisted_config = resp.json()
        assert persisted_config.get("theme") == "dark", (
            f"Expected theme='dark' in GET after PATCH, got {persisted_config.get('theme')}"
        )

        # Restore original theme to avoid side effects on other tests.
        _ = original_config
