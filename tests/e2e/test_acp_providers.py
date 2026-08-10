"""L4 subprocess E2E tests for ACP providers methods.

Covers 3 provider methods: providers/list, providers/set, providers/disable.

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from acp import (
    ClientSideConnection,
    DefaultACPClient,
    DisableProvidersRequest,
    InitializeRequest,
    ListProvidersRequest,
    SetProvidersRequest,
)
from acp.exceptions import RequestError
from acp.stdio import spawn_agent_process
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from acp.schema import InitializeResponse


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper fixture
# ---------------------------------------------------------------------------


class ACPProvidersHandle:
    """Handle to a spawned ACP server with a client connection for provider tests."""

    def __init__(self, conn: ClientSideConnection, process: Any, client: DefaultACPClient) -> None:
        self.conn = conn
        self.process = process
        self.client = client

    async def initialize(self) -> InitializeResponse:
        return await self.conn.initialize(InitializeRequest(protocol_version=1))


@pytest.fixture
async def acp_server(e2e_config: Path) -> AsyncIterator[ACPProvidersHandle]:
    """Spawn an ACP stdio server and initialize it.

    Suppresses ``RequestError`` during teardown to avoid spurious test errors
    from background task cleanup.
    """
    import os

    env = os.environ.copy()
    env["OBSERVABILITY_ENABLED"] = "false"
    env["LOGFIRE_DISABLE"] = "true"

    client = DefaultACPClient(allow_file_operations=False)
    cm = spawn_agent_process(
        lambda _conn: client,
        "wolfharness",
        "serve-acp",
        str(e2e_config),
        "--agent",
        "test_agent",
        env=env,
        log_stderr=False,
    )
    try:
        conn, process = await cm.__aenter__()
        handle = ACPProvidersHandle(conn=conn, process=process, client=client)
        await handle.initialize()
        yield handle
    finally:
        with contextlib.suppress(RequestError, Exception):
            await cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# B4.1 — providers/list
# ---------------------------------------------------------------------------


async def test_providers_list(acp_server: ACPProvidersHandle) -> None:
    """B4.1: Call providers/list, verify a list is returned."""
    response = await acp_server.conn.list_providers(ListProvidersRequest())
    assert response is not None
    assert hasattr(response, "providers")
    # The default e2e_config (model: test) has no model_variants, so the
    # provider list may be empty. We verify the call succeeds and returns
    # a list (possibly empty).
    _ = list(response.providers)


# ---------------------------------------------------------------------------
# B4.2 — providers/set
# ---------------------------------------------------------------------------


async def test_providers_set(acp_server: ACPProvidersHandle) -> None:
    """B4.2: Call providers/set with a valid provider, verify success."""
    response = await acp_server.conn.set_provider(
        SetProvidersRequest(
            id="openai",
            api_type="openai",
            base_url="https://api.openai.com/v1",
        )
    )
    assert response is not None


# ---------------------------------------------------------------------------
# B4.3 — providers/disable
# ---------------------------------------------------------------------------


async def test_providers_disable(acp_server: ACPProvidersHandle) -> None:
    """B4.3: Call providers/disable, verify success.

    The ``disable_provider`` method silently ignores unknown provider IDs
    (per provider_router.py: "Unknown providers are silently ignored").
    """
    response = await acp_server.conn.disable_provider(
        DisableProvidersRequest(id="nonexistent-provider")
    )
    assert response is not None

    # Verify the provider list is still returned correctly after disable.
    listed = await acp_server.conn.list_providers(ListProvidersRequest())
    assert listed is not None
