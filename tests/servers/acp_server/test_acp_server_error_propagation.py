"""Unit tests for ACPServer startup error propagation.

Regression test for silent serve-acp startup failures: ``ACPServer._start_async``
used to swallow all exceptions from ``serve()`` (e.g. ``OSError: Address already
in use``), so ``asyncio.run`` returned normally and the process exited 0 with no
terminal output. Startup errors must now propagate so the CLI can report them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness_server.acp_server import ACPServer


@pytest.mark.unit
async def test_start_async_propagates_serve_oserror() -> None:
    """A serve() failure (e.g. port bind error) must propagate out of _start_async."""
    from wolfharness_server.acp_server import server as server_module

    pool = MagicMock()
    server = ACPServer(pool=pool, transport="stdio")

    dummy_agent = MagicMock()
    dummy_agent.name = "test_agent"

    # The `finally: await viking_archive.close()` block always runs, so the
    # patched archive must expose an async close().
    viking_archive = AsyncMock()

    with (
        patch.object(server, "_resolve_default_agent", new=AsyncMock(return_value=dummy_agent)),
        patch.object(server_module, "ACPSessionManager", return_value=MagicMock()),
        patch.object(
            server_module.ACPVikingEventArchive, "from_config", return_value=viking_archive
        ),
        patch.object(server_module, "serve", side_effect=OSError(48, "Address already in use")),
        pytest.raises(OSError, match="Address already in use"),
    ):
        await server._start_async()
