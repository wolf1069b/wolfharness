"""Shared fixtures for viking capability tests.

The canned ``AsyncMock`` client is the single source of truth for the
fake OpenViking SDK surface. It used to be duplicated verbatim in
``test_viking.py`` and ``test_viking_integration.py`` — every SDK change
required two edits and they drifted. New SDK methods / response shapes
must be added here once.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from wolfharness.capabilities.viking import VikingCapability


def build_mock_client() -> AsyncMock:
    """Create a fully populated mock AsyncHTTPClient for viking tests.

    All SDK methods return canned success values; individual tests
    override the method they exercise (``return_value=...`` or
    ``side_effect=...``) to shape behavior.
    """
    client = AsyncMock()
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.search = AsyncMock(return_value={"results": []})
    client.find = AsyncMock(return_value={"results": []})
    client.grep = AsyncMock(return_value={"matches": []})
    client.glob = AsyncMock(return_value={"matches": []})
    client.ls = AsyncMock(return_value=[])
    client.read = AsyncMock(return_value="file content")
    client.abstract = AsyncMock(return_value="abstract summary")
    client.overview = AsyncMock(return_value="overview content")
    client.write = AsyncMock(return_value={"status": "ok"})
    client.mkdir = AsyncMock(return_value=None)
    client.rm = AsyncMock(return_value=None)
    client.link = AsyncMock(return_value=None)
    client.set_tags = AsyncMock(return_value={"status": "ok"})
    client.add_resource = AsyncMock(return_value={"status": "ok"})
    client.create_session = AsyncMock(return_value={"session_id": "test-session"})
    client.add_message = AsyncMock(return_value={"status": "ok"})
    client.commit_session = AsyncMock(return_value={"status": "ok"})
    client.get_session_context = AsyncMock(return_value={})
    client._request = AsyncMock(return_value={})
    return client


@pytest.fixture
def mock_client() -> AsyncMock:
    """Provide a fresh canned mock AsyncHTTPClient per test."""
    return build_mock_client()


@pytest.fixture
def viking_cap(mock_client: AsyncMock) -> VikingCapability:
    """Create a VikingCapability with a mock client pre-injected.

    Enables link and memory features so all tools are available for testing.
    """
    cap = VikingCapability(mode="all", enable_link=True, enable_memory=True, enable_forget=True)
    cap._client = mock_client
    return cap
