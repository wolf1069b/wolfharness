"""Test SSE anti-buffering headers on event endpoints.

Verifies that /global/event and /event responses include headers
required to prevent reverse proxies (nginx, Cloudflare) from
buffering SSE events, which causes delayed delivery and TUI black screens.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from wolfharness_server.opencode_server.routes.global_routes import get_events, get_global_events


pytestmark = pytest.mark.integration


EXPECTED_HEADERS = {
    "cache-control": "no-cache",
    "x-accel-buffering": "no",
    "x-content-type-options": "nosniff",
}


@pytest.mark.anyio
async def test_global_event_anti_buffering_headers() -> None:
    """/global/event response must include anti-buffering headers."""
    state = Mock()
    response = await get_global_events(state)
    for header_name, expected_value in EXPECTED_HEADERS.items():
        actual = response.headers.get(header_name)
        assert actual == expected_value, (
            f"/global/event: expected header {header_name!r}={expected_value!r}, got {actual!r}"
        )


@pytest.mark.anyio
async def test_event_anti_buffering_headers() -> None:
    """/event response must include anti-buffering headers."""
    state = Mock()
    response = await get_events(state)
    for header_name, expected_value in EXPECTED_HEADERS.items():
        actual = response.headers.get(header_name)
        assert actual == expected_value, (
            f"/event: expected header {header_name!r}={expected_value!r}, got {actual!r}"
        )
