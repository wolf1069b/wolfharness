"""Tests for the OpenCode Web UI proxy path blocking.

Regression tests for the crash where the OpenCode TUI did ``@``-mention
autocomplete, called ``GET /api/fs/find``, and wolfharness forwarded the
request to the hosted Web UI instead of returning 404. The Web UI responded
with ``index.html`` (``200``, ``Content-Type: text/html;charset=UTF-8``), which
the SDK parsed as ``text``, so the TUI received ``result.data`` as a string and
crashed on ``result.data.data.map(...)`` with

    TypeError: undefined is not an object (evaluating 'e.data.data.map')

Root cause: the Web UI proxy's API-prefix block list did not include ``api/``,
so unmatched ``/api/*`` routes fell through to the cloud proxy.
"""

from __future__ import annotations

import pytest

from wolfharness_server.opencode_server.server import (
    PROXY_BLOCKED_PREFIXES,
    is_proxy_path_blocked,
)


pytestmark = pytest.mark.unit


class TestIsProxyPathBlocked:
    """Tests for :func:`is_proxy_path_blocked`."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            # New: all unmatched /api/* routes must 404, never forward
            ("api/fs/find", True),
            ("api/fs/list", True),
            ("api/session", True),
            ("api/session/abc123/message", True),
            # Existing API prefixes (real API paths always carry the trailing
            # slash, e.g. "session/abc123" rather than "session")
            ("session/abc", True),
            ("config", False),
            ("config/path/to/file", True),
            ("agent/foo", True),
            ("model/foo", True),
            ("provider/foo", True),
            ("command/foo", True),
            ("skill/foo", True),
            ("location/foo", True),
            ("integration/foo", True),
            ("file/content", True),
            ("todo/foo", True),
            ("diff/foo", True),
            ("snapshot/foo", True),
            ("v1/metrics", True),
            ("experimental/workspace", True),
            # Non-API paths must still proxy to the Web UI
            ("", False),
            ("favicon.ico", False),
            ("manifest.json", False),
            ("static/app.js", False),
            ("some/deep/route", False),
            # A bare prefix with no trailing slash does not match (pre-existing
            # behavior, kept to avoid changing proxy semantics for odd paths)
            ("session", False),
            ("file", False),
        ],
    )
    def test_paths(self, path: str, expected: bool) -> None:
        """Assert the blocking decision for each path."""
        assert is_proxy_path_blocked(path) is expected

    def test_api_prefixed_but_not_exact(self) -> None:
        """``api`` followed by a non-slash is not an API route."""
        assert is_proxy_path_blocked("api-docs") is False

    def test_prefix_set_contains_api(self) -> None:
        """The ``api/`` prefix must be present so unmatched API routes 404."""
        assert "api/" in PROXY_BLOCKED_PREFIXES
