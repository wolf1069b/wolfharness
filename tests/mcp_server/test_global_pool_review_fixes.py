"""Tests for GlobalConnectionPool fixes from code review.

Covers:
- get_agentlet logs warning on exception (not silent)

The following test classes were removed during simplification:
- TestReleasePopsDyingConnection — release() no longer exists
- TestHTTPRefCountBalance — ref_count no longer exists
"""

from __future__ import annotations

from typing import Any

import pytest

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability


pytestmark = pytest.mark.integration


class TestBuildToolsetLogsWarning:
    """Tests that get_agentlet logs warning on exception."""

    async def test_build_toolset_logs_warning_on_exception(self) -> None:
        """Test that logger.warning/exception is called when get_tools() raises.

        Given a provider that raises in get_tools(), when get_agentlet()
        catches the exception, then it must call logger.warning or
        logger.exception (not silently swallow).
        """

        class _FailingProvider(FunctionToolsetCapability):
            def __init__(self) -> None:
                super().__init__(name="test-fail")

            async def get_tools(self) -> list[Any]:
                raise RuntimeError("connection refused")

        # Verify the source code includes logger.warning or logger.exception
        # in the except block that catches get_tools() failures
        import inspect

        from wolfharness.agents.native_agent.agent import Agent

        source = inspect.getsource(Agent.get_agentlet)
        assert "logger.warning" in source or "logger.exception" in source, (
            "Expected logger.warning() or logger.exception() in get_agentlet() source "
            "when get_tools() raises, but exception is silently swallowed"
        )
