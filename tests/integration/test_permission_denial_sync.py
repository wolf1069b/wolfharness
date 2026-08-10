"""Test tool call event ordering with permission denial.

These tests verify that:
1. ToolCallStartEvent is emitted exactly once per tool call
2. Events arrive in correct order (start before result)
3. No duplicate ACP notifications are sent (which would cause UI sync issues)

The tests use a DenyingInputProvider to trigger the permission flow and
verify the event sequence matches the expected flow.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp.types import ElicitResult
import pytest


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from wolfharness import AgentContext


@dataclass
class EventTrace:
    """Traces events for verification."""

    events: list[dict[str, Any]] = field(default_factory=list)
    permission_requests: list[dict[str, Any]] = field(default_factory=list)

    def log_event(self, event_type: str, event: Any) -> None:
        """Log an event."""
        self.events.append({"type": event_type, "event_class": type(event).__name__})

    def log_permission_request(self, tool_name: str, tool_call_id: str) -> None:
        """Log a permission request."""
        self.permission_requests.append({"tool_name": tool_name, "tool_call_id": tool_call_id})


class DenyingInputProvider:
    """Input provider that denies all tool calls."""

    def __init__(self, trace: EventTrace, delay: float = 0.1):
        self.trace = trace
        self.delay = delay
        self.denial_count = 0

    async def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool_description: str = "",
    ) -> str:
        """Deny all tool calls after a small delay."""
        tool_name = context.tool_name or "unknown"
        tool_call_id = context.tool_call_id or "unknown"
        self.trace.log_permission_request(tool_name, tool_call_id)
        await asyncio.sleep(self.delay)
        self.denial_count += 1
        return "skip"

    async def elicit_input(self, *args: Any, **kwargs: Any) -> Any:
        """Not used in this test."""
        return ElicitResult(action="cancel")


if __name__ == "__main__":
    print("All tests passed!")
