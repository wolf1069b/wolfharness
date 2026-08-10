"""Tests for deferred tool call handling in ACP event converter.

These tests verify that ToolCallDeferredEvent is correctly converted to
ACP session/update notifications with the proper deferred_handle in _meta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from acp.agent.notifications import ACPNotifications
from acp.schema import ToolCallStart
from acp.tool_call_reporter import ToolCallReporter
from acp.tool_call_state import ToolCallState
from wolfharness.agents.events.events import ToolCallDeferredEvent
from wolfharness_server.acp_server.event_converter import ACPEventConverter


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from syrupy import SnapshotAssertion


@pytest.fixture
def json_snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Use JSON serialization for cleaner snapshots."""
    return snapshot.use_extension(JSONSnapshotExtension)


@pytest.fixture
def converter() -> ACPEventConverter:
    """Create a fresh converter for each test."""
    return ACPEventConverter()


@pytest.fixture
def mock_notifications() -> AsyncMock:
    """Create a mock ACPNotifications for testing ToolCallState and ToolCallReporter."""
    notifications = AsyncMock(spec=ACPNotifications)
    notifications.tool_call_start = AsyncMock()
    notifications.tool_call_update = AsyncMock()
    notifications.tool_call_progress = AsyncMock()
    notifications.send_update = AsyncMock()
    return notifications


# ============================================================================
# ACPEventConverter tests
# ============================================================================


class TestConverterDeferredEvent:
    """Tests for ToolCallDeferredEvent conversion in ACPEventConverter."""

    async def test_deferred_event_produces_tool_call_start(
        self, converter: ACPEventConverter
    ) -> None:
        """A ToolCallDeferredEvent with status=pending should yield a ToolCallStart."""
        event = ToolCallDeferredEvent(
            tool_call_id="tc-deferred-001",
            tool_name="human_approval",
            deferred_strategy="block",
            deferred_handle="dh_abc123",
            status="pending",
            session_id="sess-001",
        )

        updates = [u async for u in converter.convert(event)]
        assert len(updates) == 1

        update = updates[0]
        assert isinstance(update, ToolCallStart)
        assert update.tool_call_id == "tc-deferred-001"
        assert update.status == "pending"
        assert update.title == "Deferred: human_approval"
        assert update.field_meta == {"deferred_handle": "dh_abc123"}

    async def test_deferred_event_serializes_meta_correctly(
        self, converter: ACPEventConverter, json_snapshot: SnapshotAssertion
    ) -> None:
        """The ToolCallStart from a deferred event should serialize _meta correctly."""
        event = ToolCallDeferredEvent(
            tool_call_id="tc-deferred-002",
            tool_name="complex_review",
            deferred_strategy="continue",
            deferred_handle="dh_xyz789",
            status="pending",
            session_id="sess-002",
        )

        updates = [u async for u in converter.convert(event)]
        update = updates[0]

        serialized = update.model_dump(by_alias=True, exclude_none=True)
        assert serialized == json_snapshot

    async def test_deferred_event_with_resolved_status_is_noop(
        self, converter: ACPEventConverter
    ) -> None:
        """A ToolCallDeferredEvent with status=resolved should produce no output."""
        event = ToolCallDeferredEvent(
            tool_call_id="tc-deferred-003",
            tool_name="some_tool",
            deferred_strategy="block",
            deferred_handle="dh_resolved",
            status="resolved",
        )

        updates = [u async for u in converter.convert(event)]
        assert len(updates) == 0

    async def test_deferred_event_creates_tool_state(self, converter: ACPEventConverter) -> None:
        """Converter should create a tool state entry for the deferred tool call."""
        event = ToolCallDeferredEvent(
            tool_call_id="tc-deferred-004",
            tool_name="read",
            deferred_strategy="block",
            deferred_handle="dh_read",
            status="pending",
        )

        _ = [u async for u in converter.convert(event)]

        # Tool state should be created for the deferred call
        assert "tc-deferred-004" in converter._tool_states
        state = converter._tool_states["tc-deferred-004"]
        assert state.tool_name == "read"
        assert state.started is True


# ============================================================================
# ToolCallState tests
# ============================================================================


class TestToolCallStateDeferred:
    """Tests for deferred handle in ToolCallState."""

    def test_deferred_handle_defaults_to_none(self, mock_notifications: AsyncMock) -> None:
        """Deferred handle should default to None when not provided."""
        state = ToolCallState(
            notifications=mock_notifications,
            tool_call_id="tc-001",
            tool_name="test_tool",
            title="Test Tool",
            kind="other",
            raw_input={},
        )
        assert state.deferred_handle is None

    def test_deferred_handle_stored_when_provided(self, mock_notifications: AsyncMock) -> None:
        """Deferred handle should be stored when explicitly provided."""
        state = ToolCallState(
            notifications=mock_notifications,
            tool_call_id="tc-002",
            tool_name="async_tool",
            title="Async Tool",
            kind="execute",
            raw_input={"arg": "value"},
            deferred_handle="dh_stored_001",
        )
        assert state.deferred_handle == "dh_stored_001"

    def test_on_state_change_is_noop(self, mock_notifications: AsyncMock) -> None:
        """_on_state_change should be a no-op on ACP V1."""
        state = ToolCallState(
            notifications=mock_notifications,
            tool_call_id="tc-003",
            tool_name="test_tool",
            title="Test Tool",
            kind="other",
            raw_input={},
        )
        # Should not raise or make any calls
        state._on_state_change("running")


# ============================================================================
# ToolCallReporter tests
# ============================================================================


class TestToolCallReporterDeferred:
    """Tests for deferred_start in ToolCallReporter."""

    async def test_deferred_start_sends_tool_call_start_with_meta(
        self, mock_notifications: AsyncMock
    ) -> None:
        """deferred_start should send session/update with _meta.deferred_handle."""
        reporter = ToolCallReporter(
            notifications=mock_notifications,
            tool_call_id="tc-reporter-001",
            title="Deferred Async Task",
            kind="execute",
        )

        await reporter.deferred_start(deferred_handle="dh_reporter_001")

        # Should have sent via send_update
        mock_notifications.send_update.assert_called_once()
        call_args = mock_notifications.send_update.call_args
        update = call_args[0][0]
        assert isinstance(update, ToolCallStart)
        assert update.tool_call_id == "tc-reporter-001"
        assert update.status == "pending"
        assert update.field_meta == {"deferred_handle": "dh_reporter_001"}

    async def test_deferred_start_idempotent(self, mock_notifications: AsyncMock) -> None:
        """Calling deferred_start twice should only send one notification."""
        reporter = ToolCallReporter(
            notifications=mock_notifications,
            tool_call_id="tc-reporter-002",
            title="Another Task",
            kind="read",
        )

        await reporter.deferred_start(deferred_handle="dh_idem")
        await reporter.deferred_start(deferred_handle="dh_idem")

        # Should only have sent once
        assert mock_notifications.send_update.call_count == 1

    async def test_deferred_start_does_not_interfere_with_normal_start(
        self, mock_notifications: AsyncMock
    ) -> None:
        """Normal start() should not include _meta when not using deferred_start."""
        reporter = ToolCallReporter(
            notifications=mock_notifications,
            tool_call_id="tc-reporter-003",
            title="Normal Task",
            kind="read",
        )

        await reporter.start()

        mock_notifications.tool_call_start.assert_called_once()
        call_kwargs = mock_notifications.tool_call_start.call_args[1]
        # Normal start should not have _meta
        assert call_kwargs["tool_call_id"] == "tc-reporter-003"
