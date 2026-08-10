"""Unit tests for StepErrorMetadata and RunErrorEvent.step_error field."""

from __future__ import annotations

import pytest

from wolfharness.agents.events.events import RunErrorEvent, StepErrorMetadata


@pytest.mark.unit
def test_run_error_event_step_error_defaults_to_none() -> None:
    """RunErrorEvent can be constructed without step_error (backward compat)."""
    event = RunErrorEvent(
        message="something went wrong",
        agent_name="test_agent",
        run_id="run-123",
    )
    assert event.step_error is None
    assert event.event_kind == "run_error"
    assert event.message == "something went wrong"
    assert event.agent_name == "test_agent"
    assert event.run_id == "run-123"


@pytest.mark.unit
def test_run_error_event_with_step_error() -> None:
    """RunErrorEvent can be constructed with step_error metadata."""
    step_error = StepErrorMetadata(
        node_type="ModelRequestNode",
        exception_type="ValueError",
        exception_message="invalid input",
    )
    event = RunErrorEvent(
        message="step failed",
        agent_name="test_agent",
        run_id="run-456",
        step_error=step_error,
    )
    assert event.step_error is not None
    assert event.step_error.node_type == "ModelRequestNode"
    assert event.step_error.exception_type == "ValueError"
    assert event.step_error.exception_message == "invalid input"


@pytest.mark.unit
def test_step_error_metadata_fields_accessible() -> None:
    """StepErrorMetadata fields are accessible and correct."""
    metadata = StepErrorMetadata(
        node_type="CallToolsNode",
        exception_type="RuntimeError",
        exception_message="tool execution failed",
    )
    assert metadata.node_type == "CallToolsNode"
    assert metadata.exception_type == "RuntimeError"
    assert metadata.exception_message == "tool execution failed"


@pytest.mark.unit
def test_step_error_metadata_is_frozen() -> None:
    """StepErrorMetadata is a frozen dataclass — cannot mutate fields."""
    metadata = StepErrorMetadata(
        node_type="End",
        exception_type="KeyError",
        exception_message="missing key",
    )
    with pytest.raises(AttributeError):
        metadata.node_type = "OtherNode"  # type: ignore[misc]


@pytest.mark.unit
def test_step_error_metadata_with_unknown_node_type() -> None:
    """StepErrorMetadata can represent errors where node context is unknown."""
    metadata = StepErrorMetadata(
        node_type="unknown",
        exception_type="ConnectionError",
        exception_message="MCP server unreachable",
    )
    assert metadata.node_type == "unknown"
    assert metadata.exception_type == "ConnectionError"
    assert metadata.exception_message == "MCP server unreachable"
