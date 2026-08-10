"""Tests for the RunState and RunOutcome enums in wolfharness.lifecycle.types."""

from __future__ import annotations

from enum import Enum

import pytest

from wolfharness.lifecycle import RunOutcome, RunState


pytestmark = pytest.mark.unit


def test_run_state_defines_exactly_3_states() -> None:
    """Given the RunState enum, it should define exactly 3 lifecycle states."""
    actual_values: set[str] = {m.name for m in RunState}
    expected_values = {"IDLE", "RUNNING", "DONE"}
    assert actual_values == expected_values


def test_run_state_is_enum() -> None:
    """Given RunState, it should be a proper Enum subclass."""
    assert issubclass(RunState, Enum)


def test_run_state_distinctness() -> None:
    """Given RunState, all states should be distinct."""
    assert RunState.IDLE is not RunState.DONE
    assert RunState.IDLE is not RunState.RUNNING
    assert RunState.RUNNING is not RunState.DONE


def test_run_state_value_strings() -> None:
    """Given RunState, the value strings should match the expected names."""
    assert RunState.IDLE.value == "idle"
    assert RunState.RUNNING.value == "running"
    assert RunState.DONE.value == "done"


def test_run_outcome_defines_exactly_3_outcomes() -> None:
    """Given the RunOutcome enum, it should define exactly 3 terminal outcomes."""
    actual_values: set[str] = {m.name for m in RunOutcome}
    expected_values = {"COMPLETED", "FAILED", "CHECKPOINTED"}
    assert actual_values == expected_values


def test_run_outcome_is_enum() -> None:
    """Given RunOutcome, it should be a proper Enum subclass."""
    assert issubclass(RunOutcome, Enum)


def test_run_outcome_distinctness() -> None:
    """Given RunOutcome, all outcomes should be distinct."""
    assert RunOutcome.COMPLETED is not RunOutcome.FAILED
    assert RunOutcome.COMPLETED is not RunOutcome.CHECKPOINTED
    assert RunOutcome.FAILED is not RunOutcome.CHECKPOINTED


def test_run_outcome_value_strings() -> None:
    """Given RunOutcome, the value strings should match the expected names."""
    assert RunOutcome.COMPLETED.value == "completed"
    assert RunOutcome.FAILED.value == "failed"
    assert RunOutcome.CHECKPOINTED.value == "checkpointed"
