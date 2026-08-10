"""Unit tests for ``diff_usage()`` helper.

Tests cover: normal diff, zero delta, all token fields, ``details`` dict
diff with new/removed keys, ``tool_calls`` field, and input immutability.
"""

from __future__ import annotations

from pydantic_ai import RunUsage
import pytest

from wolfharness.utils.usage import diff_usage


@pytest.mark.unit
def test_diff_usage_normal() -> None:
    """Normal diff: curr has more tokens than prev."""
    prev = RunUsage(input_tokens=100, output_tokens=50, requests=1)
    curr = RunUsage(input_tokens=250, output_tokens=120, requests=2)
    diff = diff_usage(curr, prev)

    assert diff.input_tokens == 150
    assert diff.output_tokens == 70
    assert diff.requests == 1


@pytest.mark.unit
def test_diff_usage_zero_delta() -> None:
    """Zero delta: curr == prev produces all-zero diff."""
    prev = RunUsage(input_tokens=100, output_tokens=50, requests=1)
    curr = RunUsage(input_tokens=100, output_tokens=50, requests=1)
    diff = diff_usage(curr, prev)

    assert diff.input_tokens == 0
    assert diff.output_tokens == 0
    assert diff.requests == 0
    assert diff.tool_calls == 0


@pytest.mark.unit
def test_diff_usage_all_token_fields() -> None:
    """All token fields are diffed, not just input/output."""
    prev = RunUsage(
        input_tokens=10,
        output_tokens=20,
        cache_write_tokens=5,
        cache_read_tokens=3,
        input_audio_tokens=1,
        cache_audio_read_tokens=2,
        output_audio_tokens=4,
    )
    curr = RunUsage(
        input_tokens=30,
        output_tokens=50,
        cache_write_tokens=15,
        cache_read_tokens=8,
        input_audio_tokens=3,
        cache_audio_read_tokens=6,
        output_audio_tokens=10,
    )
    diff = diff_usage(curr, prev)

    assert diff.input_tokens == 20
    assert diff.output_tokens == 30
    assert diff.cache_write_tokens == 10
    assert diff.cache_read_tokens == 5
    assert diff.input_audio_tokens == 2
    assert diff.cache_audio_read_tokens == 4
    assert diff.output_audio_tokens == 6


@pytest.mark.unit
def test_diff_usage_details_new_keys() -> None:
    """Details dict: keys only in curr are included as-is."""
    prev = RunUsage(details={"existing": 10})
    curr = RunUsage(details={"existing": 25, "new_key": 5})
    diff = diff_usage(curr, prev)

    assert diff.details["existing"] == 15
    assert diff.details["new_key"] == 5


@pytest.mark.unit
def test_diff_usage_details_removed_keys() -> None:
    """Details dict: keys only in prev are dropped from result."""
    prev = RunUsage(details={"kept": 10, "removed": 5})
    curr = RunUsage(details={"kept": 20})
    diff = diff_usage(curr, prev)

    assert diff.details["kept"] == 10
    assert "removed" not in diff.details


@pytest.mark.unit
def test_diff_usage_details_empty() -> None:
    """Details dict: empty dicts produce empty diff."""
    prev = RunUsage()
    curr = RunUsage()
    diff = diff_usage(curr, prev)

    assert diff.details == {}


@pytest.mark.unit
def test_diff_usage_tool_calls() -> None:
    """tool_calls field is diffed correctly."""
    prev = RunUsage(tool_calls=3)
    curr = RunUsage(tool_calls=7)
    diff = diff_usage(curr, prev)

    assert diff.tool_calls == 4


@pytest.mark.unit
def test_diff_usage_does_not_modify_inputs() -> None:
    """diff_usage SHALL NOT modify either input argument."""
    prev = RunUsage(input_tokens=100, details={"key": 10})
    curr = RunUsage(input_tokens=200, details={"key": 30, "new": 5})

    # Snapshot originals
    prev_input_before = prev.input_tokens
    prev_details_before = dict(prev.details)
    curr_input_before = curr.input_tokens
    curr_details_before = dict(curr.details)

    diff_usage(curr, prev)

    # Inputs unchanged
    assert prev.input_tokens == prev_input_before
    assert prev.details == prev_details_before
    assert curr.input_tokens == curr_input_before
    assert curr.details == curr_details_before


@pytest.mark.unit
def test_diff_usage_negative_values() -> None:
    """Negative diff: curr < prev produces negative deltas.

    This can happen if usage resets between runs (e.g., a new request
    snapshot has fewer tokens than the previous one).
    """
    prev = RunUsage(input_tokens=200, output_tokens=100, requests=2)
    curr = RunUsage(input_tokens=100, output_tokens=50, requests=1)
    diff = diff_usage(curr, prev)

    assert diff.input_tokens == -100
    assert diff.output_tokens == -50
    assert diff.requests == -1


@pytest.mark.unit
def test_diff_usage_requests_zero_for_tool_only() -> None:
    """When requests didn't change (tool-only iteration), diff.requests == 0.

    This is the key check used in NativeTurn to skip StepUsageEvent
    emission for non-LLM iterations.
    """
    prev = RunUsage(input_tokens=100, output_tokens=50, requests=1, tool_calls=0)
    curr = RunUsage(input_tokens=100, output_tokens=50, requests=1, tool_calls=2)
    diff = diff_usage(curr, prev)

    assert diff.requests == 0
    assert diff.tool_calls == 2
