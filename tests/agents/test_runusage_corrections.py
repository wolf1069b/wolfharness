"""Test suite for RunUsage instantiation corrections.

Tests that RunUsage is instantiated with correct field names and values.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sys

import pytest


# Add src to path for imports
sys_path = Path(__file__).parent.parent.parent / "src"

sys.path.insert(0, str(sys_path))

from pydantic_ai.usage import RequestUsage, RunUsage  # noqa: E402

from wolfharness.messaging.messages import TokenCost  # noqa: E402


pytestmark = pytest.mark.unit


def test_runusage_with_cache_tokens():
    """Test RunUsage with cache_read_tokens and cache_write_tokens."""
    usage = RunUsage(
        input_tokens=100,
        output_tokens=200,
        cache_read_tokens=50,
        cache_write_tokens=25,
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 200
    assert usage.cache_read_tokens == 50
    assert usage.cache_write_tokens == 25

    print("✓ RunUsage with cache tokens works correctly")


def test_runusage_from_usage_dict():
    """Test RunUsage instantiation from Claude API usage dict."""
    # Simulate Claude API usage dict (actual keys may vary)
    usage_dict = {
        "input_tokens": 150,
        "output_tokens": 300,
        "cache_read_tokens": 75,
        "cache_write_tokens": 37,
    }

    run_usage = RunUsage(
        input_tokens=usage_dict.get("input_tokens", 0),
        output_tokens=usage_dict.get("output_tokens", 0),
        cache_read_tokens=usage_dict.get("cache_read_tokens", 0),
        cache_write_tokens=usage_dict.get("cache_write_tokens", 0),
    )

    assert run_usage.input_tokens == 150
    assert run_usage.output_tokens == 300
    assert run_usage.cache_read_tokens == 75
    assert run_usage.cache_write_tokens == 37

    print("✓ RunUsage from usage dict works correctly")


def test_requestusage_with_cache_tokens():
    """Test RequestUsage with cache_read_tokens and cache_write_tokens."""
    request_usage = RequestUsage(
        input_tokens=100,
        output_tokens=200,
        cache_read_tokens=50,
        cache_write_tokens=25,
    )

    assert request_usage.input_tokens == 100
    assert request_usage.output_tokens == 200
    assert request_usage.cache_read_tokens == 50
    assert request_usage.cache_write_tokens == 25

    print("✓ RequestUsage with cache tokens works correctly")


def test_tokencost_with_runusage():
    """Test TokenCost creation with RunUsage."""
    run_usage = RunUsage(
        input_tokens=100,
        output_tokens=200,
        cache_read_tokens=50,
        cache_write_tokens=25,
    )

    cost_info = TokenCost(
        token_usage=run_usage,
        total_cost=Decimal("0.015"),
    )

    assert cost_info.token_usage.input_tokens == 100
    assert cost_info.token_usage.output_tokens == 200
    assert cost_info.token_usage.cache_read_tokens == 50
    assert cost_info.token_usage.cache_write_tokens == 25
    assert cost_info.total_cost == Decimal("0.015")

    print("✓ TokenCost with RunUsage works correctly")


def test_runusage_default_values():
    """Test RunUsage with default values."""
    usage = RunUsage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0

    print("✓ RunUsage default values work correctly")


def test_runusage_partial_values():
    """Test RunUsage with partial values (missing some fields)."""
    usage = RunUsage(
        input_tokens=100,
        output_tokens=200,
        # cache_read_tokens and cache_write_tokens omitted
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 200
    assert usage.cache_read_tokens == 0  # Default
    assert usage.cache_write_tokens == 0  # Default

    print("✓ RunUsage with partial values works correctly")


def test_usage_dict_with_missing_keys():
    """Test RunUsage instantiation from usage dict with missing keys."""
    usage_dict = {
        "input_tokens": 100,
        "output_tokens": 200,
        # cache_read_tokens and cache_write_tokens missing
    }

    run_usage = RunUsage(
        input_tokens=usage_dict.get("input_tokens", 0),
        output_tokens=usage_dict.get("output_tokens", 0),
        cache_read_tokens=usage_dict.get("cache_read_tokens", 0),
        cache_write_tokens=usage_dict.get("cache_write_tokens", 0),
    )

    assert run_usage.input_tokens == 100
    assert run_usage.output_tokens == 200
    assert run_usage.cache_read_tokens == 0
    assert run_usage.cache_write_tokens == 0

    print("✓ Usage dict with missing keys handled correctly")


if __name__ == "__main__":
    print("Testing RunUsage instantiation corrections...\n")
    test_runusage_with_cache_tokens()
    test_runusage_from_usage_dict()
    test_requestusage_with_cache_tokens()
    test_tokencost_with_runusage()
    test_runusage_default_values()
    test_runusage_partial_values()
    test_usage_dict_with_missing_keys()
    print("\n✓ All RunUsage instantiation tests passed!")
