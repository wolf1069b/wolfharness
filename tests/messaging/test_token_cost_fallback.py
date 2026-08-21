"""Tests for the guarded TokenCost.from_usage cost fallback.

The tokonomics fallback performs a network download of the LiteLLM pricing
table on a cache miss. These tests verify the fallback is bounded by a
timeout, that failures are negative-cached per model so later turns skip
the network call, and that the startup prefetch helper is safe.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import time
from typing import TYPE_CHECKING

from pydantic_ai import RunUsage
import pytest

from wolfharness.messaging.messages import (
    _COST_FALLBACK_FAILED,
    TokenCost,
    prefetch_token_cost_cache,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_fallback_failed() -> Iterator[None]:
    """Reset the module-global negative cache before and after each test."""
    _COST_FALLBACK_FAILED.clear()
    yield
    _COST_FALLBACK_FAILED.clear()


async def test_known_model_uses_local_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known models resolve from the local genai_prices table without the fallback."""
    fail_msg = "calculate_token_cost must not be called for known models"

    async def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError(fail_msg)

    monkeypatch.setattr("tokonomics.calculate_token_cost", _fail)

    cost = await TokenCost.from_usage(
        RunUsage(input_tokens=100, output_tokens=50),
        model="gpt-4o",
        provider="openai",
    )

    assert cost is not None
    assert cost.total_cost > 0


async def test_fallback_failure_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing fallback is recorded so later turns skip the network call."""
    unreachable_msg = "unreachable pricing host"
    call_count: list[int] = []

    async def _unreachable(model: str, input_tokens: int | None, output_tokens: int | None) -> None:
        call_count.append(1)
        raise RuntimeError(unreachable_msg)

    monkeypatch.setattr("tokonomics.calculate_token_cost", _unreachable)

    usage = RunUsage(input_tokens=10, output_tokens=5)
    first = await TokenCost.from_usage(usage, model="internal/test-custom")
    assert first is not None
    assert first.total_cost == Decimal(0)
    assert "internal/test-custom" in _COST_FALLBACK_FAILED

    second = await TokenCost.from_usage(usage, model="internal/test-custom")
    assert second is not None
    assert second.total_cost == Decimal(0)
    assert len(call_count) == 1


async def test_slow_fallback_is_bounded_by_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hanging fallback must not block end_turn beyond the configured timeout."""

    async def _slow(model: str, input_tokens: int | None, output_tokens: int | None) -> None:
        await asyncio.sleep(0.5)

    monkeypatch.setattr("tokonomics.calculate_token_cost", _slow)

    start = time.monotonic()
    cost = await TokenCost.from_usage(
        RunUsage(input_tokens=10, output_tokens=5),
        model="internal/slow-model",
    )
    elapsed = time.monotonic() - start

    assert cost is not None
    assert cost.total_cost == Decimal(0)
    assert "internal/slow-model" in _COST_FALLBACK_FAILED
    assert elapsed < 0.45, f"expected timeout to bound latency, took {elapsed:.3f}s"


async def test_prefetch_helper_seeds_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefetch warms the cache for the default plus common provider models."""
    calls: list[str] = []

    async def _fake_get_model_costs(model: str, **kwargs: object) -> None:
        calls.append(model)

    monkeypatch.setattr("tokonomics.get_model_costs", _fake_get_model_costs)

    await prefetch_token_cost_cache()

    assert calls == ["gpt-4o", "anthropic/claude-3-5-sonnet", "google/gemini-1.5-pro"]


async def test_prefetch_helper_suppresses_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefetch must never raise, even when the pricing download fails."""
    failure_msg = "pricing host unreachable"

    async def _raising(model: str, **kwargs: object) -> None:
        raise RuntimeError(failure_msg)

    monkeypatch.setattr("tokonomics.get_model_costs", _raising)

    await prefetch_token_cost_cache()  # must not raise
