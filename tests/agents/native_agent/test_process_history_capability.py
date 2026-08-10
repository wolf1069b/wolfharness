"""Tests for ProcessHistoryAdapter."""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import RunContext
import pytest

from wolfharness.agents.native_agent.process_history_capability import (
    ProcessHistoryAdapter,
    _is_run_context_annotation,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages() -> list[ModelMessage]:
    """Sample message list for testing processors."""
    return [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelRequest(parts=[UserPromptPart(content="World")]),
    ]


@pytest.fixture
def mock_run_context() -> RunContext[Any]:
    """Create a mock RunContext for testing context-aware processors."""
    model = MagicMock()
    model.system = "test"
    model.model_name = "test-model"
    return RunContext(
        deps=None,
        model=model,
        usage=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Tests for wrap_processor — single-parameter processors
# ---------------------------------------------------------------------------


class TestWrapSingleParamProcessors:
    """Single-parameter processors should pass through unchanged."""

    def test_sync_no_ctx(self, sample_messages: list[ModelMessage]) -> None:
        """Sync processor with one param passes through."""

        def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is processor
        result = wrapped(sample_messages)
        assert len(result) == 1

    async def test_async_no_ctx(self, sample_messages: list[ModelMessage]) -> None:
        """Async processor with one param passes through."""

        async def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is processor
        result = await wrapped(sample_messages)
        assert len(result) == 1

    def test_untyped_single_param(self, sample_messages: list[ModelMessage]) -> None:
        """Untyped single-param processor passes through."""

        def processor(messages):  # type: ignore[no-untyped-def]
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is processor


# ---------------------------------------------------------------------------
# Tests for wrap_processor — two-parameter processors with typed RunContext
# ---------------------------------------------------------------------------


class TestWrapTypedContextProcessors:
    """Two-param processors with RunContext-typed first param pass through."""

    def test_sync_with_typed_ctx(
        self,
        sample_messages: list[ModelMessage],
        mock_run_context: RunContext[Any],
    ) -> None:
        """Sync processor with typed RunContext passes through."""

        def processor(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is processor
        result = wrapped(mock_run_context, sample_messages)
        assert len(result) == 1

    async def test_async_with_typed_ctx(
        self,
        sample_messages: list[ModelMessage],
        mock_run_context: RunContext[Any],
    ) -> None:
        """Async processor with typed RunContext passes through."""

        async def processor(
            ctx: RunContext[Any], messages: list[ModelMessage]
        ) -> list[ModelMessage]:
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is processor
        result = await wrapped(mock_run_context, sample_messages)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests for wrap_processor — two-parameter processors with UNtyped context
# ---------------------------------------------------------------------------


class TestWrapUntypedContextProcessors:
    """Two-param processors with untyped first param must be wrapped."""

    def test_sync_untyped_ctx_gets_wrapped(
        self,
        sample_messages: list[ModelMessage],
        mock_run_context: RunContext[Any],
    ) -> None:
        """Sync untyped 2-param processor is wrapped with RunContext annotation."""
        captured: list[Any] = []

        def processor(ctx, messages):  # type: ignore[no-untyped-def]
            captured.append((ctx, len(messages)))
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is not processor

        result = wrapped(mock_run_context, sample_messages)
        assert len(result) == 1
        assert captured == [(mock_run_context, 2)]

    async def test_async_untyped_ctx_gets_wrapped(
        self,
        sample_messages: list[ModelMessage],
        mock_run_context: RunContext[Any],
    ) -> None:
        """Async untyped 2-param processor is wrapped with RunContext annotation."""
        captured: list[Any] = []

        async def processor(ctx, messages):  # type: ignore[no-untyped-def]
            captured.append((ctx, len(messages)))
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        assert wrapped is not processor
        assert inspect.iscoroutinefunction(wrapped)

        result = await wrapped(mock_run_context, sample_messages)
        assert len(result) == 1
        assert captured == [(mock_run_context, 2)]

    def test_untyped_ctx_preserves_return_value(
        self,
        sample_messages: list[ModelMessage],
        mock_run_context: RunContext[Any],
    ) -> None:
        """Wrapped processor returns exactly what the original returns."""

        def processor(ctx, messages):  # type: ignore[no-untyped-def]
            return [ModelRequest(parts=[UserPromptPart(content="replaced")])]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        result = wrapped(mock_run_context, sample_messages)
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert result[0].parts[0].content == "replaced"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Tests for wrap_processor — invalid signatures
# ---------------------------------------------------------------------------


class TestWrapInvalidSignatures:
    """Invalid processor signatures should raise ValueError."""

    def test_zero_params(self) -> None:
        """Processor with no params is invalid."""

        def processor():  # type: ignore[no-untyped-def]
            return []

        with pytest.raises(ValueError, match="must take 1 or 2 arguments, got 0"):
            ProcessHistoryAdapter.wrap_processor(processor)

    def test_three_params(self) -> None:
        """Processor with three params is invalid."""

        def processor(a, b, c):  # type: ignore[no-untyped-def]
            return []

        with pytest.raises(ValueError, match="must take 1 or 2 arguments, got 3"):
            ProcessHistoryAdapter.wrap_processor(processor)

    def test_async_three_params(self) -> None:
        """Async processor with three params is invalid."""

        async def processor(a, b, c):  # type: ignore[no-untyped-def]
            return []

        with pytest.raises(ValueError, match="must take 1 or 2 arguments, got 3"):
            ProcessHistoryAdapter.wrap_processor(processor)


# ---------------------------------------------------------------------------
# Tests for from_processors
# ---------------------------------------------------------------------------


class TestFromProcessors:
    """from_processors converts a sequence to ProcessHistory capabilities."""

    def test_empty_list(self) -> None:
        """Empty processor list returns empty capability list."""
        result = ProcessHistoryAdapter.from_processors([])
        assert result == []

    def test_single_processor(self) -> None:
        """Single processor returns single ProcessHistory."""

        def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages

        result = ProcessHistoryAdapter.from_processors([processor])
        assert len(result) == 1
        assert isinstance(result[0], ProcessHistory)

    def test_multiple_processors(self) -> None:
        """Multiple processors return multiple ProcessHistory instances in order."""
        order: list[int] = []

        def p1(messages: list[ModelMessage]) -> list[ModelMessage]:
            order.append(1)
            return messages

        def p2(messages: list[ModelMessage]) -> list[ModelMessage]:
            order.append(2)
            return messages

        result = ProcessHistoryAdapter.from_processors([p1, p2])
        assert len(result) == 2
        assert all(isinstance(r, ProcessHistory) for r in result)

    def test_mixed_typed_and_untyped(self) -> None:
        """from_processors handles a mix of typed and untyped processors."""

        def typed_p(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages

        def untyped_p(ctx, messages):  # type: ignore[no-untyped-def]
            return messages

        result = ProcessHistoryAdapter.from_processors([typed_p, untyped_p])
        assert len(result) == 2
        assert all(isinstance(r, ProcessHistory) for r in result)


# ---------------------------------------------------------------------------
# Tests for _is_run_context_annotation
# ---------------------------------------------------------------------------


class TestIsRunContextAnnotation:
    """Unit tests for the annotation checker helper."""

    def test_bare_run_context(self) -> None:
        """Bare RunContext type is detected."""
        assert _is_run_context_annotation(RunContext) is True

    def test_run_context_with_deps(self) -> None:
        """RunContext[SomeDeps] is detected."""
        assert _is_run_context_annotation(RunContext[str]) is True

    def test_non_run_context(self) -> None:
        """Non-RunContext types are rejected."""
        assert _is_run_context_annotation(str) is False
        assert _is_run_context_annotation(int) is False
        assert _is_run_context_annotation(list) is False

    def test_none_annotation(self) -> None:
        """None is rejected."""
        assert _is_run_context_annotation(None) is False

    def test_annotated_wrapper(self) -> None:
        """typing.Annotated[RunContext, ...] is detected."""
        from typing import Annotated

        assert _is_run_context_annotation(Annotated[RunContext, "meta"]) is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration-style test: ProcessHistory capability accepts wrapped processor
# ---------------------------------------------------------------------------


class TestProcessHistoryIntegration:
    """Verify wrapped processors work inside actual ProcessHistory capabilities."""

    def test_process_history_with_wrapped_untyped_ctx(
        self,
        sample_messages: list[ModelMessage],
        mock_run_context: RunContext[Any],
    ) -> None:
        """A ProcessHistory built from a wrapped untyped processor is valid."""
        captured: list[Any] = []

        def processor(ctx, messages):  # type: ignore[no-untyped-def]
            captured.append(ctx)
            return messages[:-1]

        wrapped = ProcessHistoryAdapter.wrap_processor(processor)
        capability = ProcessHistory(wrapped)

        # Verify the capability was constructed and stores the wrapped processor.
        assert capability.processor is wrapped
