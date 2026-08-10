"""Test that `agent.run_stream()` cancellation cleans up correctly.

Verifies that when a stream is cancelled mid-flight:
- `run_ctx.cancelled` is set to `True`
- `_iteration_task` is reset to `None`
- No dangling asyncio tasks remain

Note: RunHandle `_status` assertions (idle vs done after cancel) are
covered in integration tests:
- tests/orchestrator/test_cancel_e2e.py
- tests/servers/acp_server/test_acp_cancel_then_prompt.py
This file tests agent-level streaming, not RunHandle lifecycle.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from pydantic_ai.models.test import TestModel, TestStreamedResponse
import pytest

from wolfharness import Agent
from wolfharness.agents.base_agent import _current_run_ctx_var
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.orchestrator.core import SessionState


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class SlowTestModel(TestModel):
    """TestModel that inserts a delay before yielding the streamed response.

    The default TestModel's request_stream yields TestStreamedResponse which
    emits all parts instantly. We override request_stream to inject a sleep
    before yielding the response, giving us a window to cancel
    while the iteration_task is still running.
    """

    def __init__(
        self,
        *,
        custom_output_text: str | None = None,
        pre_stream_delay: float = 0.5,
    ) -> None:
        super().__init__(custom_output_text=custom_output_text)
        self.pre_stream_delay = pre_stream_delay

    @asynccontextmanager
    async def request_stream(  # type: ignore[override]
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> Any:
        """Yield the streamed response after a configurable delay."""
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)

        await asyncio.sleep(self.pre_stream_delay)
        yield TestStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _structured_response=model_response,
            _messages=messages,
            _provider_name=self._system,
        )


@pytest.fixture
async def slow_agent() -> AsyncGenerator[Agent[None]]:
    """Agent with SlowTestModel for cancellation testing."""
    model = SlowTestModel(custom_output_text="Hello world slow response", pre_stream_delay=0.5)
    return Agent(name="cancel-test-agent", model=model, session=False)


def _mock_session_pool(agent: Agent[Any], run_ctx: Any) -> None:
    """Mock agent_pool.session_pool so _get_session_run_ctx() returns run_ctx."""
    from wolfharness.orchestrator.run import RunHandle

    session_state = SessionState(session_id="test-session", agent_name="test")
    session_state.current_run_id = run_ctx.run_id
    session_controller = MagicMock()
    session_controller.get_session.return_value = session_state
    run_handle = MagicMock(spec=RunHandle)
    run_handle.run_ctx = run_ctx
    session_pool = MagicMock()
    session_pool.sessions = session_controller
    session_pool.get_run.return_value = run_handle
    agent_pool = MagicMock()
    agent_pool.session_pool = session_pool
    host_ctx = MagicMock()
    host_ctx.session_pool = session_pool
    agent_pool.get_context.return_value = host_ctx
    agent.agent_pool = agent_pool


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_stream_cancellation_sets_cancelled_and_cleans_up(
    slow_agent: Agent[None],
) -> None:
    """Cancelling the async generator mid-stream sets run_ctx.cancelled.

    Also cleans up iteration_task.

    Steps:
    1. Start streaming with a slow model
    2. Capture the run_ctx while stream is active
    3. Cancel the stream via agent.interrupt() (sets run_ctx.cancelled = True)
    4. Verify run_ctx.cancelled is True
    5. Verify _iteration_task is None after cleanup
    6. Verify no dangling tasks remain
    """
    stream_started = asyncio.Event()
    captured_run_ctx: list[Any] = []

    async def consume_stream() -> None:
        async for event in slow_agent.run_stream("Test prompt"):
            run_ctx = _current_run_ctx_var.get()
            if run_ctx is not None and not captured_run_ctx:
                captured_run_ctx.append(run_ctx)
            stream_started.set()
            if isinstance(event, StreamCompleteEvent):
                break

    task = asyncio.create_task(consume_stream())

    # Wait for stream to start and run_ctx to be captured
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    assert len(captured_run_ctx) == 1, "Should have captured the run_ctx"
    run_ctx = captured_run_ctx[0]
    assert run_ctx.cancelled is False, "run_ctx should not be cancelled before interrupt"

    # Set up SessionPool fallback for cross-task access
    _mock_session_pool(slow_agent, run_ctx)

    # Cancel the stream mid-flight via interrupt (simulates real abort flow)
    await slow_agent.interrupt(session_id="test-session")

    # Wait for the consumer task to finish
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)

    # Assert run_ctx.cancelled was set to True
    assert run_ctx.cancelled is True, "run_ctx.cancelled must be True after stream cancellation"

    # Assert _iteration_task is None after cleanup
    assert slow_agent._iteration_task is None, (
        "_iteration_task must be None after cleanup in finally block"
    )

    # Assert no dangling tasks remain
    all_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    # The iteration task should be gone; if any agent-related task remains, it's a leak
    agent_tasks = [t for t in all_tasks if "agent" in t.get_name() or "iteration" in t.get_name()]
    assert len(agent_tasks) == 0, (
        f"Dangling agent tasks found: {[t.get_name() for t in agent_tasks]}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_stream_raw_task_cancellation_cleans_up(slow_agent: Agent[None]) -> None:
    """Raw consumer task cancellation still cleans up _iteration_task.

    This tests the finally-block path in _stream_events() when the
    consumer task is cancelled directly (e.g. via task.cancel()).
    """
    stream_started = asyncio.Event()

    async def consume_stream() -> None:
        async for _event in slow_agent.run_stream("Test prompt"):
            stream_started.set()

    task = asyncio.create_task(consume_stream())

    # Wait for stream to start
    await asyncio.wait_for(stream_started.wait(), timeout=2.0)

    # Give iteration_task time to be created
    await asyncio.sleep(0.05)

    # Cancel the consumer task directly (raw cancellation)
    task.cancel()

    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3.0)

    # _iteration_task must be cleaned up
    assert slow_agent._iteration_task is None, (
        "_iteration_task must be None after consumer task cancellation"
    )

    # No dangling agent tasks
    all_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    agent_tasks = [t for t in all_tasks if "agent" in t.get_name() or "iteration" in t.get_name()]
    assert len(agent_tasks) == 0, (
        f"Dangling agent tasks found: {[t.get_name() for t in agent_tasks]}"
    )
