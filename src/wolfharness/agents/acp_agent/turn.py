"""ACP Turn — wraps ACP session/prompt stream into a single reactive Turn.

This module provides :class:`ACPTurn`, a :class:`~wolfharness.orchestrator.turn.Turn`
subclass that drives an ACP client through a single prompt → stream → complete
cycle, yielding :class:`~wolfharness.agents.events.RichAgentStreamEvent` items
and populating ``message_history`` / ``final_message`` after execution.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

import logfire
from pydantic import ValidationError

from acp.exceptions import RequestError
from wolfharness.agents.events import (
    RunErrorEvent,
    StreamCompleteEvent,
)
from wolfharness.observability.spans import safe_span
from wolfharness.orchestrator.turn import HookAwareTurn, Turn
from wolfharness.utils.pydantic_ai_helpers import flatten_prompts


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence
    from typing import Any

    from pydantic_ai import ModelMessage, UserContent

    from acp.schema import ContentBlock, PromptResponse, SessionUpdate
    from wolfharness.agents.context import AgentRunContext
    from wolfharness.agents.events import RichAgentStreamEvent
    from wolfharness.hooks import AgentHooks
    from wolfharness.messaging import ChatMessage


@runtime_checkable
class ACPClientProtocol(Protocol):
    """Protocol defining the ACP client interface expected by ACPTurn.

    The ACP client must provide three methods:

    - :meth:`prompt` — send a prompt to the remote agent, return a response handle
    - :meth:`stream_events` — return an async iterator of session updates
    - :meth:`get_messages` — return the full list of session updates for history
    """

    async def prompt(self, session_id: str, content: list[ContentBlock]) -> PromptResponse: ...

    def stream_events(self, response: PromptResponse) -> AsyncIterator[SessionUpdate]: ...

    async def get_messages(self, session_id: str) -> list[SessionUpdate]: ...


def _convert_updates_to_model_messages(
    updates: Sequence[SessionUpdate],
    *,
    session_id: str,
    agent_name: str | None = None,
    model_name: str | None = None,
) -> tuple[list[ModelMessage], ChatMessage[str] | None]:
    """Convert ACP session updates to model messages and final chat message.

    Uses :class:`~wolfharness.agents.acp_agent.acp_converters.ACPMessageAccumulator`
    to build :class:`~wolfharness.messaging.ChatMessage` objects from the raw
    session updates, then flattens the model messages.

    Returns:
        A tuple of (model_messages, final_chat_message). The final chat message
        is the last assistant message, or None if no messages were produced.
    """
    from wolfharness.agents.acp_agent.acp_converters import ACPMessageAccumulator

    accumulator = ACPMessageAccumulator(
        session_id=session_id,
        agent_name=agent_name,
        model_name=model_name,
    )
    for update in updates:
        accumulator.process(update)
    chat_messages = accumulator.finalize()

    model_messages: list[ModelMessage] = []
    for msg in chat_messages:
        model_messages.extend(msg.messages)

    final_msg: ChatMessage[str] | None = None
    for msg in reversed(chat_messages):
        if msg.role == "assistant":
            final_msg = msg
            break

    return model_messages, final_msg


class ACPTurn(HookAwareTurn, Turn):
    """Single reactive turn wrapping an ACP session/prompt stream.

    Encapsulates one complete ACP interaction cycle: sending a prompt to the
    remote agent, streaming session updates as native events, and collecting
    the final message history.
    """

    def __init__(
        self,
        acp_client: ACPClientProtocol,
        prompts: list[UserContent],
        run_ctx: AgentRunContext,
        session_id: str,
        agent_name: str | None = None,
        hooks: AgentHooks | None = None,
        env: Any | None = None,
    ) -> None:
        super().__init__()
        self._acp_client = acp_client
        self._prompts = prompts
        self._run_ctx = run_ctx
        self._session_id = session_id
        self._agent_name = agent_name
        self._hooks = hooks
        self._agent_env = env
        self._prompt_response: PromptResponse | None = None
        self._message_id = uuid4().hex

    @property
    def _hook_env(self) -> Any | None:
        """Execution environment for command hooks."""
        return self._agent_env

    @property
    def _hook_agent_name(self) -> str:
        """Agent name passed to hook invocations."""
        return self._agent_name or ""

    @property
    def _hook_prompt(self) -> str:
        """The user prompt for this turn."""
        return str(self._prompts)

    async def execute(self) -> AsyncGenerator[RichAgentStreamEvent[Any]]:  # noqa: PLR0915
        """Execute one ACP prompt → stream → complete cycle.

        Yields:
            Native streaming events mapped from ACP session updates.

        Raises:
            asyncio.CancelledError: Re-raised if the turn is cancelled.
        """
        from pydantic_ai import RunUsage

        from wolfharness.agents.acp_agent.acp_converters import (
            acp_to_native_event,
            convert_to_acp_content,
        )
        from wolfharness.agents.events import (
            StepUsageEvent,
            ToolCallCompleteEvent,
            ToolCallStartEvent,
        )

        run_id = self._run_ctx.run_id

        with safe_span(
            "turn.acp",
            turn_id=self._run_ctx.turn_id,
            session_id=self._run_ctx.session_id,
        ):
            from wolfharness.observability.trace import get_trace_id

            logfire.info(
                "Turn started",
                trace_id=get_trace_id(),
                turn_id=self._run_ctx.turn_id,
                session_id=self._run_ctx.session_id,
                agent_type="acp",
            )
            turn_start = time.perf_counter()
            try:
                # --- Phase 0: Fire pre_turn hooks ---
                pre_turn_result = await self._fire_pre_turn_hooks()
                if pre_turn_result is not None and pre_turn_result.get("decision") == "deny":
                    self._run_ctx.cancelled = True
                    from wolfharness.messaging import ChatMessage

                    self._final_message = ChatMessage[str](
                        content="",
                        role="assistant",
                        message_id=self._message_id,
                        session_id=self._session_id,
                    )
                    yield StreamCompleteEvent(cancelled=True, message=self._final_message)
                    return
                # Flatten prompts into a list of UserContent items for ACP
                # conversion. String prompts are valid UserContent items. List
                # prompts contain structured content blocks (TextContent,
                # ImageUrl, BinaryContent, etc.) that must be flattened into
                # the top-level sequence.
                flattened_prompts: list[Any] = (
                    flatten_prompts(self._prompts) if self._prompts else [""]
                )
                content = convert_to_acp_content(flattened_prompts)

                # --- Phase 1: Send prompt ---
                try:
                    response = await self._acp_client.prompt(self._session_id, content)
                    self._prompt_response = response
                except asyncio.CancelledError:
                    raise
                except (RequestError, ConnectionError, RuntimeError, ValidationError) as exc:
                    yield RunErrorEvent(
                        message=str(exc),
                        run_id=run_id,
                        agent_name=self._agent_name,
                    )
                    return

                # --- Phase 2: Stream events ---
                step_index = 0
                cumulative_usage = RunUsage()
                try:
                    async for update in self._acp_client.stream_events(response):
                        native_event = acp_to_native_event(
                            update,
                            step_index=step_index,
                            cumulative_usage=cumulative_usage,
                        )
                        if native_event is not None:
                            # Fire advisory tool hooks for tool-related events.
                            # These are advisory — they log and augment but cannot
                            # prevent the external agent from calling tools.
                            match native_event:
                                case ToolCallStartEvent(
                                    tool_name=tn,
                                    raw_input=ti,
                                    tool_call_id=tcid,
                                ):
                                    await self._fire_pre_tool_hooks(tn, ti, tcid)  # ty: ignore[invalid-argument-type]
                                case ToolCallCompleteEvent(
                                    tool_name=tn,
                                    tool_input=ti,
                                    tool_result=tr,
                                    tool_call_id=tcid,
                                ):
                                    await self._fire_post_tool_hooks(
                                        tn,
                                        ti,
                                        tr,
                                        0.0,
                                        tcid,
                                    )
                                case StepUsageEvent(step_usage=su):
                                    cumulative_usage = RunUsage(
                                        input_tokens=cumulative_usage.input_tokens
                                        + su.input_tokens,
                                        output_tokens=cumulative_usage.output_tokens
                                        + su.output_tokens,
                                    )
                                    step_index += 1
                                case _:
                                    pass
                            yield native_event
                except asyncio.CancelledError:
                    raise
                except (RequestError, ConnectionError, RuntimeError, ValueError) as exc:
                    yield RunErrorEvent(
                        message=str(exc),
                        run_id=run_id,
                        agent_name=self._agent_name,
                    )
                    return

                # --- Phase 3: Collect message history ---
                try:
                    raw_updates = await self._acp_client.get_messages(self._session_id)
                except asyncio.CancelledError:
                    raise
                except (RequestError, ConnectionError, ValidationError) as exc:
                    yield RunErrorEvent(
                        message=str(exc),
                        run_id=run_id,
                        agent_name=self._agent_name,
                    )
                    return

                model_messages, final_msg = _convert_updates_to_model_messages(
                    raw_updates,
                    session_id=self._session_id,
                )
                self._message_history = model_messages

                if final_msg is not None:
                    self._final_message = final_msg
                else:
                    from wolfharness.messaging import ChatMessage

                    self._final_message = ChatMessage[str](
                        content="",
                        role="assistant",
                        message_id=self._message_id,
                        session_id=self._session_id,
                    )

                yield StreamCompleteEvent(message=self._final_message)
            finally:
                duration_ms = (time.perf_counter() - turn_start) * 1000
                await self._fire_post_turn_hooks(self._final_message, duration_ms=duration_ms)
