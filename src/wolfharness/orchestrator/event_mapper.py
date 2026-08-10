"""Event mapper for PydanticAI to AgentPool event translation.

Extracts the inline event mapping logic into a reusable,
testable class. Maps PydanticAI stream events to AgentPool
:class:`RichAgentStreamEvent` types.

Mapping rules:
    - ``FunctionToolCallEvent`` → :class:`ToolCallStartEvent`
    - ``PartStartEvent`` with ``BaseToolCallPart`` → :class:`ToolCallStartEvent`
    - ``FunctionToolResultEvent`` → :class:`ToolCallCompleteEvent`
    - pydantic-ai ``PartDeltaEvent`` → AgentPool :class:`PartDeltaEvent` subclass
    - pydantic-ai ``PartStartEvent`` (non-tool) → AgentPool :class:`PartStartEvent` subclass
    - ``EnqueuedMessagesEvent`` → :class:`UserMessageInsertedEvent`
    - Already-mapped :class:`RichAgentStreamEvent` instances pass through.
    - Unknown objects return ``None``.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
import json
import logging
from typing import Any, Literal, cast

from pydantic_ai import (
    BaseToolCallPart,
    BaseToolReturnPart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent as PyAIPartDeltaEvent,
    PartStartEvent as PyAIPartStartEvent,
    RetryPromptPart,
)
from pydantic_ai.messages import (
    ModelRequest,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)


try:
    from pydantic_ai.messages import EnqueuedMessagesEvent
except ImportError:
    EnqueuedMessagesEvent = None  # type: ignore[assignment,misc]

from wolfharness.agents.events.events import (
    PartDeltaEvent,
    PartStartEvent,
    RichAgentStreamEvent,
    ToolCallCompleteEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
    UserMessageInsertedEvent,
)
from wolfharness.tools.base import ToolKind
from wolfharness.utils.pydantic_ai_helpers import safe_args_as_dict


ENQUEUED_MESSAGES_AVAILABLE = EnqueuedMessagesEvent is not None


class EventMapper:
    """Maps PydanticAI stream events to AgentPool RichAgentStreamEvent types.

    Tracks in-progress tool calls by ``tool_call_id`` so that
    :class:`FunctionToolResultEvent` can be correlated with the originating
    :class:`FunctionToolCallEvent` or :class:`PartStartEvent`.

    Attributes:
        tool_kind_map: Optional mapping of tool name to ToolKind string.
            Populate after construction to enable kind lookup.  Defaults to
            empty, in which case all tools receive ``"other"``.
    """

    def __init__(
        self,
        agent_name: str,
        message_id: str,
        *,
        _enqueue_message_ids: list[str] | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._message_id = message_id
        self._pending_tool_calls: dict[str, str] = {}
        self._pending_tool_inputs: dict[str, dict[str, Any]] = {}
        self.tool_kind_map: dict[str, str] = {}
        self._enqueue_message_ids = _enqueue_message_ids if _enqueue_message_ids is not None else []

    def map_event(  # noqa: PLR0911
        self,
        event: Any,
        *,
        current_node_type: str = "unknown",
    ) -> RichAgentStreamEvent[Any] | None:
        """Map a stream event to a RichAgentStreamEvent.

        Dispatches to per-event-type ``handle_*`` methods, borrowed from
        pydantic-ai's UIEventStream pattern.  Pre-mapped
        :class:`RichAgentStreamEvent` instances pass through unchanged.

        Args:
            event: A PydanticAI stream event or an AgentPool event.
            current_node_type: The pydantic-graph node type currently
                executing (e.g. ``"ModelRequestNode"``,
                ``"CallToolsNode"``).  Used by
                :meth:`handle_enqueued_messages` to infer delivery mode.

        Returns:
            Mapped event, the original event if it is already a
            RichAgentStreamEvent, or ``None`` if the event is unrecognized.
        """
        if isinstance(event, FunctionToolCallEvent):
            return self.handle_tool_call(event)
        if isinstance(event, PyAIPartStartEvent):
            return self.handle_part_start(event)
        if isinstance(event, FunctionToolResultEvent):
            return self.handle_tool_result(event)
        if isinstance(event, PyAIPartDeltaEvent):
            return self.handle_part_delta(event)
        if EnqueuedMessagesEvent is not None and isinstance(event, EnqueuedMessagesEvent):
            return self.handle_enqueued_messages(event, current_node_type=current_node_type)

        # Pre-mapped RichAgentStreamEvent instances pass through unchanged.
        if self._is_rich_event(event):
            return event  # type: ignore[no-any-return]

        return None

    def handle_tool_call(
        self,
        event: FunctionToolCallEvent,
    ) -> RichAgentStreamEvent[Any] | None:
        """Handle a ``FunctionToolCallEvent``.

        Delegates to :meth:`_emit_tool_call_start` which deduplicates
        by ``tool_call_id`` and emits a :class:`ToolCallStartEvent`
        (or :class:`ToolCallProgressEvent` for updated args).
        """
        tool_part = event.part
        if isinstance(tool_part, BaseToolCallPart):
            return self._emit_tool_call_start(tool_part)
        return None

    def handle_tool_result(
        self,
        event: FunctionToolResultEvent,
    ) -> RichAgentStreamEvent[Any] | None:
        """Handle a ``FunctionToolResultEvent``.

        Delegates to :meth:`_emit_tool_call_complete` which correlates
        with the originating tool call start by ``tool_call_id``.
        """
        return self._emit_tool_call_complete(event.part)

    def handle_part_start(
        self,
        event: PyAIPartStartEvent,
    ) -> RichAgentStreamEvent[Any] | None:
        """Handle a pydantic-ai ``PartStartEvent``.

        If the part is a ``BaseToolCallPart``, emits a
        :class:`ToolCallStartEvent` (via :meth:`_emit_tool_call_start`).
        Otherwise, converts the pydantic-ai ``PartStartEvent`` to the
        AgentPool :class:`PartStartEvent` subclass so downstream
        isinstance checks work correctly.
        """
        tool_part = event.part
        if isinstance(tool_part, BaseToolCallPart):
            return self._emit_tool_call_start(tool_part)
        if isinstance(event, PartStartEvent):
            return event
        return _normalize_thinking_event(
            PartStartEvent(
                index=event.index,
                part=event.part,
                message_id=self._message_id,
            )
        )

    def handle_part_delta(
        self,
        event: PyAIPartDeltaEvent,
    ) -> RichAgentStreamEvent[Any] | None:
        """Handle a pydantic-ai ``PartDeltaEvent``.

        Converts the pydantic-ai ``PartDeltaEvent`` to the AgentPool
        :class:`PartDeltaEvent` subclass so downstream isinstance checks
        (e.g. EventBus coalescing) work correctly.
        """
        if isinstance(event, PartDeltaEvent):
            return event
        return _normalize_thinking_event(
            PartDeltaEvent(
                index=event.index,
                delta=event.delta,
                message_id=self._message_id,
            )
        )

    def handle_enqueued_messages(
        self,
        event: Any,
        *,
        current_node_type: str,
    ) -> RichAgentStreamEvent[Any] | None:
        """Handle an ``EnqueuedMessagesEvent`` from pydantic-ai.

        Maps to :class:`UserMessageInsertedEvent` with delivery inference
        based on the current node type:

        - ``"ModelRequestNode"`` → ``delivery="steer"`` (mid-model-request)
        - ``"CallToolsNode"`` or ``"End"`` → ``delivery="followup"`` (between turns)
        - Unknown node types default to ``"steer"``

        Extracts text content from ``ModelRequest`` objects containing
        ``UserPromptPart`` instances in ``event.messages``.

        Returns ``None`` if no ``UserPromptPart`` is found.
        """
        messages: tuple[Any, ...] = event.messages
        if not messages:
            return None

        content: str | list[Any] | None = None
        for msg in messages:
            if not isinstance(msg, ModelRequest):
                continue
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = cast(str | list[Any], part.content)
                    break
            if content is not None:
                break

        if content is None:
            return None

        if current_node_type == "ModelRequestNode":
            delivery: Literal["initial", "steer", "followup"] = "steer"
        elif current_node_type in ("CallToolsNode", "End"):
            delivery = "followup"
        else:
            delivery = "steer"

        # Reuse message_id from the FIFO queue (set by steer()/followup()
        # before calling agent_run.enqueue()). This ensures the
        # UserMessageInsertedEvent from EnqueuedMessagesEvent shares the
        # same message_id as the fire-and-forget emission, enabling
        # converter-level dedup.
        if self._enqueue_message_ids:
            message_id = self._enqueue_message_ids.pop(0)
        else:
            # FIFO queue is empty — this EnqueuedMessagesEvent came from
            # pydantic-ai's internal flow (e.g. model retries, tool result
            # processing), not from our steer()/followup(). Drop it to avoid
            # spurious display events with random UUID message IDs.
            return None

        return UserMessageInsertedEvent(
            session_id="",
            message_id=message_id,
            content=content,
            delivery=delivery,
            source="processed",
            timestamp=datetime.now(UTC).timestamp(),
            meta=None,
        )

    def _emit_tool_call_start(
        self,
        tool_part: BaseToolCallPart,
    ) -> ToolCallStartEvent | ToolCallProgressEvent | None:
        """Create a ToolCallStartEvent from a tool call part.

        Returns ``None`` if a start event was already emitted for the same
        ``tool_call_id`` and the args are identical (deduplication).

        If the ``tool_call_id`` is already tracked but the args differ
        (e.g., streaming assembled a more complete version), returns a
        :class:`ToolCallProgressEvent` with ``status="in_progress"`` and
        the updated ``tool_input``.
        """
        call_id = tool_part.tool_call_id
        if call_id in self._pending_tool_calls:
            new_input = safe_args_as_dict(tool_part, default={})
            stored_input = self._pending_tool_inputs.get(call_id, {})
            if new_input == stored_input:
                return None
            self._pending_tool_inputs[call_id] = new_input
            return ToolCallProgressEvent(
                tool_call_id=call_id,
                status="in_progress",
                tool_name=tool_part.tool_name,
                tool_input=new_input,
            )
        tool_name = tool_part.tool_name
        tool_input = safe_args_as_dict(tool_part, default={})
        self._pending_tool_calls[call_id] = tool_name
        self._pending_tool_inputs[call_id] = tool_input
        kind = cast(ToolKind, self.tool_kind_map.get(tool_name, "other"))
        return ToolCallStartEvent(
            tool_call_id=call_id,
            tool_name=tool_name,
            title=f"Executing: {tool_name}",
            kind=kind,
            raw_input=tool_input,
        )

    def _emit_tool_call_complete(
        self,
        tool_return: BaseToolReturnPart | RetryPromptPart,
    ) -> ToolCallCompleteEvent | None:
        """Create a ToolCallCompleteEvent from a tool return part.

        Returns ``None`` if no matching tool call start was seen (i.e. the
        ``tool_call_id`` is not in ``_pending_tool_calls``).

        Note:
            ``RetryPromptPart`` is not a ``BaseToolReturnPart`` but shares
            the ``tool_call_id`` and ``content`` attributes. When the part
            is a ``RetryPromptPart``, ``metadata={"is_error": True}`` is
            set so downstream consumers can distinguish failures from
            successful completions.
        """
        call_id = tool_return.tool_call_id
        tool_name = self._pending_tool_calls.pop(call_id, None)
        if tool_name is None:
            return None
        tool_input = self._pending_tool_inputs.pop(call_id, {})
        is_error = isinstance(tool_return, RetryPromptPart)
        return ToolCallCompleteEvent(
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_input=tool_input,
            tool_result=tool_return.content,
            agent_name=self._agent_name,
            message_id=self._message_id,
            metadata={"is_error": True} if is_error else None,
        )

    def flush_cancelled_tool_calls(self) -> list[ToolCallCompleteEvent]:
        """Generate ``ToolCallCompleteEvent`` for all pending tool calls.

        Called when a turn is cancelled mid-tool-execution to ensure
        downstream consumers receive a completion event for every
        ``ToolCallStartEvent`` that was emitted. Each event carries
        ``metadata={"is_error": True, "cancelled": True}``.

        Returns:
            A list of ``ToolCallCompleteEvent`` instances, one per
            pending tool call.
        """
        events: list[ToolCallCompleteEvent] = []
        for call_id, tool_name in self._pending_tool_calls.items():
            tool_input = self._pending_tool_inputs.get(call_id, {})
            events.append(
                ToolCallCompleteEvent(
                    tool_name=tool_name,
                    tool_call_id=call_id,
                    tool_input=tool_input,
                    tool_result="Tool execution was cancelled.",
                    agent_name=self._agent_name,
                    message_id=self._message_id,
                    metadata={"is_error": True, "cancelled": True},
                ),
            )
        self._pending_tool_calls.clear()
        self._pending_tool_inputs.clear()
        return events

    @staticmethod
    def _is_rich_event(event: object) -> bool:
        """Check if *event* is a RichAgentStreamEvent.

        Both PydanticAI stream events and AgentPool events are dataclasses
        with an ``event_kind`` field.  This check covers both families
        without needing ``isinstance`` against the ``AgentStreamEvent``
        union (which is a ``typing.Annotated`` and cannot be used with
        ``isinstance`` at runtime).
        """
        if dataclasses.is_dataclass(event):
            return any(f.name == "event_kind" for f in dataclasses.fields(event))
        return False


def _extract_raw_content_text(
    provider_details: Any,
) -> str | None:
    """Extract reasoning text from provider_details['raw_content'].

    For raw CoT providers (vLLM, LM Studio, litellm), pydantic-ai stores
    reasoning text in ``provider_details['raw_content']`` instead of
    ``ThinkingPart.content``.  This function extracts the latest delta
    text from either a dict or a callable provider_details.

    Args:
        provider_details: A dict, a callable, or None.

    Returns:
        The extracted text, or None if no text could be extracted.
    """
    resolved: dict[str, Any] | None = None
    if callable(provider_details):
        try:
            result = provider_details(None)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(result, dict):
            resolved = result
    elif isinstance(provider_details, dict):
        resolved = provider_details

    if resolved is None:
        return None

    raw = resolved.get("raw_content")
    if not raw or not isinstance(raw, list):
        return None
    text = raw[-1]
    if not text or not isinstance(text, str):
        return None
    return text


def _normalize_thinking_event(
    event: PartStartEvent | PartDeltaEvent,
) -> PartStartEvent | PartDeltaEvent:
    """Normalize ThinkingPart/ThinkingPartDelta events from raw CoT providers.

    When ``content``/``content_delta`` is empty/None and
    ``provider_details`` contains ``raw_content``, populate
    ``content``/``content_delta`` from the raw reasoning text so that
    protocol converters can read it directly.

    This handles pydantic-ai's by-design behavior where raw CoT providers
    (vLLM, LM Studio, litellm bridge, gpt-oss via OpenRouter) store
    reasoning in ``provider_details['raw_content']`` instead of
    ``ThinkingPart.content``.

    Events with populated ``content``/``content_delta`` are returned
    unchanged.
    """
    match event:
        case PartStartEvent(part=ThinkingPart(content="") as part):
            text = _extract_raw_content_text(part.provider_details)
            if text:
                new_part = dataclasses.replace(part, content=text)
                return dataclasses.replace(event, part=new_part)
        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=None) as delta):
            text = _extract_raw_content_text(delta.provider_details)
            if text:
                new_delta = dataclasses.replace(delta, content_delta=text)
                return dataclasses.replace(event, delta=new_delta)
    return event


def normalize_thinking_parts_in_messages(
    messages: list[Any],
) -> None:
    """Normalize ``ThinkingPart`` instances in a message list in-place.

    After streaming completes, pydantic-ai's ``StreamedResponse.get()`` builds
    a final ``ModelResponse`` from ``_parts_manager.get_parts()``.  For raw
    CoT providers (vLLM, LM Studio, litellm bridge, gpt-oss via OpenRouter),
    the resulting ``ThinkingPart`` has ``content=""`` while the actual
    reasoning text lives in ``provider_details['raw_content']``.

    The streaming ``_normalize_thinking_event`` fixes individual stream events
    but never touches the assembled ``ModelResponse`` stored in
    ``message_history``.  This function walks every ``ModelResponse`` in the
    list and copies ``raw_content`` text into ``content`` for each affected
    ``ThinkingPart``, ensuring downstream consumers (OTel, next-round
    requests, protocol converters) see the reasoning text.

    This is idempotent: parts with non-empty ``content`` are left unchanged,
    and parts without ``raw_content`` are skipped.

    Args:
        messages: A list of pydantic-ai messages (e.g. from
            ``agent_run.all_messages()``).  Only ``ModelResponse`` messages
            with ``ThinkingPart`` instances are affected; other messages
            pass through untouched.
    """
    from pydantic_ai.messages import ModelResponse

    for idx, msg in enumerate(messages):
        if not isinstance(msg, ModelResponse):
            continue
        new_parts: list[Any] | None = None
        for i, part in enumerate(msg.parts):
            if not isinstance(part, ThinkingPart) or part.content:
                continue
            # The final ModelResponse holds ALL accumulated chunks in
            # provider_details['raw_content'] (unlike streaming deltas where
            # each event's list holds only the latest incremental chunk), so
            # join the full list rather than taking the last entry.
            pd = part.provider_details
            resolved = pd(None) if callable(pd) else pd
            if not isinstance(resolved, dict):
                continue
            raw = resolved.get("raw_content")
            if not raw or not isinstance(raw, list):
                continue
            text = "".join(chunk for chunk in raw if isinstance(chunk, str))
            if not text:
                continue
            if new_parts is None:
                new_parts = list(msg.parts)
            new_parts[i] = dataclasses.replace(part, content=text)
        if new_parts is not None:
            messages[idx] = dataclasses.replace(msg, parts=new_parts)


_logger = logging.getLogger(__name__)


def _extract_first_json_object(s: str) -> str | None:
    """Extract the first valid JSON object from a string that may contain concatenated JSONs.

    Uses a brace-depth scanner to find the boundary of the first top-level ``{...}``
    object, then validates it parses cleanly.  Returns ``None`` if the string is
    already valid JSON or if no valid first object can be extracted.
    """
    s = s.strip()
    if not s or s[0] != "{":
        return None

    # Fast path: already valid JSON — no repair needed.
    try:
        json.loads(s)
    except json.JSONDecodeError:
        pass
    else:
        return None

    # Scan for the first complete top-level JSON object using brace depth.
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[: i + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                else:
                    return candidate
    return None


def sanitize_tool_call_args_in_messages(
    messages: list[Any],
) -> None:
    """Repair duplicated tool call arguments in message history in-place.

    Some inference backends (vLLM with the ``glm47`` parser, SGLang with GLM
    detectors) have a known streaming bug where tool call arguments are emitted
    twice, producing concatenated JSON like::

        {"path": "/foo"}{"path": "/foo"}

    This corrupts downstream model requests (HTTP 400 "Extra data") and tool
    execution.  This function walks every ``ModelResponse`` in the list, checks
    each ``BaseToolCallPart`` whose ``args`` is a ``str``, and — when the string
    contains concatenated JSON objects — replaces it with just the first valid
    object.

    This is idempotent: parts with valid JSON args, ``dict`` args, or ``None``
    args are left unchanged.

    See: vllm-project/vllm#47504, vllm-project/vllm#44098,
    sgl-project/sglang#23071, sgl-project/sglang#16371.

    Args:
        messages: A list of pydantic-ai messages (e.g. from
            ``agent_run.all_messages()``).  Only ``ModelResponse`` messages
            with ``BaseToolCallPart`` instances whose ``args`` is a corrupted
            JSON string are affected; other messages pass through untouched.
    """
    from pydantic_ai.messages import ModelResponse

    for msg in messages:
        if not isinstance(msg, ModelResponse):
            continue
        needs_repair = False
        new_parts = list(msg.parts)
        for i, part in enumerate(new_parts):
            if not isinstance(part, BaseToolCallPart):
                continue
            if not isinstance(part.args, str):
                continue
            repaired = _extract_first_json_object(part.args)
            if repaired is None:
                continue
            _logger.warning(
                "Repaired duplicated tool call arguments",
                extra={
                    "tool_name": part.tool_name,
                    "tool_call_id": part.tool_call_id,
                    "original_len": len(part.args),
                    "repaired_len": len(repaired),
                },
            )
            needs_repair = True
            new_parts[i] = dataclasses.replace(part, args=repaired)
        if needs_repair:
            # Assign the full list so the mutation is visible to any reference
            # holding the same ModelResponse (e.g. a CallToolsNode's
            # ``model_response`` attribute).  ModelResponse is a non-frozen
            # dataclass, so attribute assignment is safe.
            msg.parts = new_parts
