"""Message-count compaction — trigger compression by message count, not tokens.

DCP's token-based compaction may never trigger because ``[pruned]`` placeholders
are ~1 token each.  A conversation with 252 pruned messages reports a low token
count, so the watermark stays below CRITICAL and ``compact_conversation`` never
fires.  The context fills with useless ``[pruned]`` noise that the token counter
cannot detect.

This capability closes that gap by triggering compaction when the **message
count** exceeds a threshold, regardless of token size.  It runs in
``before_model_request`` alongside DCP and uses ``KeepLastMessages`` to slash
the conversation to the most recent N pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

from wolfharness.messaging.compaction import (
    CompactionPipeline,
    FilterToolCalls,
    KeepLastMessages,
    TruncateToolCallInputs,
    compact_conversation,
)


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.models import ModelRequestContext

    from wolfharness.agents.context import AgentContext


logger = logging.getLogger(__name__)

_EXCLUDE_TOOLS = (
    "team_status",
    "task_list",
    "read_blackboard",
    "read_resource",
    "list_chapters",
    "inspect_wiki_state",
    "inspect_build_checkpoint",
    "send_message",
    "task_create",
    "task_create_batch",
    "task_update",
    "task_get",
    "team_add_member",
    "shutdown_request",
    "prune_tool",
    "distill_tool",
    "decompress_tool",
    "discover_opa",
    "create_opa",
    "create_ops",
    "update_ops",
    "apply_ops",
    "ingest_external_ops",
    "patch_entities_batch",
    "create_opl",
    "resolve_opa",
    "refine_opa_reason_code",
    "ops_dispatch_plan",
)


class MessageCountCompactionCapability(AbstractCapability[Any]):
    """Compact the persistent conversation when message count exceeds a limit.

    Runs independently of DCP's token-watermark compaction.  The pipeline first
    strips noisy tool calls (``FilterToolCalls``) then keeps only the last N
    request-response pairs (``KeepLastMessages``), giving the model a clean,
    recent context window.
    """

    def __init__(
        self,
        *,
        max_messages: int = 40,
        keep_last_pairs: int = 12,
        max_input_length: int = 1500,
    ) -> None:
        self._max_messages = max_messages
        self._pipeline = CompactionPipeline(
            steps=[
                FilterToolCalls(exclude_tools=list(_EXCLUDE_TOOLS)),
                TruncateToolCallInputs(max_length=max_input_length),
                KeepLastMessages(count=keep_last_pairs, count_pairs=True),
            ],
        )

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Compact the persistent conversation if message count exceeds limit."""
        conversation = None
        try:
            conversation = ctx.deps.native_agent.conversation
        except (AttributeError, AssertionError):
            return request_context

        if conversation is None:
            return request_context

        chat_messages = conversation.get_history()
        total_model_msgs = sum(len(cm.messages) if cm.messages else 0 for cm in chat_messages)
        if total_model_msgs <= self._max_messages:
            return request_context

        original, compacted = await compact_conversation(self._pipeline, conversation)
        logger.info(
            "MessageCountCompaction: %d → %d model messages (threshold=%d)",
            original,
            compacted,
            self._max_messages,
        )
        return request_context
