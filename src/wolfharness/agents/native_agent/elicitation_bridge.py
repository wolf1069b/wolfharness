"""Bridge pydantic-ai deferred elicitation calls to AgentPool's durable execution layer.

When an MCP server's elicitation request cannot be resolved immediately
(durable elicitation), the ``HandleDeferredToolCalls`` capability intercepts
the deferred tool call and:

1. Checkpoints the session state via ``CheckpointManager``.
2. Emits an ``ElicitationDeferredEvent`` to the event bus.
3. Registers an ``asyncio.Future`` in ``ElicitationFutureRegistry`` so the
   external elicitation response can later resolve the blocked call.
4. Returns ``None`` — the call remains unresolved (blocked) in pydantic-ai's
   ``FinalResult``, enabling the CheckpointManager to persist state for
   later resumption.

This capability MUST be registered AFTER ``deferred_bridge`` and BEFORE
``approval_bridge`` in the pydantic-ai capability chain.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mcp import types
from pydantic_ai.capabilities import HandleDeferredToolCalls

from wolfharness.agents.events.events import ElicitationDeferredEvent
from wolfharness.log import get_logger
from wolfharness.sessions.models import PendingDeferredCall


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

    from wolfharness.agents.context import AgentContext
    from wolfharness.agents.native_agent.checkpoint import CheckpointManager
    from wolfharness.sessions.models import ElicitationResumePayload


logger = get_logger(__name__)


class ElicitationFutureRegistry:
    """Per-session registry of pending elicitation futures.

    Maps ``deferred_handle`` strings to ``asyncio.Future`` instances so
    that external elicitation responses can resolve blocked tool calls.
    The registry is instantiated per-session and cleaned up on session
    close via ``reject_all()``.

    Lifecycle:
        - ``register()``: Called when a tool call is deferred for elicitation.
        - ``resolve()``: Called when the user responds to the elicitation.
        - ``reject_all()``: Called on session close to reject all pending futures.
    """

    def __init__(self) -> None:
        """Initialize an empty future registry."""
        self._futures: dict[str, asyncio.Future[Any]] = {}

    def __contains__(self, deferred_handle: str) -> bool:
        """Check if a future exists for the given handle.

        Args:
            deferred_handle: Identifier matching a registered future.

        Returns:
            True if a future exists for the handle, False otherwise.
        """
        return deferred_handle in self._futures

    def __len__(self) -> int:
        """Return the number of pending futures."""
        return len(self._futures)

    def register(self, deferred_handle: str) -> asyncio.Future[Any]:
        """Create and store a new future for a deferred elicitation call.

        Args:
            deferred_handle: Identifier matching the pending call's tool_call_id.

        Returns:
            The newly created future that will be resolved when the user
            responds to the elicitation request.

        Raises:
            ValueError: If a future for ``deferred_handle`` already exists.
        """
        if deferred_handle in self._futures:
            msg = f"Future already registered for handle: {deferred_handle}"
            raise ValueError(msg)
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._futures[deferred_handle] = future
        return future

    def resolve(self, deferred_handle: str, response: ElicitationResumePayload) -> None:
        """Resolve a pending elicitation future with the user's response.

        Removes the future from the registry after resolving. If the handle
        is not in the registry, a warning is logged and the call is a no-op.

        Args:
            deferred_handle: Identifier matching the registered future.
            response: The user's elicitation response payload.
        """
        future = self._futures.pop(deferred_handle, None)
        if future is None:
            logger.warning(
                "No pending elicitation future for handle",
                deferred_handle=deferred_handle,
            )
            return
        if not future.done():
            future.set_result(response)
        else:
            logger.warning(
                "Elicitation future already done",
                deferred_handle=deferred_handle,
            )

    def reject_all(self, exception: Exception) -> None:
        """Reject all pending futures with an exception.

        Called on session close to ensure no futures are left unresolved.
        Clears the registry after rejecting all futures.

        Args:
            exception: The exception to set on each pending future.
        """
        for handle, future in self._futures.items():
            if not future.done():
                future.set_exception(exception)
                logger.debug(
                    "Rejected pending elicitation future",
                    deferred_handle=handle,
                )
        self._futures.clear()

    def remove(self, deferred_handle: str) -> None:
        """Remove a future from the registry without resolving it.

        Called in ``finally`` blocks to ensure cleanup on timeout or
        cancellation. If the handle is not in the registry, this is a no-op.

        Args:
            deferred_handle: Identifier matching the registered future.
        """
        self._futures.pop(deferred_handle, None)


def _extract_elicitation_params(
    call_meta: dict[str, Any],
) -> dict[str, Any]:
    """Extract elicitation parameters from deferred call metadata.

    The metadata ``"elicitation"`` key contains a dict with fields like
    ``message``, ``requestedSchema`` (form mode), ``url``/``elicitationId``
    (URL mode), and ``mode``.

    Args:
        call_meta: Per-call metadata from ``DeferredToolRequests.metadata``.

    Returns:
        The elicitation parameters dict, or an empty dict if not present.
    """
    elicitation: Any = call_meta.get("elicitation")
    if isinstance(elicitation, dict):
        return elicitation
    return {}


def _build_pending_call(
    tool_call_id: str,
    tool_name: str,
    elicitation_params: dict[str, Any],
) -> PendingDeferredCall:
    """Build a PendingDeferredCall from elicitation parameters.

    Args:
        tool_call_id: The pydantic-ai tool call identifier.
        tool_name: Name of the tool that was deferred.
        elicitation_params: Extracted elicitation metadata dict.

    Returns:
        A PendingDeferredCall configured for elicitation deferral.
    """
    return PendingDeferredCall(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        deferred_kind="elicitation",
        deferred_strategy="block",
        elicitation_message=elicitation_params.get("message"),
        elicitation_schema=elicitation_params.get("requestedSchema"),
        elicitation_mode=elicitation_params.get("mode"),
        mcp_server_id=elicitation_params.get("mcp_server_id"),
    )


async def _emit_elicitation_event(
    ctx: RunContext[AgentContext[Any]],
    event: ElicitationDeferredEvent,
) -> None:
    """Publish an ElicitationDeferredEvent to the event bus.

    Args:
        ctx: pydantic-ai RunContext with AgentContext as deps.
        event: The elicitation deferred event to emit.
    """
    run_ctx = ctx.deps.run_ctx
    if run_ctx is None:
        logger.debug(
            "No run_ctx available — elicitation event dropped",
            deferred_handle=event.deferred_handle,
        )
        return

    if run_ctx.event_bus is not None:
        await run_ctx.event_bus.publish(run_ctx.session_id, event)
    else:
        logger.warning(
            "No event_bus available — elicitation event dropped",
            deferred_handle=event.deferred_handle,
        )


async def _handle_elicitation_deferred(
    ctx: RunContext[AgentContext[Any]],
    requests: DeferredToolRequests,
    registry: ElicitationFutureRegistry,
    checkpoint_manager: CheckpointManager | None,
    agent_config_hash: str | None,
) -> DeferredToolResults | None:
    """Handle deferred tool calls with elicitation metadata.

    Inspects ``DeferredToolRequests.calls`` for entries whose metadata
    contains ``deferred_kind == "elicitation"``. For matching calls:
    checkpoints the session, emits an ``ElicitationDeferredEvent``,
    registers a future in the registry, and marks the run as checkpointed.

    For non-matching calls: returns ``None`` to pass through to the next
    capability in the chain (typically ``approval_bridge``).

    Args:
        ctx: pydantic-ai RunContext with AgentContext as deps.
        requests: Deferred tool requests from pydantic-ai.
        registry: Per-session future registry for elicitation responses.
        checkpoint_manager: Optional CheckpointManager for state persistence.
        agent_config_hash: Optional hash for detecting config drift on resume.

    Returns:
        ``None`` — elicitation calls remain unresolved (blocked) and
        non-elicitation calls pass through to the next capability.
    """
    run_ctx = ctx.deps.run_ctx
    session_id = run_ctx.session_id if run_ctx is not None else ""

    has_elicitation = False
    pending_calls: list[PendingDeferredCall] = []

    for call in requests.calls:
        call_meta = requests.metadata.get(call.tool_call_id, {})
        if call_meta.get("deferred_kind") != "elicitation":
            continue

        has_elicitation = True
        elicitation_params = _extract_elicitation_params(call_meta)

        logger.info(
            "Elicitation bridge: intercepted deferred elicitation call",
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            mode=elicitation_params.get("mode"),
        )

        pending_call = _build_pending_call(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            elicitation_params=elicitation_params,
        )
        pending_calls.append(pending_call)

        # (b) Emit ElicitationDeferredEvent to the event bus.
        logger.info(
            "Elicitation bridge: emitting ElicitationDeferredEvent",
            tool_call_id=call.tool_call_id,
            session_id=session_id,
        )
        event = ElicitationDeferredEvent(
            deferred_handle=call.tool_call_id,
            message=elicitation_params.get("message", ""),
            requested_schema=elicitation_params.get("requestedSchema", {}),
            mode=elicitation_params.get("mode", ""),
            session_id=session_id,
        )
        await _emit_elicitation_event(ctx, event)

        # (c) Register future in ElicitationFutureRegistry.
        logger.info(
            "Elicitation bridge: registering future in registry",
            tool_call_id=call.tool_call_id,
        )
        future = registry.register(call.tool_call_id)

        # (d) Broadcast the elicitation question to the input provider so
        # the user can reply via the REST endpoint (POST /question/{id}/reply).
        # Without this, the pending question is not registered in
        # _pending_questions_dict and the user gets a 404 when trying to
        # answer — causing the TUI to hang indefinitely.
        provider = ctx.deps.input_provider
        if provider is not None:
            try:
                form_params = types.ElicitRequestFormParams(
                    message=elicitation_params.get("message", ""),
                    requestedSchema=elicitation_params.get("requestedSchema", {}),
                    mode=elicitation_params.get("mode", "form"),
                )
                await provider.broadcast_elicitation_question(
                    call.tool_call_id,
                    form_params,
                    shared_future=future,
                )
            except Exception:
                logger.warning(
                    "Failed to broadcast elicitation question",
                    tool_call_id=call.tool_call_id,
                    exc_info=True,
                )

        logger.debug(
            "Elicitation deferred tool call",
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
        )

    if not has_elicitation:
        # No elicitation calls found — pass through to next capability
        return None

    # (a) Checkpoint session state once with ALL pending calls.
    # Must be after the loop to avoid overwriting the checkpoint with
    # each individual call — only the last call would survive otherwise.
    if checkpoint_manager is not None and run_ctx is not None and pending_calls:
        logger.info(
            "Elicitation bridge: checkpointing session",
            session_id=session_id,
            pending_call_count=len(pending_calls),
        )
        messages: list[ModelMessage] = list(ctx.messages)
        await checkpoint_manager.checkpoint(
            session_id=session_id,
            message_history=messages,
            pending_calls=pending_calls,
            agent_config_hash=agent_config_hash,
        )

    # Mark the run as checkpointed so the orchestrator can transition
    # the RunHandle to checkpointed status.
    if run_ctx is not None:
        run_ctx.checkpointed = True
        # Update session store status to "checkpointed" so resume_session()
        # can find it without relying on the allow_active_run workaround.
        pool = ctx.deps.node.host_context if ctx.deps is not None else None
        if pool is not None and pool.session_pool is not None:
            store = pool.session_pool.sessions.store
            if store is not None:
                try:
                    data = await store.load_session(session_id)
                    if data is not None and data.status == "active":
                        data = data.model_copy(
                            update={
                                "status": "checkpointed",
                                "pending_deferred_calls": pending_calls,
                            }
                        )
                        data.touch()
                        await store.save_session(data)
                except Exception:
                    logger.debug(
                        "Failed to update session status to checkpointed",
                        session_id=session_id,
                        exc_info=True,
                    )

    # Elicitation calls remain unresolved (blocked) — return None so the
    # calls stay in DeferredToolRequests.calls for the next capability.
    # approval_bridge only handles requests.approvals, so these calls
    # will naturally pass through and remain blocked.
    return None


def create_elicitation_bridge_capability(
    registry: ElicitationFutureRegistry,
    checkpoint_manager: CheckpointManager | None = None,
    agent_config_hash: str | None = None,
) -> HandleDeferredToolCalls[AgentContext[Any]]:
    """Create a ``HandleDeferredToolCalls`` capability for elicitation bridging.

    This capability intercepts deferred tool calls with
    ``metadata["deferred_kind"] == "elicitation"`` and checkpoints the
    session, emits an event, and registers a future for later resolution.

    This capability MUST be registered AFTER ``deferred_bridge`` and BEFORE
    ``approval_bridge`` in the pydantic-ai capability chain.

    Args:
        registry: Per-session ElicitationFutureRegistry for tracking
            pending elicitation futures.
        checkpoint_manager: Optional CheckpointManager for persisting
            agent state. If None, checkpointing is skipped.
        agent_config_hash: Optional SHA-256 hash for drift detection
            on resume.

    Returns:
        ``HandleDeferredToolCalls`` capability configured with the
        elicitation bridge handler.
    """

    async def handler(
        ctx: RunContext[AgentContext[Any]],
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        return await _handle_elicitation_deferred(
            ctx,
            requests,
            registry,
            checkpoint_manager,
            agent_config_hash,
        )

    return HandleDeferredToolCalls(handler=handler)
