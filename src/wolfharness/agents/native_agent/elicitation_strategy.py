"""Elicitation resolution strategies for durable tool call deferral.

Defines the abstract strategy protocol and concrete implementations for
resolving deferred elicitation calls. Strategies determine how elicitation
state is persisted and retrieved when a tool call is deferred pending
external user input.

Strategies:
    CheckpointResolutionStrategy - Persists state via CheckpointManager.
    ProtocolResolutionStrategy - Placeholder for future MRTR/SEP-2663 support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from wolfharness.agents.native_agent.checkpoint import CheckpointManager
    from wolfharness.sessions.models import ElicitationResumePayload, PendingDeferredCall


@runtime_checkable
class ElicitationResolutionStrategy(Protocol):
    """Abstract strategy for resolving deferred elicitation calls.

    Implementations determine how elicitation state is persisted and
    how responses are processed when a deferred tool call receives
    user input.

    Implementations must be async and return the resolved value
    that will be injected back into the agent's tool result stream.
    """

    async def resolve(
        self,
        pending_call: PendingDeferredCall,
        response: ElicitationResumePayload,
    ) -> Any:
        """Resolve a deferred elicitation call with a user response.

        Args:
            pending_call: The deferred tool call awaiting resolution.
            response: The user's elicitation response (accept/decline/cancel).

        Returns:
            The resolved value to inject into the tool result stream.
        """
        ...


class CheckpointResolutionStrategy:
    """Resolution strategy that persists elicitation state via CheckpointManager.

    Delegates state persistence to CheckpointManager.checkpoint(), ensuring
    the deferred call and response are durably stored before the agent
    runtime proceeds. On resolve, the checkpoint is updated to reflect
    the resolved state of the pending call.

    Attributes:
        _checkpoint_manager: The manager responsible for state persistence.
        _session_id: Session identifier for checkpoint scoping.
        _message_history: Current pydantic-ai message history at resolution time.
        _agent_config_hash: Optional hash for detecting config drift on resume.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        session_id: str,
        message_history: list[ModelMessage],
        agent_config_hash: str | None = None,
    ) -> None:
        """Initialize the checkpoint resolution strategy.

        Args:
            checkpoint_manager: CheckpointManager for persisting agent state.
            session_id: Session identifier for checkpoint scoping.
            message_history: Current pydantic-ai message history.
            agent_config_hash: Optional SHA-256 hash for drift detection.
        """
        self._checkpoint_manager = checkpoint_manager
        self._session_id = session_id
        self._message_history = message_history
        self._agent_config_hash = agent_config_hash

    async def resolve(
        self,
        pending_call: PendingDeferredCall,
        response: ElicitationResumePayload,
    ) -> Any:
        """Resolve a deferred elicitation call by checkpointing state.

        Persists the current agent state (including the pending call)
        via CheckpointManager so the session can be resumed if needed.
        The response is returned for injection into the tool result stream.

        Args:
            pending_call: The deferred tool call awaiting resolution.
            response: The user's elicitation response.

        Returns:
            The elicitation response content for tool result injection.
        """
        await self._checkpoint_manager.checkpoint(
            session_id=self._session_id,
            message_history=self._message_history,
            pending_calls=[pending_call],
            agent_config_hash=self._agent_config_hash,
        )
        return response


class ProtocolResolutionStrategy:
    """Placeholder strategy for future MRTR (Model Request Token Resume) support.

    This strategy will implement resolution via the MRTR protocol (SEP-2663),
    allowing elicitation responses to be processed through a standardized
    protocol-level mechanism rather than application-level checkpointing.

    !!! note
        Not yet implemented. MRTR protocol support is tracked in SEP-2663
        and will be added in a future release.
    """

    async def resolve(
        self,
        pending_call: PendingDeferredCall,
        response: ElicitationResumePayload,
    ) -> Any:
        """Resolve a deferred elicitation call via MRTR protocol.

        Args:
            pending_call: The deferred tool call awaiting resolution.
            response: The user's elicitation response.

        Raises:
            NotImplementedError: MRTR support is not yet available.
        """
        raise NotImplementedError("MRTR support not yet available")
