"""Prompt injection manager for agents.

Provides unified handling for immediate injection (consumed by agent
hooks mid-run).
"""

from __future__ import annotations

from wolfharness.log import get_logger


logger = get_logger(__name__)


class PromptInjectionManager:
    """Manages prompt injection for agents.

    This class handles immediate injections consumed by agent hooks
    during a run. When a tool executes, the hook consumes the injection
    and adds it as additional context.
    """

    def __init__(self) -> None:
        """Initialize the injection manager."""
        self._pending_injections: list[str] = []

    def inject(self, message: str) -> None:
        """Queue a message for immediate injection.

        The message will be consumed by the next tool hook (if supported).

        Args:
            message: Message to inject
        """
        self._pending_injections.append(message)
        logger.debug("Queued injection", message_len=len(message))

    async def consume(self) -> str | None:
        """Consume the next pending injection.

        Called by agent-specific hooks (e.g., post-tool hooks) to get
        the next message to inject into the conversation. The message
        is wrapped in XML tags for clear delineation.

        Returns:
            The next injection message wrapped in XML tags, or None if none pending
        """
        if self._pending_injections:
            msg = self._pending_injections.pop(0)
            logger.debug("Consumed injection", message_len=len(msg))
            return f"<injected-context>\n{msg}\n</injected-context>"
        return None

    async def consume_all(self) -> list[str]:
        """Consume all pending injections.

        Returns:
            List of all pending injection messages wrapped in XML tags (may be empty)
        """
        result = [f"<injected-context>\n{i}\n</injected-context>" for i in self._pending_injections]
        self._pending_injections.clear()
        if result:
            logger.debug("Consumed all injections", count=len(result))
        return result

    def has_pending(self) -> bool:
        """Check if there are pending injections."""
        return bool(self._pending_injections)

    def clear(self) -> None:
        """Clear all pending injections.

        Called when run_stream exits (normally, cancelled, or on error).
        """
        self._pending_injections.clear()

    def __repr__(self) -> str:
        return f"PromptInjectionManager(pending={len(self._pending_injections)})"
