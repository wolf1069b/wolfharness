"""TriggerSource implementations: ImmediateTrigger, ProtocolTrigger, and stubs.

The TriggerSource dimension defines how prompts arrive at the RunLoop.
Four implementations are defined here:

- ``ImmediateTrigger`` — single-prompt delivery for standalone execution.
- ``ProtocolTrigger`` — bridges protocol handlers (ACP, OpenCode, AG-UI,
  OpenAI API) to the RunLoop via an internal ``asyncio.Queue``.
- ``ScheduledTrigger`` — stub; triggers on a schedule (cron or interval).
  Implementation is deferred beyond M2.
- ``ChannelTrigger`` — stub; listens on external channels (Telegram,
  Discord, Slack, webhook). Implementation is deferred beyond M2.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from wolfharness.lifecycle.types import Prompt


class ImmediateTrigger:
    """Delivers a single prompt immediately, then returns ``None``.

    The default TriggerSource for standalone ``agent.run()`` execution.
    ``subscribe()`` and ``close()`` are no-ops because the prompt is
    already set in the constructor.

    Attributes:
        _prompt: The prompt content string provided at construction.
        _delivered: Whether the single prompt has already been polled.
    """

    def __init__(self, prompt: str) -> None:
        """Initialize with the prompt content.

        Args:
            prompt: The prompt text to deliver on the first ``poll()`` call.
        """
        self._prompt: str = prompt
        self._delivered: bool = False

    def subscribe(self, run_loop: Any) -> None:
        """No-op — the prompt is already set in the constructor.

        Args:
            run_loop: The RunLoop instance (unused).
        """

    def poll(self) -> Prompt | None:
        """Return the ``Prompt`` on the first call, ``None`` thereafter.

        Returns:
            A ``Prompt`` with the constructor-provided content, or ``None``
            if the prompt was already delivered.
        """
        if self._delivered:
            return None
        self._delivered = True
        from wolfharness.lifecycle.types import Prompt

        return Prompt(content=self._prompt)

    def close(self) -> None:
        """No-op — no resources to release."""


class ProtocolTrigger:
    """Bridges protocol handlers to the RunLoop via an ``asyncio.Queue``.

    Protocol handlers (ACP, OpenCode, AG-UI, OpenAI API) call ``deliver()``
    to enqueue prompts. The RunLoop calls ``poll()`` to dequeue them
    non-blockingly.

    Attributes:
        _queue: Internal asyncio queue holding delivered prompts.
        _run_loop: Reference to the RunLoop, set by ``subscribe()``.
    """

    def __init__(self) -> None:
        """Initialize with an empty internal queue."""
        self._queue: asyncio.Queue[Prompt] = asyncio.Queue()
        self._run_loop: Any = None

    def subscribe(self, run_loop: Any) -> None:
        """Store a reference to the RunLoop.

        Args:
            run_loop: The RunLoop instance to attach to.
        """
        self._run_loop = run_loop

    async def deliver(self, content: str, priority: str = "normal") -> None:
        """Enqueue a prompt for delivery to the RunLoop.

        Args:
            content: The prompt text to deliver.
            priority: Delivery priority (``"normal"`` or ``"asap"``).
        """
        from wolfharness.lifecycle.types import Prompt

        prompt = Prompt(content=content, priority=priority)
        await self._queue.put(prompt)

    def poll(self) -> Prompt | None:
        """Non-blockingly dequeue the next prompt.

        Returns:
            The next ``Prompt`` if available, or ``None`` if the queue
            is empty.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        """Cancel the internal queue to prevent further deliveries."""
        # Drain remaining items and mark queue as done.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break


class ScheduledTrigger:
    """Stub: triggers the RunLoop on a schedule (cron or interval).

    Renders a prompt from a Jinja2 template on each trigger. This
    implementation is deferred beyond M2. All methods raise
    ``NotImplementedError``.

    Attributes:
        config: Configuration dict storing schedule and template settings.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Store configuration for deferred implementation.

        Args:
            config: Configuration dict with schedule and template settings.
        """
        self.config: dict[str, Any] = config or {}

    def subscribe(self, run_loop: Any) -> None:
        """Not implemented — deferred beyond M2.

        Args:
            run_loop: The RunLoop instance (unused).
        """
        msg = "ScheduledTrigger is not implemented yet (deferred beyond M2)."
        raise NotImplementedError(msg)

    def poll(self) -> Prompt | None:
        """Not implemented — deferred beyond M2."""
        msg = "ScheduledTrigger is not implemented yet (deferred beyond M2)."
        raise NotImplementedError(msg)

    def close(self) -> None:
        """Not implemented — deferred beyond M2."""
        msg = "ScheduledTrigger is not implemented yet (deferred beyond M2)."
        raise NotImplementedError(msg)


class ChannelTrigger:
    """Stub: triggers the RunLoop on external channel messages.

    Listens on an external channel (Telegram, Discord, Slack, webhook)
    for incoming messages and delivers them as prompts. This
    implementation is deferred beyond M2. All methods raise
    ``NotImplementedError``.

    Attributes:
        config: Configuration dict storing channel and listener settings.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Store configuration for deferred implementation.

        Args:
            config: Configuration dict with channel and listener settings.
        """
        self.config: dict[str, Any] = config or {}

    def subscribe(self, run_loop: Any) -> None:
        """Not implemented — deferred beyond M2.

        Args:
            run_loop: The RunLoop instance (unused).
        """
        msg = "ChannelTrigger is not implemented yet (deferred beyond M2)."
        raise NotImplementedError(msg)

    def poll(self) -> Prompt | None:
        """Not implemented — deferred beyond M2."""
        msg = "ChannelTrigger is not implemented yet (deferred beyond M2)."
        raise NotImplementedError(msg)

    def close(self) -> None:
        """Not implemented — deferred beyond M2."""
        msg = "ChannelTrigger is not implemented yet (deferred beyond M2)."
        raise NotImplementedError(msg)


__all__ = [
    "ChannelTrigger",
    "ImmediateTrigger",
    "ProtocolTrigger",
    "ScheduledTrigger",
]
