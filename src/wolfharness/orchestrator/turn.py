"""Abstract base class for a single reactive cycle of agent execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai.messages import ModelMessage

    from wolfharness.agents.context import AgentRunContext
    from wolfharness.agents.events.events import RichAgentStreamEvent
    from wolfharness.hooks import AgentHooks
    from wolfharness.hooks.base import HookResult
    from wolfharness.messaging import ChatMessage


class Turn(ABC):
    """Abstract base class for a single reactive cycle of agent execution.

    A Turn encapsulates one complete reactive cycle: receiving input, executing
    through an agent (or agent team), and producing output events. Subclasses
    implement :meth:`execute` to drive the agent loop and yield stream events.

    After execution completes, :attr:`message_history` and :attr:`final_message`
    become available.
    """

    _message_history: list[ModelMessage] | None = None
    """Message history populated after execute() completes."""

    _final_message: ChatMessage[Any] | None = None
    """Final message populated after execute() completes."""

    @abstractmethod
    async def execute(self) -> AsyncGenerator[RichAgentStreamEvent[Any]]:
        """Execute one reactive cycle of agent interaction.

        Yields stream events during execution (text deltas, tool calls,
        lifecycle notifications) and populates ``_message_history`` and
        ``_final_message`` before returning.
        """
        ...  # pragma: no cover
        yield  # type: ignore[misc]  # pragma: no cover  # Makes this an async generator

    @property
    def message_history(self) -> list[ModelMessage]:
        """Return the message history after execute() completes.

        Returns:
            The list of model messages from the completed turn.

        Raises:
            RuntimeError: If accessed before :meth:`execute` completes.
        """
        if self._message_history is None:
            raise RuntimeError("message_history is not available until execute() completes")
        return self._message_history

    @property
    def final_message(self) -> ChatMessage[Any]:
        """Return the final chat message after execute() completes.

        Returns:
            The final :class:`ChatMessage` produced by the turn.

        Raises:
            RuntimeError: If accessed before :meth:`execute` completes.
        """
        if self._final_message is None:
            raise RuntimeError("final_message is not available until execute() completes")
        return self._final_message


class HookAwareTurn:
    """Mixin providing unified hook firing for both native and ACP turns.

    This mixin does **not** inherit from :class:`Turn`. Host classes use it via
    cooperative multiple inheritance::

        class NativeTurn(HookAwareTurn, Turn): ...
        class ACPTurn(HookAwareTurn, Turn): ...

    Host classes must:
    - Set ``self._hooks`` (an :class:`AgentHooks` or ``None``) in ``__init__``
      via a new ``hooks`` parameter.
    - Set ``self._run_ctx`` (an :class:`AgentRunContext`) in ``__init__``.
    - Implement the three abstract properties: :attr:`_hook_env`,
      :attr:`_hook_agent_name`, and :attr:`_hook_prompt`.

    All methods are no-ops when ``self._hooks`` is ``None``.

    Tool execution logging idempotency is tracked via :attr:`_logged_tools`,
    a per-Turn-instance set. A new Turn is created for each turn, so the set
    does not need cross-turn reset.
    """

    _hooks: AgentHooks | None
    """Hooks container, set by host class ``__init__``. ``None`` = no hooks."""

    _run_ctx: AgentRunContext
    """Per-run context, set by host class ``__init__``."""

    _logged_tools: set[str]
    """Per-Turn set of tool log keys already logged to the journal.

    This Turn instance is not reused across turns — _logged_tools does not
    need reset. If Turn instances are ever reused (e.g., for retry), add a
    ``reset()`` method to clear this set.
    """

    def __init__(self) -> None:
        """Initialize the HookAwareTurn mixin.

        Host classes must call ``super().__init__()`` in their own
        ``__init__`` to ensure ``_logged_tools`` is initialized.
        """
        super().__init__()
        self._logged_tools: set[str] = set()

    @property
    @abstractmethod
    def _hook_env(self) -> Any | None:
        """Execution environment for command hooks.

        Host classes return their agent's :class:`ExecutionEnvironment` or
        ``None`` if not applicable (e.g., ACP agents without an env).
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def _hook_agent_name(self) -> str:
        """Agent name passed to hook invocations."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def _hook_prompt(self) -> str:
        """The user prompt for this turn."""
        ...  # pragma: no cover

    async def _fire_pre_turn_hooks(self) -> HookResult | None:
        """Fire pre_turn hooks if not already fired this turn.

        Returns:
            Combined :class:`HookResult`, or ``None`` if hooks are not
            configured or already fired.
        """
        if self._hooks is None:
            return None
        return await self._hooks.run_pre_turn_hooks(
            agent_name=self._hook_agent_name,
            prompt=self._hook_prompt,
            session_id=self._run_ctx.session_id,
            env=self._hook_env,
        )

    async def _fire_post_turn_hooks(
        self, result: ChatMessage[Any] | None, duration_ms: float = 0.0
    ) -> HookResult | None:
        """Fire post_turn hooks if not already fired this turn.

        Must be called in a ``finally`` block by host classes to ensure
        hooks fire even when the turn raises or is cancelled.

        Args:
            result: The final chat message from the turn, or ``None`` if
                the turn failed before producing one.
            duration_ms: Elapsed wall-clock time for this turn in
                milliseconds. Falls back to ``0.0`` when not provided.

        Returns:
            Combined :class:`HookResult`, or ``None`` if hooks are not
            configured or already fired.
        """
        if self._hooks is None:
            return None
        return await self._hooks.run_post_turn_hooks(
            agent_name=self._hook_agent_name,
            prompt=self._hook_prompt,
            result=result,
            session_id=self._run_ctx.session_id,
            env=self._hook_env,
            duration_ms=duration_ms,
        )

    async def _fire_pre_tool_hooks(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> HookResult | None:
        """Fire pre_tool_use hooks for a tool call.

        Args:
            tool_name: Name of the tool being called.
            tool_input: Input arguments for the tool.
            tool_call_id: Unique ID for this tool call, if available.

        Returns:
            Combined :class:`HookResult`, or ``None`` if hooks are not
            configured or already fired for this tool call.
        """
        if self._hooks is None:
            return None
        return await self._hooks.run_pre_tool_hooks(
            agent_name=self._hook_agent_name,
            tool_name=tool_name,
            tool_input=tool_input,
            session_id=self._run_ctx.session_id,
            env=self._hook_env,
        )

    def _log_tool_execution(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        tool_call_id: str | None = None,
        *,
        status: str = "completed",
    ) -> None:
        """Log a tool execution record to the journal for crash recovery.

        Creates a :class:`ToolExecutionRecord` and stores it via
        ``journal.log_tool_execution()``. Uses :attr:`_logged_tools` to
        prevent double-logging within a single Turn instance.

        Skips silently if any of the following are missing:
        - ``run_ctx._run_handle`` (not running inside a pooled session)
        - ``run_handle._journal`` (no journal configured)
        - ``run_ctx.turn_id`` (turn_id not yet set)

        Args:
            tool_name: Name of the tool that was called.
            tool_input: Input arguments that were passed to the tool.
            tool_output: Output from the tool.
            tool_call_id: Unique ID for this tool call, if available.
            status: Execution status (``"completed"``, ``"cancelled"``,
                ``"failed"``, etc.). Defaults to ``"completed"``.
        """
        from wolfharness.lifecycle.types import ToolExecutionRecord

        # Per-Turn idempotency guard using _logged_tools set.
        # This Turn instance is not reused across turns — _logged_tools
        # does not need reset.
        if tool_call_id is not None:
            log_key = f"tool_log:{tool_call_id}"
        else:
            log_key = f"tool_log:{tool_name}"
        if log_key in self._logged_tools:
            return
        self._logged_tools.add(log_key)

        run_handle = self._run_ctx._run_handle
        if run_handle is None:
            return
        session = run_handle.session
        if session is None:
            return
        comm = session._comm_channel
        if comm is None:
            return
        journal = comm.journal
        turn_id = self._run_ctx.turn_id
        if turn_id is None:
            return
        record = ToolExecutionRecord(
            turn_id=turn_id,
            tool_name=tool_name,
            args=tool_input,
            result=tool_output,
            status=status,
        )
        journal.log_tool_execution(record)

    def _log_cancelled_tool_executions(
        self,
        cancelled_events: list[Any],
    ) -> None:
        """Log cancelled tool executions to the journal.

        Called from cancellation paths in ``NativeTurn.execute()`` after
        ``EventMapper.flush_cancelled_tool_calls()`` to ensure the journal
        records ``status="cancelled"`` for tools that were in-progress when
        the turn was cancelled. Without this, crash recovery with the
        ``"retry"`` strategy cannot distinguish between tools that completed
        and tools that were interrupted.

        Args:
            cancelled_events: List of ``ToolCallCompleteEvent`` instances
                returned by ``flush_cancelled_tool_calls()``.
        """
        for event in cancelled_events:
            self._log_tool_execution(
                tool_name=event.tool_name,
                tool_input=event.tool_input,
                tool_output=event.tool_result,
                tool_call_id=event.tool_call_id,
                status="cancelled",
            )

    async def _fire_post_tool_hooks(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        duration_ms: float,
        tool_call_id: str | None = None,
    ) -> HookResult | None:
        """Fire post_tool_use hooks for a completed tool call.

        Also logs the tool execution to the journal for crash recovery,
        independent of whether hooks are configured.

        Args:
            tool_name: Name of the tool that was called.
            tool_input: Input arguments that were passed to the tool.
            tool_output: Output from the tool.
            duration_ms: How long the tool took to execute in milliseconds.
            tool_call_id: Unique ID for this tool call, if available.

        Returns:
            Combined :class:`HookResult`, or ``None`` if hooks are not
            configured or already fired for this tool call.
        """
        # Log tool execution to journal (independent of hooks).
        self._log_tool_execution(tool_name, tool_input, tool_output, tool_call_id)

        if self._hooks is None:
            return None
        return await self._hooks.run_post_tool_hooks(
            agent_name=self._hook_agent_name,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            session_id=self._run_ctx.session_id,
            env=self._hook_env,
        )
