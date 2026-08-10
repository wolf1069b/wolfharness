"""Hook manager for NativeAgent.

Centralizes hook-related logic:
- AgentHooks integration (pre/post tool)
- Injection consumption from PromptInjectionManager
- Combined hook result handling

Tool interception (confirmation, error wrapping, pre/post tool hooks) is
handled by :class:`ToolInterceptCapability` in ``tool_intercept.py``, which
is still required because ``NativeTurn.execute()`` does not call
``HookAwareTurn._fire_pre_tool_hooks()`` / ``_fire_post_tool_hooks()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness.hooks.base import HookResult
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.hooks import AgentHooks

logger = get_logger(__name__)


class NativeAgentHookManager:
    """Manages hooks and injection for NativeAgent.

    Responsibilities:
    - Wraps AgentHooks and delegates pre/post tool hooks to it
    - Consumes injections from PromptInjectionManager (via agent's run context)
    - Combined hook result handling
    """

    def __init__(
        self,
        *,
        agent: BaseAgent[Any, Any],
        agent_hooks: AgentHooks | None = None,
    ) -> None:
        """Initialize hook manager.

        Args:
            agent: The agent instance (for accessing per-run injection manager)
            agent_hooks: Optional AgentHooks for pre/post hooks
        """
        self.agent_name = agent.name
        self.agent_hooks = agent_hooks
        self._agent = agent

    def has_hooks(self) -> bool:
        """Check if any hooks are configured."""
        return bool(self.agent_hooks and self.agent_hooks.has_hooks())

    async def run_pre_tool_hooks(
        self,
        *,
        agent_name: str,
        tool_name: str,
        tool_input: dict[str, Any],
        session_id: str | None = None,
        env: ExecutionEnvironment | None = None,
        agent_context: Any | None = None,
    ) -> HookResult:
        """Execute pre-tool-use hooks.

        Args:
            agent_name: Name of the agent.
            tool_name: Name of the tool being called.
            tool_input: Input arguments for the tool.
            session_id: Optional conversation identifier.
            env: Agent's execution environment, passed to command hooks.
            agent_context: Optional AgentContext for hooks that need pool access.

        Returns:
            Hook result. If decision is "deny", the tool call should be blocked.
        """
        if self.agent_hooks:
            return await self.agent_hooks.run_pre_tool_hooks(
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                session_id=session_id,
                env=env,
                agent_context=agent_context,
            )
        return HookResult(decision="allow")

    async def run_post_tool_hooks(
        self,
        *,
        agent_name: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        duration_ms: float,
        session_id: str | None = None,
        env: ExecutionEnvironment | None = None,
        agent_context: Any | None = None,
    ) -> HookResult:
        """Execute post-tool-use hooks and consume pending injection.

        Combines:
        - Results from ``AgentHooks.run_post_tool_hooks()``
        - Pending injection from ``PromptInjectionManager`` (if any)

        Args:
            agent_name: Name of the agent.
            tool_name: Name of the tool that was called.
            tool_input: Input arguments that were passed to the tool.
            tool_output: Output from the tool.
            duration_ms: How long the tool took.
            session_id: Optional conversation identifier.
            env: Agent's execution environment, passed to command hooks.
            agent_context: Optional AgentContext for hooks that need pool access.

        Returns:
            Combined hook result. May include additional_context from hooks
            and/or pending injection.
        """
        if self.agent_hooks:
            result = await self.agent_hooks.run_post_tool_hooks(
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                duration_ms=duration_ms,
                session_id=session_id,
                env=env,
                agent_context=agent_context,
            )
        else:
            result = HookResult(decision="allow")

        # Consume pending injection from run context (isolated per-call)
        run_ctx = self._agent.get_active_run_context()
        injection_manager = run_ctx.injection_manager if run_ctx else None
        if injection_manager:
            injection = await injection_manager.consume()
            if injection:
                logger.debug(
                    "Consuming injection after tool use",
                    agent=self.agent_name,
                    tool=tool_name,
                    injection_len=len(injection),
                )
                existing_context = result.get("additional_context")
                if existing_context:
                    result["additional_context"] = f"{existing_context}\n\n{injection}"
                else:
                    result["additional_context"] = injection

        return result
