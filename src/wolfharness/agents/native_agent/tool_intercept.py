"""Tool interception capability for NativeAgent.

Provides uniform tool interception across all tool sources (direct tools,
MCP tools, ACP MCP tools) through pydantic-ai's ``AbstractCapability`` chain.

Owns:
- ``get_wrapper_toolset``: confirmation mode via ``ApprovalRequiredToolset``
- ``prepare_tools``: schema modification (placeholder for future use)
- ``wrap_tool_execute``: error handling with failure annotation
- ``before_tool_execute``: pre-tool hooks + deny via ``ModelRetry``
- ``after_tool_execute``: post-tool hooks + result modification + injection

.. note::
    This capability is still required because ``NativeTurn.execute()`` does
    not call ``HookAwareTurn._fire_pre_tool_hooks()`` / ``_fire_post_tool_hooks()``.
    Once NativeTurn fires tool hooks directly, this class can be removed.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelRetry
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ToolRetryError

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext, ToolDefinition
    from pydantic_ai.toolsets import AbstractToolset

    from wolfharness.agents.native_agent.hook_manager import NativeAgentHookManager

logger = get_logger(__name__)


@dataclass
class ToolInterceptCapability(AbstractCapability[Any]):
    """Unified tool interception capability.

    Provides uniform tool interception across all tool sources (direct tools,
    MCP tools, ACP MCP tools) through pydantic-ai's ``AbstractCapability`` chain.
    """

    _: KW_ONLY
    hook_manager: NativeAgentHookManager
    id: str | None = None
    description: str | None = None
    defer_loading: bool = False

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
        """Wrap the assembled toolset with ``ApprovalRequiredToolset`` based on mode.

        Reads ``tool_confirmation_mode`` from the agent's node config:
        - ``"always"``: all tools require approval
        - ``"never"``: no wrapper (tools execute directly)
        - ``"per_tool"``: only tools with ``requires_confirmation=True``
        """
        from pydantic_ai.toolsets import ApprovalRequiredToolset

        mode = self._get_confirmation_mode()

        if mode == "never":
            return None

        if mode == "always":
            return ApprovalRequiredToolset(
                wrapped=toolset,
                approval_required_func=lambda *_: True,
            )

        # mode == "per_tool": check each tool's requires_confirmation flag
        confirm_tool_names = self._get_confirmation_tool_names()

        def _check_per_tool(
            ctx: RunContext[Any],
            tool_def: ToolDefinition,
            tool_args: dict[str, Any],
        ) -> bool:
            return tool_def.name in confirm_tool_names

        return ApprovalRequiredToolset(
            wrapped=toolset,
            approval_required_func=_check_per_tool,
        )

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Modify tool definitions before the model sees them.

        Currently a pass-through. Reserved for future schema modification.
        """
        return tool_defs

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """Wrap tool execution with error handling.

        Catches exceptions and returns annotated ``ToolReturn`` with failure
        details, enabling the model to recover or try alternatives.
        """
        from time import perf_counter

        from pydantic_ai.messages import ToolReturn

        from wolfharness.tools.base import ToolResult

        agent_ctx = ctx.deps
        tool_start_times: dict[str, float] | None = getattr(agent_ctx, "_tool_start_times", None)
        if tool_start_times is None:
            tool_start_times = {}
            agent_ctx._tool_start_times = tool_start_times
        tool_start_times[call.tool_call_id] = perf_counter()

        from wolfharness.tasks.exceptions import RunAbortedError, ToolSkippedError

        try:
            result = await handler(args)
        except (
            # Pydantic-AI control-flow exceptions — must propagate to
            # _run_execute_hooks and the framework's retry/defer/approval logic.
            CallDeferred,
            ApprovalRequired,
            ToolRetryError,
            ModelRetry,
            # AgentPool control-flow exceptions — must propagate to
            # NativeTurn.execute()'s except handlers.
            RunAbortedError,
            ToolSkippedError,
        ):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Tool execution failed",
                tool_name=call.tool_name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return ToolReturn(
                return_value=f"Error: {exc}",
                content=f"Tool '{call.tool_name}' failed: {exc}",
            )

        # Convert AgentPool ToolResult to pydantic-ai ToolReturn
        if isinstance(result, ToolResult):
            val = result.structured_content or result.content
            result = ToolReturn(
                return_value=val,
                content=result.content,
                metadata=result.metadata,
            )

        return result

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Execute pre-tool hooks and handle deny.

        Runs pre-tool hooks from ``AgentHooks``. If a hook denies the tool
        call, raises ``ModelRetry`` to ask the model to try a different
        approach.
        """
        from pydantic_ai import ModelRetry

        from wolfharness.agents.context import AgentContext

        agent_ctx = ctx.deps
        env = agent_ctx.agent.env if isinstance(agent_ctx, AgentContext) else None
        session_id = (
            agent_ctx.run_ctx.session_id
            if isinstance(agent_ctx, AgentContext) and agent_ctx.run_ctx
            else None
        )

        hook_result = await self.hook_manager.run_pre_tool_hooks(
            agent_name=self.hook_manager.agent_name,
            tool_name=call.tool_name,
            tool_input=dict(args),
            session_id=session_id,
            env=env,
            agent_context=agent_ctx,
        )

        if hook_result["decision"] == "deny":
            reason = hook_result.get("reason", "Blocked by pre-tool hook")
            raise ModelRetry(f"Tool '{call.tool_name}' blocked: {reason}")

        # Apply modified input if provided
        if modified := hook_result.get("modified_input"):
            return {**dict(args), **modified}

        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Execute post-tool hooks, apply modifications, and consume injections.

        Runs post-tool hooks from ``AgentHooks`` and explicitly applies:
        - ``modified_output``: replaces the tool result entirely
        - ``additional_context``: appended to the tool result

        Also consumes pending prompt injections from
        ``PromptInjectionManager``.
        """
        from wolfharness.agents.context import AgentContext
        from wolfharness.agents.native_agent.tool_wrapping import (
            _inject_additional_context,
        )

        agent_ctx = ctx.deps
        env = agent_ctx.agent.env if isinstance(agent_ctx, AgentContext) else None
        session_id = (
            agent_ctx.run_ctx.session_id
            if isinstance(agent_ctx, AgentContext) and agent_ctx.run_ctx
            else None
        )

        from time import perf_counter

        tool_start_times: dict[str, float] | None = getattr(agent_ctx, "_tool_start_times", None)
        if tool_start_times is not None:
            start_time = tool_start_times.pop(call.tool_call_id, None)
            duration_ms = (perf_counter() - start_time) * 1000 if start_time else 0.0
        else:
            duration_ms = 0.0

        hook_result = await self.hook_manager.run_post_tool_hooks(
            agent_name=self.hook_manager.agent_name,
            tool_name=call.tool_name,
            tool_input=dict(args),
            tool_output=result,
            duration_ms=duration_ms,
            session_id=session_id,
            env=env,
            agent_context=agent_ctx,
        )

        # Apply modified_output (replaces result entirely)
        if "modified_output" in hook_result:
            result = hook_result["modified_output"]

        # Apply additional_context (appended to result)
        if additional := hook_result.get("additional_context"):
            result = _inject_additional_context(result, additional)

        # Consume pending injection from PromptInjectionManager
        run_ctx = self.hook_manager._agent.get_active_run_context()
        injection_manager = run_ctx.injection_manager if run_ctx else None
        if injection_manager:
            injection = await injection_manager.consume()
            if injection:
                logger.debug(
                    "Consuming injection after tool use",
                    agent=self.hook_manager.agent_name,
                    tool=call.tool_name,
                    injection_len=len(injection),
                )
                result = _inject_additional_context(result, injection)

        return result

    def _get_confirmation_mode(self) -> str:
        """Read tool_confirmation_mode from the agent's node config."""
        run_ctx = self.hook_manager._agent.get_active_run_context()
        if run_ctx is not None:
            node = run_ctx.node if hasattr(run_ctx, "node") else None
            if node is not None:
                return str(node.tool_confirmation_mode)  # ty: ignore[unresolved-attribute]
        agent = self.hook_manager._agent
        if hasattr(agent, "tool_confirmation_mode"):
            return str(agent.tool_confirmation_mode)
        return "per_tool"

    def _get_confirmation_tool_names(self) -> set[str]:
        """Get the set of tool names that require confirmation."""
        agent = self.hook_manager._agent
        tools = agent._builtin_provider._tools
        return {t.name for t in tools if t.requires_confirmation}
