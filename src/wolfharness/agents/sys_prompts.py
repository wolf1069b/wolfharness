"""System prompt management."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from wolfharness import text_templates
from wolfharness.agents.exceptions import PromptResolutionError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jinjarope import Environment
    from pydantic_ai import RunContext
    from toprompt import AnyPromptType

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.prompts.manager import PromptManager


ToolInjectionMode = Literal["off", "all"]
ToolUsageStyle = Literal["suggestive", "strict"]


@cache
def get_jinja_env() -> Environment:
    from jinjarope import Environment
    from toprompt import to_prompt

    env = Environment(enable_async=True)
    env.filters["to_prompt"] = to_prompt
    return env


class SystemPrompts:
    """Manages system prompts for an agent."""

    def __init__(
        self,
        prompts: AnyPromptType | list[AnyPromptType] | None = None,
        template: str | None = None,
        dynamic: bool = True,
        prompt_manager: PromptManager | None = None,
        inject_agent_info: bool = True,
        inject_tools: ToolInjectionMode = "off",
        tool_usage_style: ToolUsageStyle = "suggestive",
    ) -> None:
        """Initialize prompt manager."""
        match prompts:
            case list():
                self.prompts = prompts
            case None:
                self.prompts = []
            case _:
                self.prompts = [prompts]
        self.prompt_manager = prompt_manager
        self.template = template
        self.dynamic = dynamic
        self.inject_agent_info = inject_agent_info
        self.inject_tools = inject_tools
        self.tool_usage_style = tool_usage_style
        self._cached = False

    def __repr__(self) -> str:
        return (
            f"SystemPrompts(prompts={len(self.prompts)}, "
            f"dynamic={self.dynamic}, inject_agent_info={self.inject_agent_info}, "
            f"inject_tools={self.inject_tools!r})"
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | slice) -> AnyPromptType | list[AnyPromptType]:
        return self.prompts[idx]

    async def add_by_reference(self, reference: str) -> None:
        """Add a system prompt using reference syntax.

        Args:
            reference: [provider:]identifier[@version][?var1=val1,...]

        Examples:
            await sys_prompts.add_by_reference("code_review?language=python")
            await sys_prompts.add_by_reference("langfuse:expert@v2")
        """
        if not self.prompt_manager:
            raise PromptResolutionError("no prompt_manager available")

        try:
            content = await self.prompt_manager.get(reference)
            self.prompts.append(content)  # ty: ignore[invalid-argument-type]
        except Exception as e:
            raise PromptResolutionError(f"failed to add prompt {reference!r}") from e

    async def add(
        self,
        identifier: str,
        *,
        provider: str | None = None,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> None:
        """Add a system prompt.

        Args:
            identifier: Prompt identifier/name
            provider: Provider name (None = builtin)
            version: Optional version string
            variables: Optional template variables

        Examples:
            await sys_prompts.add("code_review", variables={"language": "python"})
            await sys_prompts.add("expert", provider="langfuse", version="v2")
        """
        if not self.prompt_manager:
            raise PromptResolutionError("no prompt_manager available")

        try:
            content = await self.prompt_manager.get_from(
                identifier,
                provider=provider,
                version=version,
                variables=variables,
            )
            self.prompts.append(content)  # ty: ignore[invalid-argument-type]
        except Exception as e:
            ref = f"{provider + ':' if provider else ''}{identifier}"
            raise PromptResolutionError(f"failed to add prompt {ref!r}") from e

    def clear(self) -> None:
        """Clear all system prompts."""
        self.prompts = []

    async def refresh_cache(self) -> None:
        """Force re-evaluation of prompts."""
        from toprompt import to_prompt

        self.prompts = [await to_prompt(prompt) for prompt in self.prompts]
        self._cached = True

    async def format_system_prompt(self, agent: BaseAgent[Any, Any]) -> str:
        """Format complete system prompt."""
        if not self.dynamic and not self._cached:
            await self.refresh_cache()
        env = get_jinja_env()
        template = env.from_string(self.template or text_templates.get_system_prompt())
        # Pre-compute tools list for the template since agent.tools was removed
        tools: list[Any] = []
        if self.inject_tools != "off":
            all_tools = await agent._get_all_tools()
            tools = [t for t in all_tools if t.enabled]
        result = await template.render_async(
            agent=agent,
            prompts=self.prompts,
            dynamic=self.dynamic,
            inject_agent_info=self.inject_agent_info,
            inject_tools=self.inject_tools,
            tool_usage_style=self.tool_usage_style,
            tools=tools,
        )
        return result.strip()

    async def to_pydantic_ai_instructions(
        self,
        agent: BaseAgent[Any, Any],
    ) -> list[str | Callable[[RunContext[Any]], Awaitable[str]]]:
        """Convert system prompts to pydantic-ai compatible instructions.

        Returns a list of instructions where:
        - Static/template prompts are rendered into a string instruction
        - Callable prompts are wrapped for pydantic-ai compatibility

        This allows SystemPrompts to produce instructions that can be passed
        directly to PydanticAgent(instructions=[...]).

        Args:
            agent: The agent to format prompts for

        Returns:
            List of pydantic-ai compatible instructions (strings and/or
            callables accepting RunContext and returning str/Awaitable[str])
        """
        import inspect

        from wolfharness.utils.context_wrapping import wrap_instruction

        instructions: list[str | Callable[[RunContext[Any]], Awaitable[str]]] = []

        # Separate renderable prompts from callable prompts that need wrapping
        renderable_prompts: list[Any] = []
        callable_prompts: list[Any] = []

        for prompt in self.prompts:
            if callable(prompt):
                sig = inspect.signature(prompt)
                param_count = len([
                    p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
                ])
                if param_count == 0:
                    # No-arg callable can be rendered by to_prompt
                    renderable_prompts.append(prompt)
                else:
                    # Callable with params needs pydantic-ai wrapping
                    callable_prompts.append(prompt)
            else:
                renderable_prompts.append(prompt)

        # Format system prompt with only renderable prompts
        original_prompts = self.prompts
        try:
            self.prompts = renderable_prompts
            formatted = await self.format_system_prompt(agent)
            if formatted:
                instructions.append(formatted)
        finally:
            self.prompts = original_prompts

        # Wrap callable prompts for pydantic-ai compatibility
        for prompt in callable_prompts:
            wrapped = wrap_instruction(prompt, fallback="", _warn=False)
            instructions.append(wrapped)

        return instructions
