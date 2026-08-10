"""Instruction function types and protocols for dynamic prompt generation.

This module defines the type system for instruction functions that can be used
to generate prompts dynamically based on runtime context.

Instruction functions can be:
- Simple: No context parameters
- AgentContext: Takes only AgentContext
- RunContext: Takes only RunContext (from pydantic-ai)
- Both: Takes both AgentContext and RunContext
- PydanticAI: Takes RunContext[AgentContext[Any]] — native pydantic-ai signature

The InstructionFunc union type accepts any of these variants, allowing
flexible prompt generation based on what context is available.

For pydantic-ai compatibility, the preferred signature is:
    def instruction(ctx: RunContext[AgentContext[Any]]) -> str:
        agent_ctx = ctx.deps
        return f"Using model: {agent_ctx.model_name}"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from pydantic_ai import RunContext

    from wolfharness.agents.context import AgentContext


# Import runtime_checkable for protocol instance checking
from typing import runtime_checkable


# Protocol definitions for type safety
@runtime_checkable
class SimpleInstruction(Protocol):
    """Instruction function with no context.

    Functions matching this protocol take no parameters and return
    either a string directly or an awaitable string.
    """

    def __call__(self) -> str | Awaitable[str]: ...


@runtime_checkable
class AgentContextInstruction(Protocol):
    """Instruction function with AgentContext only.

    Functions matching this protocol receive AgentContext, which provides
    access to agent-specific runtime information like the current tool,
    model name, conversation history, and filesystem access.

    Useful when you need access to agent-level context but don't need
    the PydanticAI run context.
    """

    def __call__(self, ctx: AgentContext[Any]) -> str | Awaitable[str]: ...


@runtime_checkable
class RunContextInstruction(Protocol):
    """Instruction function with RunContext only.

    Functions matching this protocol receive RunContext from PydanticAI,
    which provides access to dependencies and other PydanticAI-specific
    runtime information.

    Useful when you need access to PydanticAI's dependency injection system
    but don't need AgentPool's agent context.
    """

    def __call__(self, ctx: RunContext[Any]) -> str | Awaitable[str]: ...


@runtime_checkable
class BothContextsInstruction(Protocol):
    """Instruction function with both AgentContext and RunContext.

    Functions matching this protocol receive both context objects, providing
    maximum flexibility for prompt generation.

    Use this when you need access to both AgentPool's agent context and
    PydanticAI's run context simultaneously.
    """

    def __call__(
        self,
        agent_ctx: AgentContext[Any],
        run_ctx: RunContext[Any],
    ) -> str | Awaitable[str]: ...


@runtime_checkable
class PydanticAIInstruction(Protocol):
    """Instruction function compatible with pydantic-ai Agent.instructions.

    Functions matching this protocol receive RunContext with AgentContext as deps,
    which is the native pydantic-ai instruction signature. This allows instruction
    functions to be passed directly to PydanticAgent(instructions=[func]).

    The AgentContext can be accessed via ctx.deps, since AgentPool passes
    AgentContext[TDeps] as the deps type when creating the PydanticAgent.
    """

    def __call__(self, ctx: RunContext[AgentContext[Any]]) -> str | Awaitable[str]: ...


# Union type for all instruction function variants
InstructionFunc = (
    SimpleInstruction
    | AgentContextInstruction
    | RunContextInstruction
    | BothContextsInstruction
    | PydanticAIInstruction
)


class InstructionMetadata:
    """Metadata for instruction functions.

    This class provides optional metadata that can be attached to instruction
    functions for documentation, debugging, and introspection purposes.

    The metadata includes:
    - name: A descriptive name for the instruction
    - description: Optional detailed description
    - fallback: Fallback text to use if the instruction fails
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        fallback: str = "",
    ) -> None:
        """Initialize instruction metadata.

        Args:
            name: A descriptive name for the instruction
            description: Optional detailed description of what the instruction does
            fallback: Fallback text to use if the instruction fails to execute
        """
        self.name = name
        self.description = description
        self.fallback = fallback
