"""Base input provider class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from mcp import types
    from pydantic import BaseModel

    from wolfharness.agents.context import AgentContext, ConfirmationResult
    from wolfharness.messaging.context import NodeContext


class InputProvider(ABC):
    """Base class for handling all UI interactions."""

    @property
    def supports_durable_elicitation(self) -> bool:
        """Whether this provider supports durable (checkpointable) elicitation.

        Returns False by default. Providers that implement checkpoint-based
        elicitation (e.g., ACP with checkpointing enabled) override this
        property to return True dynamically based on runtime configuration.

        Subclasses (StdlibInputProvider, MockInputProvider) inherit this
        default and always return False — they do not support durability.
        """
        return False

    async def get_input(
        self,
        context: NodeContext,
        prompt: str,
        output_type: type[BaseModel] | None = None,
    ) -> Any:
        """Get normal input (used by HumanProvider).

        Args:
            context: Current agent context
            prompt: The prompt to show to the user
            output_type: Optional type for structured responses
            message_history: Optional conversation history
        """
        if output_type:
            return await self.get_structured_input(context, prompt, output_type)
        return await self.get_text_input(context, prompt)

    async def get_text_input(self, context: NodeContext[Any], prompt: str) -> str:
        """Get normal text input."""
        raise NotImplementedError

    async def get_structured_input(
        self,
        context: NodeContext[Any],
        prompt: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        """Get structured input."""
        raise NotImplementedError

    @abstractmethod
    def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool_description: str = "",
    ) -> Coroutine[Any, Any, ConfirmationResult]:
        """Get tool execution confirmation.

        Tool name and arguments are read from context.tool_name and context.tool_input.

        Args:
            context: Current node context with tool_name, tool_call_id, tool_input set
            tool_description: Human-readable description of the tool
        """

    @abstractmethod
    def get_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> Coroutine[Any, Any, types.ElicitResult | types.ErrorData]:
        """Get user response to elicitation request.

        Args:
            context: Current agent context
            params: MCP elicit request parameters
        """

    async def broadcast_elicitation_question(
        self,
        handle: str,
        params: types.ElicitRequestParams,
        shared_future: Any = None,
    ) -> bool:
        """Broadcast a durable elicitation question to the UI.

        Default implementation returns False (not supported). Providers
        that support durable elicitation override this to broadcast the
        question and store it for later resolution.

        Args:
            handle: The elicitation handle (tool_call_id).
            params: The elicitation request parameters.
            shared_future: Optional future to share for resolution.

        Returns:
            True if broadcast succeeded, False if not supported.
        """
        return False

    def cleanup_elicitation_question(self, handle: str) -> None:  # noqa: B027
        """Clean up a pending elicitation question after timeout or cancellation.

        Default implementation does nothing. Providers that store pending
        questions override this to remove the stale entry.

        Args:
            handle: The elicitation handle (tool_call_id) to clean up.
        """
