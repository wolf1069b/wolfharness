"""Token budget capability — enforces token budget per agent run.

Tracks cumulative token usage via ``wrap_model_request`` and raises
``TokenBudgetExceededError`` when the budget is exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import WrapModelRequestHandler
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.usage import RunUsage


class TokenBudgetExceededError(Exception):
    """Raised when cumulative token usage exceeds the configured budget."""

    def __init__(self, used: int, budget: int) -> None:
        self.used = used
        self.budget = budget
        super().__init__(f"Token budget exceeded: {used} tokens used, budget is {budget}.")


@dataclass
class TokenBudgetCapability(AbstractCapability[Any]):
    """Enforce a token budget per agent run.

    Wraps ``model_request`` and accumulates token usage from each model
    response. When cumulative usage exceeds ``max_tokens``, raises
    ``TokenBudgetExceededError``.
    """

    max_tokens: int = 100_000
    _used_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            msg = f"max_tokens must be >= 1, got {self.max_tokens}"
            raise ValueError(msg)

    @property
    def has_wrap_node_run(self) -> bool:
        return False

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        response: ModelResponse = await handler(request_context)
        usage: RunUsage | None = getattr(response, "usage", None)
        if usage is not None:
            self._used_tokens += usage.total_tokens
            if self._used_tokens > self.max_tokens:
                raise TokenBudgetExceededError(self._used_tokens, self.max_tokens)
        return response

    async def for_run(self, ctx: RunContext[Any]) -> TokenBudgetCapability:
        return TokenBudgetCapability(max_tokens=self.max_tokens)
