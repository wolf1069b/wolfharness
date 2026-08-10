"""Base class for message processing nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from wolfharness import AgentPool
    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.messaging import MessageNode
    from wolfharness.prompts.manager import PromptManager
    from wolfharness.ui.base import InputProvider


@dataclass(kw_only=True)
class NodeContext[TDeps = object]:
    """Context for message processing nodes."""

    node: MessageNode[TDeps, Any]
    """Current Node."""

    pool: AgentPool[Any] | None = None
    """The agent pool the node is part of."""

    input_provider: InputProvider | None = None
    """Provider for human-input-handling."""

    data: TDeps | None = None
    """Custom context data."""

    @property
    def node_name(self) -> str:
        """Name of the current node."""
        return self.node.name

    @property
    def agent(self) -> BaseAgent[TDeps, Any]:
        """Return agent node, type-narrowed to BaseAgent."""
        from wolfharness.agents.base_agent import BaseAgent

        assert isinstance(self.node, BaseAgent)
        return self.node  # ty: ignore[invalid-return-type]

    def get_input_provider(self) -> InputProvider:
        # 1. Direct context provider (highest priority)
        if self.input_provider:
            return self.input_provider
        # 2. Session-bound provider (authoritative for pooled sessions)
        from wolfharness.agents.context import AgentContext

        if isinstance(self, AgentContext):
            session_state = self.get_session_state()
            if session_state is not None:
                session_provider = getattr(session_state, "input_provider", None)
                if session_provider is not None:
                    return session_provider  # type: ignore[no-any-return]
        # 3. Pool-level fallback
        if self.pool and self.pool._input_provider:
            return self.pool._input_provider
        # 4. ContextVar fallback — set by _run_turn_unlocked for the current
        # turn. This catches cases where session.input_provider was not set
        # (e.g., run_stream path) or the agent was cached before the provider
        # was available.
        from wolfharness.mcp_server.manager import _current_input_provider

        contextvar_provider = _current_input_provider.get()
        if contextvar_provider is not None:
            return contextvar_provider
        raise RuntimeError(
            f"No InputProvider configured for node {self.node_name!r}. "
            f"When running under ACP/OpenCode protocols, an input provider must be "
            f"explicitly set via session configuration or agent initialization."
        )

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from pool."""
        if self.pool is None:
            raise RuntimeError("Cannot access prompt_manager: no agent pool available")
        return self.pool.prompt_manager
