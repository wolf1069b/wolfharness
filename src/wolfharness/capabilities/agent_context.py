"""AgentContextDeps — frozen dataclass carrying per-turn runtime state.

Constructed by RunLoop at Turn time (M2 task group 15), not by
AgentFactory at compile time. Provides typed references to all
per-turn services that agent tools and capabilities need.

This class is the ``deps`` type for pydantic-ai's ``RunContext``.
It is distinct from ``wolfharness.agents.context.AgentContext`` which IS
the ``RunContext`` itself. The name ``AgentContextDeps`` makes this
relationship explicit: ``RunContext[AgentContextDeps]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
import warnings


if TYPE_CHECKING:
    from wolfharness.capabilities.delegation import DelegationService
    from wolfharness.capabilities.extension_registry import ExtensionRegistry
    from wolfharness.host.context import HostContext, RunScope
    from wolfharness.host.registry import AgentRegistry
    from wolfharness.orchestrator.session_controller import SessionState
    from wolfharness_config.team_mode import TeamModeConfig


@dataclass(frozen=True, slots=True)
class AgentContextDeps:
    """Immutable per-turn context injected into pydantic-ai RunContext.deps.

    Carries typed references to per-turn runtime state. A new instance
    is created for each Turn — no reuse across turns.

    Attributes:
        agent_registry: Read-only access to compiled agents for delegation.
        delegation: Limited interface for spawning subagents.
        session: Current session state (message history, metadata).
        scope: Run scope (config_id, tenant_id, user_id, session_id).
        host: Infrastructure handles (mcp, storage, skills, etc.).
        extension_registry: ExtensionRegistry for scoped capability access.
        team_mode_config: Global team mode config from manifest, if enabled.
        agent_name: Name of the agent this context belongs to. Used for
            AGENT scope queries in the ExtensionRegistry.
    """

    agent_registry: AgentRegistry
    delegation: DelegationService
    session: SessionState
    scope: RunScope
    host: HostContext
    extension_registry: ExtensionRegistry | None = None
    team_mode_config: TeamModeConfig | None = None
    agent_name: str = ""


def resolve_agent_context_from_deps(
    deps: Any, *, capability_name: str = "Capability"
) -> AgentContextDeps:
    """Unwrap the M2 ``AgentContextDeps`` from pydantic-ai runtime deps.

    In production, ``ctx.deps`` is ``agents.context.AgentContext`` (the
    PydanticAI runtime context). Our ``capabilities.agent_context.AgentContextDeps``
    is stored at ``deps.data``, set by ``NativeTurn`` (turn.py:
    ``agent_deps.data = run_ctx.deps``). In tests, deps may be directly
    our ``AgentContextDeps``.

    Args:
        deps: The ``ctx.deps`` value from a pydantic-ai ``RunContext``.
        capability_name: Name of the calling capability, used in error messages.

    Returns:
        The ``AgentContextDeps`` instance from ``deps`` (or ``deps.data``).

    Raises:
        RuntimeError: If deps is None, ``.data`` is None, or deps is
            neither ``RuntimeAgentContext`` nor ``AgentContextDeps``.
    """
    from wolfharness.agents.context import AgentContext as RuntimeAgentContext

    if deps is None:
        msg = f"{capability_name} requires AgentContextDeps as deps. Got: None"
        raise RuntimeError(msg)
    # Production path: deps is RuntimeAgentContext, M2 AgentContextDeps at .data
    if isinstance(deps, RuntimeAgentContext):
        inner = deps.data
        if inner is None:
            msg = f"{capability_name} requires AgentContextDeps at deps.data. Got: None"
            raise RuntimeError(msg)
        return cast(AgentContextDeps, inner)
    # Test path: deps is directly our AgentContextDeps
    if isinstance(deps, AgentContextDeps):
        return deps
    msg = f"{capability_name} requires AgentContextDeps as deps. Got: {type(deps).__name__}"
    raise RuntimeError(msg)


# --- Deprecated alias for backward compatibility -----------------------------
# Remove in the next minor release after external consumers have migrated.


def __getattr__(name: str) -> Any:
    if name == "AgentContext":
        warnings.warn(
            "wolfharness.capabilities.agent_context.AgentContext is deprecated; "
            "use AgentContextDeps instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AgentContextDeps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
