"""Skills tools — backward-compatible re-exports.

The standalone ``SkillsTools`` toolset has been consolidated into
:class:`~wolfharness.capabilities.skill_manager_cap.SkillManagerCap`
(unify-skill-loading change). The production ``load_skill`` / ``list_skills``
tools are owned by ``SkillManagerCap``.

The module-level functions here are kept as thin backward-compatible
wrappers that **delegate to the cap's implementation** (they do not
re-implement loading logic). No production path calls these wrappers; they
exist so tests and external code that imported ``load_skill`` /
``list_skills`` / ``load_skill_for_node`` from here keep working.
"""

from __future__ import annotations

from typing import Any

from wolfharness.agents.context import AgentContext  # noqa: TC001
from wolfharness.capabilities.skill_manager_cap import (
    SkillManagerCap,
    _load_reference_content,
    _substitute_arguments,
)


__all__ = [
    "_load_reference_content",
    "_substitute_arguments",
    "list_skills",
    "load_skill",
    "load_skill_for_node",
]


def _get_skill_cap(ctx: AgentContext) -> SkillManagerCap[Any] | None:
    """Return the pool's first ``SkillManagerCap`` or ``None``."""
    if ctx.pool is None:
        return None
    caps = ctx.pool.skill_capabilities
    if caps:
        cap = caps[0]
        return cap if isinstance(cap, SkillManagerCap) else None
    return None


def _as_run_ctx(ctx: AgentContext) -> Any:
    """Wrap an ``AgentContext`` as a minimal RunContext-like with ``.deps``.

    ``AgentContext`` (``RuntimeAgentContext``) is what ``_resolve_pool``
    expects as ``ctx.deps``; exposing it through ``.deps`` lets the cap's
    tool implementations read the pool and node without re-implementation.
    """
    return _DepsCtx(ctx)


class _DepsCtx:
    """Minimal ``RunContext`` stand-in exposing ``.deps`` for delegation."""

    __slots__ = ("deps",)

    def __init__(self, deps: Any) -> None:
        self.deps = deps


async def load_skill(
    ctx: AgentContext,
    skill_name: str,
    arguments: str | None = None,
    *,
    node_name: str | None = None,
    include_assembly: bool = True,
) -> str:
    """Load a Claude Code Skill and return its instructions.

    Delegates to ``SkillManagerCap._load_skill_impl``.

    Args:
        ctx: Agent context providing access to pool and skills.
        skill_name: Name of the skill to load, or a skill:// URI.
        arguments: Optional space-separated arguments for substitution.
        node_name: Optional node name for package-scoped skill visibility.
        include_assembly: When False, skip MCP/tool status rendering and
            tool import — returns pure instruction text (used for
            instruction-only injection, e.g. team-member skills).

    Returns:
        The full skill instructions for execution.
    """
    if ctx.pool is None:
        return "No agent pool available - skills require pool context"
    cap = _get_skill_cap(ctx)
    if cap is None:
        return "SkillManagerCap not available"
    return await cap._load_skill_impl(
        _as_run_ctx(ctx),
        skill_name,
        arguments,
        node_name=node_name,
        include_assembly=include_assembly,
    )


async def load_skill_for_node(
    ctx: AgentContext,
    skill_name: str,
    node_name: str,
    arguments: str | None = None,
    *,
    include_assembly: bool = True,
) -> str:
    """Load a skill using a target node's package-level skill scope.

    Args:
        ctx: Agent context providing access to pool and skills.
        skill_name: Name of the skill to load, or a skill:// URI.
        node_name: The node whose package scope governs visibility.
        arguments: Optional space-separated arguments for substitution.
        include_assembly: When False, skip MCP/tool status rendering and
            tool import — returns pure instruction text.

    Returns:
        The full skill instructions for execution.
    """
    return await load_skill(
        ctx,
        skill_name,
        arguments,
        node_name=node_name,
        include_assembly=include_assembly,
    )


async def list_skills(ctx: AgentContext) -> str:
    """List all available skills.

    Delegates to ``SkillManagerCap._list_skills_impl``.

    Args:
        ctx: Agent context providing access to pool and skills.

    Returns:
        Formatted list of available skills with descriptions and URIs.
    """
    if ctx.pool is None:
        return "No agent pool available - skills require pool context"
    cap = _get_skill_cap(ctx)
    if cap is None:
        return "No skills available"
    return await cap._list_skills_impl(_as_run_ctx(ctx))
