"""Runtime hook classes for agent lifecycle events."""

from __future__ import annotations

from wolfharness.hooks.agent_hooks import AgentHooks
from wolfharness.hooks.base import Hook, HookEvent, HookInput, HookResult
from wolfharness.hooks.callable import CallableHook
from wolfharness.hooks.command import CommandHook
from wolfharness.hooks.prompt import PromptHook

__all__ = [
    "AgentHooks",
    "CallableHook",
    "CommandHook",
    "Hook",
    "HookEvent",
    "HookInput",
    "HookResult",
    "PromptHook",
]
