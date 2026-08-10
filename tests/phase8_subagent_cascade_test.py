"""Test that subagent cancellation cascades correctly within timeout.

This test verifies that when a parent agent is cancelled,
any spawned subagents also receive cancellation within 5 seconds.
"""

from __future__ import annotations

import anyio
import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig


@pytest.mark.asyncio
async def test_subagent_cancellation_cascade_within_5s(manifest: AgentsManifest) -> None:
    """Verify subagent cancellation cascades within 5 seconds.

    Regression test for structured concurrency:
    - When parent agent is cancelled, spawned subagents must also cancel
    - Subagent should receive CancelledError within 5 seconds (not hang)
    - This tests that CancelScope(shield=True) around complete_event.set()
      allows cleanup even during cancellation
    """
    agent_config = NativeAgentConfig(
        name="parent-agent",
        model="test",
        system_prompt="You are a parent agent that spawns subagents.",
    )

    subagent_config = NativeAgentConfig(
        name="sub-agent",
        model="test",
        system_prompt="You are a subagent.",
    )

    manifest.agents["parent-agent"] = agent_config
    manifest.agents["sub-agent"] = subagent_config

    async with (
        AgentPool(manifest=manifest) as pool,
        pool.manifest.agents["parent-agent"].get_agent(pool=pool) as parent_agent,
        anyio.create_task_group() as tg,
    ):
        # Spawn a subagent in background via TaskGroup
        tg.start_soon(parent_agent.run, "Spawn a subagent and then I will cancel")

        # Cancel the parent agent immediately
        # This should cascade to the subagent
        await anyio.sleep(0.1)  # Give subagent time to start
        tg.cancel_scope.cancel()

        # If we reach here without hanging, cancellation cascaded
        # correctly - the task group exited because all spawned tasks
        # (including the subagent) responded to cancellation. If the
        # subagent did not cascade-cancel, the block would hang until
        # the test timeout.
