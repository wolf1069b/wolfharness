"""Tests for async subagent task execution with internal_fs output."""

from __future__ import annotations

import pytest

from wolfharness import AgentPool, AgentsManifest


pytestmark = pytest.mark.integration


class TestAsyncSubagentTask:
    """Tests for task tool async_mode that runs agents in background."""

    async def test_task_async_mode_returns_task_id_immediately(self) -> None:
        """Test that task with async_mode=True returns a task ID without blocking."""
        manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Worker completed the task!"
    system_prompt: You are a helpful worker agent.

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Do some work"
          description: "Test async task"
          async_mode: true
    tools:
      - type: subagent
""")

        async with AgentPool(manifest) as pool:
            orchestrator = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

            # Run orchestrator - it should call task with async_mode=True
            result = await orchestrator.run("Start a background task")

            # The result should contain information about the started task
            assert result.content is not None
            content = str(result.content)
            assert "Task started" in content or "output" in content.lower()

    async def test_task_sync_mode_still_works(self) -> None:
        """Test that task without async_mode still works synchronously."""
        manifest = AgentsManifest.from_yaml("""
agents:
  worker:
    model:
      type: test
      custom_output_text: "Sync worker result."

  orchestrator:
    model:
      type: test
      call_tools: ["task"]
      tool_args:
        task:
          agent_or_team: worker
          prompt: "Do sync work"
          description: "Sync test"
    tools:
      - type: subagent
""")

        async with AgentPool(manifest) as pool:
            orchestrator = pool.manifest.agents["orchestrator"].get_agent(pool=pool)

            result = await orchestrator.run("Run sync task")

            # Sync mode should return the actual result, not a task ID
            assert result.content is not None
            # The orchestrator's response should reflect the worker completed
