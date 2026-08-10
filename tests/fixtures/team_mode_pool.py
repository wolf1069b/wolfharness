"""Shared team-mode AgentPool fixture for L2 integration tests.

Provides a real ``AgentPool`` with ``team_mode`` enabled and ``TestModel``
agents.  Replaces inline config construction in team-mode test files.

Usage::

    async def test_something(team_mode_pool):
        agent = team_mode_pool.get_agent("coordinator")
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yamling

from wolfharness import AgentPool, AgentsManifest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# YAML configs
# ---------------------------------------------------------------------------

TEAM_MODE_CONFIG = """\
agents:
  coordinator:
    type: native
    model: test
    system_prompt: "You are a team coordinator"
  worker:
    type: native
    model: test
    system_prompt: "You are a team worker"
  reviewer:
    type: native
    model: test
    system_prompt: "You are a team reviewer"

team_mode:
  enabled: true
  member_eligible: [worker, reviewer]
  lead_eligible: [coordinator]
  base_dir: {base_dir}
"""

TEAM_MODE_CONFIG_WITH_DEFAULTS = """\
agents:
  coordinator:
    type: native
    model: test
    system_prompt: "You are a team coordinator"
  worker:
    type: native
    model: test
    system_prompt: "You are a team worker"
  reviewer:
    type: native
    model: test
    system_prompt: "You are a team reviewer"

team_mode:
  enabled: true
  member_eligible: [worker, reviewer]
  lead_eligible: [coordinator]
  base_dir: {base_dir}
  defaults:
    team_name: "default_team"
    members:
      - name: "worker_1"
        agent: "worker"
      - name: "reviewer_1"
        agent: "reviewer"
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_manifest(yaml_text: str, base_dir: str) -> AgentsManifest:
    """Parse inline YAML into an ``AgentsManifest``."""
    raw = yamling.load_yaml(yaml_text.format(base_dir=base_dir), verify_type=dict)
    return AgentsManifest.model_validate(raw)


@pytest.fixture
async def team_mode_pool(tmp_path: Path) -> AsyncIterator[AgentPool]:
    """Real ``AgentPool`` with ``team_mode`` enabled and ``TestModel`` agents.

    Three agents: ``coordinator`` (lead-eligible), ``worker`` and ``reviewer``
    (member-eligible).  No MCP, no storage, no external deps.
    """
    manifest = _build_manifest(TEAM_MODE_CONFIG, str(tmp_path))
    async with AgentPool(manifest) as pool:
        yield pool


@pytest.fixture
async def team_mode_pool_with_defaults(tmp_path: Path) -> AsyncIterator[AgentPool]:
    """Real ``AgentPool`` with ``team_mode`` defaults configured.

    Same as ``team_mode_pool`` but with ``defaults.members`` pre-configured.
    """
    manifest = _build_manifest(TEAM_MODE_CONFIG_WITH_DEFAULTS, str(tmp_path))
    async with AgentPool(manifest) as pool:
        yield pool
