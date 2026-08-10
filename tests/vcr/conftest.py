"""Shared fixtures for L3 VCR tests.

The ``vcr_pool`` fixture builds a real ``AgentPool`` from inline YAML config
with a single native agent using the ``openai:gpt-4o-mini`` model. VCR
intercepts model API HTTP calls — the pool, agents, capabilities, EventBus,
SessionController, and protocol stacks all run for real in-process.

See ``tests/AGENTS.md`` for the VCR recording workflow and
``openspec/changes/layered-testing-infrastructure/design.md`` for design D6
(VCR scope: model API HTTP only) and D15 (``vcr_pool`` fixture).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yamling

from wolfharness import AgentPool, AgentsManifest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from typing import Any


# Inline YAML config used by the vcr_pool fixture. Uses openai:gpt-4o-mini
# because that is the reference model for cassette recording (D14). The
# `remap_hardcoded_test_models` session fixture in tests/conftest.py will
# transparently remap this to TEST_MODEL_OVERRIDE if that env var is set,
# so custom endpoints without gpt-4o access still work for recording.
VCR_POOL_CONFIG = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    system_prompt: "You are a helpful test assistant for VCR cassette replay."
"""

# Config with a single agent exposing a tool (used by tool-call tests).
VCR_POOL_CONFIG_WITH_TOOL = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    system_prompt: "You are a helpful test assistant. Use tools when asked."
    tools:
      - name: echo
        enabled: true
"""

# Config with a coordinator + worker agent for subagent delegation tests.
VCR_POOL_CONFIG_WITH_SUBAGENT = """\
agents:
  coordinator:
    type: native
    model: openai:gpt-4o-mini
    system_prompt: "You coordinate tasks. Delegate to the worker when helpful."
    tools:
      - type: subagent
  worker:
    type: native
    model: openai:gpt-4o-mini
    system_prompt: "You are a worker agent. Complete tasks concisely."
"""

# Config with team_mode enabled for dynamic team mode VCR tests.
VCR_POOL_CONFIG_TEAM_MODE = """\
agents:
  team_lead:
    type: native
    model: openai:gpt-4o-mini
    system_prompt: "You are a team lead. Use team tools to coordinate members."
  team_member:
    type: native
    model: openai:gpt-4o-mini
    system_prompt: "You are a team member. Follow instructions from the lead."

team_mode:
  enabled: true
  lead_eligible:
    - team_lead
  member_eligible:
    - team_lead
    - team_member
"""


def _build_manifest(yaml_text: str) -> AgentsManifest:
    """Parse inline YAML into an ``AgentsManifest``."""
    raw = yamling.load_yaml(yaml_text, verify_type=dict)
    return AgentsManifest.model_validate(raw)


async def _precreate_agents(pool: AgentPool) -> dict[str, Any]:
    """Pre-create agent instances for all configured agents in the manifest.

    The old ``AgentPool.get_agent()`` was synchronous. The new session-based
    API is async (``get_or_create_session_agent``). To preserve the sync
    ``pool.get_agent(name)`` call pattern in VCR tests, we pre-create all
    agents during fixture setup (async) and cache them for sync access.
    """
    agents: dict[str, Any] = {}
    assert pool._session_pool is not None
    for name in pool.manifest.agents:
        session_id = f"vcr-{name}"
        agent = await pool._session_pool.sessions.get_or_create_session_agent(
            session_id, agent_name=name
        )
        agents[name] = agent
    return agents


def _attach_get_agent_compat(pool: AgentPool, agents_cache: dict[str, Any]) -> None:
    """Attach a synchronous ``get_agent`` compatibility method to a pool instance.

    The VCR tests were written against the old ``AgentPool.get_agent()``
    API which was removed when the pool shifted to session-based agent
    creation. This shim returns pre-created agent instances from the cache,
    preserving the sync ``pool.get_agent(name)`` usage pattern.
    """

    def get_agent(name: str) -> Any:
        if name not in agents_cache:
            raise KeyError(f"Agent {name!r} not found. Available: {list(agents_cache.keys())}")
        return agents_cache[name]

    pool.get_agent = get_agent  # type: ignore[method-assignment]


@pytest.fixture
async def vcr_pool() -> AsyncIterator[AgentPool]:
    """Real ``AgentPool`` with VCR-replayed model responses.

    The pool, agents, capabilities, EventBus, SessionController are all real.
    Only the model API HTTP calls are intercepted by VCR (design D6). Use
    ``--record-mode=once`` with ``OPENAI_API_KEY`` set to record a cassette;
    CI replays the cassette with no network access.
    """
    manifest = _build_manifest(VCR_POOL_CONFIG)
    async with AgentPool(manifest) as pool:
        _agents = await _precreate_agents(pool)
        _attach_get_agent_compat(pool, _agents)
        yield pool


@pytest.fixture
async def vcr_pool_with_tool() -> AsyncIterator[AgentPool]:
    """Real ``AgentPool`` with a single agent that exposes the ``echo`` tool.

    Used by tool-call VCR tests (P2 pattern). The ``echo`` tool is provided
    programmatically by the test (not via the YAML tool config) because the
    inline config cannot define a Python callable. Tests that need the tool
    attach it to the agent after pool construction.
    """
    manifest = _build_manifest(VCR_POOL_CONFIG_WITH_TOOL)
    async with AgentPool(manifest) as pool:
        _agents = await _precreate_agents(pool)
        _attach_get_agent_compat(pool, _agents)
        yield pool


@pytest.fixture
async def vcr_pool_with_subagent() -> AsyncIterator[AgentPool]:
    """Real ``AgentPool`` with a coordinator + worker for delegation tests."""
    manifest = _build_manifest(VCR_POOL_CONFIG_WITH_SUBAGENT)
    async with AgentPool(manifest) as pool:
        _agents = await _precreate_agents(pool)
        _attach_get_agent_compat(pool, _agents)
        yield pool


@pytest.fixture
async def vcr_team_pool(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AgentPool]:
    """Real ``AgentPool`` with ``team_mode`` enabled for VCR team-mode tests.

    Two agents: ``team_lead`` (lead-eligible) and ``team_member``
    (member-eligible). VCR intercepts model API HTTP calls — the pool,
    agents, capabilities, EventBus, SessionController all run for real.

    A dummy ``OPENAI_API_KEY`` is set via ``monkeypatch`` so the OpenAI
    client can initialize without error. VCR intercepts all HTTP requests,
    so the key is never actually used for authentication.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-vcr-replay")
    manifest = _build_manifest(VCR_POOL_CONFIG_TEAM_MODE)
    async with AgentPool(manifest) as pool:
        yield pool


@pytest.fixture
def vcr_config_path(tmp_path: Path) -> Path:
    """Path to a minimal YAML config file suitable for VCR recording.

    Writes the inline ``VCR_POOL_CONFIG`` to a temp file so protocol-server
    tests that load config from a path (e.g. ``ACPServer.from_config``) have
    a file to point at.
    """
    config_path = tmp_path / "vcr_test_config.yml"
    config_path.write_text(VCR_POOL_CONFIG)
    return config_path


@pytest.fixture
def vcr_cassettes_dir() -> Path:
    """Root directory holding VCR cassettes (``tests/cassettes/vcr/``)."""
    return Path(__file__).parent.parent / "cassettes" / "vcr"


def cassette_exists(test_module_stem: str, test_name: str) -> bool:
    """Check if a VCR cassette exists for the given test.

    Cassettes follow the convention
    ``tests/cassettes/vcr/<test_module_stem>/<test_name>.yaml`` (see
    ``tests/AGENTS.md``). Tests that have not yet had their cassette
    recorded ([HUMAN-REQUIRED]) use this to skip gracefully in CI.

    When ``VCR_RECORDING`` env var is set (recording mode), always returns
    True so the skipif guard doesn't prevent the test from running to
    record the cassette.
    """
    if os.getenv("VCR_RECORDING"):
        return True
    cassette_path = (
        Path(__file__).parent.parent / "cassettes" / "vcr" / test_module_stem / f"{test_name}.yaml"
    )
    return cassette_path.exists()


@pytest.fixture(autouse=True)
def _enable_model_requests_for_vcr_recording(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Enable real model API requests for VCR tests.

    The global ``ALLOW_MODEL_REQUESTS=False`` gate blocks model calls at the
    pydantic-ai level (before httpx/VCR can intercept). VCR intercepts at the
    HTTP transport level, so for VCR tests we must lift the gate to let the
    request reach VCR. During recording (``--record-mode=once``), VCR forwards
    the request to the real API. During replay (``--record-mode=none``), VCR
    returns the recorded response without any network access.
    """
    if request.node.get_closest_marker("vcr") is None:
        yield
        return
    import pydantic_ai.models

    with pydantic_ai.models.override_allow_model_requests(True):
        yield
