"""L2 integration test: member session leak when lead run terminates.

Reproduces the bug where member sessions spawned by ``team_create`` leak
when the lead agent's RunHandle terminates without ``close_session`` being
called on the lead session (e.g., the run finishes but the session stays
alive for follow-ups).

The test uses a **real** AgentPool + SessionPool (no mocks) with a
manually created RunHandle so the full session lifecycle (RunHandle,
background tasks, complete_event) is exercised without depending on
TestModel timing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
import uuid

import pytest
import yamling

from wolfharness import AgentPool, AgentsManifest
from wolfharness.capabilities.agent_context import AgentContextDeps
from wolfharness.capabilities.runloop_delegation import RunLoopDelegationService
from wolfharness.capabilities.team_comm_capability import TeamCommCapability
from wolfharness.host.context import RunScope
from wolfharness.host.registry import AgentRegistry
from wolfharness.orchestrator.core import SessionState  # noqa: TC001
from wolfharness.orchestrator.run import RunHandle


if TYPE_CHECKING:
    from wolfharness_config.team_mode import TeamModeConfig


def _make_manifest(
    tmp_path: Any,
    *,
    idle_timeout: float = 300.0,
    poll_interval: float = 30.0,
) -> AgentsManifest:
    """Create a manifest with team_mode enabled and TestModel agents."""
    yaml_str = """
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
  idle_timeout: {idle_timeout}
  poll_interval: {poll_interval}
"""
    yaml_str = yaml_str.format(
        base_dir=str(tmp_path),
        idle_timeout=idle_timeout,
        poll_interval=poll_interval,
    )
    config_dict = yamling.load_yaml(yaml_str, verify_type=dict)
    return AgentsManifest(**config_dict)


def _make_agent_context(
    pool: AgentPool[Any],
    session_id: str,
    team_mode_config: TeamModeConfig,
) -> AgentContextDeps:
    """Construct a real AgentContextDeps for calling team_create directly."""
    session_pool = pool.session_pool
    assert session_pool is not None
    session = session_pool.sessions.get_session(session_id)
    assert session is not None

    host_ctx = pool.get_context()
    registry = AgentRegistry(dict.fromkeys(pool.manifest.agents))
    delegation = RunLoopDelegationService(
        registry=registry,
        host=host_ctx,
        session_id=session_id,
    )
    scope = RunScope(
        config_id=None,
        tenant_id=None,
        user_id=None,
        session_id=session_id,
    )
    return AgentContextDeps(
        agent_registry=registry,
        delegation=delegation,
        session=session,
        scope=scope,
        host=host_ctx,
        team_mode_config=team_mode_config,
    )


def _make_mock_run_context(agent_ctx: AgentContextDeps) -> MagicMock:
    """Create a mock pydantic-ai RunContext with AgentContextDeps as deps.

    ``_resolve_agent_context`` checks ``isinstance(deps, AgentContextDeps)``
    from ``capabilities.agent_context`` — our AgentContextDeps matches that
    check and is returned directly.
    """
    ctx = MagicMock()
    ctx.deps = agent_ctx
    return ctx


def _inject_run_handle(
    session_pool: Any,
    session: SessionState,
) -> RunHandle:
    """Manually create and register a RunHandle for the given session.

    This avoids depending on TestModel timing — the RunHandle is created
    in IDLE state and can be closed directly to simulate run termination.
    """
    run_id = str(uuid.uuid4())
    run_handle = RunHandle(
        run_id=run_id,
        session_id=session.session_id,
        agent_type="native",
    )
    session_pool.sessions._runs[run_id] = run_handle
    session.current_run_id = run_id
    return run_handle


@pytest.mark.integration
async def test_member_sessions_closed_when_lead_idle_and_no_active_runs(
    tmp_path: Any,
) -> None:
    """Given: real AgentPool with team_mode, lead session idle with no active runs.

    When: team_create spawns member sessions, then the lead goes idle
        (last_active_at set far in the past) and neither the lead nor
        any member has an active run.

    Then: member sessions are automatically closed (not leaked).
        The cleanup polls last_active_at every _poll_interval seconds
        and closes members after _idle_timeout seconds of inactivity,
        but only when no session has an active run.
    """
    # Use short intervals for testing.
    manifest = _make_manifest(tmp_path, idle_timeout=0.5, poll_interval=0.1)
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        lead_session_id = "lead-session-001"

        # Create lead session.
        await session_pool.create_session(
            lead_session_id,
            agent_name="coordinator",
            team_role="lead",
            team_member_name="coordinator",
        )

        lead_session = session_pool.sessions.get_session(lead_session_id)
        assert lead_session is not None

        # Manually inject a RunHandle so the session has an active run
        # (needed for team_create's cleanup scheduling).
        run_handle = _inject_run_handle(session_pool, lead_session)

        # Construct AgentContextDeps and call team_create.
        agent_ctx = _make_agent_context(pool, lead_session_id, team_mode_config)
        cap = TeamCommCapability(
            team_mode_config,
            "coordinator",
            session_metadata={
                "team_role": "lead",
                "team_member_name": "coordinator",
            },
        )

        mock_ctx = _make_mock_run_context(agent_ctx)

        create_result = await cap.team_create(
            mock_ctx,
            "test_team",
            [
                {"agent": "worker", "name": "worker_1"},
                {"agent": "reviewer", "name": "reviewer_1"},
            ],
        )
        assert "Team 'test_team' created with 2 members" in create_result.return_value

        # Extract member session IDs from team state (exclude lead).
        from wolfharness.capabilities.file_team_state import FileTeamState

        team_id = create_result.return_value.split("team_id=")[1].strip()
        team_state = FileTeamState(str(tmp_path))
        state = team_state._read_json(team_state._state_path(team_id))
        lead_sid = session_pool.sessions.get_session("lead-session-001")
        lead_session_id = lead_sid.session_id if lead_sid else "lead-session-001"
        member_session_ids: list[str] = [
            m["session_id"]
            for m in state.get("members", {}).values()
            if m.get("session_id") and m["session_id"] != lead_session_id
        ]
        assert len(member_session_ids) == 2

        # Verify member sessions exist in SessionPool.
        for msid in member_session_ids:
            ms = session_pool.sessions.get_session(msid)
            assert ms is not None, f"Member session {msid} should exist after team_create"

        # Simulate the lead's run completing and going idle:
        # 1. Clear the RunHandle (run finished, no active run).
        # 2. Set last_active_at far in the past.
        import time

        lead_session.current_run_id = None
        session_pool.sessions._runs.pop(run_handle.run_id, None)
        run_handle.complete_event.set()
        lead_session.last_active_at = time.monotonic() - 100.0

        # Wait for the polling cleanup to detect idle and close members.
        # _poll_interval=0.1, _idle_timeout=0.5 → cleanup fires within ~1s.
        await asyncio.sleep(2.0)

        # Assert member sessions are closed (not leaked).
        for msid in member_session_ids:
            ms = session_pool.sessions.get_session(msid)
            assert ms is None, (
                f"Member session {msid} should be closed after lead idle with no active runs, "
                "but it is still active — session leak detected"
            )

        # Cleanup: close the lead session (still alive without a run).
        await session_pool.close_session(lead_session_id)


@pytest.mark.integration
async def test_cleanup_deferred_when_lead_has_active_run(
    tmp_path: Any,
) -> None:
    """Given: lead session is idle (last_active_at stale) but has an active run.

    When: the polling cleanup checks idle status.

    Then: cleanup is deferred — member sessions stay alive because the
        lead's ``current_run_id`` is not None, indicating the lead is
        actively processing (e.g. a long model call or tool execution
        that doesn't update ``last_active_at``).
    """
    manifest = _make_manifest(tmp_path, idle_timeout=0.5, poll_interval=0.1)
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        lead_session_id = "lead-session-002"

        await session_pool.create_session(
            lead_session_id,
            agent_name="coordinator",
            team_role="lead",
            team_member_name="coordinator",
        )

        lead_session = session_pool.sessions.get_session(lead_session_id)
        assert lead_session is not None

        _inject_run_handle(session_pool, lead_session)

        agent_ctx = _make_agent_context(pool, lead_session_id, team_mode_config)
        cap = TeamCommCapability(
            team_mode_config,
            "coordinator",
            session_metadata={
                "team_role": "lead",
                "team_member_name": "coordinator",
            },
        )

        mock_ctx = _make_mock_run_context(agent_ctx)

        create_result = await cap.team_create(
            mock_ctx,
            "test_team",
            [{"agent": "worker", "name": "worker_1"}],
        )
        assert "Team 'test_team' created with 1 members" in create_result.return_value

        from wolfharness.capabilities.file_team_state import FileTeamState

        team_id = create_result.return_value.split("team_id=")[1].strip()
        team_state = FileTeamState(str(tmp_path))
        state = team_state._read_json(team_state._state_path(team_id))
        member_session_ids: list[str] = [
            m["session_id"]
            for m in state.get("members", {}).values()
            if m.get("session_id") and m["session_id"] != lead_session_id
        ]
        assert len(member_session_ids) == 1

        # Make lead appear idle but keep active run.
        import time

        lead_session.last_active_at = time.monotonic() - 100.0
        # current_run_id is still set (from _inject_run_handle).

        # Wait long enough for cleanup to have fired if it weren't deferred.
        await asyncio.sleep(2.0)

        # Member sessions should still be alive — cleanup deferred.
        for msid in member_session_ids:
            ms = session_pool.sessions.get_session(msid)
            assert ms is not None, (
                f"Member session {msid} should NOT be closed while lead has active run"
            )

        await session_pool.close_session(lead_session_id)


@pytest.mark.integration
async def test_cleanup_deferred_when_member_has_active_run(
    tmp_path: Any,
) -> None:
    """Given: lead is idle, no lead run, but a member has an active run.

    When: the polling cleanup checks idle status.

    Then: cleanup is deferred — member sessions stay alive because at
        least one member has ``current_run_id`` set, indicating it is
        still processing.  The cleanup will fire on a future poll once
        all member runs complete.
    """
    manifest = _make_manifest(tmp_path, idle_timeout=0.5, poll_interval=0.1)
    team_mode_config: TeamModeConfig | None = manifest.team_mode
    assert team_mode_config is not None

    async with AgentPool(manifest) as pool:
        session_pool = pool.session_pool
        assert session_pool is not None

        lead_session_id = "lead-session-003"

        await session_pool.create_session(
            lead_session_id,
            agent_name="coordinator",
            team_role="lead",
            team_member_name="coordinator",
        )

        lead_session = session_pool.sessions.get_session(lead_session_id)
        assert lead_session is not None

        run_handle = _inject_run_handle(session_pool, lead_session)

        agent_ctx = _make_agent_context(pool, lead_session_id, team_mode_config)
        cap = TeamCommCapability(
            team_mode_config,
            "coordinator",
            session_metadata={
                "team_role": "lead",
                "team_member_name": "coordinator",
            },
        )

        mock_ctx = _make_mock_run_context(agent_ctx)

        create_result = await cap.team_create(
            mock_ctx,
            "test_team",
            [{"agent": "worker", "name": "worker_1"}],
        )
        assert "Team 'test_team' created with 1 members" in create_result.return_value

        from wolfharness.capabilities.file_team_state import FileTeamState

        team_id = create_result.return_value.split("team_id=")[1].strip()
        team_state = FileTeamState(str(tmp_path))
        state = team_state._read_json(team_state._state_path(team_id))
        member_session_ids: list[str] = [
            m["session_id"]
            for m in state.get("members", {}).values()
            if m.get("session_id") and m["session_id"] != lead_session_id
        ]
        assert len(member_session_ids) == 1

        # Inject a RunHandle into the member session (simulating
        # the member still processing).
        member_session = session_pool.sessions.get_session(member_session_ids[0])
        assert member_session is not None
        _inject_run_handle(session_pool, member_session)

        # Lead: clear run (lead finished) and set idle.
        import time

        lead_session.current_run_id = None
        session_pool.sessions._runs.pop(run_handle.run_id, None)
        run_handle.complete_event.set()
        lead_session.last_active_at = time.monotonic() - 100.0

        # Wait long enough for cleanup to have fired if it weren't deferred.
        await asyncio.sleep(2.0)

        # Member session should still be alive — cleanup deferred because
        # the member has an active run.
        ms = session_pool.sessions.get_session(member_session_ids[0])
        assert ms is not None, "Member session should NOT be closed while it has an active run"

        await session_pool.close_session(lead_session_id)
