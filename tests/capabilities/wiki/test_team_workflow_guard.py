"""Tests for the conductor team-workflow guard harness capability.

Regression: a checkpoint_build call that omits build_id silently lets the
server mint a fresh auto-generated id over an in-progress build, detaching
every plan and source-packet owner (all chapters stay pending forever).
The guard must reject such calls with explicit feedback so the conductor
passes the canonical build identity.
"""

from __future__ import annotations

import pytest

from wolfharness.capabilities.wiki.harness.team_workflow_guard import (
    TeamWorkflowGuardCapability,
)


@pytest.fixture
def conductor_guard() -> TeamWorkflowGuardCapability:
    return TeamWorkflowGuardCapability(role="wiki_conductor")


def _build_args(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "doc_id": "manual",
        "device_id": "SY75C",
        "series_id": "SY75",
        "stage": "phase1a",
    }
    args.update(overrides)
    return args


def test_checkpoint_build_without_build_id_rejected(conductor_guard: TeamWorkflowGuardCapability) -> None:
    """A checkpoint_build call missing build_id must be rejected."""
    feedback = conductor_guard.validate("checkpoint_build", _build_args())
    assert feedback, "checkpoint_build without build_id must be rejected"
    assert "build_id" in feedback


def test_checkpoint_build_with_build_id_accepted(conductor_guard: TeamWorkflowGuardCapability) -> None:
    """A checkpoint_build call with explicit build_id must pass."""
    feedback = conductor_guard.validate(
        "checkpoint_build",
        _build_args(build_id="sy215c_repair_manual"),
    )
    assert feedback == ""


def test_checkpoint_build_non_dict_args_rejected(conductor_guard: TeamWorkflowGuardCapability) -> None:
    """Non-dict args must be rejected defensively."""
    feedback = conductor_guard.validate("checkpoint_build", None)  # type: ignore[arg-type]
    assert feedback


def test_checkpoint_build_guard_only_for_conductor() -> None:
    """Non-conductor roles must pass through untouched."""
    guard = TeamWorkflowGuardCapability(role="wiki_extraction_worker")
    feedback = guard.validate("checkpoint_build", _build_args())
    assert feedback == ""


# ── OP DAG blocked_by contract (regression) ───────────────────────────────

_OPS_DESCRIPTION = (
    "worker_role: wiki_ops_worker\n"
    "parent_opa: viking://resources/824/OP/OpA/gate.md\n"
    "retrieval_query: verify evidence\n"
    "evidence_uris: []\n"
    "expected_artifacts: [ops_draft]\n"
    "depends_on_stage: opa_discovered\n"
)


def _ops_task_args(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "subject": "OPA evidence pre-check gate for OPS dispatch",
        "owner": "ops_worker_01",
        "description": _OPS_DESCRIPTION,
    }
    args.update(overrides)
    return args


class _StubTeamState:
    """Minimal FileTeamState stand-in: roster has the declared OP worker."""

    def get_member_agent(self, team_id: str, owner: str) -> str:
        return "wiki_ops_worker"

    def list_tasks(self, team_id: str) -> list[dict[str, object]]:
        return []

    def list_member_names(self, team_id: str) -> set[str]:
        return {"ops_worker_01"}


def _runtime_agent_ctx() -> object:
    """AgentContextDeps stand-in with a team session bound to the stub state."""
    from types import SimpleNamespace

    session = SimpleNamespace(
        metadata={"team_id": "team_1", "team_base_dir": "/tmp/team"}
    )
    agent_ctx = SimpleNamespace(
        session=session,
        team_mode_config=None,
        _wiki_finalize_receipt=None,
    )
    return agent_ctx


def test_ops_gate_task_with_empty_blocked_by_accepted(
    conductor_guard: TeamWorkflowGuardCapability,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OP DAG root (gate) may legitimately declare an empty blocked_by
    list — it has no predecessor.  Regression: the guard rejected [] too,
    making every OPS dispatch deadlock on the first task."""
    import wolfharness.capabilities.wiki.harness.team_workflow_guard as guard_module

    monkeypatch.setattr(guard_module, "FileTeamState", lambda _base: _StubTeamState())
    agent_ctx = _runtime_agent_ctx()
    feedback = conductor_guard._validate_runtime_assignment(
        _ops_task_args(blocked_by=[]),
        agent_ctx,  # type: ignore[arg-type]
    )
    assert feedback == ""


def test_ops_task_with_missing_blocked_by_rejected(
    conductor_guard: TeamWorkflowGuardCapability,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-root OPS task that omits blocked_by entirely stays rejected."""
    import wolfharness.capabilities.wiki.harness.team_workflow_guard as guard_module

    monkeypatch.setattr(guard_module, "FileTeamState", lambda _base: _StubTeamState())
    args = _ops_task_args()
    args.pop("blocked_by", None)
    feedback = conductor_guard._validate_runtime_assignment(
        args,
        _runtime_agent_ctx(),  # type: ignore[arg-type]
    )
    assert "blocked_by" in feedback