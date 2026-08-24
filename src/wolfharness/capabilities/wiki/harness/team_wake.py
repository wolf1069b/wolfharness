"""TeamWakeCapability — surface stalled / errored members to the lead.

A long-running team build silently hangs when a member stops working:

1. A member errors: ``NativeTurn.execute()`` emits ``RunErrorEvent``, the run
   ends, and ``after_run`` is never called.  ``TeamCommCapability.after_run``
   (which would remind the member) only fires on success, so the lead never
   learns the member stopped — unless some other hook notifies it.

2. A member idles with unfinished work: ``TeamCommCapability.after_run`` sends
   one QUEUE reminder per turn via the member's own session, but stops once
   ``_task_reminder_count >= max_task_reminders``.  After that the member is
   silent and the build hangs.

This capability plugs both holes at the harness level (no team-mode rewrite):

- ``on_run_error`` fires whenever an agent run fails (unlike ``after_run``,
  which only fires on success).  It STEERs a ``<team-message>`` notification
  to the lead agent's session so the conductor can reassign, retry, or take
  over the failing member's task.
- ``after_run`` detects the reminder-exhaustion window (unfinished tasks +
  the TeamComm reminder counter pinned at the fingerprint) and notifies the
  lead instead of staying silent; it re-arms the fingerprint tracking so the
  lead is kept abreast of a member that keeps failing to finish.

Both paths are full-defensive: if the current agent is not in a team, has no
``team_id``/conductor session, or no ``session_pool`` is available, the
capability no-ops and never modifies run control flow.  ``on_run_error`` still
re-raises the original error so the run's failure semantics are unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from wolfharness.capabilities.agent_context import (
    AgentContextDeps,
    resolve_agent_context_from_deps,
)
from wolfharness.capabilities.file_team_state import FileTeamState
from wolfharness.lifecycle.types import DeliveryMode


if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.run import AgentRunResult

_MAX_CONSECUTIVE_EMPTY_RUNS = 3
_PHASE_DONE_RE = re.compile(r'"phase"\s*:\s*"done"')


@dataclass(frozen=True, slots=True)
class StalledTaskLease:
    """A worker lease that the patrol released or permanently blocked."""

    task_id: str
    owner: str
    blocked: bool


class TeamWakeCapability(AbstractCapability[Any]):
    """Notify the lead when a member errors or exhausts its task reminders.

    Adds no tools and no instructions — it is a pure run-lifecycle observer
    that routes alerts to the team lead (conductor) via
    ``SessionPool.send_message(..., mode=DeliveryMode.STEER)``.
    """

    def __init__(
        self,
        *,
        conductor_name: str = "wiki_conductor",
        patrol_interval: int = 60,
        stall_patrol_limit: int = 3,
        max_task_takeovers: int = 2,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        if stall_patrol_limit < 1:
            raise ValueError("stall_patrol_limit must be positive")
        if max_task_takeovers < 0:
            raise ValueError("max_task_takeovers must not be negative")
        self._conductor_name = conductor_name
        self._patrol_interval = patrol_interval
        self._stall_patrol_limit = stall_patrol_limit
        self._max_task_takeovers = max_task_takeovers
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _is_empty_run(result: AgentRunResult[Any]) -> bool:
        """Check if a run produced no text content and no tool calls."""
        try:
            new_msgs = result.new_messages()
        except (RuntimeError, ValueError, TypeError, KeyError):
            return False
        for msg in new_msgs:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        return False
                    if isinstance(part, TextPart) and part.content.strip():
                        return False
        return True

    @staticmethod
    def _resolve_agent_context(
        ctx: RunContext[Any],
    ) -> AgentContextDeps:
        """Extract agent context from a pydantic-ai ``RunContext``."""
        return resolve_agent_context_from_deps(
            ctx.deps,
            capability_name="TeamWakeCapability",
        )

    def _team_state(self, agent_ctx: AgentContextDeps) -> FileTeamState | None:
        """Build a FileTeamState for the current team, if any."""
        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is None:
            return None
        base_dir: str = agent_ctx.session.metadata.get(
            "team_base_dir",
            "",
        )
        if not base_dir and agent_ctx.team_mode_config is not None:
            base_dir = agent_ctx.team_mode_config.effective_base_dir
        if not base_dir:
            return None

        return FileTeamState(base_dir)

    @staticmethod
    def _read_build_state(
        team_state: FileTeamState,
        team_id: str,
        *,
        max_chars: int = 3000,
    ) -> str:
        """Read build_state from the blackboard for post-compaction recovery.

        Returns a truncated text snapshot of the conductor's persisted build
        state, or an empty string if unavailable.  This is injected into
        patrol messages so the conductor can recover its plan after DCP
        compaction destroys conversation history.
        """
        try:
            entry = team_state.read_blackboard(team_id, "build_state")
        except (ValueError, OSError, TypeError, KeyError):
            return ""
        if not entry:
            return ""
        text = ""
        value = entry.get("value", {})
        if isinstance(value, dict):
            text = value.get("text", "")
        elif isinstance(value, str):
            text = value
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated, call read_blackboard for full)"
        return text

    @staticmethod
    def _build_is_done(build_state_text: str) -> bool:
        """Return whether the durable build state records terminal success."""
        return bool(_PHASE_DONE_RE.search(build_state_text))

    @staticmethod
    def _unfinished_team_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return tasks that still require execution or reassignment."""
        return [
            task for task in tasks if task.get("status") in {"pending", "in_progress", "blocked"}
        ]

    @staticmethod
    def _patrol_attention_fingerprint(
        tasks: list[dict[str, Any]],
        released: list[StalledTaskLease],
    ) -> str:
        """Describe only states that require a conductor wake-up.

        Active workers report completion through team messages, so a patrol
        must not inject another prompt merely because work is progressing.
        Pending/blocked work, a released lease, or an empty task board before
        terminal ``build_state`` all require conductor action.
        """
        released_ids = sorted(
            f"{lease.task_id}:{'blocked' if lease.blocked else 'pending'}" for lease in released
        )
        attention_tasks = sorted(
            f"{task.get('task_id', '')}:{task.get('status', '')}:{task.get('owner', '')}"
            for task in tasks
            if task.get("status") in {"pending", "blocked"}
        )
        if released_ids:
            return "released=" + "|".join(released_ids)
        if attention_tasks:
            return "attention=" + "|".join(attention_tasks)
        if not TeamWakeCapability._unfinished_team_tasks(tasks):
            return "needs-next-phase"
        return ""

    async def _notify_lead(
        self,
        agent_ctx: AgentContextDeps,
        *,
        team_id: str,
        body: str,
    ) -> bool:
        """STEER a team-message notice to the conductor, best-effort.

        Returns ``True`` when the notification was handed to the session pool.
        """
        team_state = self._team_state(agent_ctx)
        if team_state is None:
            return False
        lead_sid = team_state.get_member_session_id(
            team_id,
            self._conductor_name,
        )
        if not lead_sid:
            return False
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return False

        wrapped = f'<team-message from="system" type="member_alert">\n\n{body}\n\n</team-message>'
        await session_pool.send_message(
            lead_sid,
            wrapped,
            mode=DeliveryMode.STEER,
            source="accepted",
            meta={"from": "system", "team_id": team_id},
        )
        return True

    def _finish_body(
        self,
        agent_ctx: AgentContextDeps,
    ) -> str:
        member_name: str = agent_ctx.session.metadata.get(
            "team_member_name",
            "",
        )
        return (
            f"member '{member_name or agent_ctx.scope.session_id}' ran but "
            f"could not clear its in_progress tasks (TaskComm reminders "
            f"exhausted). Please reassign, retry, or take over via "
            f"task_create / send_message."
        )

    @staticmethod
    def _task_progress_fingerprint(task: dict[str, Any]) -> str:
        """Fingerprint the task board's real, observable progress fields.

        Only fields a working worker can actually mutate are included.  The
        legacy fingerprint also read ``artifact_revision`` and
        ``tool_call_count``, neither of which any code writes, so they were
        permanently empty noise that could not reflect artifact activity.
        """
        fields = (
            "status",
            "owner",
            "progress_current",
            "progress_total",
            "last_note",
            "updated_at",
        )
        return "|".join(f"{field}={task.get(field, '')}" for field in fields)

    def _release_stalled_tasks(
        self,
        agent_ctx: AgentContextDeps,
        team_state: FileTeamState,
        team_id: str,
        tasks: list[dict[str, Any]],
    ) -> list[StalledTaskLease]:
        """Release or block task leases after configurable no-progress patrols."""
        previous: dict[str, str] = agent_ctx.session.metadata.setdefault(
            "_task_progress_fingerprints",
            {},
        )
        counts: dict[str, int] = agent_ctx.session.metadata.setdefault(
            "_task_stall_patrol_counts",
            {},
        )
        active_ids: set[str] = set()
        released: list[StalledTaskLease] = []
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            owner = str(task.get("owner", ""))
            if (
                not task_id
                or not owner
                or owner == self._conductor_name
                or task.get("status") != "in_progress"
            ):
                continue
            active_ids.add(task_id)
            fingerprint = self._task_progress_fingerprint(task)
            counts[task_id] = (
                counts.get(task_id, 0) + 1 if previous.get(task_id) == fingerprint else 1
            )
            previous[task_id] = fingerprint
            if counts[task_id] < self._stall_patrol_limit:
                continue
            takeover_count = int(task.get("takeover_count", 0)) + 1
            blocked = takeover_count > self._max_task_takeovers
            status = "blocked" if blocked else "pending"
            team_state.update_task(
                team_id,
                task_id,
                {
                    "status": status,
                    "owner": "",
                    "takeover_count": takeover_count,
                    "last_note": (
                        "automatic takeover limit exceeded; conductor must diagnose and replan"
                        if blocked
                        else "lease released after patrols observed no external progress"
                    ),
                },
            )
            released.append(StalledTaskLease(task_id=task_id, owner=owner, blocked=blocked))
            counts.pop(task_id, None)
            previous.pop(task_id, None)
        for task_id in set(previous) - active_ids:
            previous.pop(task_id, None)
            counts.pop(task_id, None)
        return released

    async def _terminate_released_workers(
        self,
        agent_ctx: AgentContextDeps,
        team_state: FileTeamState,
        team_id: str,
        released: list[StalledTaskLease],
    ) -> None:
        """Stop sessions whose task lease was revoked before reassignment."""
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return
        for owner in {lease.owner for lease in released}:
            session_id = team_state.get_member_session_id(team_id, owner)
            if session_id:
                await session_pool.close_session(session_id)

    # -- lifecycle hooks --------------------------------------------------

    async def on_run_error(
        self,
        ctx: RunContext[Any],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Notify the lead that the member errored, then re-raise.

        For the lead itself, re-arm the patrol so a timeout/error doesn't
        leave it permanently asleep with no pending self-wake.
        """
        try:
            agent_ctx = self._resolve_agent_context(ctx)
        except RuntimeError:
            raise error from None

        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is not None:
            if agent_ctx.session.metadata.get("team_role") == "lead":
                self._schedule_patrol(agent_ctx)
            else:
                self._release_owned_tasks(agent_ctx, team_id)
                member_name: str = agent_ctx.session.metadata.get(
                    "team_member_name",
                    "",
                )
                body_lines = (
                    f"member '{member_name or agent_ctx.scope.session_id}' "
                    f"errored during a run:"
                    f"\n\n{type(error).__name__}: {error}"
                    f"\n\nPlease reassign, retry, or take over this member's "
                    f"task via task_create / send_message."
                )
                await self._notify_lead(
                    agent_ctx,
                    team_id=team_id,
                    body=body_lines,
                )

        raise error

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Notify the lead when a member idles with unfinished work.

        For the lead itself, arms the recurring patrol AND runs an
        immediate post-turn health check so stalled members are caught
        without waiting 60 s for the first patrol round.
        """
        try:
            agent_ctx = self._resolve_agent_context(ctx)
        except RuntimeError:
            return result

        if agent_ctx.session.metadata.get("team_role") == "lead":
            if self._is_empty_run(result):
                count = (
                    agent_ctx.session.metadata.get(
                        "_consecutive_empty_runs",
                        0,
                    )
                    + 1
                )
                agent_ctx.session.metadata["_consecutive_empty_runs"] = count
            else:
                agent_ctx.session.metadata["_consecutive_empty_runs"] = 0
            agent_ctx.session.metadata.pop("_last_patrol_attention_fp", None)
            self._schedule_patrol(agent_ctx)
            empty_count = agent_ctx.session.metadata.get(
                "_consecutive_empty_runs",
                0,
            )
            if empty_count >= _MAX_CONSECUTIVE_EMPTY_RUNS:
                sent = await self._send_recovery_prompt(agent_ctx)
                if sent:
                    agent_ctx.session.metadata["_consecutive_empty_runs"] = 0
            else:
                await self._immediate_health_check(agent_ctx)
            return result
        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is None:
            return result

        unfinished = self._unfinished_tasks(agent_ctx, team_id)
        if not unfinished:
            return result

        fingerprint = self._worker_progress_fingerprint(unfinished)
        last_fingerprint: str | None = agent_ctx.session.metadata.get(
            "_task_reminder_fingerprint",
        )
        stall_turns: int = (
            agent_ctx.session.metadata.get("_task_stall_turn_count", 0) + 1
            if last_fingerprint == fingerprint
            else 1
        )
        agent_ctx.session.metadata["_task_reminder_fingerprint"] = fingerprint
        agent_ctx.session.metadata["_task_stall_turn_count"] = stall_turns

        # Only treat the worker as stalled once its unfinished tasks have made
        # NO progress for `stall_patrol_limit` consecutive turns. A worker that
        # finishes a turn with its task still `in_progress` but is actively
        # making progress (a normal multi-turn task) must NOT be reported as a
        # stall: the old `last_fingerprint is None / != fingerprint` condition
        # fired on almost every turn, wiping the worker's OWN task ownership
        # (`_release_owned_tasks`) and flooding the lead with false
        # member_alerts — whose "reminders exhausted" text was inaccurate —
        # which drove the conductor to repeatedly reassign and spawn new
        # workers (the fuse/reassign storm during phase 1B).
        if stall_turns < self._stall_patrol_limit:
            return result

        # Genuinely stalled: release ownership so the lead can reassign, then
        # notify. Reset the counter so repeated alerts only fire after another
        # full no-progress window.
        self._release_owned_tasks(agent_ctx, team_id)
        await self._notify_lead(
            agent_ctx,
            team_id=team_id,
            body=self._finish_body(agent_ctx),
        )
        agent_ctx.session.metadata["_task_stall_turn_count"] = 0

        return result

    async def _send_recovery_prompt(self, agent_ctx: AgentContextDeps) -> bool:
        """Break the conductor zombie loop with a focused recovery message.

        When the conductor produces ``_MAX_CONSECUTIVE_EMPTY_RUNS`` empty
        turns, the model is likely stuck (compacted context, lost plan).
        Send a single recovery message with build_state + task summary via
        QUEUE so the next run has a concrete prompt to act on, instead of
        relying on the 60 s patrol timer.

        Returns ``True`` when the message was handed to the session pool.
        """
        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is None:
            return False
        team_state = self._team_state(agent_ctx)
        if team_state is None:
            return False
        own_sid = agent_ctx.scope.session_id
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return False

        build_state_text = self._read_build_state(
            team_state,
            team_id,
            max_chars=3000,
        )
        state_block = ""
        if build_state_text:
            state_block = f"\n== build_state (from blackboard) ==\n{build_state_text}\n\n"
        tasks = team_state.list_tasks(team_id)
        pending = [t for t in tasks if t.get("status") in {"pending", "blocked"}]
        completed = [t for t in tasks if t.get("status") == "completed"]
        in_progress = [t for t in tasks if t.get("status") == "in_progress"]
        summary = f"Tasks: {len(completed)} completed, {len(in_progress)} in_progress, {len(pending)} pending/blocked"
        await session_pool.send_message(
            own_sid,
            (
                '<team-message from="system" type="recovery">\n\n'
                f"Conductor zombie loop detected — {summary}.{state_block}\n"
                "== 恢复指令（强制，最高优先级）==\n"
                "1. 读取上方 build_state 确定当前阶段\n"
                "2. 调用 task_list(include_children=True) 获取任务板\n"
                "3. 根据 build_state + 任务板派发下一步工作\n"
                "4. 有 pending/blocked task → 立即创建 1B task 并派发\n"
                "5. 全部 in_progress → end turn 等待完成通知\n"
                "6. 禁止只调用 team_status 就 end turn\n\n"
                "</team-message>"
            ),
            mode=DeliveryMode.QUEUE,
            source="accepted",
            meta={"from": "system", "team_id": team_id},
        )
        return True

    async def _immediate_health_check(self, agent_ctx: AgentContextDeps) -> None:
        """Wake the conductor only when it must make a dispatch decision.

        Workers report completion through team messages, and the patrol
        handles stalled members, so the conductor must not be woken merely
        because tasks are ``in_progress`` (that produced a tight self-feedback
        loop: conductor wakes, re-queries the whole board to re-verify an
        already-known state, and is immediately re-nudged).

        This fires only for states that genuinely require conductor action:
        pending/blocked tasks awaiting dispatch, or an empty-but-not-done
        board.  It deliberately omits the stale build_state snapshot, which
        disagreed with the conductor's own (newer) memory and forced repeated
        re-verification.
        """
        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is None:
            return
        team_state = self._team_state(agent_ctx)
        if team_state is None:
            return

        tasks = team_state.list_tasks(team_id)
        released: list[StalledTaskLease] = []
        attention_fingerprint = self._patrol_attention_fingerprint(tasks, released)
        if not attention_fingerprint:
            return

        last_fp: str | None = agent_ctx.session.metadata.get(
            "_conductor_health_fp",
        )
        if last_fp == attention_fingerprint:
            return
        agent_ctx.session.metadata["_conductor_health_fp"] = attention_fingerprint

        own_sid = agent_ctx.scope.session_id
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return

        actionable = [
            t
            for t in sorted(tasks, key=lambda t: t.get("task_id", ""))
            if t.get("status") in {"pending", "blocked"}
        ]
        summary = (
            "\n".join(
                f"- {t.get('task_id', '')}: {t.get('status', '')} owner={t.get('owner', '')} {t.get('subject', '')}"
                for t in actionable
            )
            or "- 任务板已清空；继续 audit、修复、backlinks、finalize。"
        )
        body = (
            '<team-message from="system" type="health_check">\n\n'
            "Post-turn: work awaits conductor action:\n"
            f"{summary}\n"
            f"attention={attention_fingerprint}\n"
            "根据下方待办派发或处理即可；若无待决策项可直接 end turn。\n\n"
            "</team-message>"
        )
        await session_pool.send_message(
            own_sid,
            body,
            mode=DeliveryMode.QUEUE,
            source="accepted",
            meta={"from": "system", "team_id": team_id},
        )

    def _schedule_patrol(self, agent_ctx: AgentContextDeps) -> None:
        """Arm a recurring self-wake so the lead patrols team status.

        The patrol loops until durable ``build_state.phase=done``. Unlike a one-shot
        timer, this does not depend on the conductor completing a turn
        to re-arm — the loop continues independently as long as there is
        unfinished work, surviving transient turn failures that would
        strand a one-shot timer.

        A completed task board is not terminal: the conductor may still need
        to audit, repair, rebuild backlinks, or finalize. A metadata flag
        prevents stacking overlapping timers across consecutive turns.
        """
        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is None:
            return
        team_state = self._team_state(agent_ctx)
        if team_state is None:
            return
        build_state_text = self._read_build_state(team_state, team_id)
        if self._build_is_done(build_state_text):
            return
        if agent_ctx.session.metadata.get("_patrol_armed"):
            return
        own_sid = agent_ctx.scope.session_id
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return

        agent_ctx.session.metadata["_patrol_armed"] = True

        async def _patrol() -> None:
            try:
                while True:
                    await asyncio.sleep(self._patrol_interval)
                    empty_count = agent_ctx.session.metadata.get(
                        "_consecutive_empty_runs",
                        0,
                    )
                    if empty_count >= _MAX_CONSECUTIVE_EMPTY_RUNS:
                        agent_ctx.session.metadata["_consecutive_empty_runs"] = 0
                        build_state_text = self._read_build_state(
                            team_state,
                            team_id,
                            max_chars=3000,
                        )
                        state_block = ""
                        if build_state_text:
                            state_block = (
                                f"\n== build_state (from blackboard) ==\n{build_state_text}\n\n"
                            )
                        await session_pool.send_message(
                            own_sid,
                            (
                                '<team-message from="system"'
                                ' type="stall_alert">\n\n'
                                f"Conductor produced {empty_count} consecutive"
                                " empty turns (复读 loop). Context was compacted"
                                " — you lost your build plan."
                                f"{state_block}\n"
                                "== 恢复指令（强制）==\n"
                                "1. 读取上方 build_state 确定当前阶段\n"
                                "2. 调用 task_list(include_children=True) 获取任务板\n"
                                "3. 根据 build_state + 任务板派发下一步工作\n"
                                "4. 禁止只调用 team_status 就 end turn\n\n"
                                "</team-message>"
                            ),
                            mode=DeliveryMode.QUEUE,
                            source="accepted",
                            meta={"from": "system", "team_id": team_id},
                        )
                        # ponytail: continue patrolling instead of breaking.
                        # Breaking permanently disarmed the conductor after 3
                        # empty turns, leaving it in a zombie state with no
                        # self-wake mechanism. Continue so the patrol keeps
                        # checking; the empty-run counter was already reset.
                        continue
                    current = team_state.list_tasks(team_id)
                    released = self._release_stalled_tasks(
                        agent_ctx,
                        team_state,
                        team_id,
                        current,
                    )
                    await self._terminate_released_workers(
                        agent_ctx,
                        team_state,
                        team_id,
                        released,
                    )
                    build_state_text = self._read_build_state(
                        team_state,
                        team_id,
                        max_chars=3000,
                    )
                    if self._build_is_done(build_state_text):
                        break
                    attention_fingerprint = self._patrol_attention_fingerprint(
                        current,
                        released,
                    )
                    if not attention_fingerprint:
                        # Workers are making progress. Their completion/error
                        # notifications will wake the conductor; another patrol
                        # prompt would only grow context and encourage repetition.
                        agent_ctx.session.metadata.pop(
                            "_last_patrol_attention_fp",
                            None,
                        )
                        continue
                    if (
                        agent_ctx.session.metadata.get("_last_patrol_attention_fp")
                        == attention_fingerprint
                    ):
                        continue
                    agent_ctx.session.metadata["_last_patrol_attention_fp"] = attention_fingerprint
                    state_block = ""
                    if build_state_text:
                        state_block = (
                            f"\n== build_state (from blackboard) ==\n{build_state_text}\n\n"
                        )
                    release_block = ""
                    if released:
                        retry_ids = [lease.task_id for lease in released if not lease.blocked]
                        blocked_ids = [lease.task_id for lease in released if lease.blocked]
                        release_block = (
                            "\n连续无进展 worker 会话已终止。"
                            + (
                                "任务已释放为 pending："
                                + ", ".join(retry_ids)
                                + "。请立即重新分配。"
                                if retry_ids
                                else ""
                            )
                            + (
                                " 超过自动接管上限、已转 blocked，conductor 必须诊断并拆小/换策略："
                                + ", ".join(blocked_ids)
                                + "。"
                                if blocked_ids
                                else ""
                            )
                            + "\n"
                        )
                    await session_pool.send_message(
                        own_sid,
                        (
                            '<team-message from="system"'
                            ' type="patrol">\n\n'
                            f"Patrol: 恢复构建状态后检查成员。{state_block}{release_block}\n"
                            "== 指令 ==\n"
                            "1. 根据 build_state 确定当前阶段和下一步\n"
                            "2. 检查 team_status 确认 worker 状态\n"
                            "3. 有未派发或 blocked 的工作 → 立即修复、重试或重新派发\n"
                            "4. 任务板已清空/全完成但 build_state 未 done → 继续 audit、修复、backlinks、finalize\n"
                            "5. 所有工作已派发且 worker 正在执行 → end turn\n"
                            "6. 禁止只检查 team_status 就 end turn\n\n"
                            "</team-message>"
                        ),
                        mode=DeliveryMode.QUEUE,
                        source="accepted",
                        meta={"from": "system", "team_id": team_id},
                    )
            finally:
                agent_ctx.session.metadata["_patrol_armed"] = False
                agent_ctx.session.metadata.pop("_patrol_task", None)

        agent_ctx.session.metadata["_patrol_task"] = asyncio.create_task(_patrol())

    def _release_owned_tasks(
        self,
        agent_ctx: AgentContextDeps,
        team_id: str,
    ) -> None:
        """Clear the owner of this member's unfinished tasks.

        When a member errors or idles with work left undone, keeping its
        ``in_progress`` tasks owned blocks other members from updating them
        (the agentpool ``task_update`` tool rejects writes to another owner).
        Releasing ownership lets the lead reassign or another worker claim
        them.  Best-effort: parsing is done here, so only grab the tasks we
        currently own and clear that field.
        """
        team_state = self._team_state(agent_ctx)
        if team_state is None:
            return
        member_name: str = agent_ctx.session.metadata.get(
            "team_member_name",
            "",
        )
        if not member_name:
            return
        for task in self._unfinished_tasks(agent_ctx, team_id):
            team_state.update_task(team_id, task["task_id"], {"owner": ""})

    def _unfinished_tasks(
        self,
        agent_ctx: AgentContextDeps,
        team_id: str,
    ) -> list[dict[str, Any]]:
        team_state = self._team_state(agent_ctx)
        if team_state is None:
            return []
        member_name: str = agent_ctx.session.metadata.get(
            "team_member_name",
            "",
        )
        if not member_name:
            return []
        return [
            task
            for task in team_state.list_tasks(team_id)
            if task.get("owner") == member_name and task.get("status") in {"pending", "in_progress"}
        ]

    @staticmethod
    def _worker_progress_fingerprint(tasks: list[dict[str, Any]]) -> str:
        """Aggregate a worker's unfinished-task progress into one fingerprint.

        Includes per-task progress observables (status/owner/notes/progress)
        so a worker that is actively making progress across turns — even
        without completing its task — produces a *different* fingerprint each
        turn.  An idle worker whose task state never changes produces the
        same fingerprint, which the stall counter then treats as stalled.
        """
        rows = []
        for task in sorted(tasks, key=lambda t: t.get("task_id", "")):
            row_fields = (
                task.get("task_id", ""),
                task.get("status", ""),
                task.get("owner", ""),
                task.get("progress_current", ""),
                task.get("progress_total", ""),
                task.get("last_note", ""),
                task.get("updated_at", ""),
            )
            rows.append("|".join(str(f) for f in row_fields))
        return "\n".join(rows)
