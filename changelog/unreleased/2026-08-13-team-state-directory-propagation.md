# Propagate team state directory to spawned members

Team sessions now pass the resolved `team_base_dir` into every initially
created or dynamically added member session. This keeps task-board and
blackboard consumers aligned with the lead when the runtime falls back to a
temporary team-state directory.

Team members are also prevented from moving a second task to `in_progress`
while another owned task is active. This preserves a single authoritative
current task for heartbeat, timeout, and reassignment logic; leads retain the
ability to repair task state explicitly.

Task ownership changes now follow the same authority boundary: members may
claim an unowned task only for themselves, while transfers between members
must be performed by the team lead. This prevents a worker with stale or
incorrect identity context from corrupting another worker's queue.

Leads can no longer shut down a member while it owns an `in_progress` task.
They must first explicitly reassign or cancel the active task, preventing
delayed status messages from terminating a worker that is still decoding or
writing an artifact. Pending work continues to produce the existing shutdown
warning so it can be reassigned normally.

Dynamic members now always receive the rendered team protocol, including
their exact `team_member_name`, before any caller-supplied task prompt. The
new member is also included in its initial roster. A custom prompt therefore
supplements the member identity instead of replacing it, avoiding workers
mistaking themselves for an earlier member that uses the same agent config.

Team messages now include a UTC `sent_at` timestamp, and the protocol requires
destructive coordination decisions to verify newer task state. In addition,
`shutdown_request` rejects members with an active run, closing the gap where a
worker could be killed during a storage call before its task heartbeat reached
`in_progress`.
