---
title: Bind initial work before waking dynamic team members
type: fixed
---

`team_add_member` now accepts either `initial_task` or `initial_task_id`. The
team task is persisted and assigned before the new member receives its first
message, preventing dynamic workers from observing an empty `mine_only` task
view and exiting before dispatch completes. Recovery can bind a released
pending or blocked task to a replacement member without creating a duplicate
task ID.
