# Fix team-mode idle auto-cleanup leaking member slots in state.json

The idle auto-cleanup (`_schedule_member_cleanup`) closed member sessions
when the lead went idle but did not remove them from `state.json`'s
`members` dict. Since `team_add_member` counts members from `state.json`
to enforce `max_members`, the freed sessions still occupied capacity
slots and subsequent `team_add_member` calls were incorrectly rejected
with "Team exceeds max_members".

- `_schedule_member_cleanup` now accepts `team_id` and `base_dir`
  parameters and calls a new `_remove_members_from_state` helper after
  closing sessions, which pops members whose `session_id` matches the
  closed set. This mirrors the existing `_schedule_ephemeral_cleanup`
  pattern.
- `team_create` passes `team_id` and `base_dir` to the cleanup scheduler.
