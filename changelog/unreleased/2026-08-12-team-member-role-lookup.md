# Expose team member agent role lookup

`FileTeamState` now exposes the registered agent role behind a member display name. Harness
capabilities can use this read-only lookup to validate that specialized tasks are assigned to
members with the required capability set.
