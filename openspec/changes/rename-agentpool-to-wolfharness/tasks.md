# Tasks — rename-agentpool-to-wolfharness

## 1. Create the mechanical rename script

- [x] 1.1 Adapt `scripts/rename_to_agentwolf.py` → `scripts/rename_to_wolfharness.py` (replace `agentpool`→`agentwolf` with `agentpool`→`wolfharness` in `SRC_DIRS_TO_RENAME` and `REPLACEMENTS`)
- [x] 1.2 Remove the obsolete `scripts/rename_to_agentwolf.py`

## 2. Execute the directory rename

- [x] 2.1 Run `python3 scripts/rename_to_wolfharness.py` (dry-run first to verify)
- [x] 2.2 Verify all 10 `src/agentpool*` directories renamed to `src/wolfharness*`
- [x] 2.3 Verify entry-point files exist post-rename (`wolfharness/__init__.py`, `wolfharness_cli/__init__.py`)

## 3. Replace references in content

- [x] 3.1 Run the reference-replacement step (1421 files updated)
- [x] 3.2 Rename `agentpool.spec` → `wolfharness.spec` via `git mv` and replace content references
- [x] 3.3 Update `alembic.ini` — `sqlite:///./agentpool.db` → `wolfharness.db`
- [x] 3.4 Update `Dockerfile` — `ENTRYPOINT ["agentpool", ...]` → `wolfharness`
- [x] 3.5 Update `docs/reference/js/run_code_config.js` — `agentpool` → `wolfharness`
- [x] 3.6 Update `scripts/restart_opencode_server.sh` — `agentpool` → `wolfharness`
- [x] 3.7 Update `src/wolfharness_server/opencode_server/.rules` — log paths → `wolfharness`
- [x] 3.8 Verify no residual `agentpool` references remain (excluding `openspec/changes` archives and the third-party `agentpool` PyPI dependency in `uv.lock`)
  - [x] 3.8.1 Rename `tests/agentpool_server/` → `tests/wolfharness_server/` (post-merge audit found the directory omitted from the original `src/*` rename list)
  - [x] 3.8.2 Rename `agentpool-session-pool/` → `wolfharness-session-pool/` (same omission)

## 4. Fix lint issues from the longer name

- [x] 4.1 Reflow docstring in `src/wolfharness/agents/native_agent/process_history_capability.py` (E501)
- [x] 4.2 Reflow docstring in `src/wolfharness/capabilities/skill_manager_cap.py` (E501)
- [x] 4.3 Split line in `src/wolfharness/repomap/languages.py` (E501)
- [x] 4.4 Split line in `src/wolfharness_config/capabilities.py` (E501)
- [x] 4.5 Add missing `# noqa: E402` to `session_controller.py` import (matches existing circular-import pattern)
- [x] 4.6 Split line in `tests/servers/opencode_server/test_dedup.py` (E501)

## 5. Regenerate lockfile

- [x] 5.1 Run `uv sync --all-extras` to regenerate `uv.lock` and verify `wolfharness==2.9.5` builds

## 6. Verify

- [x] 6.1 `uv run ruff check src/ tests/` passes
- [x] 6.2 `uv run ruff format src/ tests/ --check` passes
- [x] 6.3 `uv run pytest -m unit` results unchanged from `origin/main` baseline

## 7. Deliverable

- [x] 7.1 Commit the rename as a single atomic commit
- [x] 7.2 Push branch and open PR against `origin/main`
- [x] 7.3 Link PR to issue #357 and milestone 2 ("Rebrand: AgentPool → WolfHarness")