# Design — rename-agentpool-to-wolfharness

## Overview

A one-shot mechanical rename of the project from **AgentPool** (`agentpool`) to **WolfHarness** (`wolfharness`). This supersedes the earlier `agentwolf` rename attempt (PR #324, closed, never merged to `main`).

## Approach

Reuse the well-tested mechanical-rename script pattern that already existed in the repository (`scripts/rename_to_agentwolf.py`), adapted for the `wolfharness` target. The rename is executed on a branch based on `origin/main` (which never contained the `agentwolf` package code, so there are no `agentwolf` package remnants to clean up).

### Rename script (`scripts/rename_to_wolfharness.py`)

Based on the prior `rename_to_agentwolf.py`, modified:

1. **Directory renames** — `src/agentpool*` → `src/wolfharness*` for all 10 source packages:
   - `src/agentpool` → `src/wolfharness`
   - `src/agentpool_bot` → `src/wolfharness_bot`
   - `src/agentpool_cli` → `src/wolfharness_cli`
   - `src/agentpool_commands` → `src/wolfharness_commands`
   - `src/agentpool_config` → `src/wolfharness_config`
   - `src/agentpool_prompts` → `src/wolfharness_prompts`
   - `src/agentpool_server` → `src/wolfharness_server`
   - `src/agentpool_storage` → `src/wolfharness_storage`
   - `src/agentpool_sync` → `src/wolfharness_sync`
   - `src/agentpool_toolsets` → `src/wolfharness_toolsets`

2. **Content replacements** — ordered longest-first to avoid partial-write collisions:
   - `agentpool_config` → `wolfharness_config`
   - `agentpool_server` → `wolfharness_server`
   - `agentpool_toolsets` → `wolfharness_toolsets`
   - `agentpool_storage` → `wolfharness_storage`
   - `agentpool_cli` → `wolfharness_cli`
   - `agentpool_commands` → `wolfharness_commands`
   - `agentpool_prompts` → `wolfharness_prompts`
   - `agentpool_sync` → `wolfharness_sync`
   - `agentpool_bot` → `wolfharness_bot`
   - `agentpool` → `wolfharness`

3. **Excluded directories** — `.git`, `.venv`, caches, `node_modules`, `.codegraph`, `openspec/changes` (preserve historical spec docs), `.omo`.

### Manual follow-ups (not covered by the script)

The script's `FILE_PATTERNS` only covers `*.py`, `*.toml`, `*.yml`, `*.yaml`, `*.md`, `*.cfg`, `*.txt`, `*.rst`, `*.json`. Post-script manual fixes handled:

- `agentpool.spec` → `wolfharness.spec` (file rename via `git mv` + content replacement).
- `alembic.ini` — `sqlalchemy.url = sqlite:///./agentpool.db` → `wolfharness.db`.
- `Dockerfile` — `ENTRYPOINT ["agentpool", ...]` → `wolfharness`.
- `docs/reference/js/run_code_config.js` — `agentpool` → `wolfharness`.
- `scripts/restart_opencode_server.sh` — `agentpool` → `wolfharness`.
- `src/wolfharness_server/opencode_server/.rules` — log paths `~/.local/state/agentpool/...` → `wolfharness`.
- Remove obsolete `scripts/rename_to_agentwolf.py`.

### Ruff line-length fixes

The longer `wolfharness` name pushed a few lines over the 100-char limit. Fixed by reflowing docstrings, splitting string constructions, and adding the missing `# noqa: E402` to one import in `session_controller.py` (matching the existing deliberate circular-import pattern).

## Verification

- `uv sync --all-extras` — resolves and builds `wolfharness==2.9.5`.
- `uv run ruff check src/ tests/` — passes.
- `uv run ruff format src/ tests/ --check` — passes.
- `uv run pytest -m unit` — results unchanged from `origin/main` baseline.

## Out of scope

- The GitHub repository rename `Leoyzen/agentpool` → `Leoyzen/wolfharness` (repository-settings operation, done after this PR lands).
- First `wolfharness` PyPI release and deprecation notice on the final `agentpool` release (publishing steps).
- Migration guide for users (follow-up).
- Historical `agentwolf` references in archived RFCs and archived openspec changes (document the superseded interim name; left intact).
- `uv.lock`'s `agentpool` dependency (a separate third-party PyPI package pulled by `mknodes`, not the project's own package; left intact).