## Why

The project is currently named **AgentPool** (`agentpool`). The new brand is **WolfHarness** (`wolfharness`). A prior rename attempt to the interim name **AgentWolf** (`agentwolf`) was never merged to `main` (PR #324 was closed). This change renames the project mechanically from `agentpool` to `wolfharness` across the repository, package distribution, Python namespace, CLI, documentation, and CI.

## What Changes

- **BREAKING**: Rename the PyPI package `agentpool` → `wolfharness` in `pyproject.toml` (`name`, `[project.scripts]`, entry points, URLs).
- **BREAKING**: Rename all `src/agentpool*` package directories to `src/wolfharness*` and update every Python import (`agentpool.*` → `wolfharness.*`).
- **BREAKING**: Rename the CLI entry point from `agentpool` → `wolfharness`.
- **BREAKING**: Rename the PyInstaller spec `agentpool.spec` → `wolfharness.spec` and update its metadata.
- Rename the Alembic database file `agentpool.db` → `wolfharness.db`.
- Rename Docker entrypoint, docs/js references, shell scripts, and `.rules` log paths from `agentpool` → `wolfharness`.
- Replace `AgentPool` → `WolfHarness` in README title, AGENTS.md, badges, and documentation links.
- Regenerate `uv.lock` so the package resolves as `wolfharness==2.9.5`.
- Add a `scripts/rename_to_wolfharness.py` mechanical-rename script (replaces the obsolete `scripts/rename_to_agentwolf.py`, which is removed).

## Capabilities

### New Capabilities

- `wolfharness-rename`: The mechanical repository, package, namespace, CLI, docs, and CI rename from `agentpool` to `wolfharness`.

### Modified Capabilities

- `agent-pool`: The core orchestration capability's package namespace changes from `agentpool` to `wolfharness`. All import paths and module references throughout the framework update to the new namespace. No behavioral requirements change — only the naming surface.

## Impact

- `pyproject.toml`: package `name`, `[project.scripts]`, `[project.entry-points."wolfharness.capabilities"]`, project URLs.
- `src/agentpool*` → `src/wolfharness*`: all 10 source package directories and their files.
- Every `.py` file importing `agentpool.*` or `agentpool_server.*` etc.
- `tests/`: all test files referencing `agentpool` import paths.
- `docs/`: README, `docs/reference/js/run_code_config.js`, AGENTS.md references.
- `Dockerfile`, `alembic.ini`, `agentpool.spec` → `wolfharness.spec`, shell scripts, `.rules`.
- `.github/workflows/`: CI references to the package/CLI name.
- `uv.lock`: regenerated to resolve `wolfharness`.
- Historical `agentwolf` references in archived RFCs and archived openspec changes are intentionally left untouched (they document the superseded interim name).
- Historical `agentpool` references in `openspec/changes` (archived) are intentionally excluded per the rename script's design.