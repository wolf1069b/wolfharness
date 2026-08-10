## ADDED Requirements

### Requirement: Package and namespace renamed to wolfharness

The PyPI package, Python import namespace, and CLI entry point SHALL be renamed from `agentpool` to `wolfharness`. All source packages under `src/wolfharness*` SHALL be importable via the `wolfharness` namespace.

#### Scenario: Package builds and installs as wolfharness

- **WHEN** `uv sync --all-extras` is run
- **THEN** the project SHALL resolve and build as `wolfharness==2.9.5`

#### Scenario: Source directories renamed

- **WHEN** the repository is inspected
- **THEN** `src/agentpool*` directories SHALL NOT exist
- **AND** `src/wolfharness*` directories SHALL exist for all 10 source packages (core, bot, cli, commands, config, prompts, server, storage, sync, toolsets)

#### Scenario: CLI entry point renamed

- **WHEN** the CLI is invoked
- **THEN** the `agentpool` command SHALL be replaced by `wolfharness`
- **AND** `wolfharness --version`, `wolfharness serve-acp`, `wolfharness serve-opencode`, `wolfharness serve-mcp`, `wolfharness serve-agui`, `wolfharness serve-api` SHALL be the canonical commands

## MODIFIED Requirements

### Requirement: AgentPool package namespace SHALL be renamed to wolfharness

The core orchestration capability's package namespace SHALL change from `agentpool` to `wolfharness`. All import paths and module references throughout the framework SHALL update to the new namespace. No behavioral requirements change — only the naming surface.

#### Scenario: Imports resolve under wolfharness namespace

- **WHEN** a consumer imports `wolfharness`
- **THEN** the import SHALL resolve to the renamed package
- **AND** no `agentpool` import path SHALL remain in `src/`, `tests/`, or active configuration files