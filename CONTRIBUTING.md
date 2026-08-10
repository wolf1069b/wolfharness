# Contributing to WolfHarness

Thank you for your interest in contributing! This guide covers the basics.

## Development Setup

```bash
uv sync --all-extras                    # Install dependencies
uv run pytest                           # Run tests
uv run ruff check src/                  # Lint
uv run --no-group docs mypy src/        # Type check
```

See `AGENTS.md` for the full development workflow, code style rules, and testing guidelines.

## Making Changes

All significant changes go through OpenSpec:

```
/opsx:explore  → Investigate problems, map codebase, compare options
/opsx:propose  → Create proposal + design + specs + tasks
/opsx:apply    → Implement tasks
/opsx:archive  → Archive completed change
```

Major architectural decisions require an RFC first. See `docs/rfcs/STATUS.md` for the RFC pipeline.

## Documentation

Unsure where to put new documentation? Read the [Documentation Guide](docs/meta/documentation-guide.md).

**Rule of thumb**: If you don't know where your docs go, open an issue and ask. Do not create a new top-level directory under `docs/`.

## AI-Assisted Contribution

If you use AI tools (OpenCode, Codex, Claude Code, Cursor, Copilot, Gemini), your tool reads `AGENTS.md` automatically. It contains coding rules, context loading guides, and tool compatibility info.
