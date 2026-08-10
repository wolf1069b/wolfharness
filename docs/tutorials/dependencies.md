---
title: Dependencies
description: Project dependencies and dependency tree
icon: material/package-variant-closed
---

# Dependencies

This page lists AgentPool's core runtime dependencies by category. For the complete, version-pinned list, see the project's `pyproject.toml`.

## Core Dependencies

| Category | Packages |
|---|---|
| **Framework & AI** | `pydantic`, `pydantic-ai-slim`, `pydantic-graph`, `pydantic-xml` |
| **Web, Server & Protocols** | `ag-ui-protocol`, `fastapi`, `fastmcp`, `httpx`, `mcp`, `starlette`, `uvicorn`, `websockets` |
| **Storage & Database** | `alembic`, `py-key-value-aio`, `sqlalchemy`, `sqlmodel` |
| **CLI & Configuration** | `platformdirs`, `python-dotenv`, `rich`, `schemez`, `typer`, `yamling` |
| **Async, IO & Execution** | `anyenv`, `anyio`, `exxec`, `filelock`, `fsspec`, `upathtools`, `watchfiles` |
| **Observability** | `jinjarope`, `logfire`, `structlog` |
| **Documents, Search & Embeddings** | `docler`, `pillow`, `ripgrep-rs`, `searchly`, `sublime-search`, `tokonomics` |
| **Tooling & Events** | `docstring-parser`, `epregistry`, `evented`, `jinja2`, `keyring`, `promptantic`, `psygnal`, `pydocket`, `slashed`, `toprompt` |

## Optional Extras

AgentPool installs optional features via extras. Some commonly used ones:

| Extra | Packages | Purpose |
|---|---|---|
| `coding` | `rustworkx`, `grep-ast`, `ast-grep-py`, `tree-sitter-*` | Code parsing and repo-map tools |
| `mcp-discovery` | `fastembed`, `lancedb`, `pyarrow` | Semantic MCP server discovery |
| `watchdog` | `watchdog` | Hot-reload skills and config files |
| `bot` | `python-telegram-bot`, `slack-sdk`, `slackify-markdown`, `croniter` | Chat-channel integrations |
| `tiktoken` | `tiktoken` | Exact token counting |
| `tts` | `anyvoice` | Text-to-Speech output |

For a full dependency tree, run:

```bash
pip show wolfharness
pipdeptree -p wolfharness
```