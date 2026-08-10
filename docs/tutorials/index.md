---
title: Installation
description: Installation instructions for AgentPool
order: 1
icon: material/school
---

# Installation

## Basic Installation

Simple installation

```bash
uv tool install wolfharness

```

Multiple extras

```bash
uv tool install wolfharness[default, coding]
```

## Available Extras

| Extra | Dependencies | Description |
|-------|--------------|-------------|
| `a2a` | `fasta2a`, `starlette` | A2A Server |
| `bot` | `python-telegram-bot[socks]>=21.0`, `slack-sdk>=3.26.0`, `slackify-markdown>=0.2.0`, `croniter>=2.0.0` | Chat channel integrations (Telegram, Slack, Discord, etc.) |
| `braintrust` | `braintrust`, `autoevals` | Braintrust Prompt Hub |
| `clipboard` | `copykitten` | Clipboard functionality |
| `coding` | `rustworkx>=0.17.1`, `grep-ast`, `ast-grep-py>=0.40.0`, `tree-sitter>=0.25.2`, `tree-sitter-python>=0.25.0`, `tree-sitter-c>=0.24.1`, `tree-sitter-javascript>=0.25.0`, `tree-sitter-typescript>=0.23.0`, `tree-sitter-cpp>=0.23.0`, `tree-sitter-rust>=0.23.0`, `tree-sitter-go>=0.23.0`, `tree-sitter-json>=0.24.0`, `tree-sitter-yaml>=0.6.0` | Packages to allow coding functionality |
| `composio` | `composio` | Composio toolsets |
| `events` | `evented[all]` | Event triggers for agents |
| `langfuse` | `langfuse` | Langfuse Prompt Hub |
| `markitdown` | `markitdown; python_version < '3.14'` | MarkItDown Media Converter |
| `mcp-discovery` | `fastembed>=0.7.4; python_version < '3.14'`, `lancedb>=0.26.0; python_version < '3.14'`, `pyarrow>=19.0.0; python_version < '3.14'` | MCP Discovery Toolset with semantic search |
| `mcp_run` | `mcpx-py>=0.7.0` | MCP.run Toolset |
| `notifications` | `apprise>=1.9.5` | Notification Toolset |
| `promptlayer` | `promptlayer` | PromptLayer Prompt Hub |
| `tiktoken` | `tiktoken` | Exact token counting |
| `tts` | `anyvoice[tts-edge,openai]>=0.0.2` | Text-to-Speech using AnyVoice (disabled: cffi doesn't support free-threaded 3.13) |
| `watchdog` | `watchdog>=4.0.0` | Filesystem watcher for skill hot-reload |
| `zed` | `zstandard>=0.23.0` | Zed IDE storage provider |


### One-Line ACP Setup

No installation needed - run directly with uvx:

```bash
uvx --python 3.13 wolfharness@latest serve-acp 

# or

uvx --python 3.13 wolfharness@latest serve-acp path/to/agents.yml
```
