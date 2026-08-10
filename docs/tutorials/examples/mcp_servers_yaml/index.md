---
title: MCP Servers (YAML)
description: MCP server integration with git tools
hide:
  - toc
---

# MCP Servers (YAML)

This example demonstrates how to configure MCP servers directly in YAML and use them with connected agents:

- Declaring an MCP server (`uvx mcp-server-git`) in `config.yml`
- Running a Python script that loads the YAML config via `AgentsManifest`
- Connecting two agents so the picker asks the analyzer for details
- Also showing the team-level MCP server configuration option

## How It Works

1. `config.yml` defines the MCP server and two agents (`picker` and `analyzer`) connected in a chain.
2. `main_yaml.py` loads the manifest with `AgentPool`, gets the two agents, and starts the picker.
3. The picker uses the MCP server tools to fetch the latest commit hash and passes it to the analyzer.
4. `main_py.py` shows the same flow built programmatically, including a team-level MCP server setup.

## Code

### `main_yaml.py`

```python
# /// script
# dependencies = ["wolfharness"]
# ///


"""Example demonstrating MCP server integration with git tools.

This example shows:
- Using MCP servers to provide git functionality to agents
- Agent connections through YAML configuration
- Message flow between connected agents
- Team-level MCP server configuration
"""

from __future__ import annotations

import os

from wolfharness import AgentPool, AgentsManifest
from wolfharness.docs.utils import get_config_path, is_pyodide, run


PROMPT = "Get the latest commit hash!"

# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


async def run_example() -> None:
    """Run example using YAML configuration."""
    # Load config from YAML
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)

    async with AgentPool(manifest) as pool:
        # Get agents (connections already set up from YAML)
        picker = pool.get_agent("picker")
        analyzer = pool.get_agent("analyzer")

        # Register handlers to see messages
        picker.message_sent.connect(lambda msg: print(msg.format()))
        analyzer.message_sent.connect(lambda msg: print(msg.format()))

        # Start the chain
        await picker.run(PROMPT)


if __name__ == "__main__":
    run(run_example())
```

### `main_py.py`

```python
# /// script
# dependencies = ["wolfharness"]
# ///

"""Example: Two agents working together to explore git commit history."""

from __future__ import annotations

from wolfharness import Agent, Team
from wolfharness.docs.utils import run


PICKER = """
You are a specialist in looking up git commits using your tools
from the current working directory."
"""
ANALYZER = """
You are an expert in retrieving and returning information
about a specific commit from the current working directoy."
"""

MODEL = "openai:gpt-5-nano"
SERVERS = ["uvx mcp-server-git"]


async def run_example() -> None:
    picker = Agent(model=MODEL, system_prompt=PICKER, mcp_servers=SERVERS)
    analyzer = Agent(model=MODEL, system_prompt=ANALYZER, mcp_servers=SERVERS)

    # Connect picker to analyzer
    picker >> analyzer

    # Register message handlers to see the messages
    picker.message_sent.connect(lambda msg: print(msg.format()))
    analyzer.message_sent.connect(lambda msg: print(msg.format()))
    # For MCP servers, we need async context.
    async with picker, analyzer:
        # Start the chain by asking picker for the latest commit
        await picker.run("Get the latest commit hash! ")

    # MCP servers also work on team level for all its members
    agent_without_mcp_server = Agent(model=MODEL, system_prompt=ANALYZER)
    team = Team([agent_without_mcp_server], mcp_servers=["uvx mcp-hn"])
    async with team:
        # this will show you the MCP server tools
        print(await agent_without_mcp_server.tools.get_tools())


if __name__ == "__main__":
    run(run_example())


"""
Output:

CommitPicker: The latest commit hash is **9bcd7718dbc33f16239d0522ca677ed75bac997b**.
CommitAnalyzer: The latest commit with hash **9bcd7718dbc33f16239d0522ca677ed75bac997b**
includes the following details:

- **Author:** Philipp Temminghoff
- **Date:** January 20, 2025, at 01:59:43 (local time)
- **Commit Message:** chore: docs

### Changes made:
...
"""
```

### `config.yml`

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/Million-mo/wolfharness/refs/heads/main/schema/config-schema.json
mcp_servers:
  - "uvx mcp-server-git"

agents:
  picker:
    type: native
    model: openai:gpt-5-nano
    description: Git commit history explorer
    system_prompt: You are a specialist in looking up git commits using your tools from the current working directory.
    connections:
      - type: node
        name: analyzer

  analyzer:
    type: native
    model: openai:gpt-5-nano
    description: Git commit analyzer
    system_prompt: You are an expert in retrieving and returning information about a specific commit from the current working directory.
```

