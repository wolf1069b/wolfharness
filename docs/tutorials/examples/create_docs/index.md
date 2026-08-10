---
title: Create Docs
description: Agent collaboration for documentation generation
hide:
  - toc
---

# Multi-Agent Documentation System

This example demonstrates a team of three agents working together to scan, document, and validate Python code. It shows different patterns of agent collaboration:

- Async delegation (fire and forget)
- Tool usage with waiting
- Chained tool calls

## How It Works

1. The File Scanner agent scans directories and identifies Python files
2. It passes these files to the Documentation Writer agent asynchronously
3. In parallel, it uses the Error Checker as a tool to validate the files
4. The Documentation Writer processes files as they come in
5. Results are printed to the console as they become available

This demonstrates different ways agents can collaborate:

- Async message passing (scanner to writer)
- Synchronous tool usage (scanner using checker)
- Event-based output handling (writer to console)
## Code

### `main.py`

```python
# /// script
# dependencies = ["wolfharness", "mypy"]
# ///


"""Agentsoft Corp. 3 agents publishing software.

This example shows:
1. Async delegation: File scanner delegates to doc writer (fire and forget)
2. Tool usage (async + wait): File scanner uses error checker as a tool (wait for result)
3. Chained tool calls.
"""

from __future__ import annotations

import os
from pathlib import Path

from mypy import api
import rich

from wolfharness import Agent, AgentPool, AgentsManifest
from wolfharness.docs.utils import run


# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


def check_types(path: str) -> str:
    """Type check Python file using mypy."""
    stdout, _stderr, _code = api.run([path])
    return stdout


async def main() -> None:
    # Load config from YAML
    config_path = Path(__file__).parent / "config.yml"
    manifest = AgentsManifest.from_file(config_path)

    async with AgentPool(manifest) as pool:
        scanner = pool.get_agent("file_scanner")
        writer = pool.get_agent("doc_writer")
        checker = pool.get_agent("error_checker")

        # Set up message logging
        for agent in (scanner, writer, checker):
            agent.message_sent.connect(lambda msg: rich.print(msg.format()))

        # Setup chain: scanner -> writer -> console output
        scanner.connect_to(writer)

        # Start async docs generation (the writer will start working in async fashion)
        await scanner.run('List all Python files in "src/wolfharness/agent"')
        assert isinstance(scanner, Agent)
        # Use error checker as tool (this blocks until complete)
        scanner.register_worker(checker)
        prompt = 'Check types for all Python files in "src/wolfharness/agent"'
        result = await scanner.run(prompt)
        rich.print(f"Type checking result:\n{result.data}")

        # Wait for documentation to finish
        await writer.task_manager.complete_tasks()


if __name__ == "__main__":
    run(main())
```

### `config.yml`

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/Million-mo/wolfharness/refs/heads/main/schema/config-schema.json
agents:
  file_scanner:
    type: native
    display_name: "File Scanner"
    model: openai:gpt-5-nano
    system_prompt: You scan directories and list source files that need documentation.
    tools:
      - type: import
        name: list_source_files
        import_path: os.listdir

  doc_writer:
    type: native
    display_name: "Documentation Writer"
    model: openai:gpt-5-nano
    system_prompt: You are a docs writer. Write markdown documentation for the files given to you.
    tools:
      - type: file_access

  error_checker:
    type: native
    display_name: "Code Validator"
    model: openai:gpt-5-nano
    system_prompt: You validate Python source files for syntax errors.
    tools:
      - type: import
        name: validate_syntax
        import_path: __main__.check_types
        description: Type check Python file using mypy.
```

