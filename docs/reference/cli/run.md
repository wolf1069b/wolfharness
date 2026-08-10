---
title: run
description: Run a node with prompts
icon: material/play
---

# run

Run a node with prompts using the `wolfharness run` command.

```bash
wolfharness run <agent_name> "prompt text"
```

The `run` command executes a single prompt against a configured agent.

## Basic Usage

```bash
# Simple run
wolfharness run assistant "Hello!"

# With streaming output
wolfharness run assistant "Tell me a story" --stream

# With explicit config file
wolfharness run assistant "Hello!" --config my-agents.yml
```

For a full list of options, run:

```bash
wolfharness run --help
```