---
title: config
description: Configuration management and diagnostics
icon: material/cog
---

The `config` command group helps you understand and manage AgentPool's layered configuration system.

## Overview

AgentPool automatically discovers and merges configuration from multiple sources:

1. **Global config** (`~/.config/wolfharness/wolfharness.yml`)
2. **Custom config** (`AGENTPOOL_CONFIG` environment variable)
3. **Project config** (`wolfharness.yml` in project/git root)
4. **Explicit config** (CLI argument)

These commands help you inspect which configs are being loaded and create new ones.

## Commands

The `config` command group includes the following commands:

```bash
# Show current configuration
wolfharness config show [config_path]

# Show config search paths
wolfharness config paths

# Initialize a new configuration
wolfharness config init [path] [--force]
```

### config show

Display the current configuration, showing which config files are found and what they contain.

```bash
# Show merged configuration
wolfharness config show

# Show with a specific explicit config
wolfharness config show my-agents.yml

# Output as YAML
wolfharness config show --format yaml
```

### config paths

Display the paths AgentPool searches for configuration files.

```bash
wolfharness config paths
```

### config init

Create a new configuration file.

```bash
# Create a starter config in current project
wolfharness config init

# Create global config for user-wide preferences
wolfharness config init global

# Create at a specific path
wolfharness config init ./configs/my-agents.yml

# Overwrite existing config
wolfharness config init --force
```

## Examples

### Inspect Configuration

```bash
# Show which config files are found and what they contain
wolfharness config show

# Show with a specific explicit config included
wolfharness config show my-agents.yml

# Output as YAML for scripting
wolfharness config show --format yaml
```

### View Config Paths

```bash
# Show where AgentPool looks for config files
wolfharness config paths
```

### Create New Config

```bash
# Create a starter config in current project
wolfharness config init

# Create a global config for user-wide preferences
wolfharness config init global

# Create at a specific path
wolfharness config init ./configs/my-agents.yml

# Overwrite existing config
wolfharness config init --force
```

## Use Cases

### Setting Global Preferences

Create a global config to set preferences that apply to all projects:

```bash
wolfharness config init global
```

Then edit `~/.config/wolfharness/wolfharness.yml`:

```yaml
# Global preferences
model_variants:
  fast:
    type: string
    identifier: openai:gpt-4o-mini
  smart:
    type: anthropic
    identifier: claude-sonnet-4-5

storage:
  provider: sql
  database_url: sqlite:///~/.local/share/wolfharness/history.db
```

### Project-Specific Agents

Create a project config that inherits global settings:

```bash
wolfharness config init
```

Then edit `./wolfharness.yml`:

```yaml
agents:
  coder:
    model: smart  # Uses global model_variant
    system_prompt: "You are an expert in this codebase."
    tools:
      - type: file_access
      - type: bash
```

### Debugging Config Issues

If your agent isn't behaving as expected, check which configs are being loaded:

```bash
wolfharness config show
```

This shows:
- Which config files were found
- What keys each layer contributes
- The final merged result
