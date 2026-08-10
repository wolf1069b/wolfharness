---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/wolfharness_config/paths.py
    - src/wolfharness_config/context.py
title: Configuration Path Resolution
description: How relative paths are resolved in configuration files
icon: material/map-marker-path
---

# Configuration Path Resolution

## Overview

AgentPool resolves relative paths in configuration files relative to the config file's directory instead of the current working directory.

## Resolution Order

Paths are resolved in the following priority:

1. **Legacy Mode**: If `AGENTPOOL_LEGACY_PATHS=1` is set, paths remain relative to CWD
2. **Environment Override**: `AGENTPOOL_CONFIG_DIR` env var overrides the config directory
3. **Config Directory**: Paths resolve relative to the config file's parent directory
4. **Current Working Directory**: Fallback when no context is set

## Usage

### Default Behavior

```yaml
# agents.yml located at /home/user/project/agents.yml
agents:
  my_agent:
    type: native
    system_prompt:
      type: file
      path: ./prompts/agent.j2  # Resolves to /home/user/project/prompts/agent.j2
    knowledge:
      paths:
        - ./docs/context.md   # Resolves to /home/user/project/docs/context.md
skills:
  paths:
    - ./skills/custom      # Resolves to /home/user/project/skills/custom
```

### Environment Override

```bash
# Force all paths to resolve to a specific directory
export AGENTPOOL_CONFIG_DIR=/custom/path
wolfharness serve-acp agents.yml
```

### Legacy Mode

```bash
# Revert to CWD-relative resolution (pre-RFC-0009 behavior)
export AGENTPOOL_LEGACY_PATHS=1
wolfharness serve-acp agents.yml
```

## Implementation Details

- Uses Pydantic `BeforeValidator` on `ConfigPath` type
- Context set via `ConfigContextManager` during config loading
- Servers (ACP, HTTP, OpenCode) automatically set context when loading manifests
- Backward compatible: `SkillsConfig.get_effective_paths()` deprecated but working

## Migration Guide

**Before (vX.Y):**
Paths resolved relative to CWD:
```bash
cd /home/user/project
wolfharness serve-acp agents.yml  # ./skills relative to CWD
```

**After (vX.Z):**
Paths resolve relative to config file:
```bash
cd /home/user/anywhere
wolfharness serve-acp /home/user/project/agents.yml  # ./skills relative to config dir
```

To maintain old behavior, set `AGENTPOOL_LEGACY_PATHS=1`.
