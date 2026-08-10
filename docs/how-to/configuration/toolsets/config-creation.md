---
title: Config Creation Toolset
description: Create agent configurations
icon: material/file-plus
---

# Config Creation Toolset

Tools for creating and managing agent configurations programmatically.

## Basic Usage

```yaml
agents:
  admin:
    tools:
      - type: config_creation
```

## Available Tools

```python exec="true"
from wolfharness_toolsets.config_creation import ConfigCreationTools
from wolfharness.docs.utils import generate_tool_docs

toolset = ConfigCreationTools()
print(generate_tool_docs(toolset))
```

## Use Cases

- Generate agent configs from templates
- Create configurations dynamically
- Manage configuration files

## Configuration Reference

/// mknodes
{{ "wolfharness_config.toolsets.ConfigCreationToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
