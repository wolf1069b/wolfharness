---
title: Code Toolset
description: Code analysis and manipulation
icon: material/code-braces
---

# Code Toolset

Tools for code analysis and manipulation.

## Basic Usage

```yaml
agents:
  coder:
    tools:
      - type: code
```

## Available Tools

```python exec="true"
from wolfharness_toolsets.builtin.code import CodeTools
from wolfharness.docs.utils import generate_tool_docs

toolset = CodeTools()
print(generate_tool_docs(toolset))
```

## Configuration Reference

/// mknodes
{{ "wolfharness_config.toolsets.CodeToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
