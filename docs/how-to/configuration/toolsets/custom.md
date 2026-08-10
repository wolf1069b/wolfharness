---
title: Custom Toolset
description: Load custom toolset implementations
icon: material/handball
---

# Custom Toolset

Load custom toolset implementations from Python code.

## Basic Usage

```yaml
agents:
  my_agent:
    tools:
      - type: custom
        import_path: mypackage.toolsets:MyCustomToolset
```

## Creating a Custom Toolset

```python
from wolfharness.resource_providers import ResourceProvider

class MyCustomToolset(ResourceProvider):
    async def get_tools(self):
        return [
            self.create_tool(self.my_tool, category="custom")
        ]
    
    async def my_tool(self, ctx, param: str) -> str:
        """My custom tool description."""
        return f"Result: {param}"
```

## Configuration Reference

/// mknodes
{{ "wolfharness_config.toolsets.CustomToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
