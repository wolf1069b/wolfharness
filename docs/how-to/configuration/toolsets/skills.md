---
title: Skills Toolset
description: Load and execute skills
icon: material/lightning-bolt
---

# Skills Toolset

Load and execute skills - reusable prompt-based capabilities.

!!! warning "Deprecated"
    The skills toolset is now **auto-provided** by `SkillManagerCap`, which
    the pool registers for every agent. An explicit `type: skills` toolset is a
    no-op and can be removed from your configuration. The `load_skill` and
    `list_skills` tools are always available when the pool has skills.

## Basic Usage

```yaml
agents:
  my_agent:
    tools:
      - type: skills
```

The `type: skills` toolset is deprecated and redundant — the `load_skill` /
`list_skills` tools are auto-provided by `SkillManagerCap`. You can remove it.

## Tool Discovery

`load_skill` and `list_skills` are provided automatically by
[`SkillManagerCap`](../../../explanation/skills-system.md) whenever the
pool loads skills. Call them directly; no separate toolset is needed.

## Configuration Reference

/// mknodes
{{ "wolfharness_config.toolsets.SkillsToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///

/// mknodes
{{ "wolfharness_config.toolsets.SkillsToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets", header_level=3) }}
///
