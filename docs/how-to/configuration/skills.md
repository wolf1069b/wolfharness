---
title: Skills Configuration
description: Configure automatic skills injection into agent prompts
order: 10
icon: material/lightning-bolt
---

Skills provide specialized instructions and techniques that agents can follow. AgentPool supports automatic injection of skills into agent system prompts using structured XML formatting.

## Overview

Skills injection allows you to:

- Automatically include relevant skill instructions in agent prompts
- Configure global defaults for all agents
- Override per-agent using skills tool configuration
- Limit the number of skills included to manage token count

## Configuration Structure

### Global Skills Configuration

```yaml
skills:
  # Skill discovery paths
  paths:
    - ~/.config/wolfharness/skills
    - ./skills
  
  # Include default AgentPool skills (default: true)
  include_default: true
  
  # Instruction injection configuration
  instruction:
    # Full-instruction injection mode: description, matcher, or all
    inject: description
    # Maximum number of skills to inject (default: 20)
    max_skills: 20
```

### Injection Modes

The static `<available-skills>` catalog (skill name + description) is **always**
emitted when the pool loads skills. The `inject` field controls whether the full
skill instructions (`<skill_content>`) are also placed into the system prompt,
which costs more tokens.

| Mode | Behavior | Use Case |
|------|----------|----------|
| `description` (default) | Catalog only — NO full `<skill_content>`. Full instructions are loaded on demand via `load_skill`. | Token-saving; the common default |
| `matcher` | Catalog + full instructions for skills selected by a runtime `matcher_fn` | Agents that should auto-load relevant skills without waiting for a tool call |
| `all` | Catalog + full instructions for every skill | When skills contain critical instructions that must always be present |

## Agent-Specific Overrides

Per-agent overrides are handled by the (deprecated) `type: skills` toolset, which
is a no-op since skills tools are auto-provided by `SkillManagerCap`. The
`inject`/`max_skills` settings are global; there is no per-agent
`injection_mode` override.

## XML Output Format

When skills injection is enabled, agents receive a skill catalog in their system prompt:

```xml
<available-skills>
  <skill name="python-style-guide" description="PEP 8 coding conventions" />
  <skill name="refactoring" description="Safe refactoring techniques" />
</available-skills>
```

With `inject: all` (or matcher-selected skills), full content is also injected:

```xml
<skill_content name="python-style-guide">
  ## Python Style Guide

  Follow PEP 8 conventions:
  - Use 4 spaces for indentation
  - Maximum line length of 100 characters
</skill_content>
```

## Complete Example

```yaml
# Global skills configuration
skills:
  paths:
    - ~/.config/wolfharness/skills
    - ./project-skills
  include_default: true
  
  # Default: catalog-only injection for all agents (token-saving)
  instruction:
    inject: description
    max_skills: 20
```

## Behavior Notes

- The `<available-skills>` catalog is emitted whenever the pool has visible
  skills — this is not disabled by default.
- `inject: matcher` requires a programmatically-provided `matcher_fn` (it is not
  serializable in YAML). Without one it falls back to `description` and logs a
  warning.
- Full instructions are always available on demand via the `load_skill` tool,
  regardless of the `inject` mode.

## Related Configuration

- [Toolsets](./node-types/index.md) - Configure agent tools including skills tool
- [Agent Pool](../../reference/core-concepts/agent-pool.md) - Global pool configuration

## See Also

- [RFC-0008: Dynamic Skills Injection](../../rfcs/implemented/RFC-0008-dynamic-skills-injection.md) - Implementation details
- [Skills Toolset](../../reference/core-concepts/toolsets.md) - Skills toolset reference
