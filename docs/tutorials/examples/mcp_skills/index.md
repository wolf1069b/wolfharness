---
title: MCP Skills
description: Using MCP-exposed skills via the skill:// URI scheme
hide:
  - toc
---

# MCP Skills

This example demonstrates how to use skills exposed by MCP servers, including both prompt-based and resource-based skills.

## Overview

MCP (Model Context Protocol) servers can expose skills in two ways:

1. **Prompt-based skills**: Traditional MCP prompts mapped to skills
2. **Resource-based skills**: FastMCP Skills Provider protocol using `skill://` URIs

AgentPool unifies access to both types through the same `skill://` URI scheme.

## MCP Skill Types

### Prompt-Based Skills

MCP servers with prompts automatically expose them as skills:

```
skill://mcp-server-name/prompt-name
```

Example:
```python
# Load an MCP prompt as a skill
await load_skill(ctx, "skill://github-copilot/code-review")
```

**Characteristics**:
- Derived from MCP prompts
- May require arguments (inferred from prompt schema)
- Rendered through MCP get_prompt

### Resource-Based Skills (FastMCP Skills Provider)

MCP servers implementing the [FastMCP Skills Provider protocol](https://gofastmcp.com/servers/providers/skills) expose skills as resources:

```
skill://mcp-skills/pdf-processing          # Short form
skill://mcp-skills/pdf-processing/SKILL.md # Explicit main file
skill://mcp-skills/pdf-processing/_manifest # JSON manifest
```

Example:
```python
# Load a resource-based skill
await load_skill(ctx, "skill://mcp-skills/pdf-processing")

# Load skill reference
await load_skill(ctx, "skill://mcp-skills/pdf-processing/examples/sample.pdf")
```

**Characteristics**:
- Exposed via MCP resources
- Main file at `skill://{name}/SKILL.md`
- Optional manifest at `skill://{name}/_manifest`
- Reference files accessible via URI paths

## Configuration

```yaml
mcp_servers:
  # MCP server with prompt-based skills
  - "uvx mcp-server-with-prompts"
  # MCP server with resource-based skills (FastMCP Skills Provider)
  - "uvx mcp-skills-provider"

skills:
  paths:
    - ./skills  # Local skills

agents:
  my_agent:
    type: native
    model: openai:gpt-4o
    tools:
      - type: skills  # Enables skill loading
```

## Usage Examples

### Listing All Skills

```python
result = await list_skills(ctx)
```

Output includes skills from all sources:

```
Available skills:

## local (2 skills)
- **python-expert**: Expert Python techniques
  URI: `skill://local/python-expert`

## mcp-server-with-prompts (3 skills)
- **code-review**: Review code changes
  URI: `skill://mcp-server-with-prompts/code-review`
- **refactor-helper**: Assist with refactoring
  URI: `skill://mcp-server-with-prompts/refactor-helper`

## mcp-skills-provider (2 skills)
- **pdf-processing**: Process PDF documents
  URI: `skill://mcp-skills-provider/pdf-processing`
- **image-analysis**: Analyze image content
  URI: `skill://mcp-skills-provider/image-analysis`
```

### Loading MCP Prompt-Based Skills

```python
# By short name (auto-routes to first provider)
await load_skill(ctx, "code-review")

# By full URI (explicit provider)
await load_skill(ctx, "skill://mcp-server-with-prompts/code-review")

# With arguments (for prompts that require them)
await load_skill(ctx, "code-review", "path/to/file.py")
```

### Loading MCP Resource-Based Skills

```python
# Short form
await load_skill(ctx, "pdf-processing")

# Explicit URI
await load_skill(ctx, "skill://mcp-skills-provider/pdf-processing")

# Load reference content
await load_skill(ctx, "skill://mcp-skills-provider/pdf-processing/examples/invoice.pdf")
```

## Provider Priority

When multiple providers have skills with the same name:

1. **Local** skills have highest priority
2. **MCP providers** are checked in registration order

Example collision resolution:

```
Local:        code-review, python-expert
MCP Server A: code-review, testing-guide
MCP Server B: documentation, deployment

Resolution:
- code-review → local (priority)
- python-expert → local
- testing-guide → MCP Server A
- documentation → MCP Server B
- deployment → MCP Server B
```

Use full URIs to override priority:

```python
# Force MCP version despite local having same name
await load_skill(ctx, "skill://mcp-server-a/code-review")
```

## FastMCP Skills Provider Protocol

MCP servers using this protocol expose skills as resources with specific URI patterns:

| URI Pattern | Purpose |
|-------------|---------|
| `skill://{server}/{skill}` | Short form (resolves to main skill) |
| `skill://{server}/{skill}/SKILL.md` | Main instruction file |
| `skill://{server}/{skill}/_manifest` | JSON manifest with metadata |
| `skill://{server}/{skill}/{file}` | Reference/supporting files |

### Manifest Format

```json
{
  "name": "pdf-processing",
  "version": "1.0.0",
  "description": "Process PDF documents",
  "files": [
    "SKILL.md",
    "examples/invoice.pdf",
    "templates/cover-page.html"
  ]
}
```

## Creating an MCP Skills Provider

To expose skills via MCP using FastMCP:

```python
from fastmcp import FastMCP
import json

mcp = FastMCP("my-skills")

@mcp.resource("skill://pdf-processing/SKILL.md")
def get_pdf_skill() -> str:
    return """
    # PDF Processing Skill

    Process PDF documents efficiently.

    ## Instructions
    ...
    """

@mcp.resource("skill://pdf-processing/_manifest")
def get_pdf_manifest() -> str:
    return json.dumps({
        "name": "pdf-processing",
        "files": ["SKILL.md", "examples/sample.pdf"]
    })

@mcp.resource("skill://pdf-processing/examples/{filename}")
def get_pdf_example(filename: str) -> bytes:
    return load_example_file(filename)
```

## Running the Example

```bash
# List all available skills (local + MCP)
wolfharness run mcp_skills/skill_discoverer "List all available skills"

# Load a specific MCP skill
wolfharness run mcp_skills/mcp_skill_user \
  "Load the skill://mcp-server-with-prompts/code-review skill"

# Auto-route to a skill by short name
wolfharness run mcp_skills/mcp_skill_user \
  "Load the code-review skill"
```

## Troubleshooting

### MCP Skill Not Found

```
Skill not found: 'my-skill'. Available: ...
```

- Verify MCP server is configured correctly
- Check MCP server exposes the expected prompts/resources
- Use `list_skills` to see available skills

### Prompt Requires Arguments

```python
# Some MCP prompts require arguments
await load_skill(ctx, "skill://mcp/prompt-name", "arg1 arg2")
```

### Resource Not Accessible

```
Reference not found: 'skill://server/skill/file.md'
```

- Check the skill manifest for available files
- Verify MCP server exposes the resource
- Use correct URI format

## Benefits

1. **Unified Access**: Same interface for local and MCP skills
2. **Dynamic Discovery**: MCP skills appear automatically when servers connect
3. **Rich Ecosystem**: Leverage skills from any MCP-compatible source
4. **Protocol Agnostic**: Works with both prompt and resource-based MCP skills

## See Also

- [FastMCP Skills Provider Documentation](https://gofastmcp.com/servers/providers/skills)
- [Skill URI Usage](../../../how-to/configuration/skill-uri-usage.md)
- [MCP Servers (YAML)](../mcp_servers_yaml/) - Basic MCP integration
## Code

### `config.yml`

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/Million-mo/wolfharness/refs/heads/main/schema/config-schema.json
# Example: MCP-Exposed Skills
#
# This example demonstrates how to use skills exposed by MCP servers.
# MCP servers can expose skills in two ways:
# 1. Prompt-based skills: MCP prompts mapped to skills
# 2. Resource-based skills: FastMCP Skills Provider protocol (skill:// URIs)

mcp_servers:
  # Example MCP servers that might expose skills
  # These are illustrative - replace with actual MCP servers that expose skills
  - "uvx mcp-server-filesystem"
  # - "uvx mcp-server-with-prompts"  # MCP server with prompt-based skills
  # - "uvx mcp-skills-provider"      # FastMCP Skills Provider server

skills:
  paths:
    - ./skills  # Local skills for comparison

agents:
  # Agent that uses MCP skills
  mcp_skill_user:
    type: native
    model: openai:gpt-4o
    description: Agent that loads skills from MCP servers
    system_prompt: |
      You are an agent that can use skills from multiple sources.

      You have access to:
      1. Local skills: skill://local/skill-name
      2. MCP prompt-based skills: skill://mcp-server/prompt-name
      3. MCP resource-based skills: skill://mcp-skills/skill-name

      When asked to use a skill, load it with the appropriate URI.
      If no source is specified, try the short name first (auto-routes).
    tools:
      - type: skills

  # Agent that discovers all available skills
  skill_discoverer:
    type: native
    model: openai:gpt-4o-mini
    description: Agent that discovers skills from all providers
    system_prompt: |
      You are a skill discovery agent. You can list all available skills
      from all providers using list_skills.

      This shows:
      - Local filesystem skills
      - MCP prompt-based skills
      - MCP resource-based skills

      Use this to help users find the right skill for their needs.
    tools:
      - type: skills
```

