---
title: Skills with References
description: Creating skills with supporting reference files
hide:
  - toc
---

# Skills with References

This example demonstrates how to create skills with supporting reference files, enabling richer skill ecosystems with bundled templates, examples, and guides.

## Overview

Skills can include a `references/` subdirectory containing supporting materials:

- **Templates**: Reusable document templates
- **Examples**: Sample outputs or implementations
- **Guides**: Detailed sub-topic documentation
- **Schemas**: Data format specifications

These references are accessible via the skill URI scheme.

## Directory Structure

```
skills/
└── documentation-style-guide/
    ├── SKILL.md                    # Main skill file
    └── references/
        ├── structure.md            # Document structure guide
        ├── formatting.md           # Formatting rules
        └── examples/
            └── api-doc.md          # Complete example
```

## Accessing References

### Load Main Skill

```python
await load_skill(ctx, "documentation-style-guide")
```

Returns the SKILL.md content with instructions.

### Load Reference Files

```python
# Load a specific reference
await load_skill(ctx, "skill://documentation-style-guide/references/structure.md")

# Load an example
await load_skill(ctx, "skill://documentation-style-guide/references/examples/api-doc.md")
```

Returns the reference content with a header indicating the source.

## Creating a Skill with References

### 1. Create the Skill Directory

```bash
mkdir -p skills/my-skill/references
```

### 2. Write the Main SKILL.md

```markdown
# Skill: my-skill

Description of what this skill does.

## License
MIT

## Compatibility
1.0.0

## Allowed Tools
tool1, tool2, tool3

## Instructions

Main instructions here...

Reference materials:
- Guide: references/guide.md
- Template: references/template.md
```

### 3. Add Reference Files

```bash
# Add supporting files
echo "# Guide" > skills/my-skill/references/guide.md
echo "# Template" > skills/my-skill/references/template.md
```

### 4. Access in Code

```python
# Load main skill
main = await load_skill(ctx, "my-skill")

# Load reference
guide = await load_skill(ctx, "skill://local/my-skill/references/guide.md")
```

## Security

Reference access includes path traversal protection:

- `..` components are rejected
- Paths must stay within the `references/` directory
- Symlinks are resolved before validation

## Use Cases

### Code Review Skill

```
skills/code-review/
├── SKILL.md
└── references/
    ├── checklists/
    │   ├── security.md
    │   ├── performance.md
    │   └── style.md
    └── examples/
        ├── good-pr.md
        └── bad-pr.md
```

### API Design Skill

```
skills/api-design/
├── SKILL.md
└── references/
    ├── rest-guidelines.md
    ├── graphql-patterns.md
    ├── schemas/
    │   ├── openapi-template.yaml
    │   └── json-schema.json
    └── examples/
        ├── crud-api.md
        └── webhook-design.md
```

### Testing Skill

```
skills/testing/
├── SKILL.md
└── references/
    ├── unit-testing-patterns.md
    ├── integration-testing.md
    ├── fixtures/
    │   └── sample-data.json
    └── examples/
        ├── pytest-example.py
        └── jest-example.js
```

## Running the Example

```bash
# Load the main skill
wolfharness run skill_with_references/documentation_helper \
  "Load the documentation-style-guide skill"

# Load a reference file
wolfharness run skill_with_references/documentation_helper \
  "Load the structure reference from documentation-style-guide"

# Explore all references
wolfharness run skill_with_references/reference_explorer \
  "Show me all references for documentation-style-guide"
```

## Benefits

1. **Modular Skills**: Keep main instructions concise, details in references
2. **Reusable Templates**: Bundle document templates with skills
3. **Comprehensive Guides**: Include detailed sub-topics without cluttering main skill
4. **Version Control**: References are tracked with the skill
5. **Consistent Access**: Same URI scheme for skills and references

## See Also

- [Skill URI Usage](../../../how-to/configuration/skill-uri-usage.md) - Complete URI documentation
- [Skill URI Loading](../skill_uri_loading/) - Basic skill loading examples
## Code

### `config.yml`

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/Million-mo/wolfharness/refs/heads/main/schema/config-schema.json
# Example: Skills with References
#
# This example demonstrates how to create skills with supporting reference files.
# References are stored in a 'references/' subdirectory within the skill folder.

skills:
  paths:
    - ./skills

agents:
  # Agent that uses skills with references
  documentation_helper:
    type: native
    model: openai:gpt-4o
    description: Agent that helps with documentation using referenced guides
    system_prompt: |
      You are a documentation specialist. You have access to the
      documentation-style-guide skill which includes reference materials.

      You can:
      1. Load the main skill: "documentation-style-guide"
      2. Load reference files: "skill://documentation-style-guide/references/headings.md"
      3. Load examples: "skill://documentation-style-guide/references/examples/api-doc.md"

      When helping with documentation, load relevant references to provide guidance.
    tools:
      - type: skills

  # Agent that shows all references
  reference_explorer:
    type: native
    model: openai:gpt-4o-mini
    description: Agent that explores skill references
    system_prompt: |
      You are a skill reference explorer. You can discover and load
      reference files bundled with skills.

      Use the load_skill tool with URI paths to access references:
      - skill://documentation-style-guide/references/structure.md
      - skill://documentation-style-guide/references/formatting.md
      - skill://documentation-style-guide/references/examples/

      When asked about a skill's references, load them and summarize
      their content.
    tools:
      - type: skills
```

