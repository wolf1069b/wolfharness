---
title: User Interaction Toolset
description: Interact with users via forms and questions
icon: material/account-question
---

# User Interaction Toolset

The user interaction toolset provides the `question` tool for agents to ask questions and collect structured responses from users. It uses the MCP Elicit protocol so forms can be rendered by compatible clients (IDEs, TUI, etc.).

## Available Tool

| Tool | Purpose |
|---|---|
| `question` | Present a multi-question XML questionnaire (enum, multi, input types) |

## `question`

Takes an XML `questions` string describing one or more questions and presents them as a form.

XML format (use single quotes for attributes to avoid JSON escaping issues):

```xml
<questions>
  <question header="Model" type="enum" required="true">
    <text>What model?</text>
    <suggest type="choice">Option 1</suggest>
    <suggest type="choice">Option 2</suggest>
  </question>
</questions>
```

For a single question you may omit the `<questions>` wrapper:

```xml
<question header="Confirm" type="enum">
  <text>Proceed?</text>
  <suggest>Yes</suggest>
  <suggest>No</suggest>
</question>
```

Supported question types:

- `enum`: single choice from a list of `<suggest>` options
- `multi`: multiple choice
- `input`: free-text input

## Responses

The tool returns the user's answers through the agent tool result. Agents can then use the answers to decide next steps or personalize their response.

For implementation details, see `wolfharness_toolsets.builtin.question_tools`.
