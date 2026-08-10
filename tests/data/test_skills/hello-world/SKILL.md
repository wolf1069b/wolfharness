---
name: hello-world
description: A simple greeting skill for testing basic skill functionality and command exposure across protocols
license: MIT
compatibility: 1.0.0
allowed-tools: bash, read
---

# hello-world

A simple greeting skill for testing

## License
MIT

## Compatibility
1.0.0

## Allowed Tools
bash, read

## Instructions

This skill outputs a friendly greeting.

When invoked, respond with a warm, friendly greeting message.

### Usage Examples

Basic greeting:
```bash
wolfharness skill hello-world
```

Expected output:
- A friendly welcome message
- Reference to the skill name
- Confirmation that the skill system is working

### Testing Scenarios

1. Protocol Consistency: This skill should be available as:
   - ACP: `/hello-world` command
   - AG-UI: `skill__hello-world` tool
   - OpenCode: `skill:hello-world` command

2. Cross-Protocol Verification:
   - Same description across all protocols
   - Same invocation behavior
   - Consistent response format
