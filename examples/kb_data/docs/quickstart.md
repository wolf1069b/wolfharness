# Quick Start

## Installation

```bash
uv tool install wolfharness
```

## Minimal config

```yaml
agents:
  assistant:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are a helpful assistant."
```

## Run

```bash
wolfharness run assistant "Hello!"
```