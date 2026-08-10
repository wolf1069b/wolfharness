---
title: Examples
description: End-to-end examples showing what you can build with WolfHarness
order: 1
---

# Examples

Real, end-to-end builds you can recreate with WolfHarness — define agents once in YAML, orchestrate them, and expose the result through any protocol.

## 🧵 Round-robin agents: a live word chain

Three agents connected in a **circle**, each appending to the previous agent's output. WolfHarness's connection system routes messages automatically, and a `cost_limit` stop condition ends the loop the moment a token budget is hit — no manual orchestration code.

<!-- excerpt-start -->

**The whole thing — configuration and a run — in two files.**

### `config.yml`

```yaml
prompts:
  system_prompts:
    word_chain:
      content: 'Append one word to the given word or sentence and continue the sentence indefinitely.'

agents:
  player1:
    type: native
    model: openai:gpt-5-mini
    system_prompt:
      - type: library
        reference: word_chain
    connections:
      - type: node
        name: player2
        connection_type: run

  player2:
    type: native
    model: openai:gpt-5-mini
    system_prompt:
      - type: library
        reference: word_chain
    connections:
      - type: node
        name: player3
        connection_type: run
        stop_condition:
          type: cost_limit
          max_cost: 0.01  # stop the circle at this spend

  player3:
    type: native
    model: openai:gpt-5-mini
    system_prompt:
      - type: library
        reference: word_chain
    connections:
      - type: node
        name: player1
        connection_type: run
```

### `main.py`

```python
from wolfharness.__main__ import run_command

run_command(
    node_name="player1",
    prompts=["Start the word chain with: tree"],
    config_path="config.yml",
    show_messages=True,
)
```

Run it:

```bash
wolfharness run player1 "Start the word chain with: tree"
```

No glue code. No message-passing boilerplate. The agents talk, the budget stops them, and you observe the whole conversation.

## More build ideas

These are fully in-repo and runnable:

| Example | What it demonstrates |
|---|---|
| [Download workers](../tutorials/examples/download_workers/) | One agent used as a **tool** by another |
| [CrewAI-style flow](../tutorials/examples/crewai_flow/) | Familiar workflow patterns from other frameworks |
| [Model comparison](../tutorials/examples/model_comparison/) | Zero-code A/B across models |
| [Human interaction](../tutorials/examples/human_interaction/) | Human-in-the-loop agents |

Explore every example in [Examples](../tutorials/examples/).

## Build your own

These patterns all bottom out in the same single source of truth — **YAML**. Start with the [Quickstart](../tutorials/quickstart.md) and combine [connections](../how-to/configuration/connections.md), [teams](../how-to/configuration/node-types/team.md), and [triggers](../how-to/configuration/events.md) to assemble your own workflows.