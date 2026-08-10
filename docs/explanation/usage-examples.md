# Complete Usage Examples

## Direct Agent Instantiation

### Native Agent

```python
from wolfharness.agents import Agent

def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

async with Agent(
    name="my_agent",
    model="openai:gpt-4o-mini",  # Required: model string or Model instance
    system_prompt="You are a helpful assistant",
    tools=[greet],  # Callables or import paths like "mymodule:my_tool"
) as agent:
    async for event in agent.run_stream("Greet Alice"):
        ...
```

### ACP Agent

```python
from wolfharness.agents.acp_agent import ACPAgent

async with ACPAgent(
    command="goose",  # Required: executable name
    args=["acp"],  # Required: command arguments
    name="goose_agent",
    cwd="/path/to/project",
) as agent:
    async for event in agent.run_stream("Write code"):
        ...
```

## Agent from Config with Streaming

### Config (config.yml)

```yaml
agents:
  coder:
    type: native
    model: "openai:gpt-4o-mini"
    system_prompt: "You are an expert Python developer"
    tools:
      - name: bash
        enabled: true
      - name: read
        enabled: true
```

### Python Code

```python
from wolfharness.delegation import AgentPool
from wolfharness.agents.events import (
    PartDeltaEvent,
    ToolCallStartEvent,
    ToolCallCompleteEvent,
    StreamCompleteEvent,
)

async with AgentPool("config.yml") as pool:
    agent = pool.get_agent("coder")
    
    # Stream events (run_stream returns AsyncIterator, not a context manager)
    async for event in agent.run_stream("Read setup.py and list dependencies"):
        match event:
            case PartDeltaEvent(delta=text):
                # Stream text chunks as they arrive
                print(text, end="", flush=True)
            
            case ToolCallStartEvent(tool_name=name):
                print(f"\n[Tool starting: {name}]")
            
            case ToolCallCompleteEvent(tool_name=name, tool_result=result):
                print(f"\n[Tool {name} completed: {result}]")
            
            case StreamCompleteEvent(message=msg):
                # Final complete message with full content
                print(f"\n\nComplete response: {msg.content}")
```
