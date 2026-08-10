# Signal Streaming Adapters: Emulating anyenv.Signal over pydantic-graph GraphRun

> **Status**: Accepted — this design has been implemented in the AgentPool codebase.
> See `docs/explanation/` for the current architecture documentation.

**Status**: DRAFT  
**Created**: 2026-06-03  
**Scope**: Design only — no implementation  

---

## 1. Overview

AgentPool currently relies on `anyenv.Signal` for loose-coupled event propagation:
- `MessageNode.message_received` — emitted when a node receives input
- `MessageNode.message_sent` — emitted when a node produces output
- `Talk.connection_processed` — emitted when a message traverses a connection
- `Talk.message_forwarded` — emitted after transformation/filtering before dispatch

The migration to `pydantic-graph` replaces the ad-hoc node/graph execution with `GraphRun`, which yields `GraphTask` sequences and `EndMarker` via `Graph.iter()`. This document specifies how to **emulate** the existing signal semantics at GraphRun step boundaries and how to **map** GraphRun yields to the existing `RichAgentStreamEvent` types so that downstream consumers (ACP, OpenCode, AG-UI) require zero changes.

---

## 2. Background & Context

### 2.1 Current Signal Definitions

In `src/wolfharness/messaging/messagenode.py`:

```python
class MessageNode[TDeps, TResult](ABC):
    message_received = Signal[ChatMessage[Any]]()
    """Signal emitted when node receives a message."""

    message_sent = Signal[ChatMessage[Any]]()
    """Signal emitted when node creates a message."""
```

In `src/wolfharness/talk/talk.py`:

```python
class Talk[TTransmittedData = Any]:
    message_received = Signal[ChatMessage[Any]]()
    message_forwarded = Signal[ChatMessage[Any]]()
    connection_processed = Signal[ConnectionProcessed]()
```

### 2.2 pydantic-graph Execution Model

`Graph.iter()` returns a `GraphRun` async iterator. Each iteration yields one of:
- `Sequence[GraphTask]` — one or more node executions to schedule
- `EndMarker[OutputT]` — graph completion with final value
- `ErrorMarker` — node raised an exception (can be recovered via `override_next()`)

Inside `_GraphIterator.iter_graph()`, the lifecycle is:
1. `_run_tracked_task()` schedules a task for each `GraphTask`
2. `_run_task()` executes the node via `node.call(step_context)`
3. Results flow through `MemoryObjectStream` back to the iterator
4. Iterator yields the next `Sequence[GraphTask]` or `EndMarker`

### 2.3 Existing Event Types

From `src/wolfharness/agents/events/events.py`:

```python
type RichAgentStreamEvent[OutputDataT] = (
    AgentStreamEvent
    | StreamCompleteEvent[OutputDataT]
    | RunStartedEvent
    | RunErrorEvent
    | ToolCallStartEvent
    | ToolCallProgressEvent
    | ToolCallCompleteEvent
    | PlanUpdateEvent
    | CompactionEvent
    | SubAgentEvent
    | SpawnSessionStart
    | ToolResultMetadataEvent
    | CustomEvent[Any]
)
```

Key events for this mapping:
- `PartStartEvent` / `PartDeltaEvent` — streaming text/tool deltas
- `ToolCallStartEvent` / `ToolCallCompleteEvent` — tool lifecycle
- `StreamCompleteEvent` — final message available
- `RunStartedEvent` — new run beginning

---

## 3. Goals & Non-Goals

### Goals
- Preserve 100 % backward compatibility for existing signal subscribers
- Map every `Graph.iter()` yield to an existing `RichAgentStreamEvent`
- Document exact emission points with code snippets
- Enable zero-change migration for ACP / OpenCode / AG-UI consumers

### Non-Goals
- Introduce new event types (reuse existing ones only)
- Implement the adapter layer (design only)
- Modify pydantic-graph internals
- Change the semantics of `Talk.connection_type` (run/context/forward)

---

## 4. Signal → GraphRun Event Point Mapping

| Current Signal | GraphRun Concept | Emission Point |
|---|---|---|
| `MessageNode.message_received` | Step start | Immediately before `node.call(step_context)` in `_run_task()` |
| `MessageNode.message_sent` | Step complete | Immediately after `node.call(step_context)` returns, before edge handling |
| `Talk.connection_processed` | Edge traversal | When `_handle_path()` resolves a `DestinationMarker` to a new `GraphTask` |
| `Talk.message_forwarded` | Edge traversal (post-transform) | After transform/filter applied, before `_process_for_target()` |

### 4.1 `message_received` → Step Start

In `pydantic-graph`, the equivalent of "a node received input" is the moment just before `Step.call()` is invoked. The adapter wraps `_run_task()`:

```python
# In adapter wrapping _GraphIterator._run_task()
async def _run_task_with_signals(task: GraphTask) -> ...:
    node = graph.nodes[task.node_id]
    if isinstance(node, Step):
        # Emulate MessageNode.message_received
        incoming_msg = _graph_task_to_chat_message(task)
        await message_node.message_received.emit(incoming_msg)

    result = await original_run_task(task)
    return result
```

**Rationale**: `GraphTask` carries `inputs` and `node_id`. We reconstruct a `ChatMessage` from the task inputs. This is the earliest point where we know the node is about to execute.

### 4.2 `message_sent` → Step Complete

After `node.call()` returns, the step has produced its output. This maps to `message_sent`:

```python
    result = await original_run_task(task)

    if isinstance(node, Step):
        outgoing_msg = _graph_result_to_chat_message(result)
        # Emulate MessageNode.message_sent
        await message_node.message_sent.emit(outgoing_msg)

    return result
```

**Rationale**: At this point the node has finished computation. The result may be a `BaseNode`, `End`, or raw data. We wrap it into a `ChatMessage` to preserve the existing signal signature.

### 4.3 `connection_processed` → Edge Traversal

In `Talk._handle_message()`, `connection_processed` captures the full routing context (source, targets, connection_type, queued). In GraphRun, the equivalent is when `_handle_path()` produces a new `GraphTask`:

```python
# In adapter intercepting _handle_path or _handle_edges
async def _handle_path_with_signals(path: Path, inputs: Any, fork_stack: ForkStack):
    tasks = original_handle_path(path, inputs, fork_stack)
    for task in tasks:
        await talk.connection_processed.emit(
            Talk.ConnectionProcessed(
                message=_inputs_to_chat_message(inputs),
                source=source_node,
                targets=[graph.nodes[task.node_id]],
                queued=False,  # GraphRun tasks are eagerly scheduled
                connection_type="run",
            )
        )
    return tasks
```

**Rationale**: `_handle_path()` is where the graph resolves a path segment into a concrete destination node task. This is the exact moment a "connection" is processed.

### 4.4 `message_forwarded` → Edge Traversal (Post-Transform)

`message_forwarded` is emitted after transform/filter but before per-target dispatch. In GraphRun, transforms are `TransformMarker` on paths. We intercept after the marker is applied:

```python
# In adapter intercepting TransformMarker application
if isinstance(item, TransformMarker):
    transformed_inputs = item.transform(StepContext(...))
    await talk.message_forwarded.emit(
        _inputs_to_chat_message(transformed_inputs)
    )
    return self._handle_path(path.next_path, transformed_inputs, fork_stack)
```

**Rationale**: `TransformMarker` is pydantic-graph's equivalent of `Talk.transform_fn`. Emitting after transform preserves the existing semantic that subscribers see the post-transform message.

---

## 5. Graph.iter() Yield → AgentPool Event Mapping

### 5.1 Yield Types and Event Mapping

| Graph.iter() Yield | Maps To | Event Type | Notes |
|---|---|---|---|
| `Sequence[GraphTask]` (first yield of a step) | PartStartEvent | `PartStartEvent` | Signals a new node/step is beginning execution |
| Step function streaming chunks | PartDeltaEvent | `PartDeltaEvent` | Each yielded chunk becomes a delta event |
| Tool call invocation inside step | ToolCallStartEvent | `ToolCallStartEvent` | When step calls a tool |
| Tool call result inside step | ToolCallCompleteEvent | `ToolCallCompleteEvent` | When tool returns |
| `EndMarker` | StreamCompleteEvent | `StreamCompleteEvent` | Final message with all content |
| `ErrorMarker` | RunErrorEvent | `RunErrorEvent` | Exception wrapped in event |

### 5.2 `GraphTask` Yield → `PartStartEvent`

When `Graph.iter()` yields `Sequence[GraphTask]`, each task represents a node about to run. We map this to `PartStartEvent`:

```python
async def _emit_for_task(task: GraphTask):
    await event_manager.emit_agent_event(
        PartStartEvent.text(
            index=task.task_id,
            content=f"Starting node {task.node_id}",
        ),
        source_session_id=session_id,
    )
```

**Sequence**: This is the first event consumers see for a given step. It aligns with `message_received` but uses the standard streaming event type.

### 5.3 Step Function Streaming → `PartDeltaEvent` Chunks

If a `Step` is defined via `GraphBuilder.stream()` (returns `AsyncIterable`), the adapter consumes the iterable and maps each chunk:

```python
async def _consume_stream_step(task: GraphTask, stream: AsyncIterable[str]):
    index = 0
    async for chunk in stream:
        await event_manager.emit_agent_event(
            PartDeltaEvent.text(
                index=index,
                content=chunk,
            ),
            source_session_id=session_id,
        )
        index += 1
```

**Rationale**: `GraphBuilder.stream()` creates a step whose `call()` returns an async iterator. The adapter wraps this iterator to emit `PartDeltaEvent` for each chunk, identical to how native agents stream today.

### 5.4 Tool Calls → `ToolCallStartEvent` + `ToolCallCompleteEvent`

When a step invokes a tool (e.g., via PydanticAI's tool framework), the adapter intercepts at the tool boundary:

```python
async def _wrap_tool_call(tool_name: str, tool_input: dict[str, Any]):
    tool_call_id = _generate_tool_call_id()

    await event_manager.emit_agent_event(
        ToolCallStartEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            title=f"Running {tool_name}",
            raw_input=tool_input,
        ),
        source_session_id=session_id,
    )

    result = await original_tool_call(tool_name, tool_input)

    await event_manager.emit_agent_event(
        ToolCallCompleteEvent(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
            tool_result=result,
            agent_name=agent_name,
            message_id=message_id,
        ),
        source_session_id=session_id,
    )

    return result
```

**Rationale**: Tool calls are opaque to GraphRun — they happen inside `node.call()`. The adapter must wrap the tool invocation layer (e.g., PydanticAI's `Tool` or AgentPool's `FunctionTool`) to emit these events.

### 5.5 `EndMarker` → `StreamCompleteEvent`

When `Graph.iter()` yields `EndMarker`, the graph is complete:

```python
if isinstance(yielded, EndMarker):
    final_message = ChatMessage(content=yielded.value)
    await event_manager.emit_agent_event(
        StreamCompleteEvent(message=final_message),
        source_session_id=session_id,
    )
```

**Rationale**: `EndMarker` carries the final output. We wrap it in `ChatMessage` to satisfy `StreamCompleteEvent`'s signature. This is the terminal event for the stream.

---

## 6. Signal Emission Points with Code Examples

### 6.1 Complete Adapter Wrapper

The adapter is a thin wrapper around `GraphRun` that intercepts key lifecycle points:

```python
class SignalEmittingGraphRun(Generic[StateT, DepsT, OutputT]):
    """Wraps a GraphRun to emit anyenv.Signal events at step boundaries."""

    def __init__(
        self,
        graph_run: GraphRun[StateT, DepsT, OutputT],
        node_mapping: dict[NodeID, MessageNode[Any, Any]],
        talk_mapping: dict[NodeID, Talk[Any]],
        event_manager: EventManager,
        session_id: str,
    ) -> None:
        self._graph_run = graph_run
        self._node_mapping = node_mapping
        self._talk_mapping = talk_mapping
        self._event_manager = event_manager
        self._session_id = session_id

    async def __anext__(self):
        result = await self._graph_run.__anext__()

        if isinstance(result, Sequence):
            for task in result:
                await self._emit_task_start(task)
        elif isinstance(result, EndMarker):
            await self._emit_stream_complete(result)

        return result

    async def _emit_task_start(self, task: GraphTask) -> None:
        """Emit message_received + PartStartEvent for a new task."""
        node = self._node_mapping.get(task.node_id)
        if node is None:
            return

        msg = ChatMessage(content=task.inputs, session_id=self._session_id)
        await node.message_received.emit(msg)
        await self._event_manager.emit_agent_event(
            PartStartEvent.text(index=task.task_id, content=str(task.inputs)),
            source_session_id=self._session_id,
        )

    async def _emit_stream_complete(self, marker: EndMarker[Any]) -> None:
        """Emit message_sent + StreamCompleteEvent on graph end."""
        msg = ChatMessage(content=marker.value, session_id=self._session_id)

        # Emit for all nodes that participated
        for node in self._node_mapping.values():
            await node.message_sent.emit(msg)

        await self._event_manager.emit_agent_event(
            StreamCompleteEvent(message=msg),
            source_session_id=self._session_id,
        )
```

### 6.2 Talk Signal Emission in Graph Context

For `Talk` signals, the adapter intercepts edge traversal:

```python
async def _handle_edge_with_talk_signals(
    source_node_id: NodeID,
    destination_node_id: NodeID,
    inputs: Any,
    talk: Talk[Any],
) -> None:
    """Emit connection_processed and message_forwarded during edge traversal."""
    msg = ChatMessage(content=inputs)
    targets = [node for node in talk.targets if node.name == destination_node_id]

    await talk.connection_processed.emit(
        Talk.ConnectionProcessed(
            message=msg,
            source=talk.source,
            targets=targets,
            queued=False,
            connection_type=talk.connection_type,
        )
    )

    if targets:
        await talk.message_forwarded.emit(msg)
```

---

## 7. Event Sequence Diagrams

### 7.1 Simple Sequential Chain (Agent A → Agent B)

```
User Input
    │
    ▼
┌─────────────────┐
│ GraphRun.start  │
│ _first_task     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     message_received (A)
│ GraphTask(A)    │ ──► PartStartEvent(A)
│ yield           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ A.node.call()   │
│ (step execution)│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
PartDelta   ToolCallStart
(chunks)    ToolCallComplete
    │
    ▼
┌─────────────────┐     message_sent (A)
│ call() returns  │ ──► connection_processed (A→B)
└────────┬────────┘     message_forwarded (A→B)
         │
         ▼
┌─────────────────┐     message_received (B)
│ GraphTask(B)    │ ──► PartStartEvent(B)
│ yield           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ B.node.call()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     message_sent (B)
│ EndMarker       │ ──► StreamCompleteEvent
│ yield           │
└─────────────────┘
```

### 7.2 Parallel Team (Fork → A & B → Join)

```
User Input
    │
    ▼
┌─────────────────┐
│ Fork node       │
│ (broadcast)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
GraphTask(A)  GraphTask(B)
    │         │
    ▼         ▼
PartStart(A)  PartStart(B)
    │         │
    ▼         ▼
A.call()      B.call()
    │         │
    ▼         ▼
message_sent  message_sent
    │         │
    ▼         ▼
┌─────────────────┐
│ Join node       │
│ (reducer)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EndMarker       │ ──► StreamCompleteEvent
└─────────────────┘
```

### 7.3 Tool Call Within a Step

```
Step Execution
    │
    ▼
┌─────────────────┐
│ Tool invocation │
│ detected        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ToolCallStartEvent
│ tool_call_id=t1 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tool executes   │
│ (may emit       │
│  ToolCallProgressEvent)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ToolCallCompleteEvent
│ tool_call_id=t1 │
└────────┬────────┘
         │
         ▼
Step continues (result injected into LLM context)
    │
    ▼
PartDeltaEvent (LLM resumes streaming)
```

---

## 8. Implementation Considerations

### 8.1 Adapter Layer Location

The adapter should live in a new module, e.g.:
- `src/wolfharness/delegation/graph_adapter.py`

It wraps `GraphRun` without subclassing it, to avoid coupling to pydantic-graph internals.

### 8.2 Node Identity Mapping

`GraphRun` uses `NodeID` (strings). AgentPool uses `MessageNode` instances. The adapter requires a bidirectional mapping:

```python
node_to_id: dict[MessageNode[Any, Any], NodeID]
id_to_node: dict[NodeID, MessageNode[Any, Any]]
```

This is built when the `AgentPool` converts its `Team` / `TeamRun` definitions into a `Graph`.

### 8.3 Session ID Propagation

`GraphRun` has no concept of "session". The adapter injects session_id into:
- Reconstructed `ChatMessage` instances for signals
- `source_session_id` parameter of `emit_agent_event()`

### 8.4 Backpressure and Queuing

`Talk.queued` and `queue_strategy` are AgentPool-specific. In GraphRun, all tasks are eagerly scheduled via `TaskGroup`. If queuing behavior must be preserved, the adapter can:
1. Buffer tasks in the adapter instead of passing to GraphRun
2. Emit `connection_processed` with `queued=True`
3. Flush buffered tasks on `Talk.trigger()`

This adds complexity; a simpler v1 can ignore queuing (treat all as non-queued) since GraphRun's `TaskGroup` handles concurrency natively.

### 8.5 Error Handling

`ErrorMarker` in GraphRun allows recovery. The adapter maps it to `RunErrorEvent`:

```python
if isinstance(yielded, ErrorMarker):
    await event_manager.emit_agent_event(
        RunErrorEvent(
            message=str(yielded.error),
            run_id=session_id,
            agent_name=node_name,
        ),
        source_session_id=session_id,
    )
    # Re-raise to preserve GraphRun semantics
    raise yielded.error
```

---

## 9. Open Questions

1. **TransformMarker ordering**: `Talk.transform_fn` runs before `filter_condition`. In GraphRun, `TransformMarker` and path-level filtering happen at different stages. Does the adapter need to replicate the exact AgentPool ordering?

2. **Connection types**: `Talk.connection_type` can be `"run"`, `"context"`, or `"forward"`. GraphRun edges always execute the destination node. How should `"context"` and `"forward"` be represented in the graph?

3. **MessageNode.run_iter()**: Current `MessageNode` has `run_iter()` which yields `ChatMessage`. GraphRun yields `GraphTask`. Should `run_iter()` be reimplemented as an async generator over the GraphRun iterator, or should the adapter provide a separate streaming API?

4. **SubAgentEvent propagation**: When a step delegates to a subagent, the subagent's events must be wrapped in `SubAgentEvent`. Does this happen inside the step's tool wrapper or at the GraphRun adapter level?

5. **Fork/Join ↔ Team/TeamRun mapping**: A `Team` (parallel) maps to Fork+Join. A `TeamRun` (sequential) maps to a linear chain of Steps. Should the adapter support dynamic graph construction from YAML configs, or is the graph built once at pool initialization?

---

## 10. Decision Record

| Decision | Rationale |
|---|---|
| Wrap `GraphRun` rather than subclass | Avoid coupling to pydantic-graph internals; GraphRun's `__init__` and iteration are complex |
| Reuse existing event types exclusively | Zero-change requirement for ACP/OpenCode/AG-UI consumers |
| Emit `message_received` / `message_sent` at step boundaries | Closest semantic match; `_run_task()` is the boundary between graph orchestration and node execution |
| Map `GraphTask` yield to `PartStartEvent` | `PartStartEvent` is the existing "something is beginning" event in the stream |
| Map `EndMarker` to `StreamCompleteEvent` | Terminal event with final `ChatMessage`; exact semantic match |
| Intercept tool calls at tool wrapper layer | GraphRun is opaque to tool calls; must wrap at the AgentPool tool framework level |
