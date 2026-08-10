---
rfc_id: RFC-0039
title: ACP Subagent Zed 协议升级 — 框架级事件自动发射与 Event+闭包完成通知
status: DRAFT
author: yuchen.liu
reviewers:
  - name: Oracle
    status: completed
    date: 2026-06-26
    session: ses_0fbc20147ffey44MYpg57mPH3T
created: 2026-06-26
last_updated: 2026-06-27
supersedes: RFC-0027
---

# RFC-0039: ACP Subagent Zed 协议升级 — 框架级事件自动发射与 Event+闭包完成通知

## 概述

本 RFC 提出对 AgentPool ACP Server 的 subagent 兼容性进行升级，修复 RFC-0027 Phase 1 实施后遗留的关键问题。RFC-0027 已实现 `zed` 显示模式、`SubagentSessionInfo` 模型、`_build_subagent_field_meta()` 辅助方法以及 `SpawnSessionStart → ToolCallStart`（带 `_meta`）的基本流程。然而 Oracle 评估（2026-06-26）通过源码级验证发现了若干关键缺陷：`tool_call_id` 断联导致完成通知无法工作、`kind` 仍为 `"other"` 而非 `"subagent"`、`ToolCallProgress` 上缺失 `_meta`、以及跨 converter 状态协调需要迁移到 `_after_consumer_loop`。

本 RFC 提出两个核心架构改进：

1. **框架级事件自动发射**：`create_child_session()` 自动发射 `SpawnSessionStart`，消除 3 处 15 行手动模板代码（subagent_tools × 1 + workers × 2）。`tool_call_id`、`depth`、`MAX_SUBAGENT_DEPTH` 检查统一在框架层处理。借鉴 Zed 的 `ThreadEnvironment::create_subagent()` 设计模式。（team/teamrun 不受影响——它们使用 `yield` 在 async generator 中产出事件，不经过 `create_child_session()`。）

2. **Event + 闭包完成通知**：利用 mixin 已有的 `_consumer_done_events: dict[str, anyio.Event]` 基础设施，在 `_on_spawn_session_start` 中抓取 `done_event` 引用，闭包捕获 parent 上下文，后台 task `await done_event.wait()` 后发射 `ToolCallProgress(completed)`。无需维护 `_subagent_map` dict——闭包天然捕获上下文，Event 自动管理生命周期。

同时，ACP 协议已于 2026-06-24 发布 v1.0.0 首个稳定版，Zed 的 ACP SDK 已升级至 `=1.0.0`，ACP Subagents RFD（PR #855）于 2026-06-25 恢复开发。本 RFC 还整合了对 Zed、OpenCode、Pydantic-AI、Hermes-Agent、Claw-Code、Pi 六个框架的 subagent 架构横向调研结果，借鉴最佳实践。

预期结果：Zed 正确检测 subagent 工具调用、渲染展开/折叠卡片 UI、在子会话完成时收到 `ToolCallProgress(status="completed")` 通知。

## 背景与上下文

### RFC-0027 已实施内容

RFC-0027（2026-04-24）实施了 Phase 1，包括：

| 已实现 | 文件位置 | 说明 |
|--------|---------|------|
| `SubagentSessionInfo` 数据模型 | `event_converter.py:113-128` | `session_id`, `message_start_index`, `message_end_index` |
| `_build_subagent_field_meta()` | `event_converter.py:186-212` | 返回 `{"subagent_session_info": ..., "tool_name": "task"}` |
| `zed` 显示模式 | `event_converter.py:648-659` | 发射 `ToolCallStart` 含 `field_meta` |
| 子会话消费循环 | `handler.py:119-124` | zed 模式下后台任务也创建子会话消费循环 |
| 旧模式废弃 | `event_converter.py` | `inline`/`tool_box` 强制转换为 `legacy` |

### 当前事件流（SessionPool + EventBus 架构）

`event_converter.py` 是**活跃的核心组件**，未被 SessionPool + EventBus 架构替代。两条代码路径均通过 `ACPEventConverter.convert()`：

```
路径 1（新 — SessionPool + EventBus）:
  Agent.run_stream() → EventBus.publish() → EventEnvelope
    → mixin._event_consumer_loop() → handler._handle_event()
    → self._converters[sid].convert(event)    ← ACPEventConverter
    → yields ACPSessionUpdate
    → client.session_update(notification)

路径 2（旧 — Legacy）:
  session.py:process_prompt()
    → converter.convert(event)
    → notifications.send_update(update)
```

每个子会话获得独立的 `ACPEventConverter` 实例（`handler.py:135`，`_before_consumer_loop` 中创建），存储在 `handler._converters: dict[str, ACPEventConverter]`。

### `_after_consumer_loop` 机制

`ProtocolEventConsumerMixin`（`mixins.py:26-258`）提供 EventBus 消费循环生命周期管理：

```
_event_consumer_loop(session_id):
  try:
    _before_consumer_loop(session_id)    ← 创建 per-session converter
    async for envelope in stream:        ← 持续消费 EventBus 事件
      _on_spawn_session_start()          ← SpawnSessionStart 时触发
      _handle_event()                    ← 每个事件都触发
  finally:
    _after_consumer_loop(session_id)     ← ← ← 无论怎么退出都会执行
```

`_after_consumer_loop` 在 `finally` 块中（line 258），无论 consumer loop 正常结束、异常还是被取消都会调用。当前 ACP handler 的实现仅做 `self._converters.pop(session_id, None)`。

### 跨框架调研发现

对 6 个框架的 subagent 架构进行了横向调研：

| 框架 | 创建方式 | 完成检测 | 事件路由 | tool_call_id 关联 | ACP 兼容 |
|------|---------|---------|---------|------------------|---------|
| **Zed** | 框架自动 (ThreadEnvironment trait) | `await subagent.send()` 同步阻塞 | `_meta` + 运行时事件双路径 | ✅ `_meta` 双向查找 | ✅ 原生 |
| **OpenCode** | `sessions.create({ parentID })` | Deferred push — `background.wait()` await `done` Deferred | 全局事件流 + `tracked()` 过滤 | `part.callID` 流转 | `kind="think"` |
| **Pydantic-AI** | 手动 tool 内 `await agent.run()` | `await` 同步阻塞 | ❌ 不传播 | N/A | ❌ |
| **Hermes-Agent** | 手动创建 AIAgent 实例 | sync: future.result() / async: 轮询队列 | callback chain (DelegateEvent) | N/A | ❌ |
| **Claw-Code** | OS 线程 spawn | 线程 join + 文件读取 | ❌ 文件 IPC | N/A | ❌ |
| **Pi** | OS 进程 spawn | 进程 exit code | JSON-lines stdout | N/A | ❌ |

**关键启示**：

1. **`_after_consumer_loop` 方案与 OpenCode 的 Deferred push 本质相同**——都是异步完成通知，不阻塞 tool 返回。AgentPool 的 mixin hook 是对等方案。

2. **OpenCode 的 message injection 模式值得借鉴**——OpenCode 通过 `ops.prompt(parentSession, synthetic: true)` 将子代理结果作为合成消息注入父会话。AgentPool 当前 tool 只返回纯文本，没有结构化结果。可考虑在 `_after_consumer_loop` 中通过 EventBus 向父会话注入 `SubagentCompletedEvent`。

3. **OpenCode 的 `kind="think"` vs AgentPool 的 `kind="subagent"`**——OpenCode 故意不用 `"subagent"` 因为它不是 ACP 客户端。AgentPool 作为 ACP server 服务 Zed，应使用 `"subagent"` 以触发 Zed 的 subagent UI。

4. **OpenCode 的递归取消是缺失项**——OpenCode 的 `cancelBackgroundJobs()` 通过 `metadata.parentSessionId` walk tree 递归取消所有子任务。AgentPool 目前无取消传播能力。

5. **Zed 的 tool_call_id 双向查找证明了关联的必要性**——Zed 通过 `_meta.subagent_session_info` 实现 `tool_call_for_subagent(session_id)` 反向查找。AgentPool 的 `tool_call_id` 断联问题必须修复。

6. **结构化结果是普遍缺失**——只有 Zed 和 Hermes 提供了结构化子代理结果。AgentPool 应改进。

### 术语表

| 术语 | 定义 |
|------|------|
| `_after_consumer_loop` | `ProtocolEventConsumerMixin` 中的钩子方法（`mixins.py:91-100`），在子会话 consumer loop 退出时的 `finally` 块中调用 |
| `tool_call_id` 断联 | converter 生成新 `uuid.uuid4()` 作为 `tool_call_id`，忽略 `SpawnSessionStart.tool_call_id` 字段的问题 |
| `_consumer_done_events` | mixin 已有的 `dict[str, anyio.Event]`（`mixins.py:60`），每个子会话的 consumer loop 退出时对应 Event 被 set（`mixins.py:248-250`） |
| Event + 闭包方案 | 在 `_on_spawn_session_start` 中抓取 `done_event` 引用，闭包捕获 parent 上下文（parent_sid, tool_call_id），后台 task `await done_event.wait()` 后发射完成通知。无需 dict 维护映射 |
| 框架级事件自动发射 | `create_child_session()` 自动构造并发射 `SpawnSessionStart`，调用方无需手动构造事件。借鉴 Zed 的 `ThreadEnvironment::create_subagent()` |
| `SubagentRunInfo` | ACP schema 中 `ToolCallStart.subagent` 字段的类型（`tool_call.py:47-60`），包含 `child_session_id`, `subagent_id`, `run_mode`, `display_name` |
| PR #855 | ACP Subagents RFD，旨在协议层面标准化 subagent 交互模式 |
| Deferred push | OpenCode 的完成通知模式——await `done` Deferred，完成后自动触发回调 |

### 相关工作

| RFC / PR | 状态 | 与本 RFC 的关系 |
|----------|------|-----------------|
| RFC-0027 | ✅ Phase 1 已实现 | 本 RFC 的前序，已实现基础 `_meta` 填充 |
| RFC-0013 | ✅ 已实现 | Subagent Event Stream Unification |
| RFC-0014 | ✅ 已实现 | SpawnSessionStart 事件 |
| ACP v1.0.0 | ✅ 已发布 (2026-06-24) | 线协议稳定为 version 1 |
| PR #855 | 🟡 Draft (2026-06-25 恢复) | ACP Subagents RFD，合并后 `_meta` 扩展可能被替代 |
| Zed #58537 | ✅ Merged (2026-06-11) | 保留 waiting tool call 状态，影响 `ToolCallProgress` 发射方式 |

## 问题陈述

### GAP 1（P0）：`tool_call_id` 断联 — 完成通知的前提条件

**文件位置**：`event_converter.py:649`

**现状**：zed 模式下 `SpawnSessionStart` 处理中，converter 生成新的 `tool_call_id = str(uuid.uuid4())`，忽略了 `SpawnSessionStart` 事件已有的 `tool_call_id: str | None = None` 字段（`events.py:705`）。

**影响**：
- `handler._on_spawn_session_start` 无法将生成的 `tool_call_id` 关联回子会话
- 即使添加了完成通知机制，也无法知道哪个 `tool_call_id` 对应哪个子会话
- 这是所有完成通知方案的前提条件

**跨框架对比**：Zed 通过 `_meta.subagent_session_info` 实现 `tool_call_for_subagent(session_id)` 双向查找。OpenCode 通过 `part.callID` 从 AI SDK 流转到 ACP 事件系统。AgentPool 是唯一存在 tool_call_id 断联的框架。

**证据**：
```python
# event_converter.py:649 — 当前代码
tool_call_id = str(uuid.uuid4())  # ❌ 忽略了 event.tool_call_id
```

```python
# events.py:705 — SpawnSessionStart 已有 tool_call_id 字段
tool_call_id: str | None = None
```

### GAP 2（P0）：`kind="other"` 而非 `"subagent"`

**文件位置**：`event_converter.py:656`

**现状**：zed 模式下 `ToolCallStart` 的 `kind` 为 `"other"`，而 `"subagent"` 已在 `ToolCallKind` literal 中定义（`tool_call.py:41`）。

**影响**：Zed 通过 `kind` 判断工具调用类型。`"other"` 不会触发 subagent UI 渲染路径。

**跨框架对比**：OpenCode 故意使用 `kind="think"` 因为它不是 ACP 客户端。AgentPool 作为 ACP server 服务 Zed，应使用 `"subagent"`。

### GAP 3（P0）：子代理结束时未发射 `ToolCallProgress(completed)`

**文件位置**：`event_converter.py` + `handler.py`

**现状**：子会话完成时，父会话的 `ToolCallStart` 永远停留在 pending/in_progress 状态。没有机制通知 Zed 子代理已完成。

**影响**：Zed 的 subagent 卡片会永远显示 loading 状态，无法显示完成图标。

**架构挑战**：SessionPool 模式下每个子会话有独立的 `ACPEventConverter` 实例。子会话的 converter 无法直接通知父会话的 converter。

**跨框架对比**：
- Zed：`await subagent.send()` 同步阻塞，tool 自然返回时即完成
- OpenCode：Deferred push — `background.wait()` await `done` Deferred，自动触发 `inject()`
- AgentPool 提议：`_after_consumer_loop` mixin hook——与 OpenCode 的 Deferred push 本质相同，都是异步完成通知

### GAP 4（P0）：`ToolCallProgress` 上未携带 `_meta`

**文件位置**：`event_converter.py`

**现状**：`_meta.subagent_session_info` 仅在 `ToolCallStart` 上设置，`ToolCallProgress` 上未设置。

**影响**：Zed 的 `ToolCall::update_acp_status()`（`acp_thread.rs:599-600`）也会从 `_meta` 读取子代理信息。如果 `_meta` 缺失，Zed 可能丢失子代理会话跟踪。

**注意**：`field_meta` 通过 `AnnotatedObject` 基类（`base.py:30`）在 `ToolCallProgress` 上可用，无需 schema 变更。`_meta` 中还包含 `"tool_name": "task"`（`event_converter.py:211`），`ToolCallProgress` 上也必须包含此字段。

### GAP 5（P1）：`message_start_index` 始终为 0，`message_end_index` 始终为 None

**文件位置**：`event_converter.py:651`

**现状**：`message_start_index` 硬编码为 0，`message_end_index` 从未设置。

**影响**：Zed 无法精确定位子会话中的条目范围。子会话历史中的展开/折叠内容可能不正确。

**跨框架对比**：Zed 使用 `subagent.num_entries(cx)` 在 spawn 时获取 start index，完成时 `num_entries(cx).saturating_sub(1)` 获取 end index。

### GAP 6（P1）：无 `MAX_SUBAGENT_DEPTH=1` enforcement

**文件位置**：`handler.py` / `session.py`

**现状**：`SpawnSessionStart` 事件已有 `depth` 字段（`events.py:713`），但 handler 未检查。

**影响**：理论上可以无限嵌套子代理，与 Zed 的 `MAX_SUBAGENT_DEPTH=1` 限制不一致。

**跨框架对比**：
- Zed：硬编码 `MAX_SUBAGENT_DEPTH=1`，在 `create_subagent_thread()` 中检查
- OpenCode：权限制——`general` agent 可嵌套，`explore` 不可，更灵活
- AgentPool 应采用 Zed 的硬编码方式以保持兼容

### GAP 7（P1）：无取消传播

**现状**：父会话取消时，不会递归取消子会话。

**影响**：子代理可能在父会话已取消后继续运行，浪费资源。

**跨框架对比**：
- Zed：`Thread::cancel()` 递归遍历 `running_subagents`
- OpenCode：`cancelBackgroundJobs()` 通过 `metadata.parentSessionId` walk tree
- Claw-Code：线程终止

### GAP 8（P2）：原生 `SubagentRunInfo` 字段从未填充

**文件位置**：`event_converter.py`

**现状**：ACP schema 中 `ToolCallStart.subagent` 字段（类型为 `SubagentRunInfo`）从未被设置。

**影响**：Zed 当前从 `_meta` 读取子代理信息，不从 `subagent` 字段读。此字段为前瞻性工作，为 ACP Subagents RFD（PR #855）合并后做准备。

### GAP 9（P2）：无结构化子代理结果

**文件位置**：`subagent_tools.py:335-346`

**现状**：tool 返回纯 `final_content` 字符串。无 child_session_id、无完成状态、无时长、无结构化元数据。

**跨框架对比**：
- Zed：`SpawnAgentToolOutput::Success { session_info, output }`
- Hermes：`DelegateEvent.TASK_COMPLETED` 含完整元数据
- OpenCode：XML 包装 `<task id="..." state="completed">` + 合成消息注入

### GAP 10（P0）：`SpawnSessionStart` 手动发射 — 15 行模板代码 ×3 处

**文件位置**：`subagent_tools.py:247-259`, `workers.py:165-176`, `workers.py:283`

**现状**：3 处调用方各自手动构造 `SpawnSessionStart` 并 `emit_event()`，模式高度一致但重复：

```python
# subagent_tools.py:247-259 — 15 行模板
spawn_event = SpawnSessionStart(
    child_session_id=child_session_id,
    parent_session_id=parent_session_id,
    tool_call_id=ctx.tool_call_id,
    spawn_mechanism="task",
    source_name=agent_or_team,
    source_type=source_type,
    depth=child_depth,
    description=f"Run {agent_or_team} task",
    metadata={"prompt": prompt[:200]} if prompt else {},
    model_id=node_model_id,
)
await ctx.events.emit_event(spawn_event)
```

**影响**：
- 重复代码，维护负担——3 处需同步修改
- `team.py:506` 使用 `yield` 在 async generator 中产出 `SpawnSessionStart`，不走 `ctx.events.emit_event()`——不受 auto-emit 影响，保持现有模式
- `team.py:506` 忘记设 `tool_call_id`——但 team 不经过 `create_child_session()`，需单独处理（不在本 RFC 范围）
- `MAX_SUBAGENT_DEPTH` 检查无处实施——应在框架层统一拦截

**跨框架对比**：Zed 的 `ThreadEnvironment::create_subagent()` 自动处理创建、深度检查、工具过滤。工具代码（`SpawnAgentTool::run()`）只调用 `environment.create_subagent()`，不关心事件发射。

**解决方案**：`create_child_session()` 自动构造并发射 `SpawnSessionStart`，调用方简化为 1 行：

```python
# 修改前（15 行）
child_session_id = await ctx.create_child_session(...)
spawn_event = SpawnSessionStart(child_session_id=..., tool_call_id=ctx.tool_call_id, ...)
await ctx.events.emit_event(spawn_event)

# 修改后（1 行）
child_session_id = await ctx.create_child_session(
    agent_name=agent_or_team,
    agent_type=source_type,
    description=f"Run {agent_or_team} task",
)
```

### 影响分析

| 功能 | 影响 | 严重程度 |
|------|------|----------|
| Zed subagent 卡片永远显示 loading | 完全不可用 | P0 |
| Zed 无法检测 subagent kind | UI 渲染异常 | P0 |
| 子代理完成无通知 | 状态卡死 | P0 |
| Progress 事件丢失子代理跟踪 | 间歇性 UI 错误 | P0 |
| 消息索引不正确 | 展开内容错误 | P1 |
| 无深度限制 | 潜在无限递归 | P1 |
| 无取消传播 | 资源浪费 | P1 |
| 原生 SubagentRunInfo 未填充 | 前瞻性缺失 | P2 |
| 无结构化结果 | 信息缺失 | P2 |

## 目标与非目标

### 目标

| ID | 目标 | 优先级 |
|----|------|--------|
| G1 | 修复 `tool_call_id` 关联：`create_child_session()` 自动从 `ctx.tool_call_id` 填充到 `SpawnSessionStart` | P0 |
| G2 | 修复 `kind` 为 `"subagent"` | P0 |
| G3 | 使用 Event + 闭包方案在子会话完成时发射 `ToolCallProgress(completed)` | P0 |
| G4 | 在 `ToolCallProgress` 上携带 `_meta.subagent_session_info` + `tool_name` | P0 |
| G5 | 追踪准确的 `message_start_index` 和 `message_end_index` | P1 |
| G6 | 强制 `MAX_SUBAGENT_DEPTH=5` — 在 `create_child_session()` 中统一检查 | P1 |
| G7 | 实现递归取消传播 | P1 |
| G8 | 填充原生 `SubagentRunInfo` 字段（`run_mode="foreground"`） | P2 |
| G9 | 返回结构化子代理结果（child_session_id, status, duration） | P2 |
| G10 | `create_child_session()` 自动发射 `SpawnSessionStart`，消除 3 处手动模板代码 | P0 |

### 非目标

| ID | 非目标 | 理由 |
|----|--------|------|
| NG1 | 支持多轮 reprompting | 高风险，依赖 session_manager 对子会话的恢复能力 |
| NG2 | 修改 ACP 协议 schema | 仅使用已有扩展机制 |
| NG3 | 实现 ACP Proxy Chains | RFD 仍在草案阶段 |
| NG4 | 支持 Zed Parallel Agents | 用户级并行架构，非 subagent 嵌套 |
| NG5 | 迁移到 ACP v2 | v2 仍在 scaffolding |
| NG6 | 前台→后台切换（promotion） | OpenCode 独有创新，AgentPool 暂不需要 |

## 评估标准

| 标准 | 权重 | 描述 | 最低阈值 |
|------|------|------|----------|
| Zed 兼容性 | 关键 | Zed 能正确渲染 subagent 卡片 UI 并显示完成状态 | `ToolCallProgress(completed)` 被 Zed 正确接收 |
| 向后兼容性 | 关键 | 现有 legacy 模式行为不变 | legacy 模式功能完全不变 |
| 代码侵入性 | 高 | 对现有文件的修改范围 | 不超过 200 行新增/修改 |
| 实施复杂度 | 中 | 开发工时估算 | P0 不超过 2 天 |
| 可测试性 | 中 | 新增代码的可测试性 | 单元测试覆盖率 ≥ 80% |
| 协议合规性 | 中 | 符合 ACP v1 扩展机制 | 仅使用 `_meta` 扩展 |

## 方案分析

### 选项 1：Converter 内部完成通知 — 在 `ACPEventConverter` 中跟踪子会话状态

**描述**：在 `ACPEventConverter` 中维护 `tool_call_id → child_session_id` 映射，当 `StreamCompleteEvent` 到达时在 converter 内部发射 `ToolCallProgress(completed)`。

**优势**：
- 修改集中在 `event_converter.py` 一个文件
- Converter 已有 `_subagent_tool_map`，可复用

**劣势**：
- **`SpawnSessionStart` 双重派发问题**：`_on_spawn_session_start`（handler）和 `_handle_event`（→ converter）都接收同一事件，必须对 `tool_call_id` 达成一致
- Converter 没有 handler 上下文，无法访问 `self._converters` 查找父 converter
- `StreamCompleteEvent` 在子会话的 converter 中处理，但需要通知父会话的 converter — 跨 converter 状态共享
- 需要额外的 `session_manager` 引用注入到 converter

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| Zed 兼容性 | 3/5 | 可工作但跨 converter 协调复杂 |
| 向后兼容性 | 5/5 | 仅影响 zed 模式 |
| 代码侵入性 | 2/5 | 需要注入 session_manager 到 converter |
| 实施复杂度 | 2/5 | 跨 converter 状态共享增加复杂度 |
| 可测试性 | 3/5 | 跨 converter 测试困难 |

**工作量估算**：中（~200 行，2-3 天）

**风险评估**：
- 技术风险：中。跨 converter 状态共享可能导致竞态条件
- 兼容性风险：低。仅影响 zed 模式

---

### 选项 2：Handler 层完成通知 — Event + 闭包（推荐）

**描述**：利用 mixin 已有的 `_consumer_done_events: dict[str, anyio.Event]` 基础设施。在 `_on_spawn_session_start` 中，`start_event_consumer(child_sid)` 后抓取 `done_event` 引用，闭包捕获 `parent_sid` 和 `tool_call_id`，启动后台 task `await done_event.wait()` 后通过父 converter 发射完成通知。无需维护 `_subagent_map` dict。

**优势**：
- **零状态管理**：闭包天然捕获 `parent_sid` 和 `tool_call_id`，无需 dict 存储和清理
- **已接线**：`_consumer_done_events`（`mixins.py:60`）已存在，consumer loop 退出时自动 set（`mixins.py:248-250`）
- **自动生命周期**：Event 对象在 set 后仍然有效，`await done_event.wait()` 照常返回；task 完成后闭包自动释放
- **`_after_consumer_loop` 无需修改**：所有逻辑在 `_on_spawn_session_start` 中完成，职责内聚
- **`_consumer_task_refs` 已存在**（`mixins.py:61`）：mixin 设计之初就考虑了"持有 task 引用防 GC"
- **无竞态条件**：consumer loop 退出在所有事件处理之后，Event set 保证 happened-after
- **跨框架验证**：与 OpenCode 的 Deferred push 模式本质相同——`anyio.Event` 等价于 OpenCode 的 `Deferred`

**劣势**：
- 修改涉及 `handler.py`、`event_converter.py` 和 `context.py` 三个文件
- `done_event` 引用必须在 `start_event_consumer` 之后、consumer loop 退出之前抓取——存在时间窗口
- 闭包捕获 `self`（handler），如果 handler 被销毁而 task 仍在运行，可能访问已释放的对象
- 递归取消仍需轻量的 `_parent_of: dict[str, str]`（child_sid → parent_sid）映射

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| Zed 兼容性 | 5/5 | 完整支持 Zed subagent 完成通知 |
| 向后兼容性 | 5/5 | 仅影响 zed 模式 |
| 代码侵入性 | 5/5 | ~80 行新增，复用已有基础设施 |
| 实施复杂度 | 5/5 | <1 天，利用已有 anyio.Event |
| 可测试性 | 4/5 | 闭包测试需要 mock done_event |
| 协议合规性 | 5/5 | 仅使用 `_meta` 扩展 |

**工作量估算**：低（~80 行，<1 天）

**风险评估**：
- 技术风险：低。`_consumer_done_events` 是已验证的基础设施
- 兼容性风险：低。仅影响 zed 模式
- 生命周期风险：中。需确保闭包捕获的 `self` 在 task 运行期间有效——通过 `_consumer_task_refs` 持有引用缓解

---

### 选项 3：EventBus 事件 — 新增 `SpawnSessionComplete` 事件类型

**描述**：在 EventBus 上新增 `SpawnSessionComplete` 事件，子会话完成时发布，父会话的 handler 订阅并处理。

**优势**：
- 使用 EventBus 发布/订阅模式，解耦清晰
- 支持多个订阅者（如日志、监控）

**劣势**：
- 需要新增事件类型定义
- 需要修改 EventBus 订阅 scope（当前为 `"session"`，父会话只接收自己的事件）
- 过度设计 — `_after_consumer_loop` 已满足需求
- 新增事件类型影响面较大

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| Zed 兼容性 | 5/5 | 可工作 |
| 向后兼容性 | 5/5 | 新增事件，不影响现有 |
| 代码侵入性 | 2/5 | 需要新增事件类型 + 修改 EventBus |
| 实施复杂度 | 2/5 | 3-4 天，涉及事件系统变更 |
| 可测试性 | 3/5 | EventBus 测试复杂度较高 |

**工作量估算**：中高（~250 行，3-4 天）

**风险评估**：
- 技术风险：中。EventBus scope 修改可能影响其他协议服务器
- 兼容性风险：低。新增事件类型

## 推荐

**推荐选项 2：Handler 层完成通知 — Event + 闭包**。

推荐理由：

1. **零状态管理**：闭包天然捕获上下文，无需 dict 存储和清理——比原 `_subagent_map` 方案更优雅
2. **复用已有基础设施**：`_consumer_done_events`（`mixins.py:60`）和 `_consumer_task_refs`（`mixins.py:61`）已存在
3. **`_after_consumer_loop` 无需修改**：所有逻辑在 `_on_spawn_session_start` 中完成，职责内聚
4. **跨框架验证**：`anyio.Event` 等价于 OpenCode 的 `Deferred`——都是异步 push 通知
5. 工作量最低（~80 行，<1 天）
6. 配合框架级事件自动发射（G10），`tool_call_id` 从源头正确传递，无需双重派发处理

**接受的权衡**：
- 闭包捕获 `self`（handler），需确保 handler 在 task 运行期间有效
- 递归取消仍需轻量 `_parent_of` 映射

## 技术设计

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│               提议的 Subagent 生命周期（三层架构）                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ── 层 1：框架层（context.py）──                                         │
│                                                                          │
│  ctx.create_child_session(agent_name, agent_type, description=...)       │
│     ├─ 创建 child session via SessionPool                                │
│     ├─ 自动填充 tool_call_id = ctx.tool_call_id                          │
│     ├─ 自动计算 depth = parent_depth + 1                                 │
│     ├─ 深度检查: depth > MAX_SUBAGENT_DEPTH → raise                      │
│     ├─ 自动构造 SpawnSessionStart(tool_call_id=..., depth=...)           │
│     └─ 自动 emit_event(spawn_event)                                      │
│                                                                          │
│  调用方简化为 1 行（消除 3 处 × 15 行模板代码）                          │
│                                                                          │
│  ── 层 2：Handler 层（handler.py）──                                     │
│                                                                          │
│  _on_spawn_session_start(session_id, envelope)                           │
│     ├─ start_event_consumer(child_sid)    ← 创建 _consumer_done_events   │
│     ├─ done_event = self._consumer_done_events.get(child_sid)            │
│     ├─ 闭包捕获: parent_sid, tool_call_id, child_sid                     │
│     └─ 后台 task: await done_event.wait() → 发射 ToolCallProgress        │
│                                                                          │
│  ── 层 3：Mixin 层（mixins.py）──                                        │
│                                                                          │
│  _event_consumer_loop(child_sid):                                        │
│     try:                                                                 │
│       _before_consumer_loop(child_sid)    ← 创建 child converter         │
│       async for envelope in stream:       ← 消费子会话事件               │
│         _handle_event()                   ← converter.convert(event)     │
│     finally:                                                             │
│       done_event.set()                    ← ← ← 闭包 task 醒来           │
│       _after_consumer_loop(child_sid)     ← 清理 converter               │
│                                                                          │
│  ── 完成通知流程 ──                                                      │
│                                                                          │
│  done_event.set()                                                        │
│     ↓                                                                    │
│  闭包 task 醒来                                                          │
│     ├─ parent_converter = self._converters.get(parent_sid)               │
│     ├─ update = parent_converter.build_subagent_completed(...)           │
│     └─ client.session_update(notification)  → Zed 收到 completed         │
│                                                                          │
│  ✅ 无 dict: 闭包捕获上下文，Event 自动管理生命周期                      │
│  ✅ tool_call_id 从 ctx → event → converter 一致传递                    │
│  ✅ _after_consumer_loop 无需修改                                        │
│  ✅ 深度检查在框架层统一拦截                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 数据模型

#### Event + 闭包完成通知（无需新增 dict）

利用 mixin 已有的 `_consumer_done_events`（`mixins.py:60`）：

```python
# mixins.py:60 — 已存在，无需修改
self._consumer_done_events: dict[str, anyio.Event] = {}

# mixins.py:248-250 — 已存在，consumer loop 退出时自动 set
done_event = self._consumer_done_events.pop(session_id, None)
if done_event is not None:
    done_event.set()
```

`anyio.Event` 引用在 set 后仍然有效——即使 dict 已 pop 条目，event 对象还在，`await done_event.wait()` 照常返回。

#### `_parent_of` 轻量映射（仅用于递归取消）

```python
# handler.py — __init__ 中新增（仅用于递归取消，不用于完成通知）
self._parent_of: dict[str, str] = {}  # child_sid → parent_sid
```

#### `build_subagent_completed()`（新增在 `ACPEventConverter` 上）

```python
# event_converter.py — ACPEventConverter 新增方法
def build_subagent_completed(
    self,
    tool_call_id: str,
    child_session_id: str,
    message_end_index: int | None = None,
) -> ToolCallProgress:
    """构建子代理完成的 ToolCallProgress。"""
    field_meta = self._build_subagent_field_meta(
        child_session_id=child_session_id,
        tool_name="task",
        message_end_index=message_end_index,
    )
    return ToolCallProgress(
        tool_call_id=tool_call_id,
        status="completed",
        field_meta=field_meta,
    )
```

### API 变更

#### `context.py` 变更 — 框架级事件自动发射（GAP 10 修复）

**`create_child_session()` 自动发射 `SpawnSessionStart`**

```python
# context.py — create_child_session 修改
async def create_child_session(
    self,
    agent_name: str,
    agent_type: str,
    parent_session_id: str | None = None,
    *,
    spawn_mechanism: str = "task",
    description: str | None = None,
    tool_call_id: str | None = None,
    **metadata: Any,
) -> str:
    """Create a child session and automatically emit SpawnSessionStart.

    Args:
        agent_name: Name of the child agent.
        agent_type: Type of the child agent.
        parent_session_id: Explicit parent session ID.
        spawn_mechanism: "task" or "spawn".
        description: Human-readable description.
        tool_call_id: Parent tool call ID (auto-filled from ctx if available).
        **metadata: Additional metadata.
    """
    # ... 现有的 session 创建逻辑 ...
    child_session_id = ...  # 现有逻辑

    effective_parent = parent_session_id or self.node._events.session_id
    # ✅ Fix #3: 直接访问类型化字段，不使用 getattr（AGENTS.md 禁止 getattr）
    effective_tool_call_id = tool_call_id or self.tool_call_id

    # 计算深度
    # ✅ Fix #3: 直接访问 run_ctx.depth（AgentRunContext 有 depth: int = 0 字段）
    parent_depth = self.run_ctx.depth if self.run_ctx is not None else 0
    child_depth = parent_depth + 1

    # ✅ 深度检查（GAP 6 修复）— 框架层统一拦截
    if child_depth > MAX_SUBAGENT_DEPTH:
        raise SubagentDepthError(child_depth, MAX_SUBAGENT_DEPTH)

    # ✅ 自动发射 SpawnSessionStart（GAP 10 修复）
    spawn_event = SpawnSessionStart(
        child_session_id=child_session_id,
        parent_session_id=effective_parent,
        tool_call_id=effective_tool_call_id,
        spawn_mechanism=spawn_mechanism,
        source_name=agent_name,
        source_type=agent_type,
        depth=child_depth,
        description=description or f"Spawn {agent_name}",
        metadata=metadata,
    )
    # ✅ Fix #2: 使用 self.events.emit_event()（含 EventBus），不是 self.node._events
    await self.events.emit_event(spawn_event)

    return child_session_id
```

**调用方简化**：

```python
# subagent_tools.py — 修改前（15 行模板）
child_session_id = await ctx.create_child_session(...)
spawn_event = SpawnSessionStart(
    child_session_id=child_session_id,
    parent_session_id=parent_session_id,
    tool_call_id=ctx.tool_call_id,
    spawn_mechanism="task",
    source_name=agent_or_team,
    source_type=source_type,
    depth=child_depth,
    description=f"Run {agent_or_team} task",
    metadata={"prompt": prompt[:200]} if prompt else {},
    model_id=node_model_id,
)
await ctx.events.emit_event(spawn_event)

# subagent_tools.py — 修改后（1 行）
child_session_id = await ctx.create_child_session(
    agent_name=agent_or_team,
    agent_type=source_type,
    description=f"Run {agent_or_team} task",
    metadata={"prompt": prompt[:200]} if prompt else {},
)
```

> ⚠️ **注意**：`team.py` 和 `teamrun.py` 的 `SpawnSessionStart` 通过 `yield` 在 async generator 中产出，不走 `ctx.events.emit_event()`。它们创建子会话时直接调用 `session_pool.create_session()`，不经过 `ctx.create_child_session()`。因此 auto-emit **不影响** team/teamrun，它们保持现有 yield 模式。仅 3 处调用方（subagent_tools × 1 + workers × 2）受影响。

#### `event_converter.py` 变更

**1. 修复 `tool_call_id` 断联（GAP 1）**

```python
# event_converter.py:649 — 修改前
tool_call_id = str(uuid.uuid4())

# event_converter.py:649 — 修改后
tool_call_id = event.tool_call_id or str(uuid.uuid4())
```

**2. 修复 `kind`（GAP 2）**

```python
# event_converter.py:656 — 修改前
kind="other",

# event_converter.py:656 — 修改后
kind="subagent",
```

**3. 在 `ToolCallProgress` 上携带 `_meta`（GAP 4）**

当前 zed 模式下 `SpawnSessionStart` 仅发射 `ToolCallStart`。需在所有后续 `ToolCallProgress`（针对同一 `tool_call_id`）上携带相同的 `field_meta`（含 `subagent_session_info` + `tool_name`）。

**4. 新增 `build_subagent_completed()` 方法**

见上方数据模型部分。

**5. 填充 `SubagentRunInfo`（GAP 8，P2）**

```python
# event_converter.py — zed 模式 ToolCallStart 构造中新增
subagent=SubagentRunInfo(
    child_session_id=child_id,
    subagent_id=child_id,
    run_mode="foreground",  # ⚠️ 不是 "async"，schema 仅允许 "foreground" | "background"
    display_name=source_name,
),
```

#### `handler.py` 变更

**1. 新增 `_parent_of` 轻量映射（仅用于递归取消）**

```python
# handler.py — __init__ 中新增
self._parent_of: dict[str, str] = {}  # child_sid → parent_sid（仅用于递归取消）
MAX_SUBAGENT_DEPTH: int = 5
```

**2. `_on_spawn_session_start` 中 Event + 闭包完成通知**

```python
# handler.py — _on_spawn_session_start 修改
async def _on_spawn_session_start(self, session_id: str, envelope: EventEnvelope) -> None:
    event = envelope.event
    if isinstance(event, SpawnSessionStart):
        child_sid = event.child_session_id
        if not child_sid or child_sid == session_id:
            return
        if getattr(event, "spawn_mechanism", None) == "task":
            if self._event_converter_template.subagent_display_mode != "zed":
                return

        # ✅ 启动子 consumer — 这会创建 _consumer_done_events[child_sid]
        await self.start_event_consumer(child_sid)

        # ✅ 抓住 done event 引用
        done_event = self._consumer_done_events.get(child_sid)

        # ✅ 闭包捕获上下文 — 不需要 dict！
        parent_sid = session_id
        tool_call_id = event.tool_call_id  # 已由 create_child_session 填充

        # ✅ 注册 parent 关系（仅用于递归取消）
        self._parent_of[child_sid] = parent_sid

        # ✅ Fix #1: 提取通知逻辑为 helper，避免 done_event 为 None 时重复
        async def _notify_completed() -> None:
            """发送 ToolCallProgress(completed) 到父会话。"""
            parent_converter = self._converters.get(parent_sid)
            if parent_converter is None:  # 父会话可能已关闭
                return
            update = parent_converter.build_subagent_completed(
                tool_call_id=tool_call_id,
                child_session_id=child_sid,
            )
            await self.client.session_update(SessionNotification(
                session_id=parent_sid,
                update=update,
            ))

        if done_event is None:
            # ✅ Fix #1: 竞态修复：consumer 已退出（done_event 被 pop+set），立即通知
            try:
                await _notify_completed()
            except Exception:
                logger.exception("Failed to send subagent completion notification (immediate)")
            finally:
                self._parent_of.pop(child_sid, None)  # Fix #7: 清理
            return

        async def _await_child_and_notify() -> None:
            """等待子 consumer 退出后发送完成通知。"""
            try:
                await done_event.wait()  # ← 阻塞直到子 consumer 退出

                # ✅ Fix #7: 正常退出时清理 _parent_of
                self._parent_of.pop(child_sid, None)

                await _notify_completed()

            except (ConnectionResetError, BrokenPipeError) as exc:
                # ✅ Fix #5: 错误处理 — 客户端连接关闭
                logger.debug("Client connection closed before subagent completion: %s", exc)
            except Exception:
                # ✅ Fix #5: 错误处理 — 其他异常不静默吞掉
                logger.exception("Failed to send subagent completion notification")
            finally:
                # ✅ Fix #6: 内存泄漏修复 — task 完成后从 _consumer_task_refs 移除
                with contextlib.suppress(ValueError):
                    self._consumer_task_refs.remove(task)

        # ✅ 后台 task — _consumer_task_refs 持有引用防 GC
        task = asyncio.ensure_future(_await_child_and_notify())
        self._consumer_task_refs.append(task)
```

**3. `_after_consumer_loop` 无需修改**

所有完成通知逻辑在 `_on_spawn_session_start` 中完成。`_after_consumer_loop` 保持现有行为（`self._converters.pop(session_id, None)`）。

**4. 递归取消传播（GAP 7）**

```python
# handler.py — 新增方法
async def _cancel_subagents(self, parent_sid: str) -> None:
    """递归取消父会话的所有子会话。

    参考 OpenCode 的 cancelBackgroundJobs() walk tree 模式。
    使用 _parent_of 轻量映射（child_sid → parent_sid）。
    """
    children = [
        child_sid for child_sid, parent in self._parent_of.items()
        if parent == parent_sid
    ]
    for child_sid in children:
        await self._cancel_subagents(child_sid)  # 递归
        await self.stop_event_consumer(child_sid)
        self._parent_of.pop(child_sid, None)
```

### `SpawnSessionStart` 双重派发处理

`SpawnSessionStart` 同时派发到 `_on_spawn_session_start`（handler）和 `_handle_event`（→ converter）。由于 `create_child_session()` 在发射事件前已设置 `event.tool_call_id`（从 `ctx.tool_call_id` 填充），converter 和 handler 读到的是同一个值——**双重派发一致性问题在源头解决**。

### Zed #58537 兼容

Zed 2026-06-11 修复（PR #58537）保留了 waiting tool call 状态。这意味着：

- `ToolCallProgress` 更新时不应将 status 重置为 `"pending"`
- 进度更新应使用 `status="in_progress"`
- 仅在完成时使用 `status="completed"`
- 失败时使用 `status="failed"`

### 子会话 Consumer Loop 退出保障

`_after_consumer_loop` 只在 EventBus 流结束时触发。需确保子会话的 `StreamCompleteEvent` 后会话被正确关闭。

**验证项**：
- `SessionPool` 在 `StreamCompleteEvent` 后是否关闭子会话
- 如果不关闭，需在 `_handle_event` 中检测 `StreamCompleteEvent` 并主动关闭子会话
- 添加超时机制：子会话超过 5 分钟无事件则自动关闭

## 实施计划

### Phase 1：P0 修复（1-2 天）

**目标**：G1-G4，使 Zed 正确显示 subagent 完成状态

**范围**：

- [ ] **G10**：`create_child_session()` 自动构造并发射 `SpawnSessionStart`（`context.py`）
- [ ] **G10**：简化 `subagent_tools.py`（移除 15 行手动模板，改为 1 行调用）
- [ ] **G10**：简化 `workers.py`（2 处）
- [ ] **G10**：team.py/teamrun.py 不受影响（使用 yield 模式，不经过 create_child_session）
- [ ] **G1**：`create_child_session()` 从 `ctx.tool_call_id` 自动填充到 `SpawnSessionStart`
- [ ] **G2**：修复 `kind="other"` → `kind="subagent"` — `event_converter.py:656`
- [ ] **G3**：`_on_spawn_session_start` 中 Event + 闭包完成通知（`handler.py`）
- [ ] **G3**：在 `event_converter.py` 新增 `build_subagent_completed()` 方法
- [ ] **G4**：在 `ToolCallProgress` 上携带 `_meta.subagent_session_info` + `tool_name`
- [ ] 编写测试：`create_child_session` 自动发射 `SpawnSessionStart` 验证
- [ ] 编写测试：`tool_call_id` 从 ctx → event → converter 一致传递
- [ ] 编写测试：Event + 闭包完成通知测试（mock done_event）
- [ ] 编写测试：并发子会话测试（两个子会话同时 spawn）
- [ ] 编写测试：错误路径测试（子会话崩溃时发射 `status="failed"`）
- [ ] 编写测试：`kind="subagent"` 验证
- [ ] 编写测试：`ToolCallProgress` 上 `_meta` 完整性验证

**预估代码量**：~100 行新增/修改
**预估工期**：1-2 天
**依赖**：无
**回滚策略**：还原 `event_converter.py` 和 `handler.py` 的修改

### Phase 2：P1 功能对齐（1-2 天）

**目标**：G5-G7

**范围**：

- [ ] **G5**：追踪 `message_start_index` — spawn 时查询子会话 entry count
- [ ] **G5**：追踪 `message_end_index` — `_after_consumer_loop` 时查询子会话 entry count
- [ ] **G5**：在 `build_subagent_completed()` 中传入 `message_end_index`
- [ ] **G6**：强制 `MAX_SUBAGENT_DEPTH=5` — 在 `create_child_session()` 中检查 `child_depth`
- [ ] **G7**：实现递归取消传播 — `_cancel_subagents()` walk `_parent_of` tree
- [ ] 编写测试：消息索引正确性测试
- [ ] 编写测试：深度限制 enforcement 测试
- [ ] 编写测试：递归取消传播测试

**预估代码量**：~80 行新增
**预估工期**：1-2 天
**依赖**：Phase 1
**回滚策略**：移除索引追踪、深度检查和取消逻辑

### Phase 3：P2 前瞻性（<1h）

**目标**：G8-G9

**范围**：

- [ ] **G8**：填充 `SubagentRunInfo(child_session_id=..., subagent_id=..., run_mode="foreground", display_name=...)`
- [ ] **G9**：返回结构化子代理结果（child_session_id, status, duration）— 参考 Zed 的 `SpawnAgentToolOutput` 和 OpenCode 的 XML 包装
- [ ] 编写测试：`SubagentRunInfo` 字段验证
- [ ] 编写测试：结构化结果验证

**预估代码量**：~30 行新增
**预估工期**：<1h
**依赖**：Phase 1
**回滚策略**：移除 `SubagentRunInfo` 构造和结构化结果

### 里程碑总览

| Phase | 目标 | 工期 | 累计 |
|-------|------|------|------|
| Phase 1 | P0 修复（tool_call_id + kind + completed + _meta） | 1-2 天 | 1-2 天 |
| Phase 2 | P1 功能对齐（消息索引 + 深度限制 + 递归取消） | 1-2 天 | 2-4 天 |
| Phase 3 | P2 前瞻性（SubagentRunInfo + 结构化结果） | <1h | 2-4 天 |

### 依赖关系

```
Phase 1 (P0) ──→ Phase 2 (P1)
         │
         └──→ Phase 3 (P2)
```

Phase 2 和 Phase 3 可并行执行，均依赖 Phase 1。

## 开放问题

1. **子会话 consumer loop 退出时机**：`_after_consumer_loop` 只在 EventBus 流结束时触发。需验证 `SessionPool` 是否在子会话 `StreamCompleteEvent` 后关闭会话。如果不关闭，需要在 `_handle_event` 中检测 `StreamCompleteEvent` 并主动触发关闭。

2. **`_get_session_entry_count` 实现**：`build_subagent_completed()` 需要 `message_end_index`，需要从 `SessionPool` 或 `SessionController` 查询子会话的 entry count。需确认此 API 是否存在或需要新增。

3. **ACP Subagents RFD (PR #855) 影响**：PR #855 于 2026-06-25 恢复开发。合并后 `_meta` 扩展可能被原生协议替代。本 RFC 的实施应考虑在 feature flag 后实现，便于未来迁移。

4. **错误路径 `ToolCallProgress(status="failed")`**：子会话崩溃时应发射 `status="failed"` 而非 `"completed"`。当前 `RunFailedEvent` 由子会话的 converter 处理并 reset 状态，但不会通知父会话。需在 `_after_consumer_loop` 中区分正常退出和异常退出。

5. **`_meta` 中 `"tool_name": "task"` 的保留**：`_build_subagent_field_meta()` 在 `event_converter.py:211` 返回的 `_meta` 中除了 `subagent_session_info` 还有 `"tool_name": "task"`。`ToolCallProgress` 上的 `_meta` 也必须包含此字段。

6. **前台→后台切换（promotion）**：OpenCode 支持 `raceFirst(wait, waitForPromotion)` 实现前台→后台无重启切换。AgentPool 暂不需要（NG6），但长期值得考虑。

7. **结果投递模式**：OpenCode 通过 `ops.prompt(parentSession, synthetic: true)` 将子代理结果作为合成消息注入父会话。AgentPool 当前 tool 只返回纯文本。是否应在 `_after_consumer_loop` 中通过 EventBus 向父会话注入 `SubagentCompletedEvent`，而非仅发 ACP `ToolCallProgress`？

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-06-26 | 创建 RFC-0039 而非更新 RFC-0027 | RFC-0027 Phase 1 已实现，直接修改会混淆历史 |
| 2026-06-26 | 推荐选项 2（`_after_consumer_loop`） | Oracle 评估确认已接线、无竞态条件、有 handler 上下文 |
| 2026-06-26 | 新增 GAP 1（tool_call_id 断联）为 P0 | Oracle 发现 converter 忽略 `SpawnSessionStart.tool_call_id`，是完成通知的前提 |
| 2026-06-26 | `run_mode` 使用 `"foreground"` 而非 `"async"` | Oracle 确认 schema 仅允许 `"foreground"` \| `"background"` |
| 2026-06-26 | 不实现多轮 reprompting (NG1) | 高风险，依赖 session_manager 对子会话的恢复能力 |
| 2026-06-26 | GAP 8（SubagentRunInfo）降级为 P2 | Oracle 确认 Zed 从 `_meta` 读取，不从 `subagent` 字段读 |
| 2026-06-26 | 不新增 SpawnSessionComplete 事件 | Oracle 确认 `_after_consumer_loop` 已满足需求 |
| 2026-06-26 | handler 设置 `event.tool_call_id` 供 converter 读取 | 解决 `SpawnSessionStart` 双重派发的一致性问题 |
| 2026-06-26 | 新增 GAP 7（递归取消传播）为 P1 | 跨框架调研发现 Zed 和 OpenCode 均有递归取消，AgentPool 缺失 |
| 2026-06-26 | 新增 GAP 9（结构化结果）为 P2 | 跨框架调研发现 Zed 和 Hermes 提供结构化结果，AgentPool 缺失 |
| 2026-06-26 | `kind="subagent"` 而非 OpenCode 的 `"think"` | AgentPool 是 ACP server 服务 Zed，应触发 Zed 的 subagent UI |
| 2026-06-26 | 递归取消参考 OpenCode 的 walk tree 模式 | OpenCode 的 `cancelBackgroundJobs()` 通过 `metadata.parentSessionId` 递归取消 |
| 2026-06-27 | 完成通知从 `_subagent_map` dict 改为 Event + 闭包方案 | 闭包天然捕获上下文，无需 dict 存储和清理；复用已有 `_consumer_done_events` 和 `_consumer_task_refs` |
| 2026-06-27 | `create_child_session()` 自动发射 `SpawnSessionStart` | 消除 3 处 × 15 行手动模板代码；借鉴 Zed 的 `ThreadEnvironment::create_subagent()` 框架级抽象 |
| 2026-06-27 | `tool_call_id` 在 `create_child_session()` 中从 `ctx.tool_call_id` 自动填充 | 从源头解决断联问题，无需双重派发处理 |
| 2026-06-27 | `MAX_SUBAGENT_DEPTH` 检查移至 `create_child_session()` | 框架层统一拦截，比在 handler 中检查更内聚 |
| 2026-06-27 | 递归取消使用轻量 `_parent_of` 映射而非 `_subagent_map` | 完成通知不再需要 dict，仅取消传播需要 child→parent 关系 |
| 2026-06-27 | `_after_consumer_loop` 无需修改 | 所有完成通知逻辑在 `_on_spawn_session_start` 中完成，职责内聚 |
| 2026-06-27 | Fix #1: done_event 为 None 时立即发射通知 | Oracle 发现 mixin finally 块先 pop 再 set，consumer 快速退出时 handler .get() 返回 None。提取 `_notify_completed()` helper 避免重复 |
| 2026-06-27 | Fix #2: emit 路径从 `self.node._events` 改为 `self.events` | Oracle 确认 `self.events` 创建含 EventBus 的 StreamEventEmitter，`self.node._events` 可能 bypass EventBus |
| 2026-06-27 | Fix #3: 不使用 getattr，直接访问类型化字段 | AGENTS.md 禁止 getattr。`AgentContext.tool_call_id` 和 `AgentRunContext.depth` 是类型化字段 |
| 2026-06-27 | Fix #4: 仅 3 处调用方受影响，非 4 处 | Oracle 确认 team.py/teamrun.py 使用 yield 模式，不调用 `create_child_session()`，不受 auto-emit 影响 |
| 2026-06-27 | Fix #5: 闭包添加 try/except 错误处理 | `self.client.session_update()` 可能抛异常（连接关闭），异常不应被静默吞掉 |
| 2026-06-27 | Fix #6: task 完成后从 `_consumer_task_refs` 移除 | 防止长期运行 ACP 服务器内存泄漏 |
| 2026-06-27 | Fix #7: `_parent_of` 在闭包正常退出时 pop | 仅 `_cancel_subagents` 中 pop 会导致正常退出时遗留条目 |

## 参考

### 调研文档

- Zed ACP Subagent 功能调研报告 — 完整的 Zed subagent 实现分析、AgentPool 差距对比和适配方案（2026-06-26 更新）

### Oracle 评估

- Oracle session: `ses_0fbc20147ffey44MYpg57mPH3T`（2026-06-26）
- 评估范围：`event_converter.py`, `handler.py`, `session.py`, `session_manager.py`, `acp_agent.py`, `acp/schema/tool_call.py`, `acp/schema/base.py`

### 跨框架调研

- **Zed** — `~/src/zed/crates/agent/src/tools/spawn_agent_tool.rs`, `~/src/zed/crates/acp_thread/src/acp_thread.rs`, `~/src/zed/crates/agent/src/thread.rs`
- **OpenCode** — `~/src/opencode/packages/opencode/src/tool/task.ts`, `~/src/opencode/packages/core/src/background-job.ts`, `~/src/opencode/packages/opencode/src/cli/cmd/run/stream.transport.ts`
- **Pydantic-AI** — `packages/pydantic-ai/`
- **Hermes-Agent** — hermes-agent 仓库
- **Claw-Code** — claw-code 仓库
- **Pi** — pi 仓库

### AgentPool 源码

- `packages/wolfharness/src/wolfharness_server/acp_server/event_converter.py` — 核心 ACP 事件转换器（723 行，活跃）
- `packages/wolfharness/src/wolfharness_server/acp_server/handler.py` — 协议处理器（521 行）
- `packages/wolfharness/src/wolfharness_server/mixins.py` — `ProtocolEventConsumerMixin`（258 行，`_consumer_done_events` 在 line 60，`_consumer_task_refs` 在 line 61，`_after_consumer_loop` 在 line 91-100）
- `packages/wolfharness/src/wolfharness/agents/context.py` — `AgentRunContext.create_child_session()`（line 231-276）
- `packages/wolfharness/src/wolfharness_server/acp_server/session.py` — ACP session 管理（929 行）
- `packages/wolfharness/src/acp/schema/tool_call.py` — `SubagentRunInfo` 定义
- `packages/wolfharness/src/acp/schema/base.py` — `AnnotatedObject` 基类，`field_meta` 字段
- `packages/wolfharness_toolsets/builtin/subagent_tools.py` — subagent tool 实现（3 处手动 SpawnSessionStart 之一）
- `packages/wolfharness_toolsets/builtin/workers.py` — worker tool 实现（3 处之二）
- `packages/wolfharness/src/wolfharness/delegation/team.py` — team 实现（使用 yield 模式，不受 auto-emit 影响）
- `packages/wolfharness/src/wolfharness/delegation/teamrun.py` — sequential team 实现（同 team.py，使用 yield 模式）

### 相关 RFC

- [RFC-0027: ACP Subagent Zed 兼容性](RFC-0027-acp-subagent-zed-compatibility.md) — 前序 RFC，Phase 1 已实现，已被本 RFC 取代
- [RFC-0013: Subagent Event Stream Unification](../implemented/RFC-0013-subagent-event-unification.md)
- [RFC-0014: SpawnSessionStart Event](../implemented/RFC-0014-spawn-session-events.md)

### ACP 协议

- [ACP v1.0.0](https://github.com/agentclientprotocol/agent-client-protocol/releases/tag/v1.0.0) — 2026-06-24 首个稳定版
- [ACP Subagents RFD (PR #855)](https://github.com/agentclientprotocol/agent-client-protocol/pull/855) — 2026-06-25 恢复开发
- [ACP 扩展性文档](https://agentclientprotocol.com/protocol/extensibility.md)

### Zed 相关 PR

- [Zed #58537](https://github.com/zed-industries/zed/pull/58537) — ACP: preserve waiting tool call status on updates (2026-06-11)
- [Zed #58308](https://github.com/zed-industries/zed/pull/58308) — ACP SDK 升级至 v0.13.1 (2026-06-02)
- [Zed #50493](https://github.com/zed-industries/zed/pull/50493) — Subagent GA (2026-02-27)
