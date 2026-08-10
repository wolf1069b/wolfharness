---
rfc_id: RFC-0027
title: ACP Subagent Zed 兼容性
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-04-24
last_updated: 2026-04-24
---

# RFC-0027: ACP Subagent Zed 兼容性

## 概述

本 RFC 提出 AgentPool ACP Server 与 Zed 编辑器 subagent UI 的兼容性改造方案。当前 AgentPool 的 ACP Server 虽然已实现三种 subagent 展示模式（legacy/inline/tool_box），但所有模式均未填充 ACP 协议中的 `_meta` 扩展字段，导致 Zed 编辑器无法检测和渲染 subagent 面板 UI。Zed 的 subagent 功能已正式 GA（2026-02-27，PR #50493），其 subagent UI 完全依赖于 `ToolCallStart`/`ToolCallProgress` 事件中的 `_meta.subagent_session_info` 字段。

本 RFC 识别了 5 个关键差距（GAP），其中 GAP 1 为 P0 阻塞性问题，并提出 3 个方案选项，通过 4 阶段实施计划（Phase 0-3）逐步实现完整的 Zed subagent 兼容性。

预期结果：Zed 编辑器能够正确检测 AgentPool 的 subagent 工具调用，渲染展开/折叠卡片 UI，并独立管理子会话的生命周期。

## 背景与上下文

### 当前系统状态

AgentPool ACP Server 实现了完整的 ACP v1 协议，包括 `initialize`、`session/new`、`session/prompt`、`session/resume`、`session/fork`、`session/close`、`session/list` 等方法。对于 subagent 场景，Server 提供了三种展示模式：

| 模式 | 描述 | 当前行为 |
|------|------|----------|
| **legacy**（默认） | 扁平化文本 | 将 subagent 文本内容以 markdown headers 嵌入 `AgentMessageChunk` |
| **inline** | 独立 ToolCall | 每个 subagent 活动创建独立的 `ToolCallStart`/`ToolCallProgress`，但全部在同一 session |
| **tool_box** | 工具箱累积 | 单个 `ToolCallStart` per subagent 调用，内容累积显示 |
| **zed** 📝 | Zed 适配（*提议*） | ToolCallStart 含 `_meta`（subagent_session_info + tool_name），Phase 2 增加子 ACP session |

> ⚠️ **重要变更（Phase 1）**：当 `display_mode == "zed"` 时，`SpawnSessionStart` 处理产出带 `_meta`（subagent_session_info + tool_name）的 `ToolCallStart`。`inline`/`tool_box`/`legacy` 模式下 SpawnSessionStart 行为不变（AgentMessageChunk，无 _meta）。子级 ToolCallStart/ToolCallProgress 在所有模式下均不携带 `_meta`。
>
> **重要变更（Phase 2）**：当 `display_mode == "zed"` 时，SubAgentEvent 事件路由到子 ACP session。其他 display_mode 下行为不变。

### 相关工作

| RFC | 状态 | 与本 RFC 的关系 |
|-----|------|-----------------|
| RFC-0013 | ✅ 已实现 | 统一了 OpenCode Server 的 main+subagent 事件处理 EventProcessor |
| RFC-0014 | ✅ 已实现 | 添加了 `SpawnSessionStart` 事件，提供显式子会话创建信号 |
| RFC-0025 | 📝 草稿 | Shared Agent Architecture，单共享 Agent + per-session 状态 |
| RFC-0026 | ✅ 已实现 | Per-Session Agent Isolation，会话级 Agent 隔离 |

### 现有基础设施

AgentPool 核心层已具备子会话管理基础，Phase 2 应桥接而非重建：

| 组件 | 位置 | 现有能力 |
|------|------|----------|
| `SessionManager.create_child_session()` | `sessions/manager.py` | 创建子会话并关联 parent_id |
| `SessionManager.get_child_sessions()` | `sessions/manager.py` | 查询父会话的所有子会话 |
| `SessionData.parent_id` | `sessions/models.py` | 会话的父会话 ID 字段 |

Phase 2 的 `ACPSessionManager` 应委托核心 `SessionManager.create_child_session()` 管理子会话生命周期，而非在 ACP 层重新实现。

### 术语表

| 术语 | 定义 |
|------|------|
| ACP | Agent Client Protocol，编辑器与 AI Agent 之间的标准化通信协议 |
| `_meta` | ACP 协议中所有类型包含的扩展字段，类型为 `{ [key: string]: unknown }` |
| SubagentSessionInfo | Zed 定义的 _meta 扩展结构，包含 `session_id`、`message_start_index`、`message_end_index` |
| ToolCallStart | ACP 事件类型，通知客户端新的工具调用已启动（Zed 中称为 `tool_call`） |
| ToolCallProgress | ACP 事件类型，工具调用的状态或结果更新（Zed 中称为 `tool_call_update`） |
| AnnotatedObject | ACP schema 基类，具有 `field_meta` 字段（序列化为 `_meta`） |
| display_mode | AgentPool 的 subagent 展示模式配置，支持 legacy/inline/tool_box/zed |
| zed (display_mode) | Zed 编辑器专属 subagent 适配模式，通过 `_meta.subagent_session_info` 扩展实现子会话 UI |

### Zed Subagent 实现方式

Zed 的 subagent **不是** ACP 协议的标准特性，而是通过 `_meta` 扩展机制实现：

```rust
// Zed: crates/acp_thread/src/acp_thread.rs:69-70
pub const SUBAGENT_SESSION_INFO_META_KEY: &str = "subagent_session_info";
pub const TOOL_NAME_META_KEY: &str = "tool_name";

// Zed: crates/agent/src/tools/spawn_agent_tool.rs:155-168
// spawn 时：创建 SubagentSessionInfo 并写入 _meta
let session_info = SubagentSessionInfo {
    session_id: subagent.id(),
    message_start_index: subagent.num_entries(cx),  // 当前条目数（非硬编码 0）
    message_end_index: None,  // spawn 时未知，后续更新
};
event_stream.update_fields_with_meta(
    acp::ToolCallUpdateFields::new(),
    Some(acp::Meta::from_iter([(
        SUBAGENT_SESSION_INFO_META_KEY.into(),
        serde_json::json!(&session_info),  // ← JSON Object, NOT string
    )])),
);

// 完成时：更新 message_end_index
session_info.message_end_index =
    cx.update(|cx| Some(subagent.num_entries(cx).saturating_sub(1)));
// ⚠️ saturating_sub(1)：0-based index，非 count
```

Zed 客户端通过 `subagent_session_info_from_meta()` 函数从 `_meta` 中提取 `SubagentSessionInfo`，进而调用 `tool_call_for_subagent(session_id)` 定位父级 ToolCall 以渲染展开/折叠卡片 UI。

## 问题陈述

### GAP 1（P0 阻塞）：`_meta` 字段从未填充

**Zed 期望**：每个 `ToolCallStart` 和 `ToolCallProgress`（Zed 中分别称为 `tool_call` 和 `tool_call_update`）均可携带 `_meta` 字段。对于 subagent 工具调用，Zed 读取：
- `subagent_session_info`：`{session_id, message_start_index, message_end_index}` — 作为 JSON 对象序列化在 `_meta` 内
- `tool_name`：底层工具名称（用于显示归因，Zed 使用 snake_case 的 `TOOL_NAME_META_KEY = "tool_name"`）

**AgentPool 现状**：`AnnotatedObject`（`ToolCallStart`/`ToolCallProgress` 的基类）具有 `field_meta` 字段（序列化为 `_meta`），但 **从未在任何位置填充**。在 `event_converter.py` 的第 449、490、506、552、584、755、781、815、991 行，所有 `ToolCallStart()` 和 `ToolCallProgress()` 构造函数均省略了 `field_meta` 参数。

> ⚠️ **注意**：上述行号中，部分对应非 subagent 的 ToolCall（如 BuiltinToolCallPart）。Phase 1 修改时，仅 subagent 相关的 ToolCallStart/ToolCallProgress 需传入包含 `subagent_session_info` 的 `field_meta`；非 subagent ToolCall 可选传入仅含 `tool_name` 的 `field_meta`（用于 Zed 工具归因显示），**不得**传入 `subagent_session_info`。

**影响**：Zed 完全无法检测 subagent 工具调用。没有 `_meta.subagent_session_info`，Zed 的 `subagent_session_info_from_meta()` 返回 `None`，`tool_call_for_subagent()` 永远找不到父级 ToolCall。整个 subagent 面板 UI 无法运作。

**代码证据**：

```python
# event_converter.py:449 — BuiltinToolCallPart ToolCallStart
yield ToolCallStart(
    tool_call_id=tool_call_id,
    title=state.title,
    kind=state.kind,
    raw_input=state.raw_input,
    status="pending",
    # ❌ field_meta 未传入
)

# event_converter.py:755 — inline 模式 subagent text ToolCallStart
yield ToolCallStart(
    tool_call_id=state.text_output_call_id,
    title=f"[`{source_name}`] Output",
    kind="other",
    status="pending",
    content=[ContentToolCallContent.text(text=full_content)]
    if full_content
    else None,
    # ❌ field_meta 未传入
)
```

### GAP 2（P1）：Subagent 事件扁平化到父流

**Zed 期望**：Subagent 在**独立的 ACP session**（通过 `session/new` 创建）中运行，父级 ToolCall 的 `_meta.subagent_session_info.session_id` 链接到该子会话。Zed 的 `AcpThread::tool_call_for_subagent(session_id)` 定位父级 ToolCall 以渲染展开/折叠卡片 UI。

**AgentPool 现状**：三种展示模式均**未创建独立的 ACP session**：
- **legacy**（服务器默认）：将 subagent 文本扁平化到 `AgentMessageChunk` 中，使用 markdown headers 分隔
- **inline**：为每个 subagent 活动创建独立的 `ToolCallStart`/`ToolCallProgress`，但全部在**同一 session**
- **tool_box**：在同一 session 中为每个 subagent 调用创建一个 `ToolCallStart`，累积内容

三种模式均不：(1) 调用 `session/new` 创建子 ACP session，(2) 通过 ACP 发射 `SubagentSpawned` 事件，或 (3) 使 subagent 内容作为独立 session 可访问。

### GAP 3（P1）：SpawnSessionStart 事件未用于 ACP subagent 信令

**AgentPool 现状**：`SpawnSessionStart` 事件（定义于 `events.py:646`）由 `subagent_tools.py`（第 109 行）发射。ACP converter 在 `event_converter.py:686-693` 处理该事件，仅发射一个带有 emoji 前缀的简单 `AgentMessageChunk.text()`。

`SpawnSessionStart` 事件携带 `child_session_id`、`parent_session_id` 和 `tool_call_id` — 恰好是构建 `SubagentSessionInfo` 所需的信息。但 ACP converter 未：(1) 创建子 ACP session，(2) 在父级 ToolCallStart 的 `_meta` 中发射 `subagent_session_info`，或 (3) 将 subagent 事件路由到子会话。

### GAP 4（P2）：无 `message_start_index` / `message_end_index` 追踪

**Zed 期望**：`SubagentSessionInfo` 包含 `message_start_index` 和 `message_end_index`（usize 类型），定义子会话线程中的条目范围，用于确定哪些条目属于特定的 subagent turn。

**AgentPool 现状**：无 message index 追踪概念。

### GAP 5（P2）：`_meta` 中未发射 `tool_name` 用于工具归因

Zed 源码 `acp_thread.rs:64-67`：

```rust
pub fn meta_with_tool_name(tool_name: &str) -> acp::Meta {
    acp::Meta::from_iter([(TOOL_NAME_META_KEY.into(), tool_name.into())])
}
```

Zed 从 `_meta` 读取 `tool_name` key（`TOOL_NAME_META_KEY = "tool_name"`，snake_case）以显示产生特定 ToolCall 的工具名称。AgentPool 从未设置此字段。

### 影响分析

若不解决上述差距，以下功能将持续不可用：

| 功能 | 影响 | 严重程度 |
|------|------|----------|
| Zed subagent 面板 UI | 完全不可用 | P0 |
| Subagent 展开/折叠卡片 | 完全不可用 | P0 |
| 子会话独立管理 | 完全不可用 | P1 |
| Subagent 工具归因显示 | 信息缺失 | P2 |
| Message 范围追踪 | 信息缺失 | P2 |

## 目标与非目标

### 目标

| ID | 目标 | 优先级 |
|----|------|--------|
| G1 | `ToolCallStart`/`ToolCallProgress` 必须在 subagent 场景下填充 `_meta.subagent_session_info` | P0 |
| G2 | `ToolCallStart`/`ToolCallProgress` 必须在 subagent 场景下填充 `_meta.tool_name` | P0 |
| G3 | Subagent 运行应在独立的 ACP session 中执行，事件路由到子会话 | P1 |
| G4 | `SubagentSessionInfo` 应包含 `message_start_index`/`message_end_index` | P2 |
| G5 | 提供 `zed` 显示模式，用户显式配置激活 Zed 适配 | P2 |
| G6 | 保持与现有 legacy/inline/tool_box 模式的向后兼容 | P0 |

### 非目标

| ID | 非目标 | 理由 |
|----|--------|------|
| NG1 | 修改 ACP 协议 schema | `_meta` 是协议扩展机制，不需要修改 schema |
| NG2 | 支持 Zed Parallel Agents | Parallel Agents 是用户级并行架构，非 subagent 嵌套，不在本 RFC 范围 |
| NG3 | 实现 ACP Proxy Chains | Proxy Chains RFD 仍在草案阶段，属于长期方向 |
| NG4 | 修改 `SpawnSessionStart` 或 `SubAgentEvent` 数据结构 | 保持事件模型稳定 |
| NG5 | 支持 MAX_SUBAGENT_DEPTH > 1 | Zed 限制为 1 层，无需支持更深层嵌套 |

## 评估标准

| 标准 | 权重 | 描述 | 最低阈值 |
|------|------|------|----------|
| Zed 兼容性 | 关键 | Zed 能正确渲染 subagent 卡片 UI | SubagentSessionInfo 被 Zed 正确解析 |
| 向后兼容性 | 关键 | 现有客户端（非 Zed）行为不变 | 现有三种模式功能正常 |
| 非Zed零影响 | 关键 | `zed` 模式下的 _meta 填充不影响其他客户端行为 | legacy/inline/tool_box 模式行为完全不变 |
| 代码侵入性 | 高 | 对现有 event_converter.py 的修改范围 | 不超过现有代码量的 30% |
| 实施复杂度 | 中 | 开发工时和代码行数估算 | Phase 1 不超过 150 行新增 |
| 可维护性 | 中 | 新增代码的清晰度和可测试性 | 单元测试覆盖率 ≥ 80% |
| 协议合规性 | 中 | 符合 ACP v1 扩展机制规范 | 仅使用 `_meta` 扩展，不引入非标准字段 |

## 方案分析

### 选项 1：最小化修复 — 仅填充 `_meta`（Phase 1）

**描述**：仅实施 Phase 1，在现有 `ToolCallStart`/`ToolCallProgress` 构造时填充 `field_meta`，包含 `SubagentSessionInfo` 和 `tool_name`。不创建独立子会话，不路由事件，不新增显示模式。

**实施范围**：
- 在 `event_converter.py` 中创建 `_build_subagent_meta()` 辅助方法
- 修改所有 subagent 相关的 `ToolCallStart`/`ToolCallProgress` 构造调用，传入 `field_meta`
- `SpawnSessionStart` 事件处理改为发射带 `_meta` 的 `ToolCallStart`，而非纯文本

**优势**：
- 修改范围最小，仅涉及 `event_converter.py` 一个文件
- 风险最低，不改变事件路由逻辑
- Zed 可立即检测到 subagent 存在（虽然内容仍在同一 session）
- 开发工时最短

**劣势**：
- Zed 检测到 `subagent_session_info.session_id` 后会尝试加载该 session，但 session 不存在
- 不满足 Zed 的完整子会话模型，展开/折叠 UI 可能显示为空或错误
- `message_start_index`/`message_end_index` 无法提供有效值
- 仅是临时方案，后续仍需实施 Phase 2

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| Zed 兼容性 | 2/5 | Zed 可检测 subagent，但子会话内容为空 |
| 向后兼容性 | 5/5 | 不影响现有模式 |
| 代码侵入性 | 5/5 | 仅新增 ~130 行 |
| 实施复杂度 | 5/5 | 1-2 天 |
| 可维护性 | 3/5 | 临时方案，需后续迭代 |
| 协议合规性 | 5/5 | 仅使用 `_meta` 扩展 |

**工作量估算**：低（~130 行新增，1-2 天）

**风险评估**：
- 技术风险：低。Zed 可能对不存在的 session_id 显示错误，但不影响父会话
- 兼容性风险：低。`_meta` 是可选字段，非 Zed 客户端会忽略

---

### 选项 2：完整 ACP 子会话支持 — `_meta` + 子会话创建 + 事件路由 + `display_mode=zed`（Phase 0+1+2+3）

**描述**：在 Phase 1 基础上，增加子会话创建（Phase 2）和 message index 追踪（Phase 3）。当 `SpawnSessionStart` 事件到达时，创建独立的子 ACP session；SubAgentEvent 中的内容事件路由到子会话；追踪 message 范围索引。通过 `display_mode=zed` 配置激活 Zed 适配，无需自动检测客户端类型。

**实施范围**：
- Phase 1：当 `display_mode == "zed"` 时填充 `_meta`（SubagentSessionInfo + tool_name）；其他 display_mode 行为不变
- Phase 2：子会话创建与事件路由（仅 `display_mode == "zed"`）
  - 修改 `SpawnSessionStart` 处理：调用 `session_manager` 创建子 ACP session
  - 修改 `SubAgentEvent` 处理：将 inner_event 路由到子会话的 session update 流
  - 子会话生命周期管理：spawn 时创建，`StreamCompleteEvent` 时关闭
- Phase 3：`message_start_index` / `message_end_index` 追踪
  - 在子会话中维护消息计数器
  - 在 `ToolCallProgress` 的 `_meta` 中更新 `message_end_index`

**优势**：
- 完全符合 Zed 的 subagent 数据模型
- 子会话内容独立可访问，Zed 可正确渲染展开/折叠 UI
- `message_start_index`/`message_end_index` 使 Zed 可精确定位条目范围
- 单一 `display_mode=zed` 配置，无需额外的路由层或自动检测
- 非 Zed 客户端零影响：legacy/inline/tool_box 模式行为完全不变

**劣势**：
- 修改范围较大，涉及 `event_converter.py`、`session_manager.py`、`session.py`
- 子会话生命周期管理增加复杂度（创建、路由、关闭、异常处理）
- 子会话的 session update 需要通过父会话的 JSON-RPC 连接发射，需确认 ACP 传输层支持
- 需要处理嵌套 subagent（虽然 Zed 限制 depth=1，但内部可能有多层）
- 增加内存占用（每个子会话需要独立的 session 状态）

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| Zed 兼容性 | 5/5 | 完整支持 Zed subagent UI |
| 向后兼容性 | 5/5 | display_mode=zed 为显式配置，非 Zed 客户端零影响 |
| 代码侵入性 | 3/5 | 涉及 3 个文件，~380 行新增 |
| 实施复杂度 | 2/5 | 5-7 天，需处理 session 生命周期 |
| 可维护性 | 4/5 | 架构清晰，但子会话管理增加维护成本 |
| 协议合规性 | 5/5 | 完全使用 ACP 标准机制 |

**工作量估算**：中高（~380 行新增，5-7 天）

**风险评估**：
- 技术风险：中。子会话的 session update 发射路径需验证；ACP 传输层是否支持从非活跃 session 发射 update 需确认
- 兼容性风险：低。子会话路由为新增逻辑，现有模式不受影响
- 性能风险：中。每个子会话增加内存和事件处理开销

---

### 选项 3：新增 `zed_subagent` 显示模式 — 全新 display mode 替代现有三种（Phase 0+1+2+3+4）

**描述**：在选项 2 基础上，新增 `zed_subagent` 显示模式（Phase 4），专为 Zed 客户端优化。同时实现客户端自动检测：当 ACP `initialize` 握手时检测客户端标识，自动选择 `zed_subagent` 模式。

> ⚠️ **注意**：选项 2 现已通过 `display_mode=zed` 实现了 Zed 专用显示模式。选项 3 提议的 `zed_subagent` 枚举值与选项 2 的 `display_mode=zed` 功能等价，且选项 3 的客户端自动检测已被明确排除（subagent 适配是 Zed 专属 ad-hoc 扩展，应作为显式配置项）。因此选项 3 相对于选项 2 无增量价值，已被选项 2 的 `display_mode=zed` 方案取代。

**实施范围**：
- Phase 1-3：同选项 2
- Phase 4：`zed_subagent` 显示模式 + 客户端自动检测
  - 新增 `zed_subagent` display_mode 枚举值
  - 实现 `_convert_subagent_zed()` 方法，完全按照 Zed 期望的事件序列输出
  - 在 `initialize` 响应中读取客户端信息，自动设置 display_mode
  - 允许 YAML 配置覆盖自动检测

**优势**：
- 专门的 Zed 优化路径，不受现有模式约束
- 自动检测减少用户配置负担
- 完整的 Zed subagent 体验：独立子会话 + 展开/折叠 + 工具归因 + message 范围
- 现有模式完全不受影响，零风险
- 未来可扩展：其他 IDE（如 JetBrains）可能有不同的 subagent 期望，可新增专用模式

**劣势**：
- 新增第 4 种 display_mode，增加维护面积
- 客户端自动检测可能不准确（依赖客户端标识字段，部分客户端可能不提供）
- 最长开发周期
- 自动检测逻辑可能在 Zed 版本更新后失效

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| Zed 兼容性 | 5/5 | 完全优化，最佳 Zed 体验 |
| 向后兼容性 | 5/5 | 新增模式，不影响现有模式 |
| 代码侵入性 | 2/5 | 新增 ~530 行，新增 1 个 display_mode |
| 实施复杂度 | 1/5 | 7-10 天，需实现完整新模式 + 自动检测 |
| 可维护性 | 3/5 | 4 种 display_mode 增加维护面积 |
| 协议合规性 | 5/5 | 完全使用 ACP 标准机制 |

**工作量估算**：高（~530 行新增，7-10 天）

**风险评估**：
- 技术风险：中高。自动检测依赖客户端信息，可能需要 fallback 机制
- 兼容性风险：低。纯新增模式
- 维护风险：中。4 种 display_mode 的维护成本

## 推荐

**推荐选项 2：完整 ACP 子会话支持**，分阶段实施。

推荐理由：

1. 选项 1 仅是临时方案，Zed 检测到 session_id 但找不到对应 session，用户体验不佳
2. 选项 2 通过 `display_mode=zed` 提供了 Zed 专用适配，无需自动检测客户端类型
3. 选项 3 已被选项 2 的 `display_mode=zed` 方案取代，无增量价值
4. 选项 2 的非 Zed 客户端零影响设计：legacy/inline/tool_box 模式行为完全不变
5. 选项 2 的 Phase 1 可立即解决 P0 阻塞问题，Phase 2/3 渐进式交付

**接受的权衡**：
- Phase 1 阶段 Zed 的 subagent 展开/折叠 UI 内容为空，需等待 Phase 2 完成
- `display_mode=zed` 需要用户显式配置，不支持自动检测客户端类型

## 技术设计

### 架构图

#### 当前事件流

```
┌─────────────────────────────────────────────────────────────────────┐
│                    当前 ACP Subagent 事件流                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  subagent_tools.py                                                   │
│     │                                                                │
│     ├─ SpawnSessionStart ──→ event_converter ──→ AgentMessageChunk  │
│     │                                │                    (emoji文本) │
│     │                                │                               │
│     └─ SubAgentEvent ──────→ event_converter                        │
│                                  │                                   │
│                                  ├── legacy ──→ AgentMessageChunk    │
│                                  │               (markdown headers)  │
│                                  │                                   │
│                                  ├── inline ──→ ToolCallStart       │
│                                  │              ToolCallProgress     │
│                                  │              (同一 session,       │
│                                  │               _meta = None ❌)    │
│                                  │                                   │
│                                  └── tool_box ─→ ToolCallStart      │
│                                                  ToolCallProgress   │
│                                                  (同一 session,     │
│                                                   _meta = None ❌)  │
│                                                                      │
│  Zed 客户端:  _meta 为空 → subagent_session_info_from_meta() = None │
│               → 无法检测 subagent → UI 不可用                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 提议事件流

```
┌─────────────────────────────────────────────────────────────────────┐
│                    提议 ACP Subagent 事件流                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  initialize 握手                                                     │
│     │                                                                │
│     └─ YAML pool_server.subagent_display_mode 配置                  │
│        ├─ "zed" → Zed 适配模式（含 _meta + 子会话）                  │
│        └─ 其他  → 传统模式（legacy/inline/tool_box）                │
│                                                                      │
│  subagent_tools.py                                                   │
│     │                                                                │
│     ├─ SpawnSessionStart ──→ event_converter                        │
│     │                           │                                    │
│     │     ┌────────────────────┼────────────────────┐               │
│     │     │ display_mode=="zed"│  display_mode!="zed"│               │
│     │     │                    │                      │               │
│     │     │ ├ ToolCallStart    │ └ AgentMessageChunk │               │
│     │     │ │  (含 session_info│   (保持原有行为，    │               │
│     │     │ │   +tool_name ✅) │    无 _meta ❌)      │               │
│     │     │ │                  │                        │               │
│     │     │ └ Phase 2:        │                        │               │
│     │     │   创建子 ACP      │                        │               │
│     │     │   session         │                        │               │
│     │     └────────────────────┴────────────────────┘               │
│     │                                                                │
│     └─ SubAgentEvent ──────→ event_converter                        │
│                                  │                                   │
│     ┌────────────────────────────┼──────────────────────┐           │
│     │ display_mode=="zed"        │  display_mode!="zed"  │           │
│     │                            │                        │           │
│     │ ├ Phase 2: 路由到子会话  │ ├ legacy →             │           │
│     │ │  session/update          │ │  AgentMessageChunk   │           │
│     │ │                          │ │  (无 _meta)          │           │
│     │ │ └ 更新父 ToolCallProgress│ │                      │           │
│     │ │   (含 session_info       │ ├ inline →             │           │
│     │ │    +tool_name ✅)        │ │  ToolCallStart       │           │
│     │ │   [Phase 2]             │ │  (无 _meta ❌)       │           │
│     │ │                          │ │                      │           │
│     │ │ Phase 1: SCEvent→TCP(含_meta)│ └ tool_box →           │           │
│     │ │ 其他P1丢弃(P2→子会话)      │   ToolCallStart        │           │
│     │ │                            │   (无 _meta ❌)        │           │
│     └────────────────────────────┴──────────────────────┘           │
│                                                                      │
│  ✅ 关键: display_mode=="zed" 与传统模式条件分支                      │
│          display_mode=="zed" 时 SpawnSessionStart 使用 ToolCallStart（含 _meta）；传统模式保持 AgentMessageChunk 不变 │
│          inline/tool_box 子级 ToolCallStart/ToolCallProgress 不携带 _meta │
│                                                                      │
│  ⚠️ 向后兼容说明：display_mode!="zed" 时 SpawnSessionStart 保持原有 AgentMessageChunk 行为，非 Zed 客户端零影响。│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### SubagentSessionInfo 数据模型

```python
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

# Module-level constants matching Zed's acp_thread.rs
TOOL_NAME_META_KEY: str = "tool_name"
SUBAGENT_SESSION_INFO_META_KEY: str = "subagent_session_info"


class SubagentSessionInfo(BaseModel):
    """ACP _meta 中 subagent_session_info 的 Python 表示。

    与 Zed 的 AcpThread 期望格式对齐。

    ⚠️ WARNING: 序列化为 JSON **对象**（非 JSON 字符串）存入 _meta["subagent_session_info"]。
    Zed 使用 serde_json::json!(&session_info) 写入 Value::Object，
    使用 serde_json::from_value(v.clone()) 读取。
    若误传为字符串，Zed 的 from_value 会静默失败（.ok() → None），
    导致整个 _meta 机制失效。
    """

    session_id: str
    """子 ACP session 的唯一标识符。"""

    message_start_index: int
    """子会话中该 turn 的起始条目索引（0-based）。

    对应 Zed SubagentSessionInfo.message_start_index (usize, required)。
    """

    message_end_index: int | None = None
    """子会话中该 turn 的结束条目索引（0-based）。

    对应 Zed SubagentSessionInfo.message_end_index (Option<usize>)。
    """

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> SubagentSessionInfo | None:
        """从 _meta 字典中提取 SubagentSessionInfo。

        Args:
            meta: _meta 字典，值应为 JSON 对象（dict）。

        Returns:
            SubagentSessionInfo 实例，若 meta 中无 subagent_session_info 则返回 None。

        ⚠️ 正确格式为 dict（JSON 对象）；若误传为 str（JSON 字符串），
        此方法尝试作为 fallback 解析，但 Zed 端 from_value 会静默失败。
        """
        raw = meta.get(SUBAGENT_SESSION_INFO_META_KEY)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        if isinstance(raw, str):
            # Fallback: 尝试解析 JSON 字符串（兼容错误格式）
            try:
                data = json.loads(raw)
                return cls.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                return None
        return None


def build_subagent_meta(
    session_info: SubagentSessionInfo,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """构建包含 SubagentSessionInfo 的 _meta 字典。

    此函数为底层实现，由 `_build_subagent_field_meta()` 调用。

    `_build_subagent_field_meta()` 为推荐调用入口，包含空值安全检查和 WARNING 日志。
    `build_subagent_meta()` 为底层实现，直接调用不安全。

    Args:
        session_info: 子会话信息
        tool_name: 工具名称（用于 Zed 的 tool_name 归因）

    Returns:
        可传入 ToolCallStart/ToolCallProgress 的 field_meta 参数的字典。
        返回类型为 dict[str, Any] 以匹配 AnnotatedObject.field_meta 的类型定义。

    ⚠️ WARNING: subagent_session_info 的值必须为 dict（JSON 对象），
    绝不能为 str（JSON 字符串）。这是 Zed 的 serde_json::from_value 要求。

    常量 TOOL_NAME_META_KEY 和 SUBAGENT_SESSION_INFO_META_KEY 为模块级定义，
    匹配 Zed 的 acp_thread.rs 中的常量。
    """
    meta: dict[str, Any] = {
        SUBAGENT_SESSION_INFO_META_KEY: session_info.model_dump(),
    }
    if tool_name is not None:
        meta[TOOL_NAME_META_KEY] = tool_name
    return meta
```

### 路由模式设计

Subagent 事件路由通过 `display_mode` 配置控制。新增 `zed` 模式专为 Zed 编辑器适配，为 Zed 专属的 `_meta.subagent_session_info` 扩展机制提供支持。

`zed` 模式是 **ad-hoc 适配**，非 ACP 协议标准特性。其他 ACP 客户端无需支持此模式，使用 legacy/inline/tool_box 即可。

#### 四种 display_mode 对比

| display_mode | _meta 填充 | 子会话创建 | 适用场景 |
|---|---|---|---|
| legacy | ❌ | ❌ | 默认模式，扁平化文本 |
| inline | ❌ | ❌ | 每个 subagent 独立 ToolCall |
| tool_box | ❌ | ❌ | 累积式工具箱 |
| **zed** | ✅ | ✅ (Phase 2) | Zed 编辑器 subagent UI 适配 |

#### YAML 配置

```yaml
pool_server:
  subagent_display_mode: zed  # legacy | inline | tool_box | zed
```

单一配置项，无需额外的路由配置。使用现有 `pool_server.subagent_display_mode` 配置键，仅新增 `"zed"` 枚举值。无需额外配置键或迁移。

#### ⚠️ 配置注意事项

- `"zed"` 模式专为 Zed 编辑器设计，非 Zed 客户端可能无法正确渲染 `_meta.subagent_session_info` 扩展字段
- 服务器启动时若检测到 `display_mode == "zed"`，应输出 WARNING 日志：`"Subagent display_mode='zed' is designed for Zed editor clients only. Other clients may not render subagent UI correctly."`
- `"zed"` **永远不会作为默认值** — 必须用户显式配置。若未配置 `subagent_display_mode` 或配置为 `legacy`/`inline`/`tool_box`，subagent 行为与当前完全一致。

```python
@dataclass
class ACPEventConverter:
    _display_mode: Literal["legacy", "inline", "tool_box", "zed"] = "legacy"
    _session_manager: SessionManager | None = None  # Phase 2 新增
    _subagent_tool_map: dict[str, str] = field(default_factory=dict)
    # 注：Phase 3 将新增 _subagent_message_counts: dict[str, int] = field(default_factory=dict)
```

### ⚠️ _meta 填充约束

**关键规则**：`_meta.subagent_session_info` 仅在 `display_mode == "zed"` 时，由 `SpawnSessionStart` 产出的 `ToolCallStart` 及其后续 `ToolCallProgress` 中设置。`_convert_subagent_inline` / `_convert_subagent_tool_box` 产出的子级 `ToolCallStart` / `ToolCallProgress` **不得**携带任何 `_meta` 字段。

此约束仅在 `display_mode == "zed"` 时有意义。其他 display_mode 下不填充任何 _meta。

原因：Zed 的 `is_subagent()` 通过 `subagent_session_info.is_some()` 检测 subagent。如果子级 ToolCall 也携带此字段，Zed 会将所有子级 ToolCall 视为独立的 subagent 父级，导致 UI 渲染混乱。

inline/tool_box 子级 ToolCallStart/ToolCallProgress **不得**携带任何 `_meta` 字段。仅 `display_mode == "zed"` 时 SpawnSessionStart 产出的 ToolCallStart/ToolCallProgress 携带 `_meta`（含 `subagent_session_info` + `tool_name`）。

> **`tool_name` 选择说明**：SpawnSessionStart 产出的 ToolCallStart 使用 `tool_name="task"`，这是 AgentPool 中创建子代理的工具名（定义于 `subagent_tools.py`），与现有 subagent_tools.py 保持一致。Zed 使用 `_meta.tool_name` 在 UI 中显示工具归因。

### 事件流规范

#### SpawnSessionStart → ToolCallStart（带 _meta）

当 `display_mode == "zed"` 时，`SpawnSessionStart` 事件到达时，event_converter 应：

1. 发射 `ToolCallStart`，`field_meta` 包含 `subagent_session_info` 和 `tool_name`
2. 记录 `tool_call_id` 与 `child_session_id` 的映射关系
3. 创建子 ACP session（Phase 2）

当 `display_mode != "zed"` 时，SpawnSessionStart 保持原有 `AgentMessageChunk` 行为不变。

```python
# event_converter.py — SpawnSessionStart 处理（提议）
case SpawnSessionStart(
    child_session_id=child_id,
    parent_session_id=parent_id,
    tool_call_id=tc_id,
    source_name=source_name,
    spawn_mechanism=mechanism,
    description=description,
):
    if self._display_mode == "zed":
        # zed 模式：发射带 _meta 的 ToolCallStart
        meta = self._build_subagent_field_meta(child_session_id=child_id, tool_name="task")

        # Phase 2: 创建子 ACP session
        # await self._session_manager.new_session(
        #     session_id=child_id,
        #     parent_session_id=parent_id,
        # )

        # 记录映射
        self._subagent_tool_map[child_id] = tc_id or f"spawn:{child_id}"

        icon = "⚡" if mechanism == "spawn" else "🚀"
        yield ToolCallStart(
            tool_call_id=tc_id or f"spawn:{child_id}",
            title=f"{icon} `{source_name}`: {description}",
            kind="other",
            status="in_progress",
            field_meta=meta,
        )
    else:
        # 非 zed 模式：保持原有 AgentMessageChunk 行为
        icon = "⚡" if mechanism == "spawn" else "🚀"
        yield AgentMessageChunk.text(f"{icon} `{source_name}`: {description}")
```

#### SubAgentEvent 处理（Phase 1）

实际实现不使用独立的 `_convert_subagent_event()` 方法，而是拆分在 `convert()` 方法的两个位置：

1. **SpawnSessionStart** → 见上方 `convert()` 方法的 SpawnSessionStart case（lines 637-671）
2. **StreamCompleteEvent** → 见下方 `convert()` 方法的 SubAgentEvent case 中 StreamCompleteEvent 分支（lines 688-697）

inline/tool_box 子级 ToolCallStart/ToolCallProgress 修改在现有 `_convert_subagent_inline` / `_convert_subagent_tool_box` 方法中进行（子级 ToolCall 不携带 _meta，无需修改）。

#### StreamCompleteEvent → ToolCallProgress（完成父级 ToolCall）

⚠️ `StreamCompleteEvent` 本身不含 `child_session_id` 字段。子代理完成事件通过 `SubAgentEvent` 包装传递，`child_session_id` 在 `SubAgentEvent` 外层。

当 `display_mode == "zed"` 时，子代理完成时 `SubAgentEvent(event=StreamCompleteEvent(...))` 到达。在 `convert()` 方法的 `SubAgentEvent` case 分支内，通过匹配 `inner_event` 类型处理，使用 `SubAgentEvent.child_session_id` 从 `_subagent_tool_map` 查找对应的 `tool_call_id`，发射 `ToolCallProgress` 完成父级 ToolCall。当 `display_mode != "zed"` 时，不发射 ToolCallProgress（无父级 ToolCallStart 需要完成）。

```python
# event_converter.py — StreamCompleteEvent 处理（Phase 1 提议）
# 在 convert() 方法的 SubAgentEvent case 分支内：
case SubAgentEvent(child_session_id=child_id, tool_call_id=_, event=inner_event)
    if self._display_mode == "zed" and child_id in self._subagent_tool_map
       and isinstance(inner_event, StreamCompleteEvent):
    # Phase 1: 仅处理 StreamCompleteEvent 以完成父级 ToolCall
    # 非 StreamCompleteEvent 的 SubAgentEvent（文本输出、工具调用等）在 Phase 1 中丢弃
    # Phase 2 将这些事件路由到子 ACP session
    tc_id = self._subagent_tool_map[child_id]
    meta = self._build_subagent_field_meta(child_session_id=child_id, tool_name="task")
    yield ToolCallProgress(tool_call_id=tc_id, status="completed", field_meta=meta)
```

**各 display_mode 行为**：

- **display_mode == "zed"**：发射 `ToolCallProgress(status="completed", field_meta=meta)` 完成父级 ToolCall
- **display_mode != "zed"**：不发射 ToolCallProgress（无父级 ToolCallStart 需要完成）

#### SubAgentEvent → 子会话路由（Phase 2）

```python
# event_converter.py — SubAgentEvent 处理（提议，Phase 2）
case SubAgentEvent(
    source_name=source_name,
    source_type=source_type,
    event=inner_event,
    depth=depth,
    child_session_id=child_id,
    parent_session_id=parent_id,
    tool_call_id=tc_id,
):
    if self._display_mode == "zed":
        # zed 模式：事件路由到子会话
        # async for update in self._convert_event_to_session_update(inner_event):
        #     await self._session_manager.send_session_update(child_id, update)
        # ⚠️ Zed 使用 num_entries.saturating_sub(1) 计算 end_index（0-based index，非 count）
        # message_end_index=message_count - 1 if message_count > 0 else None
        # ...
        pass  # Phase 2 实现
    else:
        # 传统模式：使用 legacy/inline/tool_box 转换
        match self._display_mode:
            case "inline":
                async for update in self._convert_subagent_inline(
                    source_name, source_type, inner_event, depth,
                    child_session_id=child_id, tool_call_id=tc_id,
                ):
                    yield update
            case "tool_box":
                async for update in self._convert_subagent_tool_box(
                    source_name, source_type, inner_event, depth,
                    child_session_id=child_id, tool_call_id=tc_id,
                ):
                    yield update
            case "legacy":
                async for update in self._convert_subagent_legacy(
                    source_name, source_type, inner_event, depth,
                    child_session_id=child_id, tool_call_id=tc_id,
                ):
                    yield update
```

### API 变更

#### event_converter.py 方法签名变更

```python
import logging
from typing import Any, Literal

from .subagent_meta import SubagentSessionInfo, build_subagent_meta

logger = logging.getLogger(__name__)

@dataclass
class ACPEventConverter:
    """ACP event converter with subagent _meta support."""

    # 以下字段为 dataclass 字段声明（ACPEventConverter 为 @dataclass 类）
    _display_mode: Literal["legacy", "inline", "tool_box", "zed"] = "legacy"
    _session_manager: SessionManager | None = None  # Phase 2 新增

    # Phase 1 新增
    _subagent_tool_map: dict[str, str] = field(default_factory=dict)  # child_session_id → tool_call_id
    # 注：Phase 3 将新增 _subagent_message_counts: dict[str, int] = field(default_factory=dict)

    def _build_subagent_field_meta(
        self,
        child_session_id: str,
        tool_name: str | None = None,
        message_start_index: int = 0,
        message_end_index: int | None = None,
    ) -> dict[str, Any] | None:
        """为 subagent 工具调用构建 _meta 字段。

        Args:
            child_session_id: 子会话 ID
            tool_name: 底层工具名称
            message_start_index: 起始消息索引（默认 0，假设子会话总是新建的）
            message_end_index: 结束消息索引

        Returns:
            field_meta 字典，若 child_session_id 为空则返回 None。
        """
        if not child_session_id:
            logger.warning("build_subagent_field_meta called with empty child_session_id, skipping _meta fill")
            return None
        session_info = SubagentSessionInfo(
            session_id=child_session_id,
            message_start_index=message_start_index,
            message_end_index=message_end_index,
        )
        return build_subagent_meta(session_info, tool_name=tool_name)
```

> **两层默认值说明**：`event_converter.py` 的字段默认值为 `"legacy"`（通过 `_get_display_mode()` 环境变量回退），而 `server.py`/`session_manager.py` 在构造时传入 `"tool_box"` 作为参数默认值。实际运行时默认为 `"tool_box"`，但 converter 层面的字段默认值必须保持 `"legacy"` 以匹配源码行为。

#### `reset()` 行为变更

`reset()` 方法在 `StreamCompleteEvent` 时被调用，用于清理单次 prompt 响应的状态。但 `_subagent_tool_map` 和 `_subagent_message_counts` 是**跨 prompt 生命周期**的状态（一个 subagent 可能跨多个 prompt 运行），不应被 `reset()` 清除。

```python
def reset(self) -> None:
    """重置单次 prompt 响应的状态，但保留跨 prompt 的 subagent 追踪状态。"""
    # 清理单次响应状态（使用实际 dataclass 字段）
    self._current_tool_inputs.clear()
    self._tool_states.clear()
    # ... 其他单次响应状态清理

    # ⚠️ 不清理 subagent 追踪状态
    # self._subagent_tool_map 保留 — 在父 session close 时清理
    # self._subagent_message_counts 保留 — 在父 session close 时清理  [Phase 3 新增]
```

> ⚠️ 当前 event_converter.py 存在 reset() body 重复声明（字段赋值与 dataclass field(default_factory=...) 重复）和 reset() 被调用两次的 bug。Phase 1 应一并清理。

`_subagent_tool_map`（及 Phase 3 的 `_subagent_message_counts`）应在父会话 `session/close` 时清理，而非 `reset()` 时。

此外，当前 `event_converter.py` 存在 `self.reset()` 在 `StreamCompleteEvent` handler 中被调用两次的 bug（lines 670, 673），应在 Phase 1 中一并修复。

#### session/close 清理

`_subagent_tool_map` 的生命周期与父会话绑定，应在父会话 `session/close` 时显式清理：

```python
# 在 session/close 处理中清理 _subagent_tool_map
# 具体实现见「EventConverter 清理钩子」小节的 cleanup() 方法
```

**设计理由**：`_subagent_tool_map` 跨多个 prompt 生命周期持久存在，`reset()` 不应清除它。清理路径必须显式绑定到 `session/close`，确保会话结束时不会遗留过期的映射条目。

#### EventConverter 清理钩子

`ACPEventConverter` 添加 `cleanup()` 方法，在 `AcpSession.close()` 中调用：

```python
# event_converter.py
def cleanup(self) -> None:
    """清理会话级别的持久状态。在 session/close 时由 AcpSession 调用。"""
    for child_id in list(self._subagent_tool_map.keys()):
        # Phase 2: await self._session_manager.close_session(child_id)
        del self._subagent_tool_map[child_id]
    # Phase 3: for child_id in list(self._subagent_message_counts.keys()):
    #     del self._subagent_message_counts[child_id]
```

> ⚠️ **Phase 2 变更**：`cleanup()` 将变更为 `async def cleanup()`，`AcpSession.close()` 中需改为 `await self._converter.cleanup()`。

```python
# session.py — AcpSession.close() 中添加
def close(self) -> None:
    # ... 现有清理逻辑 ...
    if self._converter is not None:
        self._converter.cleanup()
```

设计理由：`ACPEventConverter` 没有 session 生命周期感知，`cleanup()` 提供显式的会话结束信号，避免 `_subagent_tool_map` 在长期运行会话中造成内存泄漏。

#### session_manager.py 变更（Phase 2）

```python
class SessionManager:
    """ACP session lifecycle management."""

    async def new_session(
        self,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        cwd: str | None = None,
    ) -> AcpSession:
        """创建新的 ACP session。

        Args:
            session_id: 可选的 session ID（用于子会话预分配）
            parent_session_id: 父会话 ID（用于 subagent 关联）
            cwd: 工作目录

        Returns:
            新创建的 AcpSession 实例。
        """
        ...

    async def send_session_update(
        self,
        session_id: str,
        update: ACPSessionUpdate,
    ) -> None:
        """向指定 session 发射 session update 事件。

        用于将子会话的事件通过 ACP 传输层发送给客户端。

        Args:
            session_id: 目标 session ID
            update: ACP session update 对象
        """
        ...

    async def get_message_count(
        self,
        session_id: str,
    ) -> int:
        """获取指定 session 的当前消息数量。

        用于计算 message_end_index。

        Args:
            session_id: 目标 session ID

        Returns:
            当前消息条目数量。
        """
        ...
```

### 子会话生命周期状态机

```
┌──────────┐  SpawnSessionStart  ┌──────────┐
│ 不存在    │ ──────────────────→ │ 已创建    │
│          │                     │ (active) │
└──────────┘                     └────┬─────┘
                                      │
                        SubAgentEvent │ (事件路由到子会话)
                                      │
                                      ▼
                                 ┌──────────┐
                   ┌─────────────│ 运行中    │
                   │             │ (running)│
                   │             └────┬─────┘
                   │                  │
      StreamCompleteEvent            │  父会话取消
      (成功/错误)                    │  (session/cancel)
                   │                  │
                   ▼                  ▼
             ┌──────────┐      ┌──────────┐
             │ 已完成    │      │ 已取消    │
             │ (closed) │      │ (closed) │
             └──────────┘      └──────────┘

异常路径：
  - 子会话运行时父会话 session/close → 遍历所有子会话，执行 close
  - ACP 服务器崩溃 → 重启后通过 SessionManager.get_child_sessions() 检测孤立会话
  - 子会话创建后 SubAgentEvent 未到达 → 子会话超时关闭（建议 5 分钟）
```

状态转换规则：

1. `SpawnSessionStart` → 创建子会话 (active)
2. `SubAgentEvent` → 事件路由到子会话 (running)
3. `StreamCompleteEvent(success)` → 子会话关闭 (closed)，父 ToolCallProgress → completed
4. `StreamCompleteEvent(error)` → 子会话关闭 (closed)，父 ToolCallProgress → failed
5. 父 `session/cancel` → 所有子会话 cancelled + closed
6. 父 `session/close` → 所有子会话 closed
7. 超时 → 子会话 closed，父 ToolCallProgress → failed

### 安全考量

1. **Session ID 注入**：`SubagentSessionInfo.session_id` 由服务端生成，不接受客户端输入。使用 `identifier.ascending("session")` 生成，确保不可预测。

2. **子会话隔离**：子会话继承父会话的权限范围。不允许子会话访问父会话之外的资源。工作目录（`cwd`）与父会话一致。

3. **_meta 数据泄露**：`_meta` 字段通过 ACP JSON-RPC 传输，可能被客户端日志记录。不应在 `_meta` 中包含敏感信息（如 API key、用户凭证）。`SubagentSessionInfo` 仅包含 session ID 和索引，无安全风险。

4. **拒绝服务**：恶意客户端可能通过频繁触发 subagent 创建大量子会话。需实施子会话数量上限（建议与 Zed 的 `MAX_SUBAGENT_DEPTH=1` 对齐，即每个父会话最多 N 个并发子会话）。

5. **子会话生命周期**：子会话必须在父会话关闭时一并关闭，避免孤立会话占用资源。在 `session/close` 处理中遍历并关闭所有子会话。

## 实施计划

### Phase 0：子会话 session/update 传输验证 — Spike（0.5-1 天）

**目标**：验证 Zed 能否接收通过父会话 client 连接发送的子会话 `session/update`

**范围**：
- [ ] 创建 PoC：在 ACP `process_prompt` 中手动创建子 session，发送 `session/update`
- [ ] 验证 Zed 能否收到并处理带有子 session_id 的 update notification
- [ ] 验证 Zed 是否需要通过 `session/new` 预先知晓子会话
- [ ] 如需 `session/new`，验证服务端代理调用的可行性
- [ ] 输出：技术可行性报告，确定 Phase 2 的传输路径设计

**Go/No-Go 准则**：
- **Go**：Zed 能接收通过父会话连接发送的、`session_id` 不同于父会话的 `session/update` notification（无需先 `session/new`）
- **Go（替代路径）**：Zed 需要 `session/new` 知晓子会话，但 ACP 传输层支持服务端主动发送 `session/new` 响应给客户端
- **No-Go**：Zed 既不接受未知 session_id 的 `session/update`，ACP 传输层也不支持服务端主动调用 `session/new` → Phase 2 必须使用替代设计（将子会话内容嵌入父 ToolCallProgress 的 content 字段）

**预估工期**：0.5-1 天
**依赖**：无
**阻塞**：Phase 2 的设计依赖此 spike 的结果

### Phase 1：填充 `_meta`（SubagentSessionInfo + tool_name）— P0

**目标**：G1 + G2，使 Zed 可检测 subagent 工具调用

**范围**：
- [ ] 创建 `SubagentSessionInfo` 数据模型（~40 行）
- [ ] 创建 `build_subagent_meta()` 和 `_build_subagent_field_meta()` 辅助方法（~35 行）
- [ ] 增加 `ACPEventConverter` 字段声明 `_subagent_tool_map`（~5 行）
- [ ] 修改 `SpawnSessionStart` 处理：当 `display_mode == "zed"` 时发射带 `_meta` 的 `ToolCallStart`；当 `display_mode != "zed"` 时保持原有 `AgentMessageChunk` 行为不变
- [ ] 修改 StreamCompleteEvent 处理：当 `display_mode == "zed"` 时对 SpawnSessionStart 产出的 ToolCallStart 发射 ToolCallProgress (status=completed)
- [ ] 非 StreamCompleteEvent 的 SubAgentEvent 在 `display_mode == "zed"` 时丢弃（Phase 2 路由到子会话）
- [ ] 修改 `_convert_subagent_inline` 中所有 `ToolCallStart`/`ToolCallProgress` 调用，不变（子级 ToolCall 不携带 _meta）
- [ ] 修改 `_convert_subagent_tool_box` 中所有 `ToolCallStart`/`ToolCallProgress` 调用，不变（子级 ToolCall 不携带 _meta）
- [ ] 编写测试：验证 `display_mode != "zed"` 时 ToolCallStart.field_meta 为 None（无 subagent_session_info 泄露）
- [ ] 编写测试：验证 inline/tool_box 子级 ToolCallStart.field_meta 为 None
- [ ] 编写测试：验证仅 `display_mode == "zed"` 时 SpawnSessionStart 产出的 ToolCallStart.field_meta 包含 subagent_session_info
- [ ] 编写测试：验证 `display_mode == "zed"` 时 StreamCompleteEvent 产出的 `ToolCallProgress.field_meta` 包含 `subagent_session_info`
- [ ] 编写测试：验证 `display_mode == "zed"` 时非 StreamCompleteEvent 的 SubAgentEvent（如文本输出事件）不产生任何 ACP 事件（Phase 1 丢弃行为）
- [ ] 为 `SubagentSessionInfo` 和 `build_subagent_meta` 编写单元测试
- [ ] 添加 ACP 快照测试，验证 subagent 工具调用的完整 JSON-RPC 输出格式
- [ ] 快照测试对比 emitted JSON-RPC messages 与参考文件，确保 `_meta.subagent_session_info` 为 JSON 对象（非字符串）
- [ ] 修复 `reset()` 在 StreamCompleteEvent 中被调用两次的 bug
- [ ] 清理 `reset()` body 中与 `dataclass field(default_factory=...)` 重复的字段赋值声明
- [ ] 确认 `reset()` 不清除 `_subagent_tool_map`
- [ ] 修复 subagent_tools.py 中 SpawnSessionStart 双重发射 bug（task() 和 _stream_task() 均 emit，同步模式下产生两个 ToolCallStart）
- [ ] 在 session/close 处理中清理 _subagent_tool_map
- [ ] 在 `ACPEventConverter` 中添加 `cleanup()` 方法
- [ ] 在 `AcpSession.close()` 中调用 `converter.cleanup()`

#### 类型定义传播清单

添加 `"zed"` 到 `display_mode` Literal 类型需更新以下文件：

| 文件 | 当前类型 | 变更 |
|------|----------|------|
| `server.py` | `SubagentDisplayMode = Literal["inline", "tool_box"]` | 添加 `"zed"` |
| `event_converter.py` (2处) | `Literal["legacy", "inline", "tool_box"]` | 添加 `"zed"` |
| `pool_server.py` | `Literal["inline", "tool_box"]` | 添加 `"zed"` |
| `session.py` | `Literal["inline", "tool_box"]` | 添加 `"zed"` |
| `session_manager.py` (2处) | `Literal["inline", "tool_box"]` | 添加 `"zed"` |
| `acp_agent.py` | `Literal["inline", "tool_box"]` | 添加 `"zed"` |
| `serve_acp.py` | CLI argument choices | 添加 `"zed"` |

- [ ] 更新上述所有文件中的 display_mode Literal 类型，添加 `"zed"` 值
- [ ] 更新 `server.py` 中 `_coerce_subagent_display_mode()` 处理 `"zed"` 值
- [ ] 更新 `event_converter.py` 中 `_get_display_mode()` 验证逻辑，接受 `"zed"`

> **关于 "legacy" 类型差异**：`event_converter.py` 的 `_display_mode` 类型包含 "legacy"（通过环境变量 `_get_display_mode()` 默认值），但 `server.py`/`pool_server.py` 等服务器层类型不包含 "legacy"。"legacy" 是 event_converter 内部回退值，非 YAML 配置可选项。本次仅新增 "zed"，不改变 "legacy" 的现有分布。

⚠️ 嵌套 SpawnSessionStart 说明：当前 `_convert_subagent_legacy` 对嵌套 SpawnSessionStart 执行 `pass`（丢弃），`_convert_subagent_inline` / `_convert_subagent_tool_box` 未处理该事件（走默认分支）。Phase 1 不修改此行为（Zed 限制 MAX_SUBAGENT_DEPTH=1，嵌套场景罕见）。Phase 2 应添加嵌套 SpawnSessionStart 的 ToolCallStart 处理。

**预估代码量**：~130 行新增
**预估工期**：1-2 天
**依赖**：无
**回滚策略**：还原 `event_converter.py` 中的 `field_meta` 参数即可

### Phase 2：子会话创建与事件路由 — P1

**目标**：G3，创建独立子 ACP session，路由事件

**范围**：
- [ ] 修改 `ACPEventConverter.__init__` 接收 `session_manager` 参数
- [ ] 当 `display_mode == "zed"` 时，`SpawnSessionStart` 处理中调用 `session_manager.new_session()` 创建子会话
- [ ] 当 `display_mode == "zed"` 时，`SubAgentEvent` 处理将 inner_event 路由到子会话
- [ ] ToolCallProgress 必须携带 `subagent_session_info`（与 ToolCallStart 保持一致）
- [ ] 处理 `StreamCompleteEvent`：更新父 `ToolCallProgress` 状态为 `completed`，关闭子会话
- [ ] 修改 `session_manager.py`：支持 `parent_session_id` 参数和 `send_session_update` 方法
- [ ] 处理子会话关闭时的资源清理
- [ ] 编写集成测试
- [ ] 添加 ACP 快照测试，验证子会话 session/update 的传输格式

**预估代码量**：~300 行新增
**预估工期**：3-5 天
**依赖**：Phase 1
**回滚策略**：移除 `session_manager` 参数和子会话路由逻辑，回退到 Phase 1 行为

### Phase 3：`message_start_index` / `message_end_index` 追踪 — P2

**目标**：G4，提供精确的消息范围索引

**范围**：
- [ ] 在 `ACPEventConverter` 中维护 `_subagent_message_counts` 映射
- [ ] 子会话创建时记录 `message_start_index`
- [ ] 每次 inner_event 路由后更新 `message_end_index`
- [ ] 在 `ToolCallProgress` 的 `field_meta` 中更新 `message_end_index`
- [ ] ⚠️ `message_end_index` 使用 0-based index（匹配 Zed 的 `saturating_sub(1)` 语义），即 `message_count - 1 if message_count > 0 else None`
- [ ] 编写单元测试验证索引正确性
- [ ] 确保 `message_start_index` 始终有值（子会话创建时设为 0）

**假设**：本 RFC 假设子会话总是新建的（非恢复），因此 `message_start_index` 始终为 0。若未来支持子会话恢复，需更新此值为 `session.get_entry_count()`。

**预估代码量**：~80 行新增
**预估工期**：1-2 天
**依赖**：Phase 2
**回滚策略**：移除 `_subagent_message_counts` 和索引更新逻辑

### 里程碑总览

| Phase | 目标 | 工期 | 累计 |
|-------|------|------|------|
| Phase 0 | Spike: 传输验证 | 0.5-1 天 | 0.5-1 天 |
| Phase 1 | P0 阻塞解决：`_meta` 填充 | 1-2 天 | 1.5-3 天 |
| Phase 2 | 子会话创建、路由、display_mode=zed 子会话路由 | 3-5 天 | 4.5-8 天 |
| Phase 3 | Message index 追踪 | 1-2 天 | 5.5-10 天 |

### 依赖关系

```
Phase 0 ──────→ Phase 2 ──→ Phase 3
Phase 1 ──────→ Phase 2
```

Phase 0 与 Phase 1 可并行执行。Phase 0 仅阻塞 Phase 2（需要传输验证结果），Phase 1 可立即开始。

## 开放问题

1. **子会话 session update 的发射路径**：ACP 传输层当前仅支持从活跃 session（即正在执行 `session/prompt` 的 session）发射 update。子会话的 update 是否需要通过父会话的 JSON-RPC 连接中继？这需要确认 ACP JSON-RPC 传输层的具体实现。Phase 0 Spike 将验证此路径。

2. **嵌套 subagent 的消息索引**：当 subagent 本身也调用 subagent 时（虽然 Zed 限制 `MAX_SUBAGENT_DEPTH=1`，但 AgentPool 内部可能有多层），`message_start_index`/`message_end_index` 的语义是否仍与 Zed 期望一致？

3. **非 Zed 客户端的 `_meta` 兼容性**：填充 `_meta` 后，其他 ACP 客户端（如 JetBrains IDE、VS Code ACP 适配器）如何处理未知的 `_meta` 字段？根据 ACP 协议规范，客户端应忽略未知的 `_meta` key，但需验证实际行为。

4. **Zed 版本兼容性**：`_meta.subagent_session_info` 格式是否为 Zed 稳定接口？若 Zed 在未来版本更改格式，AgentPool 需要如何适配？

5. **子会话的超时清理策略**：若 `SubAgentEvent` 长时间未到达（如 LLM 响应超时），子会话应在多少时间后自动关闭？

6. **子会话的 session/prompt**（已解决）：AgentPool 的子会话**不支持**独立的 `session/prompt` 调用。子会话仅作为事件接收容器，通过 `send_session_update()` 接收父会话路由的事件。这与 Zed 的 `SubagentHandle.send()` 内部 API 语义一致（Zed 的子会话不通过 ACP `session/prompt` 发送消息）。

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-04-24 | 推荐选项 2（完整 ACP 子会话支持） | 平衡了 Zed 兼容性、代码侵入性和可维护性 |
| 2026-04-24 | Phase 1 优先交付 | P0 阻塞问题需立即解决 |
| 2026-04-24 | 不推荐选项 1 作为最终方案 | 临时方案，Zed 子会话内容为空，用户体验不佳 |
| 2026-04-24 | subagent_session_info 为 JSON 对象（非字符串） | Oracle 审查确认 Zed 使用 serde_json::json! 写入 Value::Object |
| 2026-04-24 | SubagentRoutingMode 机制替换为 display_mode=zed | 用户决策：subagent 适配是 Zed 专属 ad-hoc 扩展，应作为显式配置项。Oracle R3/Metis R4/用户决策 |
| 2026-04-24 | display_mode=zed 时 SpawnSessionStart 使用 ToolCallStart（含 _meta）；其他 display_mode 保持 AgentMessageChunk 不变 | 非 Zed 客户端零影响设计 |
| 2026-04-24 | 移除 _routing_mode 传播链，display_mode=zed 通过现有 _display_mode 传播 | Metis F1 (Round 3)/Oracle R4/Metis R4/用户决策 |
| 2026-04-24 | Zed 客户端通过 display_mode=zed 显式配置激活适配，非 Zed 客户端保持 legacy 默认 | legacy（非 tool_box）为当前默认 display_mode；display_mode != "zed" 时 SpawnSessionStart 行为不变 |
| 2026-04-24 | 桥接核心 SessionManager.create_child_session() | Metis 审查发现已有基础设施，避免重建 |
| 2026-04-24 | 仅 SpawnSessionStart 的 ToolCallStart 携带 subagent_session_info | Metis S2/F1：子级 ToolCall 携带 subagent_session_info 会导致 Zed 将子级视为独立 subagent 父级，UI 渲染混乱 |
| 2026-04-24 | _meta key 使用 snake_case（tool_name 而非 toolName） | Oracle S1/Metis P1：Zed 的 TOOL_NAME_META_KEY = "tool_name" 为 snake_case |
| 2026-04-24 | YAML 配置简化为 pool_server.subagent_display_mode 单层结构（4 值） | Oracle M3/Metis S3：display_mode=zed 替代原 routing_mode + display_mode 双层结构 |
| 2026-04-24 | 子会话不支持独立 session/prompt 调用 | Metis M2：子会话仅作为事件接收容器，与 Zed SubagentHandle.send() 语义一致 |
| 2026-04-24 | reset() 不清除 subagent 追踪状态 | Metis F2：_subagent_tool_map 和 _subagent_message_counts 为跨 prompt 生命周期状态 |
| 2026-04-24 | Phase 0 与 Phase 1 可并行执行，Phase 0 仅阻塞 Phase 2 | Oracle M1：Phase 0 的 spike 验证结果仅 Phase 2 需要 |
| 2026-04-24 | 当前默认 display_mode 为 legacy（非 tool_box） | Oracle S1 (Round 3)：源码确认 legacy 为默认模式，tool_box 不是默认 |
| 2026-04-24 | routing_mode 移至 Phase 2 里程碑 → 已替换为 display_mode=zed | Oracle M2/Metis M2 (Round 3)：Phase 1 仅在 display_mode==zed 时填充 _meta；Phase 2 添加子会话路由 |
| 2026-04-24 | ACPEventConverter 使用 @dataclass 字段声明而非 __init__ body | Metis S2 (Round 3)：dataclass 类中状态字段必须使用 field(default_factory=...) 声明 |
| 2026-04-24 | inline/tool_box 子级 ToolCall 不携带 _meta 字段 | Round 7 审查 + 用户决策：仅 SpawnSessionStart 的 ToolCallStart 携带 _meta；子级 ToolCall 不携带 _meta 简化实现并避免 Zed UI 混乱 |
| 2026-04-24 | SubAgentEvent 处理器通过 _display_mode == "zed" guard 区分路由 | Metis S4 (Round 3)：zed 模式路由到子会话，其他 display_mode 使用传统转换 |
| 2026-04-24 | message_end_index 使用 0-based index（saturating_sub(1) 语义） | Oracle M3 (Round 3)：Zed 使用 num_entries.saturating_sub(1)，非 count |
| 2026-04-24 | message_start_index 硬编码为 0（非 current_entry_count） | Metis F2 (Round 3)：Phase 1/2 子会话新建，始终为 0；Phase 3 需改为 session.get_entry_count() |
| 2026-04-24 | _build_subagent_field_meta() 为主要辅助方法 | Metis M3 (Round 3)：更安全的 API（None fallback），build_subagent_meta() 为底层实现 |
| 2026-04-24 | GAP 1 行号区分 subagent vs 非 subagent ToolCall | Metis M4 (Round 3)：非 subagent ToolCall 不得传入 subagent_session_info |
| 2026-04-24 | Zed 源码片段更新为更准确的表示 | Oracle M4 (Round 3)：包含 TOOL_NAME_META_KEY 常量定义、spawn 时 message_end_index=None、完成时 saturating_sub(1) |
| 2026-04-24 | TOOL_NAME_META_KEY 提升为模块级常量 | Oracle S1/Metis F1 (Round 4)：build_tool_name_meta() 引用局部变量会导致 NameError（⚠️ build_tool_name_meta() 已在 Round 7 移除） |
| 2026-04-24 | _display_mode 字段默认值修正为 "legacy" | Oracle M1/Metis S1 (Round 4)：源码 _get_display_mode() 默认返回 "legacy"，code example 使用 "tool_box" 为行为回归 |
| 2026-04-24 | 概述注释拆分 Phase 1/Phase 2 | Metis M1 (Round 4)：避免 Phase 1 引用 Phase 2 的路由机制概念 |
| 2026-04-24 | Phase 1 scope 移除 _subagent_message_counts | Metis M2 (Round 4)：该字段为 Phase 3 新增，Phase 1 不应引用 |
| 2026-04-24 | 架构图 _meta 标记区分内容类型 | Metis M3 (Round 4)：session_info+tool_name 与无 _meta 的区别（Round 7 更新：inline/tool_box 子级 ToolCall 不携带 _meta） |
| 2026-04-24 | message_start_index 注释修正为未来考虑 | Oracle M2 (Round 4)：Phase 3 scope 和假设均保持 0，非 Phase 3 交付项 |
| 2026-04-24 | _display_mode 类型更新为 Literal["legacy", "inline", "tool_box", "zed"] | Oracle M3 (Round 4)：匹配更新后的类型注解，新增 zed 枚举值 |
| 2026-04-24 | SpawnSessionStart 行为条件化：display_mode==zed 使用 ToolCallStart（含 _meta），其他保持 AgentMessageChunk | R5 决策修订：非 Zed 客户端零影响，display_mode != "zed" 时 SpawnSessionStart 行为不变 |
| 2026-04-24 | tool_name 使用 "task"（AgentPool 子代理工具名） | R6 决策："task" 是 AgentPool 中创建子代理的工具名（定义于 subagent_tools.py），与现有命名保持一致 |
| 2026-04-24 | YAML 配置简化为 pool_server.subagent_display_mode 单层结构（4 值：legacy/inline/tool_box/zed） | R7 决策修订：display_mode=zed 替代原 routing_mode + display_mode 双层结构 |
| 2026-04-24 | Phase 2 ToolCallProgress 与 ToolCallStart 保持一致的 _meta 内容 | R8 决策：ToolCallProgress 必须携带 subagent_session_info，确保 Zed subagent 卡片在 progress 事件时仍能获取会话信息 |
| 2026-04-24 | SubagentRoutingMode 枚举已整体移除，display_mode=zed 替代其全部功能 | R3 决策修订：display_mode=zed 替代原 SubagentRoutingMode 枚举，用户显式配置，无需自动检测 |
| 2026-04-24 | Phase 1 不处理嵌套 SpawnSessionStart | R4 决策：Zed 限制 MAX_SUBAGENT_DEPTH=1，嵌套场景罕见；Phase 2 应补充嵌套 SpawnSessionStart 的 ToolCallStart 处理 |
| 2026-04-24 | Phase 1 修复 subagent_tools.py 双重 SpawnSessionStart 发射 | Oracle P1-3 (Round 7)：task() 和 _stream_task() 均会 emit SpawnSessionStart，同步模式下产生两个 ToolCallStart |
| 2026-04-24 | model_dump() 替代 model_dump(exclude_none=True) | Metis P2-4 (Round 7)：exclude_none 会省略 message_end_index=None，Zed Rust struct 可能缺少 #[serde(default)] |
| 2026-04-24 | Phase 1 在 session/close 中清理 _subagent_tool_map | Oracle P1-2 (Round 7)：_subagent_tool_map 生命周期与父会话绑定，需显式清理路径 |
| 2026-04-24 | 架构图 Phase 1 zed 分支补充 StreamCompleteEvent → ToolCallProgress(含 _meta) | R1 修订 (P1-1)：原描述"非 SpawnSessionStart 事件与传统模式相同"不准确，Phase 1 zed 模式下 StreamCompleteEvent 也产出带 _meta 的 ToolCallProgress |
| 2026-04-24 | 移除概念性 _convert_subagent_event() 代码块，改为引用 convert() 中的实际代码位置 | R2 修订 (P1-2)：概念性方法与其免责声明矛盾（免责声明称实现在 convert() 中），统一为实际代码位置引用消除歧义 |
| 2026-04-24 | StreamCompleteEvent 处理模式匹配中 tool_call_id=tc_id 改为 tool_call_id=_ | R3 修订 (P2-1)：tc_id 从模式绑定后立即被 _subagent_tool_map[child_id] 遮蔽，改为 _ 消除变量遮蔽 |
| 2026-04-24 | cleanup() 方法标注 Phase 2 将变更为 async def，AcpSession.close() 需 await | R4 修订 (P2-2)：Phase 2 需在 cleanup() 中 await 子会话关闭，同步→异步为 breaking change，需预先标注 |
| 2026-04-24 | API 变更部分添加 Any 类型导入和 subagent_meta 模块导入 | R5 修订 (P2-3)：_build_subagent_field_meta 返回 dict[str, Any] 需 Any 导入；SubagentSessionInfo/build_subagent_meta/常量需从 .subagent_meta 导入 |
| 2026-04-24 | Phase 1 guardrail 测试项拆分为 3 条可验证的具体测试 | R6 修订 (P2-4)：原 guardrail 描述为行为约束而非可验证测试项，拆分为 display_mode≠zed 泄露检查、子级 _meta 检查、zed 模式正向检查 |
| 2026-04-24 | Phase 1 zed 模式下非 StreamCompleteEvent 的 SubAgentEvent 显式丢弃 | R11-1：用户决策：Phase 1 仅处理 StreamCompleteEvent，其他事件丢弃；Phase 2 路由到子会话 |
| 2026-04-24 | isinstance(inner_event, StreamCompleteEvent) 添加到外层 guard | R11-2：Oracle P1-1/Metis M11-1：避免非 StreamCompleteEvent 被静默匹配后丢弃 |

## 参考

### 调研文档

- Zed ACP Subagent 功能调研报告 — 完整的 Zed subagent 实现分析和 AgentPool 差距对比

### Zed 源码

- `~/src/zed/crates/agent/src/tools/spawn_agent_tool.rs` — Zed 的 spawn_agent tool：通过 _meta 发射 SubagentSessionInfo
- `~/src/zed/crates/acp_thread/src/acp_thread.rs` — Zed 客户端：从 _meta 提取 SubagentSessionInfo 以渲染 subagent UI
- `~/src/zed/crates/agent_ui/src/acp/thread_view/active_thread.rs` — Zed subagent 卡片 UI 渲染

### ACP Schema 文件

- `packages/wolfharness/src/acp/schema/base.py` — `AnnotatedObject` 基类，`field_meta` 字段（序列化为 `_meta`）
- `packages/wolfharness/src/acp/schema/session_updates.py` — `ToolCallStart`/`ToolCallProgress` 定义

### AgentPool 源码

- `packages/wolfharness/src/wolfharness_server/acp_server/event_converter.py` — 核心 ACP 事件转换器
- `packages/wolfharness/src/wolfharness_server/acp_server/session.py` — ACP session 管理
- `packages/wolfharness/src/wolfharness_server/acp_server/session_manager.py` — Session 生命周期管理
- `packages/wolfharness/src/wolfharness_server/acp_server/server.py` — ACP server 入口
- `packages/wolfharness_toolsets/builtin/subagent_tools.py` — 发射 SubAgentEvent 和 SpawnSessionStart
- `packages/wolfharness/src/wolfharness/agents/events/events.py` — 事件定义：SubAgentEvent（line 614）、SpawnSessionStart（line 646）

### 相关 RFC

- [RFC-0013: Subagent Event Stream Unification for OpenCode Protocol](../implemented/RFC-0013-subagent-event-unification.md)
- [RFC-0014: SpawnSessionStart Event for Explicit Subsession Creation](../implemented/RFC-0014-spawn-session-events.md)
- [RFC-0025: Shared Agent Architecture](RFC-0025-shared-agent-architecture.md)
- [RFC-0026: Per-Session Agent Isolation](../implemented/RFC-0026-per-session-agent-isolation.md)

### ACP 协议

- [ACP 协议官网](https://agentclientprotocol.com/)
- [ACP 扩展性文档](https://agentclientprotocol.com/protocol/extensibility.md)
- [ACP Proxy Chains RFD](https://agentclientprotocol.com/rfds/proxy-chains.md)
