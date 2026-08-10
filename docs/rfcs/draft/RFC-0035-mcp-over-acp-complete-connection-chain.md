---
rfc_id: RFC-0035
title: "MCP-over-ACP: Complete Connection Chain with Tool Registration"
status: DRAFT
author: yuchen.liu
reviewers:
  - name: TBD
    status: pending
created: 2026-05-27
last_updated: 2026-05-27
decision_date:
related_prds: []
related_rfcs:
  - RFC-0033-mcp-over-acp-transport.md
---

# RFC-0035: MCP-over-ACP Complete Connection Chain with Tool Registration

## Overview

本 RFC 解决 RFC-0033 实现后的一个关键遗留问题：**ACP-transport MCP server 在 mcp/connect 成功后，如何完成完整的 MCP 客户端生命周期，包括 fastmcp Client 构建、MCP initialize 握手、tools/list 获取和工具注册到 Agent 运行时**。

RFC-0033 已经实现了：
- Agent 向 Client 主动发起 `mcp/connect`（Agent→Client）
- Client 返回 `connectionId`
- Agent 建立 `AcpMcpConnection` 和内存流桥接
- Agent 通过 `mcp/message` 和 `mcp/disconnect` 与 Client 通信

**但缺少最关键的一步**：从 `AcpMcpConnection` 的内存流到 Agent 能够实际调用 MCP 工具之间的桥梁。当前代码在 `connect_acp_mcp_server()` 返回 `connectionId` 后就停止了，没有任何代码创建 fastmcp `Client` 实例、发起 MCP `initialize` 握手、获取 `tools/list` 并将工具注册到 session 的 Agent 运行时。

本 RFC 提出三种设计方案，对比其优劣，并给出推荐实现。

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### MCP 客户端完整生命周期（Non-ACP 传输）

对于 stdio/SSE/HTTP 传输的 MCP server，wolfharness 已经有一套成熟的连接链：

```
session/new
  └─→ initialize_mcp_servers()
        └─→ convert_acp_mcp_server_to_config() → MCPServerConfig
              └─→ MCPResourceProvider.__aenter__()
                    └─→ MCPClient._connect()
                          └─→ MCPClient._get_client() → fastmcp.Client(transport)
                                └─→ fastmcp.Client.__aenter__()
                                      ├─→ transport.connect() → 建立底层连接
                                      ├─→ MCP initialize 握手
                                      ├─→ tools/list → 获取工具列表
                                      ├─→ prompts/list (可选)
                                      └─→ resources/list (可选)
                    └─→ MCPResourceProvider.list_tools() → 注册到 Agent
```

关键组件：
- **`MCPResourceProvider`**（`wolfharness/resource_providers/mcp_provider.py`）：MCP server 的 ResourceProvider 包装，管理工具/提示/资源的生命周期
- **`MCPClient`**（`wolfharness/mcp_server/client.py`）：fastmcp Client 的包装，处理连接、重连、消息路由
- **`fastmcp.Client`**：底层 MCP 客户端，负责 JSON-RPC 协议握手

### ACP-transport MCP 当前状态

```
session/new
  └─→ initialize_mcp_servers()
        └─→ 遇到 AcpMcpServer:
              └─→ connect_acp_mcp_server(server)
                    ├─→ Agent→Client: mcp/connect (acpId)
                    ├─→ Client→Agent: connectionId
                    └─→ AcpMcpConnectionManager.create_connection()
                          └─→ AcpMcpConnection.open()
                                └─→ 创建内存流 (to_session/from_session)
                    🔴 [在此停止 — 没有后续步骤]
```

已实现的组件（RFC-0033）：
- **`AcpMcpConnectionManager`**：管理 `connectionId → AcpMcpConnection` 映射
- **`AcpMcpConnection`**：持有内存流和 `send_to_client` 回调
- **`AcpMcpTransport`**（`wolfharness_server/acp_server/acp_mcp_transport.py`）：实现了 `fastmcp.ClientTransport` 接口，将内存流包装为 fastmcp 可用的传输层

未实现的组件：
- **没有代码创建 fastmcp Client**：`AcpMcpTransport` 已经就绪，但没有任何地方调用 `fastmcp.Client(AcpMcpTransport(...))`
- **没有 MCP initialize 握手**：即使创建了 Client，也需要进入 async context 才能触发 fastmcp 的自动初始化
- **没有工具注册**：`tools/list` 的结果需要转换为 wolfharness 的 `Tool` 对象并注册到 Agent 运行时

### 现有代码的关键限制

1. **`MCPClient._get_client()` 对 ACP 抛出 NotImplementedError**（`client.py:203-207`）
   ```python
   case AcpMCPServerConfig():
       raise NotImplementedError(
           "ACP-transport MCP servers are managed by the ACP agent directly. "
           "Use AcpMcpConnectionManager to establish connections."
       )
   ```
   这表明原始设计者有意让 ACP transport 走不同路径，但从未完成该路径。

2. **`MCPResourceProvider` 依赖 `MCPClient`**，而 `MCPClient` 依赖 `MCPServerConfig`。ACP transport 的 `AcpMcpConnection` 不是 `MCPServerConfig`，无法直接复用 `MCPResourceProvider`。

3. **`AcpMcpTransport` 需要 `AcpMcpConnection` 实例**，但 `AcpMcpConnection` 由 `AcpMcpConnectionManager` 创建，与 `MCPClient` 的构建时机不同步。

---

## Problem Statement

### The Problem

ACP-transport MCP server 在 `mcp/connect` 成功后，**工具对 Agent 不可见**。

具体表现：
1. Client 发送 `session/new` 并传入 `mcpServers: [{type: "acp", name: "workspace-fs", id: "xxx"}]`
2. Agent 正确响应 `mcp/connect` 并收到 `connectionId`
3. `AcpMcpConnection` 的内存流已建立
4. **Agent 运行时的可用工具列表中不包含该 MCP server 的工具**
5. 当 LLM 尝试调用工具时，Agent 不知道该 MCP server 存在

### Evidence

- `connect_acp_mcp_server()` 返回后，没有任何代码调用 `list_tools()` 或注册工具
- `session.py` 中 ACP server 的处理路径只调用 `connect_acp_mcp_server()`，不创建 `MCPResourceProvider`
- `MCPClient._get_client()` 对 `AcpMCPServerConfig` 抛出 `NotImplementedError`
- `AcpMcpTransport` 存在但从未被实例化（没有代码调用它）

### Impact of Inaction

- **功能不可用**：ACP-transport MCP server 虽然连接建立，但工具无法被调用
- **用户体验断裂**：客户端看到 `mcpCapabilities.acp: true`，传入 ACP server，但 Agent 完全不用它
- **RFC-0033 不完整**：虽然协议层 mcp/connect/mcp/message/mcp/disconnect 已实现，但端到端功能不工作

---

## Goals & Non-Goals

### Goals (In Scope)

1. 在 `mcp/connect` 成功后，创建 fastmcp `Client` 并通过 `AcpMcpTransport` 与 Client 通信
2. 完成 MCP `initialize` 握手和 `tools/list` 获取
3. 将获取的工具注册到 session 的 Agent 运行时（与 stdio/SSE/HTTP MCP server 的工具注册方式一致）
4. 支持工具列表变更通知（`tools/list_changed`）
5. 在 session 关闭时正确断开 MCP 连接并清理资源
6. 复用现有 `MCPResourceProvider` 的能力（enable/disable 工具、缓存等）

### Non-Goals (Out of Scope)

1. **修改 ACP 协议本身**：mcp/connect/mcp/message/mcp/disconnect 的消息格式不变
2. **MCP Server 的 prompts/resources 支持**：本 RFC 聚焦工具调用链，prompts/resources 可作为后续扩展
3. **多连接负载均衡**：每个 ACP MCP server 只建立一个 connection，不实现连接池
4. ** Bridging**：将 ACP transport 转为 stdio/HTTP shim 不在范围内

### Success Criteria

- [ ] ACP-transport MCP server 的工具在 Agent 运行时可用（`agent.list_tools()` 包含）
- [ ] LLM 可以通过 ACP channel 调用 MCP 工具并收到结果
- [ ] 工具 enable/disable 状态与 stdio/SSE/HTTP MCP 一致
- [ ] Session 关闭时断开连接，无内存泄漏
- [ ] 现有 stdio/SSE/HTTP MCP 功能无回归

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| 架构一致性 | 高 | 与现有 MCP 架构（MCPResourceProvider、MCPClient）的集成方式 | 不应引入全新的工具注册路径 |
| 复用性 | 高 | 能复用现有 MCPResourceProvider 的能力（缓存、enable/disable、变更通知） | 必须复用，不重新实现 |
| 实现复杂度 | 中 | 代码改动范围和引入的技术风险 | - |
| 可维护性 | 中 | 代码清晰度和调试便利性 | - |
| 可测试性 | 中 | 是否可通过单元/集成测试覆盖核心路径 | 核心路径覆盖率 ≥ 80% |

---

## Options Analysis

### Option 1: 扩展 MCPClient 支持 ACP Transport（推荐）

**Description**

修改 `MCPClient._get_client()`，为 `AcpMCPServerConfig` 创建 `AcpMcpTransport`，让 `MCPClient` 能够正常管理 ACP-transport 的 MCP 连接：

```python
case AcpMCPServerConfig(acp_id=acp_id):
    # 从 AcpMcpConnectionManager 获取已建立的连接
    transport = AcpMcpTransport.from_manager(
        connection_manager, acp_id=acp_id
    )
```

然后让 `session.py` 的 `initialize_mcp_servers()` 对 `AcpMcpServer` 也走 `MCPResourceProvider` 路径：

```python
if isinstance(server, AcpMcpServer):
    # 先建立 ACP 连接（获取 connectionId）
    connection_id = await self.acp_agent.connect_acp_mcp_server(server)
    # 再创建 MCPResourceProvider（复用现有路径）
    cfg = AcpMCPServerConfig(name=server.name, acp_id=server.id)
    provider = MCPResourceProvider(server=cfg, ...)
    provider = await provider.__aenter__()  # 这会触发 MCPClient → fastmcp Client → AcpMcpTransport
    self.session_mcp_providers.append(provider)
```

**Advantages**

- **完全复用现有架构**：`MCPResourceProvider` 的 enable/disable、缓存、变更通知全部可用
- **最小侵入性**：只需修改 `MCPClient._get_client()` 一处核心逻辑，其余路径不变
- **行为一致性**：ACP 和 stdio/SSE/HTTP MCP 对上层 Agent 完全透明

**Disadvantages**

- **`AcpMcpTransport` 需要访问 `AcpMcpConnectionManager`**：需要解决 transport 如何获取 manager 和 connection_id 的问题
- **连接建立时机问题**：`mcp/connect` 必须在 `MCPResourceProvider.__aenter__()` 之前完成（因为 transport 需要 connection_id）
- **`AcpMcpConnectionManager` 的生命周期耦合**：`MCPClient` 通常是 per-session，但 `AcpMcpConnectionManager` 是 per-Agent

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| 架构一致性 | ⭐⭐⭐⭐⭐ | 完全复用 MCPResourceProvider |
| 复用性 | ⭐⭐⭐⭐⭐ | 100% 复用现有能力 |
| 实现复杂度 | ⭐⭐⭐⭐ | 需解决 transport↔manager 引用问题 |
| 可维护性 | ⭐⭐⭐⭐⭐ | 代码路径与现有 MCP 一致 |
| 可测试性 | ⭐⭐⭐⭐ | 可 mock AcpMcpConnectionManager |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 人，预计 **2-3 天**
- Dependencies: 需确认 `AcpMcpTransport` 是否已完成且可用

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `AcpMcpTransport` 接口不兼容 | Low | High | Pre-Phase 0 验证 |
| `AcpMcpConnectionManager` 引用循环 | Medium | Medium | 使用弱引用或回调注入 |

---

### Option 2: 手动构建 fastmcp Client（不通过 MCPResourceProvider）

**Description**

在 `connect_acp_mcp_server()` 中手动创建 `AcpMcpTransport` + `fastmcp.Client`，然后自己调用 `list_tools()` 并手动注册到 Agent：

```python
async def connect_acp_mcp_server(self, server: AcpMcpServer) -> str:
    # ... 现有代码：发送 mcp/connect，获取 connectionId ...
    
    # 创建 Transport + fastmcp Client
    transport = AcpMcpTransport(connection_id, self._mcp_manager)
    mcp_client = fastmcp.Client(transport)
    
    # 进入 context，触发 MCP initialize + tools/list
    async with mcp_client:
        tools = await mcp_client.list_tools()
        # 手动注册工具到 Agent
        for tool in tools:
            await self._register_tool_to_agent(tool)
```

**Advantages**

- **不修改 `MCPClient` 和 `MCPResourceProvider`**：避免影响现有 stdio/SSE/HTTP 路径
- **直接控制**：完全掌握每个步骤的执行时机

**Disadvantages**

- **重复实现**：需要重新实现 `MCPResourceProvider` 已经做的所有事情（工具缓存、enable/disable、变更通知）
- **维护负担**：两个并行的工具注册路径，未来修改需要维护两份代码
- **缺少 provider 抽象**：Agent 运行时通过 `ResourceProvider` 管理工具，手动注册可能不兼容

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| 架构一致性 | ⭐⭐ | 绕过现有 MCPResourceProvider |
| 复用性 | ⭐ | 不复用任何现有能力 |
| 实现复杂度 | ⭐⭐⭐ | 看似简单，但需重新实现大量功能 |
| 可维护性 | ⭐⭐ | 两条并行的工具注册路径 |
| 可测试性 | ⭐⭐⭐ | 需要全新测试覆盖 |

**Effort Estimate**

- Complexity: Medium-High
- Resources: 1 人，预计 **4-5 天**（因为需要重新实现工具注册、缓存、变更通知）

**Risk Assessment**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 工具注册与 Agent 运行时不兼容 | High | High | 需要深入了解 Agent 运行时 |
| 变更通知缺失 | High | Medium | 需手动实现 tools/list_changed 监听 |

---

### Option 3: 新建 AcpMCPResourceProvider（MCPResourceProvider 子类）

**Description**

创建 `AcpMCPResourceProvider` 继承 `MCPResourceProvider`，重写 `__aenter__` 以支持 ACP 连接建立：

```python
class AcpMCPResourceProvider(MCPResourceProvider):
    async def __aenter__(self) -> Self:
        # 先建立 ACP 连接
        connection_id = await self.acp_agent.connect_acp_mcp_server(self.acp_server)
        # 再创建 Transport + fastmcp Client
        transport = AcpMcpTransport(connection_id, self.acp_agent._mcp_manager)
        self._mcp_client = fastmcp.Client(transport)
        # 进入 context 触发 initialize + list_tools
        await self._mcp_client.__aenter__()
        # 注册工具
        await self._register_tools()
        return self
```

**Advantages**

- **复用 MCPResourceProvider 框架**：缓存、enable/disable 等能力继承自父类
- **隔离 ACP 特定逻辑**：不修改 `MCPClient._get_client()`

**Disadvantages**

- **MCPResourceProvider 不是为继承设计的**：`__aenter__` 中直接创建 `MCPClient`，子类需要大量重写
- **`MCPClient` 仍然需要修改**：`AcpMCPServerConfig` 的 `NotImplementedError` 需要处理
- **代码分散**：ACP 逻辑分散在 `AcpMCPResourceProvider`、`AcpMcpTransport`、`AcpMcpConnectionManager` 三个类中

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| 架构一致性 | ⭐⭐⭐ | 继承但不完全复用 |
| 复用性 | ⭐⭐⭐ | 复用部分能力 |
| 实现复杂度 | ⭐⭐⭐ | 需重写 MCPResourceProvider 核心方法 |
| 可维护性 | ⭐⭐⭐ | 新增类增加复杂度 |
| 可测试性 | ⭐⭐⭐ | 需测试子类特定逻辑 |

---

## Recommendation

### Recommended Option

**Option 1: 扩展 MCPClient 支持 ACP Transport**

### Justification

1. **架构一致性最高**：ACP MCP server 与 stdio/SSE/HTTP MCP server 对上层完全透明，走同一套 `MCPResourceProvider → MCPClient → fastmcp.Client` 路径
2. **100% 复用现有能力**：工具缓存、enable/disable、变更通知不需要重新实现
3. **维护成本最低**：未来对 MCP 功能的改进（如 prompts/resources 支持）自动惠及 ACP transport
4. **实现复杂度可控**：核心改动只有两处：
   - `MCPClient._get_client()` 中处理 `AcpMCPServerConfig`
   - `AcpMcpTransport` 获取 connection 的方式（从 manager 查找改为直接传入或延迟初始化）

### Accepted Trade-offs

1. **需要解决 `AcpMcpConnectionManager` 引用问题**：可以通过在 `AcpMCPServerConfig` 中存储 `connection_id`，然后让 `AcpMcpTransport` 在 `connect()` 时从 manager 查找（而不是在构造函数时就需要）
2. **连接建立时机提前**：`mcp/connect` 需要在 `MCPResourceProvider.__aenter__()` 之前完成。这可以通过让 `session.py` 先调用 `connect_acp_mcp_server()`，再创建 `MCPResourceProvider` 来解决

---

## Technical Design

### 架构图

```
ACPSession
  └── session_mcp_providers: list[MCPResourceProvider]
        ├── MCPResourceProvider (stdio server)
        │     └── MCPClient → fastmcp.Client(StdioTransport) → subprocess
        ├── MCPResourceProvider (SSE server)
        │     └── MCPClient → fastmcp.Client(SSETransport) → HTTP
        └── MCPResourceProvider (ACP server)  ← 新增，与上面完全一致
              └── MCPClient → fastmcp.Client(AcpMcpTransport) → AcpMcpConnection
                                                    ↓
                                              memory streams
                                                    ↓
                                              AcpMcpConnectionManager
                                                    ↓
                                              AgentSideConnection.send_request("mcp/message")
                                                    ↓
                                              Client (IDE)
```

### 关键修改点

#### 1. MCPClient._get_client() 支持 ACP

```python
# wolfharness/mcp_server/client.py

case AcpMCPServerConfig(acp_id=acp_id):
    # AcpMcpTransport 延迟初始化：在 connect() 时从 manager 获取连接
    transport = AcpMcpTransport(acp_id=acp_id)
```

#### 2. AcpMcpTransport 支持延迟初始化

```python
# wolfharness_server/acp_server/acp_mcp_transport.py

class AcpMcpTransport(ClientTransport):
    def __init__(self, acp_id: str) -> None:
        self._acp_id = acp_id
        self._connection: AcpMcpConnection | None = None
    
    async def connect(self) -> tuple[MemoryObjectReceiveStream, MemoryObjectSendStream]:
        # 从全局 manager 查找已建立的连接
        manager = _get_active_manager()  # 或通过其他方式获取
        self._connection = manager.get_connection_by_acp_id(self._acp_id)
        return self._connection.to_session, self._connection.from_session
```

#### 3. session.py 中 ACP server 的初始化流程

```python
# wolfharness_server/acp_server/session.py

async def initialize_mcp_servers(self) -> None:
    for server in self.mcp_servers:
        if isinstance(server, AcpMcpServer):
            # Step 1: 建立 ACP 连接（获取 connectionId）
            connection_id = await self.acp_agent.connect_acp_mcp_server(server)
            # Step 2: 复用 MCPResourceProvider 路径
            cfg = AcpMCPServerConfig(name=server.name, acp_id=server.id)
            provider = MCPResourceProvider(server=cfg, ...)
            provider = await provider.__aenter__()
            self.session_mcp_providers.append(provider)
            continue
        # ... 现有 stdio/SSE/HTTP 逻辑不变 ...
```

#### 4. 数据流

```
1. LLM 决定调用工具 "workspace-fs/read_file"
2. Agent.run() → Tool.call() → MCPResourceProvider.call_tool()
3. MCPClient.call_tool() → fastmcp.Client.call_tool()
4. fastmcp.Client 序列化 JSON-RPC request
5. AcpMcpTransport.send() → AcpMcpConnection.send_to_client()
6. AgentSideConnection.send_request("mcp/message", {connectionId, message})
7. Client 收到请求，执行工具，返回 result
8. 原路返回：Client → AgentSideConnection → AcpMcpConnection → fastmcp.Client
9. fastmcp.Client 反序列化 response → MCPClient → ToolResult
```

### Error Handling

| 场景 | 处理 |
|------|------|
| `mcp/connect` 失败 | 记录错误，该 MCP server 不可用，不影响其他 server |
| MCP `initialize` 握手失败 | `MCPResourceProvider.__aenter__()` 抛出异常，由 session.py catch 并记录 |
| `tools/list` 返回空 | 正常注册（0 个工具），后续 tools/list_changed 可能更新 |
| `mcp/message` 超时 | fastmcp.Client 超时处理，返回 timeout error |
| Client 断开连接 | `AcpMcpConnection` 检测流关闭，触发断开回调 |

---

## Implementation Plan

### Phase 1: AcpMcpTransport 延迟初始化改造（1 天）

- **Scope**：
  - 修改 `AcpMcpTransport` 支持通过 `acp_id` 延迟查找连接（而非构造函数传入 connection）
  - 或改为 `connect()` 时通过回调/全局注册表获取 `AcpMcpConnection`
- **Deliverables**：AcpMcpTransport 可独立工作，不依赖构造函数传入完整连接

### Phase 2: MCPClient._get_client() 支持 ACP（1 天）

- **Scope**：
  - 修改 `client.py` 中 `_get_client()` 的 `AcpMCPServerConfig` 分支
  - 创建 `AcpMcpTransport` 实例并传入 fastmcp Client
  - 确保 oauth、timeout、client_info 等参数正确传递
- **Deliverables**：`MCPClient` 可成功创建 ACP-transport 的 fastmcp Client

### Phase 3: session.py 集成（1 天）

- **Scope**：
  - 修改 `session.py` 的 `initialize_mcp_servers()`
  - AcpMcpServer 先调用 `connect_acp_mcp_server()`，再创建 `MCPResourceProvider`
  - 确保 `AcpMCPServerConfig` 正确生成
- **Deliverables**：ACP MCP server 成功注册工具到 Agent 运行时

### Phase 4: 端到端验证（1-2 天）

- **Scope**：
  - 集成测试：`session/new` → `mcp/connect` → `tools/list` → tool call → `mcp/disconnect`
  - 验证工具 enable/disable 工作正常
  - 验证 session 关闭时资源清理
  - 回归测试：stdio/SSE/HTTP MCP 不受影呴
- **Deliverables**：完整可用 + 测试覆盖

---

## Open Questions

1. **`AcpMcpTransport` 如何获取 `AcpMcpConnectionManager`？**
   - Option A: 全局注册表（registry pattern）
   - Option B: 在 `AcpMCPServerConfig` 中存储 manager 引用
   - Option C: `connect()` 时通过回调注入

2. **`AcpMcpConnection` 的内存流生命周期是否与 fastmcp Client 匹配？**
   - fastmcp Client 会在 `__aexit__` 时关闭 transport
   - 需要确认 `AcpMcpConnection` 的流是否支持 reopen 或需要重建

3. **`tools/list_changed` notification 的处理方式？**
   - Client 可能发送 `_mcp/message` with `method: notifications/tools/list_changed`
   - 当前 `AcpMcpConnection.handle_client_message()` 需要能路由这类 notification

4. **多个 session 复用同一 ACP connection 的情况？**
   - 当前设计是每个 session 独立连接，是否支持 connection 复用？

---

## Decision Record

### Decision

**Status**: DRAFT

**Date**: —

**Approvers**: —

### Decision Summary

—

### Key Discussion Points

—

### Conditions of Approval

—

### Dissenting Opinions

—

---

## References

### Related Documents

- [RFC-0033: MCP-over-ACP Transport](../implemented/RFC-0033-mcp-over-acp-transport.md)
- [ACP RFD: mcp-over-acp](../../../../agent-client-protocol/docs/rfds/mcp-over-acp.mdx)

### Key Files

| 文件 | 说明 |
|------|------|
| `wolfharness/mcp_server/client.py` | MCPClient，需修改 `_get_client()` |
| `wolfharness/resource_providers/mcp_provider.py` | MCPResourceProvider，工具注册 |
| `wolfharness_server/acp_server/acp_mcp_transport.py` | AcpMcpTransport，需支持延迟初始化 |
| `wolfharness_server/acp_server/acp_agent.py` | connect_acp_mcp_server() |
| `wolfharness_server/acp_server/session.py` | initialize_mcp_servers() |
| `wolfharness_server/acp_server/acp_mcp_manager.py` | AcpMcpConnectionManager |

### External Resources

- [fastmcp ClientTransport Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://modelcontextprotocol.io/)
