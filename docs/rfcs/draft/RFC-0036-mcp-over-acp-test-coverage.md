---
rfc_id: RFC-0036
title: "MCP-over-ACP: Comprehensive Test Coverage Design"
status: DRAFT
author: yuchen.liu
reviewers:
  - name: Metis - Plan Consultant
    status: approved
    date: 2026-05-29
    feedback: |
      1. Add real fastmcp Client contract test (catches ClientTransport inheritance bugs)
      2. Fix MockAcpMcpConnection.send_to_client signature (return Any, wrap with connectionId)
      3. Test observable errors, not silent swallowing (I1.2)
      4. Rename Layer 3 from E2E to High-level Integration
      5. Add provider init failure cleanup test
      6. Add client without send_request test (catches Bug #2)
      7. Add forwarder error handling test
      8. Use anyio.create_memory_object_stream(0) for realistic backpressure
  - name: oracle
    status: approved
    date: 2026-05-29
    feedback: |
      1. Mock-first is sound but dangerously thin on integration coverage
      2. Need 2-3 in-memory tests with real fastmcp.Client(AcpMcpTransport(...))
      3. Add test_transport_clienttransport_contract: session_kwargs passthrough, close(), _set_auth ValueError
      4. Add concurrency stress test with capacity-0 streams
      5. Verify cleanup ordering (forwarder cancelled before streams closed)
      6. Simulate realistic send_to_client (AsyncMock that suspends)
      7. Watch for: capacity-0 deadlock, session_kwargs passthrough, close() not implemented
created: 2026-05-29
last_updated: 2026-05-29
decision_date:
related_prds: []
related_rfcs:
  - RFC-0033-mcp-over-acp-transport.md
  - RFC-0035-mcp-over-acp-complete-connection-chain.md
---

# RFC-0036: MCP-over-ACP Comprehensive Test Coverage Design

## Overview

本 RFC 针对 MCP-over-ACP 功能设计完整的测试覆盖方案，确保 RFC-0033（传输协议）和 RFC-0035（完整连接链）的所有组件在各种边界条件下都能正确工作。MCP-over-ACP 涉及跨层协作（schema → transport → session → agent），任何一层的异常都可能导致端到端功能不可用。通过系统性的测试设计，我们希望在代码变更前捕获协议偏差、接口不匹配和资源泄漏等问题。

核心测试策略采用三层金字塔：单元测试验证隔离组件，集成测试验证层间协作，端到端测试验证真实场景。

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Test Architecture](#test-architecture)
- [Test Case Catalog](#test-case-catalog)
- Implementation Plan
- [Open Questions](#open-questions)
- Decision Record
- [References](#references)

---

## Background & Context

### MCP-over-ACP 组件架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Test Layer 3: End-to-End                                       │
│  [Real ACP Client] ↔ [AgentPoolACPAgent] → [Agent + LLM]        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Test Layer 2: Integration                                      │
│  [ACPSession] → [initialize_mcp_servers] → [MCPResourceProvider]│
│                          ↓                                      │
│  [AcpMcpConnectionManager] ↔ [AcpMcpTransport] ↔ [fastmcp]      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Test Layer 1: Unit                                             │
│  [AcpMcpConnection]  [AcpMcpTransport]  [AcpMcpConnectionManager]│
│  [MCPClient._get_client_from_transport]                         │
│  [MCPResourceProvider with external transport]                  │
└─────────────────────────────────────────────────────────────────┘
```

### 已发现的测试盲区（来自 RFC-0035 实施过程）

| 问题 | 影响 | 发现时机 |
|------|------|---------|
| `AcpMcpTransport` 未继承 `ClientTransport` | fastmcp.Client 拒绝 transport，工具无法注册 | 运行时错误 |
| `AcpMcpTransport` 调用了 `session.initialize()` | 与 fastmcp.Client 的自动初始化冲突，导致双初始化 | 代码审查 |
| `AgentSideConnection` 缺少 `send_request` | `mcp/connect` 发送时抛出 `AttributeError`，异常被 session.py 的 `except Exception` 吞掉 | 运行时静默失败 |
| `connect_acp_mcp_server` 异常被静默捕获 | 工具注册失败，但 session 创建仍返回成功 | 日志分析 |

### 现有测试状态

| 组件 | 现有测试 | 覆盖度 | 缺失 |
|------|---------|--------|------|
| `AcpMcpConnectionManager` | `test_acp_mcp_manager.py` | ✅ create/remove/get/close | 无 |
| `AcpMcpTransport` | `test_acp_mcp_transport.py` | ⚠️ 消息流传输 | ❌ `ClientTransport` ABC 接口合规性 |
| `AgentPoolACPAgent.ext_method` | `test_mcp_integration.py` | ✅ 被动 mcp/message 处理 | ❌ 主动 mcp/connect/mcp/disconnect |
| `ACPSession.initialize_mcp_servers` | 间接通过 `test_acp_per_session_agent.py` | ⚠️ stdio/SSE 路径 | ❌ ACP-transport 路径 |
| `MCPClient._get_client` | `test_mcp_client.py` | ✅ stdio/SSE/HTTP | ❌ ACP-transport 路径 |
| `MCPResourceProvider.__aenter__` | 间接通过集成测试 | ⚠️ 通用路径 | ❌ 外部 transport 注入路径 |

---

## Problem Statement

### The Problem

MCP-over-ACP 的测试覆盖存在系统性缺口。当前测试主要覆盖：
1. 底层 connection manager 的状态机
2. 非 ACP transport 的 MCP 集成（stdio/SSE/HTTP）

但**缺少**以下关键路径的测试：
1. AcpMcpTransport 作为 fastmcp ClientTransport 的接口合规性
2. Session 创建时 ACP MCP server 的完整初始化链（connect → transport → client → tools/list）
3. Agent 主动向 Client 发送 mcp/connect 的协议方向正确性
4. 异常路径（connect 失败、transport 初始化失败、client 初始化失败）的处理
5. 资源清理（session 关闭时 disconnect + provider cleanup）

### Evidence

- `AcpMcpTransport` 未继承 `ClientTransport` 的问题在代码运行时才暴露，编译期和静态检查都无法发现
- `AgentSideConnection.send_request` 缺失的问题导致异常被静默吞掉，只有通过日志分析才能发现
- 没有测试验证 `MCPClient._get_client_from_transport` 是否能正确构建 fastmcp Client

### Impact of Inaction

- 未来对 AcpMcpTransport、MCPClient 或 session.py 的修改可能破坏 MCP-over-ACP 功能而不被察觉
- 重构 fastmcp 版本升级时无法验证兼容性
- 新加入的开发者难以理解 MCP-over-ACP 的正确行为边界

---

## Goals & Non-Goals

### Goals (In Scope)

1. 为 `AcpMcpTransport` 提供 `ClientTransport` ABC 接口合规性测试
2. 为 `AcpMcpConnectionManager` 补充并发和异常路径测试
3. 为 `AgentPoolACPAgent` 的主动 mcp/connect/mcp/disconnect 提供单元测试
4. 为 `ACPSession.initialize_mcp_servers` 的 ACP 路径提供集成测试
5. 为 `MCPClient._get_client_from_transport` 提供单元测试
6. 为 `MCPResourceProvider` 的外部 transport 注入路径提供集成测试
7. 提供端到端测试验证完整消息流（session/new → mcp/connect → mcp/message → tool call → mcp/disconnect）
8. 所有测试能够在 CI 中运行（不依赖真实外部 ACP client）

### Non-Goals (Out of Scope)

1. **不测试** fastmcp 库本身的正确性（假设 fastmcp Client 行为正确）
2. **不测试** ACP 协议的其他 method（fs/read_text_file、terminal/create 等）
3. **不测试** 客户端（Zed IDE、SEED 等）的行为
4. **不测试** 性能/负载（吞吐量、并发连接数）
5. **不测试** Bridging 功能（RFC-0033 已明确 defer）

### Success Criteria

- [ ] 新增测试在 CI 中全部通过
- [ ] 代码覆盖率：MCP-over-ACP 相关代码行覆盖 ≥ 90%
- [ ] 关键路径（session create → tool call → session close）有端到端测试覆盖
- [ ] 异常路径（connect 失败、initialize 失败、transport 无效）有测试覆盖
- [ ] 现有 stdio/SSE/HTTP MCP 测试不受影响（回归测试通过）

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| 可执行性 | 高 | 测试能否在 CI/本地环境中稳定运行 | 不使用真实外部服务 |
| 覆盖度 | 高 | 覆盖正常路径、异常路径和边界条件 | ≥ 90% 行覆盖 |
| 可维护性 | 中 | 测试代码清晰，失败时提供可操作的诊断信息 | 断言失败消息明确 |
| 速度 | 中 | 测试执行时间合理 | 单测 < 1s，集成测试 < 10s |
| 确定性 | 高 | 测试结果不依赖时序或外部状态 | 无 flaky test |

---

## Options Analysis

### Option 1: Mock-based 测试（推荐）

**Description**

使用 Python unittest.mock 和 anyio 内存流来模拟所有外部依赖：
- Mock `Client`（ACP client）来验证 mcp/connect 和 mcp/disconnect 的发送
- Mock `AcpMcpConnection` 的内存流来验证 transport 的消息路由
- 使用 `AsyncMock` 来模拟 fastmcp ClientSession 的响应

**Advantages**

- **完全隔离**：不依赖真实 fastmcp Client、网络或外部进程
- **可控**：可以精确模拟各种异常场景（timeout、connection refused、invalid response）
- **快速**：毫秒级执行，适合 CI
- **确定性**：无 race condition，无 flaky test

**Disadvantages**

- **Mock 与实际行为偏差**：如果 fastmcp Client 的内部实现改变，mock 可能不再准确
- **无法捕获接口变化**：如 `ClientTransport` ABC 添加新方法，mock 测试不会失败
- **需要维护 mock 的复杂性**：模拟 fastmcp Client 的完整生命周期较复杂

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| 可执行性 | ⭐⭐⭐⭐⭐ | 纯 Python，无外部依赖 |
| 覆盖度 | ⭐⭐⭐⭐ | 可覆盖正常和异常路径 |
| 可维护性 | ⭐⭐⭐ | Mock 复杂，需要维护 |
| 速度 | ⭐⭐⭐⭐⭐ | 极快 |
| 确定性 | ⭐⭐⭐⭐⭐ | 完全确定性 |

**Effort Estimate**

- Complexity: Medium
- Resources: 1 人，预计 **3-4 天**

---

### Option 2: In-memory 集成测试

**Description**

使用真实的 fastmcp Client 和 AcpMcpTransport，但通过内存流连接到 mock server：
- 创建真实的 `fastmcp.Client(AcpMcpTransport(connection))`
- `AcpMcpConnection` 的内存流连接到 mock handler（模拟 ACP client 的响应）
- Mock handler 根据接收到的 MCP JSON-RPC 请求返回预设响应

**Advantages**

- **测试真实 fastmcp Client**：验证 fastmcp Client 与 AcpMcpTransport 的集成
- **捕获接口变化**：如果 fastmcp Client 的 API 改变，测试会失败
- **更真实**：测试的是实际代码路径，而非 mock

**Disadvantages**

- **依赖 fastmcp 内部行为**：fastmcp Client 的 async 生命周期较复杂（background task、reentrant context）
- **调试困难**：fastmcp Client 的错误可能难以定位到具体原因
- **速度较慢**：需要启动 fastmcp Client 的 background task

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| 可执行性 | ⭐⭐⭐⭐ | 无外部网络依赖 |
| 覆盖度 | ⭐⭐⭐⭐⭐ | 测试真实代码路径 |
| 可维护性 | ⭐⭐⭐ | fastmcp 内部复杂 |
| 速度 | ⭐⭐⭐ | 比 mock 慢 |
| 确定性 | ⭐⭐⭐⭐ | 可能有 race condition |

**Effort Estimate**

- Complexity: Medium-High
- Resources: 1 人，预计 **4-5 天**

---

### Option 3: 端到端进程测试

**Description**

启动完整的 AgentPool ACP server 和 mock ACP client 进程，通过真实 ACP 协议通信：
- 使用 `wolfharness serve-acp` 启动 server
- 使用 Python subprocess 启动 mock client（通过 stdio 或 WebSocket 连接）
- 发送真实的 JSON-RPC 请求，验证响应

**Advantages**

- **最真实的测试**：验证完整的协议栈
- **捕获进程级问题**：如资源泄漏、signal 处理、stdio 缓冲

**Disadvantages**

- **极慢**：启动进程、建立连接需要秒级时间
- **不稳定**：进程间通信容易受系统状态影响（flaky）
- **难以调试**：问题可能出现在任何一层，难以定位
- **不适合 CI**：超时、端口占用等问题

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| 可执行性 | ⭐⭐ | 需要进程管理 |
| 覆盖度 | ⭐⭐⭐⭐⭐ | 最真实的覆盖 |
| 可维护性 | ⭐⭐ | 极难调试 |
| 速度 | ⭐ | 秒级到分钟级 |
| 确定性 | ⭐⭐ | 容易 flaky |

**Effort Estimate**

- Complexity: High
- Resources: 1-2 人，预计 **5-7 天**

---

## Recommendation

### Recommended Option

**Option 1 (Mock-based) 为主，Option 2 (In-memory) 为辅**

### Justification

1. **Mock-based 覆盖 80% 场景**：单元测试和集成测试使用 mock，确保快速、稳定、可控
2. **In-memory 覆盖关键路径**：1-2 个端到端测试使用真实 fastmcp Client，验证 transport ↔ client 集成
3. **不采用 Option 3**：进程级测试的收益与成本不成正比，且已有 `--show-events-detailed` 可用于手动验证

### Accepted Trade-offs

- Mock 测试可能无法捕获 fastmcp 版本升级带来的接口变化 → 通过 1-2 个 in-memory 测试作为安全网
- Mock 的复杂性 → 通过良好的抽象（`MockACPClient`、`MockMcpConnection`）降低维护成本

---

## Review-Driven Changes Summary

This section documents all changes made to the original RFC after @Metis and @oracle review. Use this as a checklist during implementation.

### Critical Additions (Must Implement)

| Change | Location | Rationale | Reviewer |
|--------|----------|-----------|----------|
| `MockAcpMcpConnection.send_to_client` returns `Any`, wraps with `connectionId` | Mock Infrastructure | Matches real signature, prevents false positives | Metis, oracle |
| `MockACPClientWithoutSendRequest` fixture | Mock Infrastructure | Tests Bug #2 (missing `send_request`) | Metis |
| U1.7 `test_transport_passes_session_kwargs` | Unit Tests | Catches silent `session_kwargs` passthrough bug | oracle |
| U1.8 `test_transport_set_auth_raises_valueerror` | Unit Tests | Verifies ABC compliance | oracle |
| U1.9 `test_transport_close_is_callable` | Unit Tests | fastmcp calls `close()` in error paths | oracle |
| U1.10 `test_forwarder_handles_client_error` | Unit Tests | Tests forwarder error propagation | Metis |
| U1.11 `test_fastmcp_client_accepts_transport` | Unit Tests | Real fastmcp Client creation (catches Bug #1) | Metis, oracle |
| U3.7 `test_connect_client_without_send_request_raises` | Unit Tests | Tests Bug #2 at unit level | Metis |
| U3.8 `test_connect_uses_mcp_message_when_available` | Unit Tests | Tests dual client path branching | Metis |
| I1.2 restructured to test error observability | Integration Tests | Original tested silent swallowing (wrong behavior) | Metis, oracle |
| I1.6 `test_provider_init_failure_cleans_up_connection` | Integration Tests | Prevents resource leak on provider init failure | Metis |
| I1.7 `test_initialize_mcp_servers_error_propagation` | Integration Tests | Errors must be observable to callers | Metis |
| I2.4 `test_real_fastmcp_client_connects_via_transport` | Integration Tests | Real Client handshake (prevents API drift) | oracle |
| I2.5 `test_real_fastmcp_client_lists_tools` | Integration Tests | Real Client tool listing | oracle |
| HI.5 `test_cleanup_ordering_forwarder_before_streams` | High-level Integration | Verifies cleanup ordering (prevents race) | oracle |
| HI.6 `test_concurrent_messages_no_deadlock` | High-level Integration | Tests capacity-0 stream deadlock | oracle |

### Design Changes

| Change | Original | New | Rationale |
|--------|----------|-----|-----------|
| Layer 3 name | "End-to-End Tests" | "High-level Integration Tests" | Still in-process, not cross-process (@Metis) |
| I1.2 test focus | "session 仍创建成功" (validates silent swallowing) | "错误可观测" (validates `mcp_init_errors` populated) | Silent swallowing hid real bugs (@Metis) |
| Mock `send_to_client` | Returns `dict` synchronously | Returns `Any`, uses `AsyncMock` with suspension | Realistic async behavior (@oracle) |
| E2E test approach | Pure mocks | Real `fastmcp.Client` + memory streams | Catches API drift (@oracle) |

### Implementation Checklist

- [ ] Fix `MockAcpMcpConnection` to match real signature
- [ ] Add `MockACPClientWithoutSendRequest` fixture
- [ ] Implement U1.7-U1.11 (transport tests)
- [ ] Implement U3.7-U3.8 (agent MCP method tests)
- [ ] Restructure I1.2 to test observability
- [ ] Implement I1.6-I1.7 (session integration tests)
- [ ] Implement I2.4-I2.5 (real fastmcp Client tests)
- [ ] Implement HI.5-HI.6 (cleanup + concurrency tests)
- [ ] Use `anyio.create_memory_object_stream(0)` in all mocks
- [ ] Use `AsyncMock` with realistic suspension for `send_to_client`

---

## Test Architecture

### 测试分层

```
Layer 1: Unit Tests (tests/unit/acp_server/)
├── test_acp_mcp_transport.py
│   └── AcpMcpTransport ABC compliance
├── test_acp_mcp_manager.py
│   └── Connection lifecycle + concurrency
└── test_acp_agent_mcp_methods.py
    └── AgentPoolACPAgent.connect/disconnect_acp_mcp_server

Layer 2: Integration Tests (tests/integration/acp_server/)
├── test_acp_mcp_session_integration.py
│   └── ACPSession.initialize_mcp_servers with ACP transport
├── test_mcp_client_acp_transport.py
│   └── MCPClient._get_client_from_transport
└── test_mcp_resource_provider_acp.py
    └── MCPResourceProvider with external transport

Layer 3: High-level Integration Tests (tests/integration/acp_server/)
└── test_acp_mcp_full_lifecycle.py
    └── Complete: session/new → connect → message → tool call → disconnect
    └── Uses real fastmcp.Client with memory streams (catches API drift)
    └── Renamed from "E2E" per @Metis/@oracle review: still in-process, not cross-process
```

### Mock 基础设施

```python
# tests/fixtures/acp_mcp_fixtures.py

import anyio
from unittest.mock import AsyncMock

class MockAcpMcpConnection:
    """Simulates an AcpMcpConnection with in-memory streams.

    ⚠️ CRITICAL: Matches real AcpMcpConnection.send_to_client signature.
    Returns Any (not dict), wraps with connectionId/message envelope.
    """
    def __init__(self, connection_id: str = "test-conn"):
        self.connection_id = connection_id
        self.sent_messages: list[dict] = []
        self._to_session_send, self._to_session_receive = anyio.create_memory_object_stream(0)
        self._from_session_send, self._from_session_receive = anyio.create_memory_object_stream(0)

    async def send_to_client(self, message: dict) -> Any:
        """Simulate AcpMcpConnection.send_to_client — returns Any, wraps envelope."""
        self.sent_messages.append(message)
        envelope = {"connectionId": self.connection_id, "message": message}
        return await self._handle_message(envelope)

    async def _handle_message(self, envelope: dict) -> Any:
        """Simulate MCP server responses based on JSON-RPC method."""
        message = envelope.get("message", {})
        if message.get("method") == "initialize":
            return {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "mock"}}
        elif message.get("method") == "tools/list":
            return {"tools": [{"name": "read_file", "description": "Read a file"}]}
        elif message.get("method") == "tools/call":
            return {"content": [{"type": "text", "text": "hello"}]}
        return None  # Notifications return None

    async def inject_response(self, message: dict) -> None:
        """Simulate server-to-client message injection (via handle_client_message)."""
        await self._from_session_send.send(message)

class MockACPClient:
    """Simulates an ACP Client for testing mcp/connect direction.

    Supports both send_request and mcp_message paths.
    """
    def __init__(self):
        self.sent_requests: list[dict] = []
        self.responses: dict[str, Any] = {}

    async def send_request(self, method: str, params: dict) -> dict:
        self.sent_requests.append({"method": method, "params": params})
        return self.responses.get(method, {})

    async def mcp_message(self, connection_id: str, message: dict) -> dict:
        self.sent_requests.append({"method": "mcp/message", "connectionId": connection_id, "message": message})
        return self.responses.get("mcp/message", {})

    def set_response(self, method: str, response: dict) -> None:
        self.responses[method] = response

class MockACPClientWithoutSendRequest:
    """Simulates a legacy ACP client that lacks send_request (catches Bug #2)."""
    def __init__(self):
        self.sent_requests: list[dict] = []
```

---

## Test Case Catalog

### Layer 1: Unit Tests

#### `test_acp_mcp_transport.py`

| ID | 测试名 | 描述 | 预期结果 | Review Source |
|----|--------|------|---------|---------------|
| U1.1 | `test_transport_is_client_transport` | `isinstance(AcpMcpTransport(), ClientTransport)` | `True` | RFC |
| U1.2 | `test_transport_connect_session_yields_session` | 进入 `connect_session()` context | 返回 `ClientSession` | RFC |
| U1.3 | `test_transport_no_initialize_called` | `connect_session()` 内部不调用 `session.initialize()` | `call_count == 0` | Metis, oracle |
| U1.4 | `test_transport_forwarder_routes_messages` | 从 `from_session_receive` 读取消息并发送到 client | `send_to_client` 被调用 | RFC |
| U1.5 | `test_transport_cleanup_on_exit` | 退出 `connect_session()` context 后 | forwarder task 被取消，流关闭 | RFC |
| U1.6 | `test_transport_connection_id_property` | `transport.connection_id` | 返回 `AcpMcpConnection.connection_id` | RFC |
| U1.7 | `test_transport_passes_session_kwargs` | `connect_session(read_timeout_seconds=30)` | `ClientSession` 接收 `read_timeout_seconds` | oracle |
| U1.8 | `test_transport_set_auth_raises_valueerror` | 调用 `transport._set_auth(...)` | 抛出 `ValueError` (ABC 默认行为) | oracle |
| U1.9 | `test_transport_close_is_callable` | 调用 `transport.close()` | 不抛出异常 | oracle |
| U1.10 | `test_forwarder_handles_client_error` | `send_to_client` 抛出异常 | forwarder 传播错误，connection 关闭 | Metis |
| U1.11 | `test_fastmcp_client_accepts_transport` | 创建 `fastmcp.Client(AcpMcpTransport(connection))` | 成功创建，不抛出异常 | Metis, oracle |

#### `test_acp_mcp_manager.py`

| ID | 测试名 | 描述 | 预期结果 |
|----|--------|------|---------|
| U2.1 | `test_create_connection` | 创建 connection | 返回 `AcpMcpConnection`，注册到 manager |
| U2.2 | `test_create_duplicate_connection_fails` | 用相同 connection_id 创建两次 | 抛出 `ValueError` |
| U2.3 | `test_get_connection` | 通过 ID 获取 connection | 返回正确 connection |
| U2.4 | `test_remove_connection` | 移除 connection | connection 关闭，从 manager 移除 |
| U2.5 | `test_close_all` | 关闭所有 connections | 所有 connection 关闭，manager 为空 |
| U2.6 | `test_concurrent_create` | 并发创建多个 connections | 全部成功，无 race condition |

#### `test_acp_agent_mcp_methods.py`

| ID | 测试名 | 描述 | 预期结果 | Review Source |
|----|--------|------|---------|---------------|
| U3.1 | `test_connect_acp_mcp_server_sends_mcp_connect` | 调用 `connect_acp_mcp_server()` | `client.send_request("mcp/connect", ...)` 被调用 | RFC |
| U3.2 | `test_connect_acp_mcp_server_validates_connection_id` | Client 返回空 connectionId | 抛出 `ValueError` | RFC |
| U3.3 | `test_connect_acp_mcp_server_creates_connection` | 成功连接后 | `AcpMcpConnectionManager` 包含新 connection | RFC |
| U3.4 | `test_disconnect_acp_mcp_server_sends_mcp_disconnect` | 调用 `disconnect_acp_mcp_server()` | `client.send_request("mcp/disconnect", ...)` 被调用 | RFC |
| U3.5 | `test_disconnect_acp_mcp_server_removes_connection` | 成功断开 | connection 从 manager 移除 | RFC |
| U3.6 | `test_disconnect_handles_client_error` | Client 返回 error | 本地 connection 仍被移除 | RFC |
| U3.7 | `test_connect_client_without_send_request_raises` | Client 缺少 `send_request` | 抛出 `AttributeError`（不静默吞掉） | Metis |
| U3.8 | `test_connect_uses_mcp_message_when_available` | Client 有 `mcp_message` 方法 | 调用 `client.mcp_message(...)` 而非 `send_request` | Metis |

### Layer 2: Integration Tests

#### `test_acp_mcp_session_integration.py`

| ID | 测试名 | 描述 | 预期结果 | Review Source |
|----|--------|------|---------|---------------|
| I1.1 | `test_initialize_mcp_servers_with_acp_transport` | `ACPSession.initialize_mcp_servers()` 处理 `AcpMcpServer` | `MCPResourceProvider` 被创建并注册 | RFC |
| I1.2 | `test_initialize_mcp_servers_connect_failure_observable` | `connect_acp_mcp_server()` 失败 | 错误被记录 **且** `session.mcp_init_errors` 包含该错误（可观测） | Metis, oracle |
| I1.3 | `test_initialize_mcp_servers_transport_failure` | `MCPResourceProvider.__aenter__()` 失败 | 记录错误，已建立的 ACP 连接断开 | RFC |
| I1.4 | `test_session_close_disconnects_acp` | `session.close()` | `mcp/disconnect` 发送，provider 清理 | RFC |
| I1.5 | `test_session_close_with_failed_provider` | provider 初始化失败后再 close | 不抛出异常，不发送 disconnect | RFC |
| I1.6 | `test_provider_init_failure_cleans_up_connection` | `MCPResourceProvider.__aenter__()` 失败 | ACP connection 从 manager 移除，无泄漏 | Metis |
| I1.7 | `test_initialize_mcp_servers_error_propagation` | `connect_acp_mcp_server()` 失败 | 失败信息可被 session 消费者获取 | Metis |

#### `test_mcp_client_acp_transport.py`

| ID | 测试名 | 描述 | 预期结果 | Review Source |
|----|--------|------|---------|---------------|
| I2.1 | `test_get_client_from_transport` | `MCPClient(transport=AcpMcpTransport(...))` | 创建 fastmcp Client 成功 | RFC |
| I2.2 | `test_get_client_from_transport_timeout` | 设置 timeout 参数 | fastmcp Client 使用指定 timeout | RFC |
| I2.3 | `test_client_info_passed_to_fastmcp` | 设置 client_name | fastmcp Client 包含 client_info | RFC |
| I2.4 | `test_real_fastmcp_client_connects_via_transport` | 使用真实 `fastmcp.Client(AcpMcpTransport(...))` | Client 通过 transport 成功连接，完成 initialize handshake | oracle |
| I2.5 | `test_real_fastmcp_client_lists_tools` | 真实 Client 连接后调用 `tools/list` | 返回 mock 工具列表 | oracle |

#### `test_mcp_resource_provider_acp.py`

| ID | 测试名 | 描述 | 预期结果 |
|----|--------|------|---------|
| I3.1 | `test_provider_with_external_transport` | `MCPResourceProvider(transport=AcpMcpTransport(...))` | `provider.client._external_transport` 不为 None |
| I3.2 | `test_provider_transport_type_acp` | `provider.transport_type` | 返回 `"acp"` |
| I3.3 | `test_provider_list_tools_acp` | `provider.list_tools()` | 返回从 mock MCP server 获取的工具列表 |
| I3.4 | `test_provider_call_tool_acp` | `provider.call_tool("read_file")` | 发送 mcp/message，返回结果 |

### Layer 3: End-to-End Tests

#### `test_acp_mcp_full_lifecycle.py` (High-level Integration)

| ID | 测试名 | 描述 | 预期结果 | Review Source |
|----|--------|------|---------|---------------|
| HI.1 | `test_full_lifecycle_real_client` | 完整链路：session/new → mcp/connect → tools/list → tool call → session/close | 每个步骤的 JSON-RPC 消息正确发送 | RFC |
| HI.2 | `test_mcp_connect_direction` | 验证 mcp/connect 方向 | `Agent→Client`（Agent 发送） | RFC |
| HI.3 | `test_mcp_disconnect_on_session_close` | session 关闭时 | Agent 发送 mcp/disconnect | RFC |
| HI.4 | `test_tools_available_after_connect` | 连接后 Agent 的工具列表 | 包含 ACP MCP server 的工具 | RFC |
| HI.5 | `test_cleanup_ordering_forwarder_before_streams` | session 关闭时 | forwarder task 在 stream 关闭前被取消 | oracle |
| HI.6 | `test_concurrent_messages_no_deadlock` | 并发发送 N 个消息 | 无 `ClosedResourceError`、无死锁 | oracle |

---

## Implementation Plan

### Phase 1: Mock 基础设施（1 天）

- **Scope**：
  - 创建 `tests/fixtures/acp_mcp_fixtures.py`
  - 实现 `MockAcpMcpConnection`（修复 send_to_client 签名，返回 Any，包装 connectionId/message）
  - 实现 `MockACPClient`（支持 send_request 和 mcp_message 双路径）
  - 实现 `MockACPClientWithoutSendRequest`（模拟缺少 send_request 的 client）
  - 实现 `MockFastmcpClientSession`（模拟 fastmcp ClientSession 响应）
- **Deliverables**：fixture 模块 + 使用文档
- **Review-Driven Changes**：
  - 使用 `anyio.create_memory_object_stream(0)` 模拟真实背压
  - `send_to_client` 使用 `AsyncMock` 模拟异步挂起

### Phase 2: Layer 1 单元测试（2 天）

- **Scope**：
  - `test_acp_mcp_transport.py`：U1.1-U1.11（新增 U1.7-U1.11）
  - `test_acp_mcp_manager.py`：U2.1-U2.6
  - `test_acp_agent_mcp_methods.py`：U3.1-U3.8（新增 U3.7-U3.8）
- **Deliverables**：6 个测试文件，全部通过
- **Review-Driven Changes**：
  - 新增 `test_fastmcp_client_accepts_transport`（真实 fastmcp Client 创建）
  - 新增 `test_transport_passes_session_kwargs`（验证 session_kwargs 透传）
  - 新增 `test_connect_client_without_send_request_raises`（验证 Bug #2）

### Phase 3: Layer 2 集成测试（2 天）

- **Scope**：
  - `test_acp_mcp_session_integration.py`：I1.1-I1.7（新增 I1.6-I1.7，重构 I1.2）
  - `test_mcp_client_acp_transport.py`：I2.1-I2.5（新增 I2.4-I2.5）
  - `test_mcp_resource_provider_acp.py`：I3.1-I3.4
- **Deliverables**：3 个测试文件，全部通过
- **Review-Driven Changes**：
  - 新增 2 个真实 fastmcp Client 集成测试（I2.4, I2.5）
  - I1.2 改为测试错误可观测性（非静默吞掉）
  - 新增 I1.6 测试 provider 初始化失败时的连接清理

### Phase 4: Layer 3 高级集成测试（1 天）

- **Scope**：
  - `test_acp_mcp_full_lifecycle.py`：HI.1-HI.6（新增 HI.5-HI.6）
- **Deliverables**：1 个测试文件，使用真实 fastmcp Client + anyio memory streams
- **Review-Driven Changes**：
  - 重命名为"高级集成测试"（非端到端）
  - 使用真实 fastmcp Client 验证完整生命周期
  - 新增并发消息无死锁测试（HI.6）
  - 新增清理顺序验证（HI.5）

### Phase 5: CI 集成与覆盖率验证（1 天）

- **Scope**：
  - 更新 pytest 配置，添加 marker：`@pytest.mark.acp_mcp`
  - 验证代码覆盖率 ≥ 90%
  - 确保现有测试无回归
- **Deliverables**：CI 通过，覆盖率报告
- **Review-Driven Changes**：
  - 确保 in-memory 集成测试在 CI 中稳定运行（监控 flaky）

---

## Open Questions

### Answered Questions

1. **fastmcp Client 的 async lifecycle 是否稳定？**
   - **Answer**：fastmcp Client 使用 background task + reentrant context manager，mock 测试难以完全模拟。应使用 2-3 个真实 fastmcp Client 的 in-memory 集成测试作为安全网（@oracle 建议）。
   - **Action**：Phase 4 使用真实 `fastmcp.Client(AcpMcpTransport(...))` 验证生命周期。

2. **`AcpMcpTransport` 的 `connect_session()` 是否需要支持 `read_timeout_seconds`？**
   - **Answer**：需要。fastmcp Client 会在 `session_kwargs` 中传递 timeout 参数，当前实现接受 `**session_kwargs` 但未透传给 `ClientSession` 是一个 silent bug（@oracle 发现）。
   - **Action**：添加 U1.7 `test_transport_passes_session_kwargs`，并在实现中修复透传。

3. **如何验证 `mcp/message` 的消息格式正确性？**
   - **Answer**：Mock 中做严格的 JSON-RPC schema validation 会过度复杂化。应通过真实 fastmcp Client 测试来隐式验证格式（因为 fastmcp 会验证 JSON-RPC）。
   - **Action**：在真实 Client 集成测试（I2.4, I2.5）中验证消息交换。

4. **并发场景：多个 session 同时连接同一 ACP server？**
   - **Answer**：当前设计是每个 session 独立连接，`acp_id` 可能相同，但 `connectionId` 是独立的。需要测试 `AcpMcpConnectionManager` 的并发创建行为。
   - **Action**：添加 U2.6 `test_concurrent_create` 和 HI.6 `test_concurrent_messages_no_deadlock`。

5. **Silent failure policy 是否正确？**
   - **Answer**：`ACPSession.initialize_mcp_servers()` 的 broad `except Exception` 是 design choice（MCP servers 是 best effort），但错误必须可观测（@Metis 指出）。
   - **Action**：I1.2 改为测试错误可观测性（`session.mcp_init_errors` 包含失败信息），而非验证静默吞掉。

---

## Decision Record

### Decision

**Status**: DRAFT → REVIEW INTEGRATED

**Date**: 2026-05-29

**Approvers**: yuchen.liu (author), Metis - Plan Consultant (reviewer), oracle (reviewer)

### Decision Summary

1. **Mock infrastructure fixed**: `MockAcpMcpConnection.send_to_client` now returns `Any` (not `dict`) and wraps messages with `connectionId` envelope, matching real `AcpMcpConnection` signature.
2. **Real fastmcp Client contract tests added**: At least 2 tests (U1.11, I2.4, I2.5) use actual `fastmcp.Client(AcpMcpTransport(...))` to catch API drift and interface mismatches.
3. **Error handling tests restructured**: I1.2 changed from "test silent swallowing" to "test error observability" (`session.mcp_init_errors` populated).
4. **Layer 3 renamed**: "End-to-End" → "High-level Integration" since it still uses in-memory streams, not cross-process communication.
5. **New test scenarios added**: `session_kwargs` passthrough (U1.7), `_set_auth` ValueError (U1.8), forwarder error handling (U1.10), client without `send_request` (U3.7), provider cleanup on failure (I1.6), cleanup ordering (HI.5), concurrency deadlock (HI.6).
6. **Mock realism improved**: `AsyncMock` with suspension for `send_to_client`, `anyio.create_memory_object_stream(0)` for backpressure simulation.

### Key Discussion Points

- **Metis**: Mock tests alone cannot catch fastmcp API drift (e.g., new abstract methods on `ClientTransport`). Need real Client instantiation tests.
- **oracle**: fastmmp's background task + reentrant context manager creates concurrency topology impossible to mock faithfully. In-memory tests with real Client are cheapest way to exercise this.
- **Both**: The bugs found during implementation (`ClientTransport` inheritance, double `initialize()`, missing `send_request`) were all contract/interface bugs, not logic bugs. Only real Client tests catch these.

### Conditions of Approval

- [x] Mock infrastructure matches real signatures
- [x] At least 2 tests use real fastmcp Client
- [x] Error handling tests verify observability, not silent swallowing
- [x] Cleanup ordering and concurrency scenarios covered
- [ ] Tests implemented and passing (pending)
- [ ] Code coverage ≥ 90% (pending)

### Dissenting Opinions

- None. Both reviewers approved mock-first strategy with real Client integration tests as safety net.

---

## References

### Related Documents

- [RFC-0033: MCP-over-ACP Transport](../implemented/RFC-0033-mcp-over-acp-transport.md)
- [RFC-0035: MCP-over-ACP Complete Connection Chain](../draft/RFC-0035-mcp-over-acp-complete-connection-chain.md)

### Key Files

| 文件 | 说明 |
|------|------|
| `tests/wolfharness_server/acp_server/test_acp_mcp_manager.py` | 现有 manager 测试 |
| `tests/wolfharness_server/acp_server/test_acp_mcp_transport.py` | 现有 transport 测试 |
| `tests/servers/acp_server/test_mcp_integration.py` | 现有 MCP 集成测试 |
| `src/wolfharness_server/acp_server/acp_mcp_transport.py` | Transport 实现 |
| `src/wolfharness_server/acp_server/acp_mcp_manager.py` | Manager 实现 |
| `src/wolfharness/mcp_server/client.py` | MCPClient |
| `src/wolfharness/resource_providers/mcp_provider.py` | MCPResourceProvider |

### External Resources

- [fastmcp ClientTransport Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [anyio MemoryObjectStream](https://anyio.readthedocs.io/en/stable/streams.html#memory-object-streams)
