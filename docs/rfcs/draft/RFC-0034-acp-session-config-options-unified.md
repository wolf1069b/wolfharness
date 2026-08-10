---
rfc_id: RFC-0034
title: "ACP Session Config Options 统一化：在 IDE 中透出模型选择与 Agent Role 切换"
status: DRAFT
author: yuchen.liu
reviewers: []
created: 2026-05-26
last_updated: 2026-05-27
decision_date:
related_rfcs:
  - RFC-0027 (ACP Subagent Zed 兼容性)
  - RFC-0031 (ACP Per-Session Agent Isolation)
  - RFC-0032 (ACP Slash Commands Protocol Compliance)
  - ACP-RFD-custom-llm-endpoint (Configurable LLM Providers, PR #648, MERGED)
---

# RFC-0034: ACP Session Config Options 统一化

## 概述

本 RFC 提出统一 AgentPool ACP Server 的 Session Config Options 透出逻辑，使 Zed、Cursor 等 ACP 兼容 IDE 能够在 UI 中：

1. **选择模型（Model）**：列出 manifest `model_variants` 和 tokonomics 发现的可用模型
2. **切换 Agent Role**：将 `agent_pool.all_agents` 中的多个 agent 作为可选 role 透出
3. **切换 Tool Permission**：already/never/per_tool 工具确认策略
4. **其他 config options**：thought_level、personality 等 agent 暴露的扩展 category

当前 AgentPool 仅通过 ACP 旧版 `SessionModeState`（`session/set_mode`）透出 tool permission，model 和 agent role 均未通过标准 ACP 机制透出。与此同时，`serve-opencode` 通过 REST API 自有机制透出了 model list，但两套透出逻辑相互独立、数据来源不同，存在显著差异。

本 RFC 的目标是对齐两套协议的数据来源，并让 ACP 通道能完整透出 model + agent role + 所有 mode categories，从而实现"在 IDE 中选择模型和 agent"的完整体验。

## 背景与上下文

### 当前系统状态

#### ACP 协议的 Mode/Config Options 层

ACP 协议提供两套机制透出可切换配置：

| 机制 | 协议方法 | 状态 | 描述 |
|------|----------|------|------|
| `SessionModeState` | `session/set_mode` | 正式规范 | 单一 mode 选择器，一次只能有一套 modes |
| `SessionConfigOption[]` | `session/set_config_option` | UNSTABLE | 多维度 config option，每个 category 独立选择器 |
| `ProviderInfo[]` | `providers/list` / `providers/set` / `providers/disable` | UNSTABLE（PR #648 MERGED） | LLM provider 路由配置，客户端可覆盖 agent 的 LLM 请求目标 |

AgentPool 当前实现：
- `NewSessionResponse.modes` → 透出 tool permission（`get_session_mode_state()`，只取 `id=="mode"` 的 category）
- `NewSessionResponse.config_options` → 透出所有 `agent.get_modes()` 返回的 categories（`get_session_config_options()`）
- `NewSessionResponse.models` → 透出 tokonomics 发现的模型 + manifest model_variants（`get_session_model_state()`）

#### wolfharness 内部的 `get_modes()` 体系

各 agent 类型的 `get_modes()` 返回如下 categories：

| Agent 类型 | Categories | 说明 |
|------------|------------|------|
| `NativeAgent` | `mode`（tool permission）+ `model` | model 来自 tokonomics |
| `ClaudeCodeAgent` | `mode` + `model` + `thought_level` | Claude 的 extended thinking |
| `CodexAgent` | `mode`（approval policy）+ `model`（effort）+ `thought_level` | Codex 特有模式 |
| `ACPAgent`（嵌套） | passthrough from remote server | 透传远端 config_options |
| `AGUIAgent` | `[]`（无） | 不支持 mode 切换 |

#### serve-opencode 的 Model List 逻辑

`opencode_server/routes/config_routes.py` 的 `_build_providers_with_fallback()` 采用如下优先级：

1. **configured variants**（manifest `model_variants`）→ 解析 provider 信息，按 provider 分组
2. **tokonomics 全局发现**（`get_all_models()`，7 天缓存）→ 全量模型
3. **agent-specific modes**（仅 Codex/ClaudeCode 的 `thought_level`）→ fallback
4. 空列表 + warning

#### ACP 的 Model List 逻辑

`acp_server/acp_agent.py` 的 `get_session_model_state()` 采用如下优先级：

1. **tokonomics first**：`agent.get_available_models()` → 逐 agent 调用，每次请求都可能触发网络
2. **configured override**：manifest `model_variants` 的 key（仅取名称，不解析 provider）

#### Agent Role 的透出现状

- `acp_server/converters.py` 存在 `agent_to_mode()` 辅助函数，可将 agent 转换为 `SessionMode`，但**从未被调用**
- `get_session_mode_state()` 只取 `id=="mode"` 的 category（tool permission），不包含 agent role
- `agent_routes.py` 的 `GET /agent` 端点可列出所有 agents，供 opencode TUI 使用，但 ACP 协议无对应端点

#### Zed IDE 对 ACP Config Options 的渲染行为（源码级分析）

通过对 Zed 源码（commit: 当前 HEAD）的调研，确认以下关键行为：

**1. Config Options 渲染机制**（`crates/agent_ui/src/config_options.rs`）

Zed 的 `ConfigOptionsView` 会将**所有** `SessionConfigOption` 渲染为独立的下拉按钮（`ConfigOptionSelector`）：

```rust
// ConfigOptionsView::build_selectors() 为每个 option 创建独立 selector
config_options
    .config_options()
    .into_iter()
    .map(|option| ConfigOptionSelector::new(config_options, option.id.clone(), ...))
    .collect()
```

这意味着**无论 category 是什么，所有 config options 都会在 UI 中可见且可点击**。

**2. Category 键盘快捷键的限制**（`first_config_option_id()`）

Zed 为 `Mode` 和 `Model` category 提供了键盘快捷键（`ToggleProfileSelector`、`ToggleModelSelector`、`CycleModeSelector`、`CycleFavoriteModels`），但这些快捷键通过 `first_config_option_id(category)` 查找目标 option：

```rust
fn first_config_option_id(&self, category: SessionConfigOptionCategory) -> Option<SessionConfigId> {
    self.config_options
        .config_options()
        .into_iter()
        .find(|option| option.category.as_ref() == Some(&category))  // ← 仅返回第一个匹配
        .map(|option| option.id)
}
```

**关键限制**：如果存在**多个相同 category** 的 config options，`first_config_option_id()` 只会返回**第一个匹配项**。后续相同 category 的 option **无法通过键盘快捷键触发**，但**仍可通过鼠标点击访问**。

**3. Zed Profile 切换的独立性**（`crates/agent_ui/src/profile_selector.rs`）

Zed 内置的 `ProfileSelector` 是完全独立的本地 UI 组件：
- 数据来源：`AgentSettings.profiles`（本地 settings.json）
- 与 ACP `config_options` **无任何关联**
- 当 ACP agent 未提供 `category="mode"` 的 config option 时，`ToggleProfileSelector` 快捷键会回退到本地 `ProfileSelector`

**4. `providers/*` 协议支持状态**（`crates/agent_servers/src/acp.rs`）

Zed 的 `AcpConnection::connect_client_future()` 中注册的所有 handler **不包含** `providers/list`、`providers/set`、`providers/disable`。当前 Zed 版本**完全不支持** ACP Configurable LLM Providers 协议。

**对 RFC 的影响**：
- `agent_role` 使用 `category="other"` 时，**UI 渲染正常**，所有 agent 都会显示为可点击按钮
- 若 agent 同时存在其他 `category="other"` 的 option（如自定义扩展），**键盘快捷键可能指向错误的 option**
- Zed 用户目前无法通过 UI 配置 provider 路由（Phase 0 的实现暂无 Zed UI 入口，但为协议合规和未来版本兼容所必需）

#### ACP Configurable LLM Providers 协议（PR #648，已 MERGED）

ACP 协议新增 `providers/*` 方法族，允许客户端发现和覆盖 agent 的 LLM 请求路由：

| 方法 | 作用 | 时机 |
|------|------|------|
| `providers/list` | 返回 agent 暴露的 LLM provider 列表（id、协议类型、当前路由、是否 required） | `initialize` 后、`session/new` 前 |
| `providers/set` | 覆盖某个 provider 的路由目标（apiType、baseUrl、headers） | `session/new` 前 |
| `providers/disable` | 禁用某个 provider（`current: null`） | `session/new` 前 |

协议关键语义：
- 客户端通过 `agentCapabilities.providers: true` 判断 agent 是否支持 provider 配置
- `providers/set` 替换 provider 的完整配置（apiType + baseUrl + headers），非增量更新
- `providers/list` 不返回 headers（可能含 secrets），只返回非敏感路由摘要
- Provider 配置为 process-scoped，不应持久化到磁盘
- Agent MAY 不将变更应用到已运行的 session，但 SHOULD 应用到后续创建的 session

**AgentPool 当前状态**：❌ 完全未实现。`AgentCapabilities` 中无 `providers` 字段，无 `providers/*` 处理器。

**与 model 选择的关系**：`providers/*` 运行在**传输层**（"LLM 请求发到哪"），而 `session/set_config_option(model=...)` 运行在**应用层**（"用什么模型"）。两者正交但存在交互：
- 被禁用的 provider 下的模型不应出现在 `SessionModelState` 中
- 客户端通过 `providers/set` 覆盖 endpoint 后，agent 的模型请求应路由到新 endpoint
- agent 的 `model_variants` 中的 `AnyModelConfig` 隐含了 provider 信息，是 `providers/list` 的数据来源

### 问题陈述

#### GAP 1（P0）：Agent Role 完全未透出到 ACP

Zed 等 IDE 通过 `SessionModeState` 或 `config_options` 中的 `category="mode"` 来展示可选的 agent role（在 Zed 中称为 "profile" 或 "agent"）。AgentPool 的 `agent_pool.all_agents` 包含所有已配置的 agent，但 ACP Server 未将其透出。

**影响**：用户无法在 Zed 等 IDE 的 UI 中切换 agent role（如从 `diag-agent` 切换到 `plan-agent`）。

#### GAP 2（P1）：ACP 和 OpenCode 的 Model List 数据来源不一致

| 维度 | ACP | OpenCode |
|------|-----|---------|
| 数据优先级 | tokonomics first，configured override | configured first，tokonomics fallback |
| tokonomics 来源 | `agent.get_available_models()`（per-agent，每次调用） | `get_all_models()`（全局，7 天缓存） |
| model_variants | 仅取 key 名，无 provider 解析 | 解析 `AnyModelConfig`，提取 provider |
| 结构 | 扁平 `SessionModelState` | 按 provider 分组的 `Provider[]` |

当用户在两个协议下使用同一配置时，看到的模型列表不同。

#### GAP 3（P1）：`/mode` 路由硬编码，未动态透出 agent modes

`opencode_server/routes/config_routes.py` 的 `GET /mode` 返回硬编码的 `[Mode(name="default")]`，没有调用 `agent.get_modes()` 动态透出 `mode` category（如 permission modes）。

#### GAP 4（P2）：ACP `get_session_mode_state()` 过滤过严

当前 `get_session_mode_state()` 只提取 `id=="mode"` 的 category 填入旧版 `SessionModeState`。其余 categories（model、thought_level 等）通过 `config_options` 透出，但旧版 `modes` 字段只有 tool permission，导致仅支持旧版 `session/set_mode` 的客户端（如部分旧版 Zed）看不到 model 选择器。

#### GAP 5（P0）：ACP Configurable LLM Providers 完全未实现

ACP 协议 PR #648（已 MERGED）定义了 `providers/list`、`providers/set`、`providers/disable` 三个方法，允许客户端发现 agent 的 LLM provider 并覆盖路由目标。AgentPool 当前完全未实现：

- `AgentCapabilities` 中无 `providers` 字段，客户端无法发现此能力
- 无 `providers/list` 处理器，客户端无法获取 agent 的 provider 列表
- 无 `providers/set` 处理器，客户端无法覆盖 provider 的路由目标（apiType、baseUrl、headers）
- 无 `providers/disable` 处理器，客户端无法禁用某个 provider

**影响**：
1. 企业用户无法通过 ACP 将 agent 的 LLM 请求路由到内部网关（合规、日志、成本控制）
2. 自托管用户无法将 agent 请求重定向到本地 vLLM / Ollama 等服务
3. 客户端无法在 UI 中显示 agent 的 LLM provider 状态
4. 更关键的是：**model 列表与 provider 状态耦合** — 如果不实现 `providers/*`，Phase 1 的 `build_model_state_for_acp()` 无法感知 provider 的启用/禁用状态，可能展示实际不可用的模型

### 相关工作

| RFC | 状态 | 关系 |
|-----|------|------|
| RFC-0027 | DRAFT | ACP Subagent Zed 兼容性，提出 `display_mode=zed` |
| RFC-0031 | IMPLEMENTED | ACP Per-Session Agent Isolation，每 session 独立 agent 实例 |
| RFC-0032 | DRAFT | ACP Slash Commands 合规性 |
| ACP PR #648 | MERGED | Configurable LLM Providers，定义 `providers/*` 方法族 |

### 术语表

| 术语 | 定义 |
|------|------|
| `SessionModeState` | ACP 旧版单一 mode 选择器，通过 `session/set_mode` 切换 |
| `SessionConfigOption` | ACP 新版多维度配置选项，通过 `session/set_config_option` 切换 |
| `SessionConfigOptionCategory` | Config option 的语义分类：`"mode"` / `"model"` / `"thought_level"` / `"other"` |
| `model_variants` | manifest YAML 中的模型变体配置，键为变体名，值为 `AnyModelConfig` |
| `get_modes()` | `BaseAgent` 抽象方法，返回所有可切换的 `ModeCategory` 列表 |
| agent role | 多 agent 配置中可切换的 agent 身份，如 `diag-agent`、`plan-agent` |
| `config_options` | `NewSessionResponse` / `LoadSessionResponse` 中的 UNSTABLE 扩展字段 |

## 目标与非目标

### 目标

| ID | 目标 | 优先级 |
|----|------|--------|
| G1 | ACP `config_options` 中透出 agent role（`agent_pool.all_agents`），支持 `session/set_config_option` 切换 | P0 |
| G2 | 统一 ACP 和 OpenCode 的 model list 数据来源（configured variants 优先，tokonomics 兜底） | P1 |
| G3 | ACP `config_options` 中的 model category 使用与 OpenCode 一致的优先级逻辑 | P1 |
| G4 | `serve-opencode` 的 `GET /mode` 路由动态透出 agent modes（调用 `agent.get_modes()`） | P1 |
| G5 | 新增 `agent_role` config option category，切换时实际替换 session 的 agent 实例 | P0 |
| G6 | 保持向后兼容：旧版 `SessionModeState.modes` 继续透出 tool permission | P0 |
| G7 | 实现 ACP `providers/*` 协议方法（`providers/list`、`providers/set`、`providers/disable`），从 `model_variants` 派生 `ProviderInfo[]` | P0 |

### 非目标

| ID | 非目标 | 理由 |
|----|--------|------|
| NG1 | 修改 ACP 协议 schema 或 `SessionConfigOptionCategory` 枚举 | 使用现有 `"other"` category 即可承载 agent role；Zed 源码确认该 category 可被正确渲染 |
| NG2 | 将 OpenCode 的 `/provider` REST API 替换为 ACP 协议 | 两套协议并行存在，各自服务不同客户端 |
| NG3 | 实现 agent role 的持久化（跨 session 记忆上次选择） | 不在本 RFC 范围 |
| NG4 | 支持动态新增/删除 agent（运行时 reload manifest） | 涉及 agent pool 热更新，属于单独 RFC |
| NG5 | 跨 session 同步 config option 变更 | 每个 session 独立配置，不需要全局同步 |
| NG6 | 实现 `providers/*` 对已运行 session 的实时路由切换 | 协议规定 Agent MAY 不应用到已运行 session，本 RFC 遵循保守策略：`providers/set` 仅影响后续新建 session |
| NG7 | 解决 Zed 中多个同 category config option 的键盘快捷键冲突 | Zed `first_config_option_id()` 的设计限制；agent 实现应避免在启用 agent_role 时同时使用 `category="other"` 的其他 option |

## 评估标准

| 标准 | 权重 | 描述 | 最低阈值 |
|------|------|------|----------|
| IDE 兼容性 | 关键 | Zed 等 IDE 能显示 agent role 和模型选择器 | config_options 能被客户端正确解析 |
| 数据一致性 | 高 | ACP 和 OpenCode 透出的模型列表相同 | 相同配置下两者列表集合一致 |
| 协议合规性 | 高 | 实现已 MERGED 的 ACP 协议特性（`providers/*`） | 通过 ACP 官方 conformance test |
| 向后兼容性 | 关键 | 旧版 SessionModeState.modes 不变 | 现有 tool permission 切换功能正常 |
| 代码复用 | 高 | ACP 和 OpenCode 复用共享的 model 构建逻辑 | 提取到 `shared/model_utils.py` |
| 实施复杂度 | 中 | 修改范围可控 | 不超过 7 个核心文件 |
| 可测试性 | 中 | 可单元测试的 config option 构建逻辑 | 覆盖率 ≥ 80% |

## 方案分析

### 选项 1：最小修复 — 仅补充 Agent Role 到 config_options

**描述**：仅在 `get_session_config_options()` 的输出中追加一个 `agent_role` config option（枚举 `agent_pool.all_agents`），不修改 model list 逻辑，不统一数据来源。

**实施范围**：
- 在 `acp_server/acp_agent.py` 中新增 `get_agent_role_config_option()` 函数
- 修改 `get_session_config_options()` 追加 agent role option
- 修改 `AgentPoolACPAgent.set_session_config_option()` 处理 `agent_role` category，切换 session agent

**优势**：
- 修改范围最小，仅涉及 `acp_server/acp_agent.py` 一个文件
- 快速解决 G1（P0 问题）
- 不影响 model list 现有逻辑，风险低

**劣势**：
- 未解决 G2/G3（ACP 与 OpenCode model list 不一致）
- 未解决 G4（OpenCode `/mode` 路由硬编码）
- 技术债务继续积累，后续需二次重构

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| IDE 兼容性 | 3/5 | Agent role 透出，但 model list 仍与 OpenCode 不一致 |
| 数据一致性 | 1/5 | 未解决 ACP/OpenCode model list 差异 |
| 向后兼容性 | 5/5 | 不影响现有 modes 逻辑 |
| 代码复用 | 2/5 | 未提取共享逻辑 |
| 实施复杂度 | 5/5 | 改动最小 |
| 可测试性 | 4/5 | 逻辑简单，易于测试 |

**工作量**：低（~80 行）

---

### 选项 2：统一数据来源 + 完整 Config Options 对齐

**描述**：统一 ACP 和 OpenCode 的 model list 构建逻辑（提取到共享模块），同时补充 agent role config option，修复 OpenCode `/mode` 路由。

**实施范围**：

**Phase 1：共享 Model List 逻辑**
- 在 `shared/model_utils.py` 中提取统一的 `build_model_config_options()` 函数
  - configured variants 优先（解析 provider，与 OpenCode 对齐）
  - tokonomics 兜底（使用 `agent.get_available_models()` 而非全局 `get_all_models()`，保持 per-agent 语义）
- 修改 `acp_server/acp_agent.py` 的 `get_session_model_state()` 使用共享逻辑
- 保持 ACP 的扁平 `SessionModelState` 结构（不改为 provider 分组，协议格式不变）

**Phase 2：Agent Role Config Option**
- 新增 `get_agent_role_config_option()` 函数，枚举 `agent_pool.all_agents`
- 将 agent role 作为 `SessionConfigOption`（`id="agent_role"`，`category="other"`）
- 修改 `set_session_config_option()` 处理 `agent_role`：调用 `AgentPoolACPAgent._swap_session_agent()` 切换 session agent 实例

**Phase 3：OpenCode /mode 路由修复**
- 修改 `config_routes.py` 的 `GET /mode`，调用 `agent.get_modes()` 动态透出 `mode` category 的 available modes
- 将 `ModeCategory.available_modes` 转换为 `list[Mode]` 返回

**优势**：
- 彻底解决 ACP 与 OpenCode 的 model list 不一致问题
- Agent role 透出完整，切换逻辑闭环
- 代码复用性高，未来新增协议支持时可复用共享逻辑
- `/mode` 路由不再硬编码，动态反映 agent 配置

**劣势**：
- 修改范围较大（5 个文件），需要更充分的测试
- Phase 1 中统一 model list 来源可能带来细微行为差异（tokonomics per-agent vs 全局）
- OpenCode `/mode` 路由修复是纯增量，不涉及破坏性变更，但需验证所有 agent 类型

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| IDE 兼容性 | 5/5 | 完整透出 model + agent role，两协议一致 |
| 数据一致性 | 5/5 | 共享 model 构建逻辑，来源统一 |
| 向后兼容性 | 5/5 | 旧版 modes 不变，config_options 为增量 |
| 代码复用 | 5/5 | shared/model_utils.py 统一模型构建 |
| 实施复杂度 | 3/5 | 三阶段，每阶段独立可交付 |
| 可测试性 | 4/5 | 共享逻辑独立可测试 |

**工作量**：中（~280 行）

---

### 选项 3：全面重构 — 引入统一 AgentCapabilities 注册机制

**描述**：在选项 2 基础上，引入统一的 `AgentCapabilities` 注册中心，ACP/OpenCode/AGUI 三套协议均从同一来源构建 model list、mode list 和 agent role list。

**实施范围**：
- 新增 `AgentCapabilitiesProvider` 抽象层，统一各协议的 capabilities 查询
- 三套协议的 model/mode/role 查询均委托给 `AgentCapabilitiesProvider`
- 引入能力缓存层，避免重复调用 tokonomics

**优势**：
- 单一真相来源，三套协议数据完全一致
- 可扩展性强，未来新协议直接复用
- 缓存层解决 tokonomics 性能问题

**劣势**：
- 工程量大，抽象层设计复杂
- 现有三套协议各有历史包袱，完全统一风险高
- 本 RFC 范围之外，需独立 RFC 立项

**评估**：

| 标准 | 评分 | 说明 |
|------|------|------|
| IDE 兼容性 | 5/5 | 完整且一致 |
| 数据一致性 | 5/5 | 单一真相来源 |
| 向后兼容性 | 3/5 | 重构风险较高 |
| 代码复用 | 5/5 | 终极复用 |
| 实施复杂度 | 1/5 | 工程量极大 |
| 可测试性 | 4/5 | 抽象层可独立测试 |

**工作量**：高（~800 行以上）

---

## 推荐

**推荐选项 2：统一数据来源 + 完整 Config Options 对齐**，分三阶段实施。

推荐理由：
1. 选项 1 仅解决 P0 问题，遗留 model list 不一致，用户在 Zed 和 OpenCode TUI 看到不同模型列表，体验差
2. 选项 2 通过提取共享逻辑，以较低成本同时解决 G1-G4，技术债务清零
3. 选项 3 范围过大，引入不必要的架构复杂度，超出本 RFC 目标
4. 三阶段设计使每阶段独立可交付，降低集成风险

**接受的权衡**：
- ACP 侧 tokonomics 来源保持 per-agent（`agent.get_available_models()`），与 OpenCode 的全局 `get_all_models()` 不完全相同。这是有意为之：ACP 是 per-session 协议，per-agent 语义更准确；OpenCode 是全局 REST API，全局缓存更合理。
- model list 的 **结构**（ACP：扁平 `SessionModelState`；OpenCode：provider 分组）保持不变，统一的是**数据来源优先级**，而非数据结构。

## 技术设计

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    当前状态（存在差异）                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ACP Server                      OpenCode Server                │
│  ──────────────────               ────────────────────────────  │
│  get_session_model_state()        _build_providers_with_fallback │
│    → tokonomics first             → configured first            │
│    → manifest.model_variants      → tokonomics fallback         │
│      （仅取 key 名）              → agent-specific modes         │
│                                                                  │
│  get_session_config_options()     GET /mode                     │
│    → get_modes() (完整)           → 硬编码 [default] ❌          │
│                                                                  │
│  get_session_mode_state()         GET /agent                    │
│    → tool permission only         → all_agents (正确)           │
│                                                                  │
│  providers/*: 未实现 ❌            （无 ACP 协议，不适用）         │
│  Agent Role: 未透出 ❌                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    目标状态（对齐后）                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ 传输层（session/new 前配置） ─────────────────────────────┐  │
│  │                                                              │  │
│  │  providers/list ✅                                           │  │
│  │    → 从 model_variants 派生 ProviderInfo[]                  │  │
│  │    → 每个 provider 含 id、supported LlmProtocol、           │  │
│  │      required、current（apiType + baseUrl）                  │  │
│  │                                                              │  │
│  │  providers/set ✅                                            │  │
│  │    → 覆盖 provider 路由表（apiType + baseUrl + headers）     │  │
│  │    → 仅影响后续新建 session                                  │  │
│  │                                                              │  │
│  │  providers/disable ✅                                        │  │
│  │    → 设置 provider.current = null                            │  │
│  │    → 被禁用 provider 下的模型从 model list 中移除            │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ 应用层（session 生命周期内） ─────────────────────────────┐  │
│  │                                                              │  │
│  │  shared/model_utils.py                                       │  │
│  │    build_model_state_for_acp()                               │  │
│  │      → configured variants first                             │  │
│  │      → tokonomics fallback                                   │  │
│  │      → 过滤 disabled provider 下的模型 ✅                    │  │
│  │                                                              │  │
│  │  ACP Server                     OpenCode Server              │  │
│  │  ───────────────                ───────────────────────────  │  │
│  │  get_session_model_state()      _build_providers_with_fall  │  │
│  │    ↑ 使用共享逻辑               ↑ 保持不变                   │  │
│  │    → configured first ✅        → configured first ✅        │  │
│  │    → 过滤 disabled ✅                                         │  │
│  │                                                              │  │
│  │  get_session_config_options()  GET /mode                     │  │
│  │    → get_modes() + agent_role  → agent.get_modes() 动态 ✅  │  │
│  │                                                              │  │
│  │  Agent Role: config_options ✅                               │  │
│  │    id="agent_role", category="other"                         │  │
│  │    options=[all_agents as ModeInfo]                          │  │
│  │    切换时: _swap_session_agent()                              │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ 运行时路由 ───────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │  Provider 路由表（wolfharness 内存维护）                       │  │
│  │    默认：manifest model_variants → provider → default URL   │  │
│  │    覆盖：providers/set → provider → client URL + headers    │  │
│  │    禁用：providers/disable → provider → null（不可路由）     │  │
│  │                                                              │  │
│  │  agent 请求模型 X → 查找 X 的 provider P                    │  │
│  │    → 从路由表获取 P 的当前 endpoint                         │  │
│  │    → providers/set override 优先，manifest default 兜底      │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 0：ACP Configurable LLM Providers 实现

#### 数据模型

从 ACP PR #648 协议定义映射到 wolfharness 的类型：

```python
# acp/schema/providers.py（新增文件）

LlmProtocol = Literal["anthropic", "openai", "azure", "vertex", "bedrock"] | str


class ProviderCurrentConfig(BaseModel):
    """Provider 当前有效路由配置（非敏感信息）。"""

    api_type: LlmProtocol = Field(alias="apiType")
    base_url: str = Field(alias="baseUrl")


class ProviderInfo(BaseModel):
    """可配置的 LLM Provider 信息。"""

    id: str
    supported: list[LlmProtocol]
    required: bool
    current: ProviderCurrentConfig | None = None  # None = 禁用
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ProvidersListRequest(BaseModel):
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ProvidersListResponse(BaseModel):
    providers: list[ProviderInfo]
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ProvidersSetRequest(BaseModel):
    id: str
    api_type: LlmProtocol = Field(alias="apiType")
    base_url: str = Field(alias="baseUrl")
    headers: dict[str, str]
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ProvidersSetResponse(BaseModel):
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ProvidersDisableRequest(BaseModel):
    id: str
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ProvidersDisableResponse(BaseModel):
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
```

#### Provider 路由表

wolfharness 内存维护一个 provider 路由表，存储 `providers/set` 的覆盖配置：

```python
# acp_server/provider_router.py（新增文件）

class ProviderRouter:
    """LLM Provider 路由表，管理客户端覆盖的 provider 配置。

    路由优先级：
    1. providers/set 覆盖的配置（客户端声明）
    2. manifest model_variants 中的默认配置（agent 配置）

    生命周期：process-scoped，不持久化。
    """

    def __init__(self, manifest: AgentsManifest | None = None) -> None:
        self._overrides: dict[str, ProviderOverride] = {}
        self._disabled: set[str] = set()
        self._manifest = manifest

    def get_provider_info_list(self) -> list[ProviderInfo]:
        """生成 providers/list 响应数据，从 model_variants 派生。"""
        providers = self._derive_providers_from_manifest()
        result = []
        for pid, info in providers.items():
            if pid in self._overrides:
                info["current"] = self._overrides[pid].to_current_config()
            if pid in self._disabled:
                info["current"] = None
            result.append(ProviderInfo(**info))
        return result

    def set_provider(self, id: str, api_type: LlmProtocol,
                     base_url: str, headers: dict[str, str]) -> None:
        """处理 providers/set 请求。"""
        self._overrides[id] = ProviderOverride(
            api_type=api_type, base_url=base_url, headers=headers
        )
        self._disabled.discard(id)  # set 隐含 re-enable

    def disable_provider(self, id: str) -> None:
        """处理 providers/disable 请求。"""
        self._disabled.add(id)

    def is_provider_disabled(self, provider_id: str) -> bool:
        """查询 provider 是否被禁用。供 model list 构建时过滤。"""
        return provider_id in self._disabled

    def get_provider_headers(self, provider_id: str) -> dict[str, str]:
        """获取 provider 的客户端覆盖 headers。供运行时路由使用。"""
        if provider_id in self._overrides:
            return self._overrides[provider_id].headers
        return {}

    def get_provider_base_url(self, provider_id: str) -> str | None:
        """获取 provider 的覆盖 base_url，无覆盖则返回 None。"""
        if provider_id in self._overrides:
            return self._overrides[provider_id].base_url
        return None

    def _derive_providers_from_manifest(self) -> dict[str, dict[str, Any]]:
        """从 manifest.model_variants 派生 ProviderInfo 列表。

        复用 shared/model_utils._extract_provider() 提取 provider 名称，
        然后按 provider 分组，推断 supported LlmProtocol 和 required 标志。
        """
        providers: dict[str, dict[str, Any]] = {}
        if not self._manifest or not self._manifest.model_variants:
            return providers

        for name, config in self._manifest.model_variants.items():
            provider_name = _extract_provider(config)
            if provider_name not in providers:
                api_type = self._infer_llm_protocol(provider_name)
                providers[provider_name] = {
                    "id": provider_name,
                    "supported": [api_type],
                    "required": True,  # manifest 中配置的 provider 默认 required
                    "current": ProviderCurrentConfig(
                        api_type=api_type,
                        base_url=self._get_default_base_url(provider_name),
                    ),
                }
        return providers

    @staticmethod
    def _infer_llm_protocol(provider_name: str) -> LlmProtocol:
        """从 provider 名称推断 LlmProtocol 类型。"""
        mapping = {
            "anthropic": "anthropic",
            "openai": "openai",
            "azure": "azure",
            "bedrock": "bedrock",
            "vertex": "vertex",
            "google": "vertex",   # google → vertex 协议
            "deepseek": "openai", # deepseek 兼容 openai 协议
            "groq": "openai",
            "openrouter": "openai",
        }
        return mapping.get(provider_name, "openai")  # 未知 provider 默认 openai 兼容

    @staticmethod
    def _get_default_base_url(provider_name: str) -> str:
        """获取 provider 的默认 base URL。"""
        # 从 PROVIDER_INFO 或环境变量推断
        ...
```

#### 修改 `AgentCapabilities`

```python
# acp/schema/capabilities.py
class AgentCapabilities(AnnotatedObject):
    # ... 现有字段 ...
    providers: bool | None = False
    """Whether the agent supports providers/list, providers/set, providers/disable."""
```

#### 修改 ACP Server 请求分发

```python
# acp_server/acp_agent.py — 在 AgentPoolACPAgent 中添加 handlers

async def handle_providers_list(
    self, params: ProvidersListRequest
) -> ProvidersListResponse:
    return ProvidersListResponse(
        providers=self._provider_router.get_provider_info_list()
    )

async def handle_providers_set(
    self, params: ProvidersSetRequest
) -> ProvidersSetResponse:
    # 验证 id 存在于当前 providers 中
    known_ids = {p.id for p in self._provider_router.get_provider_info_list()}
    if params.id not in known_ids:
        raise JsonRpcError(code=-32602, message=f"Unknown provider: {params.id}")

    self._provider_router.set_provider(
        id=params.id,
        api_type=params.api_type,
        base_url=params.base_url,
        headers=params.headers,
    )
    return ProvidersSetResponse()

async def handle_providers_disable(
    self, params: ProvidersDisableRequest
) -> ProvidersDisableResponse:
    # 验证不是 required provider
    known = {p.id: p for p in self._provider_router.get_provider_info_list()}
    if params.id in known and known[params.id].required:
        raise JsonRpcError(code=-32602, message=f"Cannot disable required provider: {params.id}")

    self._provider_router.disable_provider(params.id)
    return ProvidersDisableResponse()
```

#### 修改 `initialize` 响应

```python
# 返回 AgentCapabilities(providers=True)
agent_capabilities = AgentCapabilities(
    # ... 现有字段 ...
    providers=True,
)
```

### Phase 1：共享 Model List 逻辑

#### `shared/model_utils.py` 新增函数

```python
async def build_model_state_for_acp(
    agent: BaseAgent,
    manifest: AgentsManifest | None = None,
    provider_router: ProviderRouter | None = None,
) -> SessionModelState | None:
    """构建 ACP SessionModelState，与 OpenCode 对齐优先级逻辑。

    优先级：
    1. manifest.model_variants（configured variants，优先）
    2. agent.get_available_models()（tokonomics，兜底）

    过滤：
    - 若 provider_router 非空，被禁用 provider 下的模型不列入列表

    Args:
        agent: 当前 session 的 agent 实例
        manifest: 可选的 AgentsManifest 配置
        provider_router: 可选的 ProviderRouter 实例，用于感知 provider 禁用状态

    Returns:
        SessionModelState，若无可用模型则返回 None
    """
    from acp.schema import ModelInfo as ACPModelInfo, SessionModelState

    # Step 1: configured variants（与 OpenCode 对齐：configured first）
    configured: dict[str, ACPModelInfo] = {}
    configured_providers: dict[str, str] = {}  # model_name → provider_id
    if manifest and manifest.model_variants:
        for name, config in manifest.model_variants.items():
            configured[name] = ACPModelInfo(model_id=name, name=name)
            configured_providers[name] = _extract_provider(config)

    # Step 2: tokonomics fallback（仅当 configured 为空时调用）
    toko_models: dict[str, ACPModelInfo] = {}
    toko_providers: dict[str, str] = {}  # model_name → provider_id
    if not configured:
        try:
            raw = await agent.get_available_models()
            for m in raw or []:
                mid = m.id_override or m.id
                toko_models[mid] = ACPModelInfo(
                    model_id=mid,
                    name=m.name,
                    description=m.description or "",
                )
                toko_providers[mid] = m.provider
        except Exception:
            logger.exception("Failed to get available models from agent")

    all_models = configured if configured else toko_models
    model_providers = configured_providers if configured else toko_providers

    # Step 3: 过滤被禁用 provider 下的模型
    if provider_router:
        disabled_models = {
            name for name, pid in model_providers.items()
            if provider_router.is_provider_disabled(pid)
        }
        all_models = {k: v for k, v in all_models.items() if k not in disabled_models}

    if not all_models:
        return None

    acp_list = list(all_models.values())
    all_ids = [m.model_id for m in acp_list]
    current = agent.model_name
    if current and current not in all_ids:
        acp_list.insert(0, ACPModelInfo(
            model_id=current, name=current, description="Currently configured model"
        ))
        current_id = current
    else:
        current_id = current if current in all_ids else all_ids[0]

    return SessionModelState(available_models=acp_list, current_model_id=current_id)
```

#### 修改 `acp_server/acp_agent.py`

将 `get_session_model_state()` 中的逻辑替换为调用 `build_model_state_for_acp()`：

```python
# 修改前（acp_agent.py:70-142）：tokonomics first，configured override
async def get_session_model_state(agent: BaseAgent) -> SessionModelState | None:
    toko_models = await agent.get_available_models()          # tokonomics first
    configured = [ACPModelInfo(model_id=name, ...) for name in manifest.model_variants]
    ...

# 修改后：委托给共享逻辑，传入 provider_router 感知 provider 状态
async def get_session_model_state(
    agent: BaseAgent,
    provider_router: ProviderRouter | None = None,
) -> SessionModelState | None:
    from wolfharness_server.shared.model_utils import build_model_state_for_acp
    manifest = getattr(getattr(agent, "agent_pool", None), "manifest", None)
    return await build_model_state_for_acp(agent, manifest, provider_router)
```

### Phase 2：Agent Role Config Option

#### 数据模型

Agent role 作为一个 `SessionConfigOption` 透出：

```python
# acp_server/acp_agent.py 中新增
async def get_agent_role_config_option(
    agent: BaseAgent,
) -> SessionConfigOption | None:
    """构建 agent role config option。

    将 agent_pool.all_agents 作为可选 role 枚举。
    若 pool 只有一个 agent，或 pool 不可访问，返回 None。

    Returns:
        SessionConfigOption（id="agent_role"，category="other"），
        若 agent 数量 <= 1 则返回 None（无需透出单一 agent）。
    """
    from acp.schema import SessionConfigOption, SessionConfigSelectOption

    pool = getattr(agent, "agent_pool", None)
    if pool is None:
        return None

    all_agents = getattr(pool, "all_agents", {})
    if len(all_agents) <= 1:
        return None  # 单一 agent 无需切换

    options = [
        SessionConfigSelectOption(
            value=name,
            name=ag.display_name or name,
            description=ag.description or f"Switch to {name} agent",
        )
        for name, ag in all_agents.items()
    ]
    current = agent.name

    return SessionConfigOption(
        id="agent_role",
        name="Agent",
        description="Select the active agent for this session",
        category="other",
        current_value=current,
        options=options,
    )
```

#### 修改 `get_session_config_options()`

```python
async def get_session_config_options(agent: BaseAgent) -> list[SessionConfigOption]:
    """Get all SessionConfigOptions from agent's modes + agent role."""
    try:
        mode_categories = await agent.get_modes()
    except Exception:
        logger.exception("Failed to get modes from agent")
        mode_categories = []

    options = [to_session_config_option(category) for category in mode_categories]

    # 追加 agent role option（若有多个 agents）
    if role_opt := await get_agent_role_config_option(agent):
        options.append(role_opt)

    return options
```

**排序建议**：为避免 Zed 中 `category="other"` 的键盘快捷键冲突，agent_role 应作为唯一的 `category="other"` option。若 agent 需要暴露其他自定义 config options，建议使用 `category="thought_level"` 或请求 ACP 协议扩展新 category，而非复用 `"other"`。

#### 修改 `set_session_config_option()` — 处理 `agent_role`

```python
async def set_session_config_option(
    self, params: SetSessionConfigOptionRequest
) -> SetSessionConfigOptionResponse | None:
    ...
    if params.config_id == "agent_role":
        # 切换 session 的 agent 实例
        await self._swap_session_agent(params.session_id, params.value)
    else:
        # 原有逻辑：转发到 agent.set_mode()
        await session.agent.set_mode(params.value, category_id=params.config_id)

    config_options = await get_session_config_options(session.agent)
    return SetSessionConfigOptionResponse(config_options=config_options)
```

#### `_swap_session_agent()` 实现

```python
async def _swap_session_agent(
    self,
    session_id: str,
    agent_name: str,
) -> None:
    """将指定 session 切换到不同的 agent 实例。

    利用 RFC-0031 的 per-session agent 机制，委托给 ACPSession.switch_active_agent()。
    整个 swap 过程在 session 级锁保护下执行，防止并发竞态。

    Args:
        session_id: 目标 session ID
        agent_name: 目标 agent 名称（须在 agent_pool.all_agents 中存在）

    Raises:
        ValueError: agent_name 不在 pool 中
        RuntimeError: session 不存在，或 swap 期间存在未完成的 prompt task
    """
    session = self.session_manager.get_session(session_id)
    if not session:
        raise RuntimeError(f"Session not found: {session_id}")

    # 获取或创建 session 级锁
    if session_id not in self._session_agent_locks:
        self._session_agent_locks[session_id] = asyncio.Lock()

    async with self._session_agent_locks[session_id]:
        # 协调 session._task_lock：若当前有 active prompt，禁止 swap
        if hasattr(session, "_task_lock") and session._task_lock.locked():
            raise RuntimeError(
                f"Cannot swap agent while a prompt is in progress (session={session_id})"
            )

        pool = self.agent_pool
        if pool is None or pool.manifest is None or agent_name not in pool.manifest.agents:
            raise ValueError(f"Agent not found: {agent_name}")

        # 委托给 ACPSession.switch_active_agent() 处理完整的 swap 流程：
        # 1. 断开旧 agent 的 state_updated 信号
        # 2. 调用 remove_session_agent() 关闭旧 per-session agent（跳过 default_agent）
        # 3. 创建新 per-session agent（get_or_create_session_agent）
        # 4. 重新应用 session mutation（env, input_provider, sys_prompts, cwd context）
        # 5. 重新连接 state_updated 信号
        # 6. 持久化 agent 切换（session_manager.update_session_agent）
        # 7. 发送 available commands update
        await session.switch_active_agent(agent_name)

        # 同步更新 ACP agent 层的 session_agents 注册表
        self._session_agents[session_id] = session.agent
        logger.info("Session agent swapped", session_id=session_id, new_agent=agent_name)
```

### Phase 3：OpenCode `/mode` 路由修复

#### 修改 `config_routes.py` 的 `GET /mode`

```python
@router.get("/mode")
async def list_modes(state: StateDep) -> list[Mode]:
    """List available modes from agent's mode category."""
    if state.agent is None:
        return [Mode(name="default", tools={})]

    try:
        mode_categories = await state.agent.get_modes()
        for category in mode_categories:
            if category.id == "mode":
                return [
                    Mode(
                        name=mode.id,
                        tools={},
                        # prompt 可选：若 ModeInfo 有 description 则填入
                    )
                    for mode in category.available_modes
                ]
    except Exception:  # noqa: BLE001
        logger.warning("Failed to get modes from agent for /mode route")

    # Fallback：返回默认值（保持向后兼容）
    return [Mode(name="default", tools={})]
```

### API 变更汇总

#### 新增文件

| 文件 | 描述 |
|------|------|
| `acp/schema/providers.py` | ACP `providers/*` 协议类型定义（`ProviderInfo`、`LlmProtocol`、请求/响应类型） |
| `acp_server/provider_router.py` | Provider 路由表（内存维护，process-scoped，不持久化） |

#### 新增函数

| 函数 | 文件 | 描述 |
|------|------|------|
| `ProviderRouter.__init__()` | `acp_server/provider_router.py` | 初始化路由表，从 manifest 派生 provider 列表 |
| `ProviderRouter.get_provider_info_list()` | `acp_server/provider_router.py` | 生成 `providers/list` 响应数据 |
| `ProviderRouter.set_provider()` | `acp_server/provider_router.py` | 处理 `providers/set` 请求，覆盖路由 |
| `ProviderRouter.disable_provider()` | `acp_server/provider_router.py` | 处理 `providers/disable` 请求 |
| `ProviderRouter.is_provider_disabled()` | `acp_server/provider_router.py` | 查询 provider 禁用状态（供 model list 过滤） |
| `handle_providers_list()` | `acp_server/acp_agent.py` | ACP `providers/list` 请求处理器 |
| `handle_providers_set()` | `acp_server/acp_agent.py` | ACP `providers/set` 请求处理器 |
| `handle_providers_disable()` | `acp_server/acp_agent.py` | ACP `providers/disable` 请求处理器 |
| `build_model_state_for_acp()` | `shared/model_utils.py` | ACP model state 构建，configured first + provider 状态过滤 |
| `get_agent_role_config_option()` | `acp_server/acp_agent.py` | 构建 agent role config option |
| `_swap_session_agent()` | `acp_server/acp_agent.py` | 切换 session 的 agent 实例（委托 `session.switch_active_agent()` + 锁保护） |

#### 修改函数

| 函数 | 文件 | 变更描述 |
|------|------|----------|
| `AgentCapabilities` | `acp/schema/capabilities.py` | 新增 `providers: bool \| None` 字段 |
| `get_session_model_state()` | `acp_server/acp_agent.py` | 委托给 `build_model_state_for_acp()`，传入 `provider_router` |
| `get_session_config_options()` | `acp_server/acp_agent.py` | 追加 agent role option |
| `set_session_config_option()` | `acp_server/acp_agent.py` | 处理 `agent_role` category |
| `initialize` 响应 | `acp_server/acp_agent.py` | 返回 `AgentCapabilities(providers=True)` |
| `list_modes()` | `opencode_server/routes/config_routes.py` | 动态调用 `agent.get_modes()` |

### Agent Role 切换状态机

```
session 初始化
    │
    ├─ NewSessionResponse.config_options 包含 agent_role option
    │    current_value = session.agent.name  # 当前 session 实际运行的 agent
    │    options = [all pool agents]
    │
    └─ 用户在 IDE 中选择 agent
         │
         ▼
    session/set_config_option
    { config_id: "agent_role", value: "plan-agent" }
         │
         ▼
    _swap_session_agent(session_id, "plan-agent")
         │
         ├─ 关闭旧 agent（__aexit__）
         └─ 创建新 agent（get_or_create_session_agent）
              │
              ▼
         SetSessionConfigOptionResponse
         { config_options: [...updated with current_value="plan-agent"] }
```

### 向后兼容保证

| 机制 | 影响 | 兼容性 |
|------|------|--------|
| `initialize` 响应 | `AgentCapabilities` 新增 `providers` 字段 | ✅ 新增可选字段，旧客户端忽略 |
| `providers/*` 方法 | 新增三个方法 | ✅ 旧客户端不调用；`agentCapabilities.providers` 为 feature flag |
| `NewSessionResponse.modes`（旧版） | tool permission，不变 | ✅ 完全兼容 |
| `NewSessionResponse.models`（旧版） | model list，数据来源统一为 configured first + provider 过滤 | ✅ 格式兼容，数据来源优先级与 OpenCode 一致 |
| `NewSessionResponse.config_options`（新） | 追加 `agent_role`，现有 config_options 不变 | ✅ 新增字段，客户端忽略未知 config_id |
| `session/set_mode` | tool permission 切换，不变 | ✅ 完全兼容 |
| `session/set_config_option` | 新增 `agent_role` 处理，现有 config_id 不变 | ✅ 扩展，不破坏现有 |
| `category="other"` 冲突 | 若 agent 存在多个 `category="other"` option，Zed 键盘快捷键仅匹配第一个 | ⚠️ 已知限制；不影响鼠标点击；agent 实现应避免同时启用多个 `category="other"` option |

### Zed 兼容性分析

基于 Zed 源码（`crates/agent_ui/src/config_options.rs`、`crates/agent_ui/src/profile_selector.rs`、`crates/agent_servers/src/acp.rs`）的完整调研，本 RFC 设计的 Zed 兼容性如下：

| 功能 | Zed 支持状态 | 兼容性 | 备注 |
|------|-------------|--------|------|
| `config_options` 渲染 | ✅ 完全支持 | 所有 option 均显示为独立下拉按钮 | `ConfigOptionsView::build_selectors()` 遍历全部 options |
| `agent_role` 鼠标点击 | ✅ 完全支持 | 点击按钮弹出 picker，选择后触发 `session/set_config_option` | 标准 `ConfigOptionSelector` 行为 |
| `agent_role` 键盘快捷键 | ⚠️ 有限支持 | 仅当 agent_role 是同 category 中**第一个** option 时可用 | `first_config_option_id()` 限制；不影响鼠标操作 |
| `category="mode"` 快捷键 | ✅ 支持 | `ToggleProfileSelector` → `Mode` category | 若 agent 同时提供 tool permission 和 agent_role 的 `mode` category，只有第一个响应快捷键 |
| `category="model"` 快捷键 | ✅ 支持 | `ToggleModelSelector` → `Model` category | 与现有 model selector 行为一致 |
| `providers/*` 协议 | ❌ 不支持 | Zed 当前无 `providers/list`、`providers/set`、`providers/disable` 处理器 | Phase 0 实现暂无 Zed UI 入口，但确保协议合规和未来兼容 |
| Zed Profile 切换 | ✅ 独立运行 | 本地 `AgentSettings.profiles`，与 ACP 无关 | 当 ACP agent 未提供 `mode` category 时，快捷键回退到本地 ProfileSelector |

**关键结论**：
1. `agent_role` 使用 `category="other"` 在 Zed 中**可正确渲染和交互**，满足 P0 目标
2. 键盘快捷键限制是 Zed 的设计行为，不影响核心功能；agent 实现可通过控制 `config_options` 列表顺序来优化体验
3. `providers/*` 的 Phase 0 实现虽暂无 Zed UI 入口，但为 ACP 协议合规性和其他客户端（如 Cursor、未来版 Zed）的兼容所必需

### Phase 0：ACP Configurable LLM Providers（优先级 P0）

**目标**：G7、GAP 5

**范围**：
- [ ] 新增 `acp/schema/providers.py`：定义 `LlmProtocol`、`ProviderInfo`、`ProviderCurrentConfig`、请求/响应类型
- [ ] 新增 `acp_server/provider_router.py`：实现 `ProviderRouter` 类
  - 从 `manifest.model_variants` 派生 `ProviderInfo[]`（复用 `_extract_provider()`）
  - 推断 `LlmProtocol` 和默认 `baseUrl`
  - 维护 `_overrides` 和 `_disabled` 内存状态
- [ ] 修改 `acp/schema/capabilities.py`：`AgentCapabilities` 新增 `providers: bool | None = False`
- [ ] 修改 `acp_server/acp_agent.py`：
  - 初始化 `ProviderRouter` 实例
  - 添加 `handle_providers_list()`、`handle_providers_set()`、`handle_providers_disable()` 处理器
  - 在请求分发 switch 中注册 `providers/list`、`providers/set`、`providers/disable`
  - 修改 `initialize` 响应返回 `AgentCapabilities(providers=True)`
- [ ] 编写单元测试：`ProviderRouter._derive_providers_from_manifest()` 正确提取 provider
- [ ] 编写单元测试：`providers/set` 覆盖后 `get_provider_info_list()` 反映新配置
- [ ] 编写单元测试：`providers/disable` 后 `is_provider_disabled()` 返回 True
- [ ] 编写单元测试：`providers/disable` 对 required provider 返回 `invalid_params`
- [ ] 编写单元测试：`providers/set` 对未知 id 返回 `invalid_params`
- [ ] 添加 ACP 快照测试：验证 `initialize` 响应包含 `agentCapabilities.providers=true`

**预估代码量**：~200 行新增
**依赖**：无（应最先实施，为 Phase 1 提供 `ProviderRouter`）

---

### Phase 1：共享 Model List 逻辑（优先级 P1）

**目标**：G2、G3

**范围**：
- [ ] 在 `shared/model_utils.py` 中新增 `build_model_state_for_acp()` 函数（configured first + provider 状态过滤）
- [ ] 修改 `acp_server/acp_agent.py` 的 `get_session_model_state()` 委托给共享函数，传入 `provider_router`
- [ ] 编写单元测试：验证 configured variants 存在时 tokonomics 不被调用
- [ ] 编写单元测试：验证 configured variants 为空时 tokonomics 作为 fallback
- [ ] 编写单元测试：验证 current model 不在列表时被插入到列表头
- [ ] 编写单元测试：验证 provider 被禁用时，该 provider 下的模型从列表中过滤
- [ ] 编写单元测试：验证 `provider_router=None` 时行为与不传一致（向后兼容）

**预估代码量**：~90 行新增，~30 行修改
**依赖**：Phase 0（需要 `ProviderRouter` 类型和接口）

---

### Phase 2：Agent Role Config Option（优先级 P0）

**目标**：G1、G5

**范围**：
- [ ] 在 `acp_server/acp_agent.py` 中新增 `get_agent_role_config_option()` 函数（`current_value=agent.name`）
- [ ] 修改 `get_session_config_options()` 追加 agent role option
- [ ] 在 `AgentPoolACPAgent` 中新增 `_swap_session_agent()` 方法
  - 获取 `_session_agent_locks[session_id]` 防止并发竞态
  - 检查 `session._task_lock` 拒绝 active prompt 期间的 swap
  - 验证 `pool.manifest` 非空且 `agent_name` 存在
  - 委托 `session.switch_active_agent()` 处理完整 swap 流程（信号、env、input_provider、sys_prompts 重连）
  - 同步更新 `_session_agents` 注册表
- [ ] 修改 `set_session_config_option()` 处理 `agent_role` category
- [ ] 预验证：确认 Zed 能正确渲染 `category="other"` 的 config options（Zed 源码确认所有 config options 均渲染为独立按钮，但键盘快捷键 `first_config_option_id()` 仅匹配同 category 的第一个 option。若 agent 存在多个 `category="other"` 的 option，需确保 agent_role 在列表中位于首位，或接受键盘快捷键不可用的限制）
- [ ] 文档更新：在 agent 开发文档中说明，启用 agent_role 的 agent 不应同时使用 `category="other"` 的其他 config option，以避免 Zed 中的键盘快捷键冲突
- [ ] 编写单元测试：单一 agent 时 `get_agent_role_config_option()` 返回 None
- [ ] 编写单元测试：多 agent 时正确枚举所有 agent，且 `current_value` 为当前 session agent
- [ ] 编写并发测试：两个同时的 `set_config_option("agent_role", ...)` 请求被序列化，不 corrupt registry
- [ ] 编写集成测试：`session/set_config_option` 切换 agent 后，后续 `session/prompt` 使用新 agent 的 system prompt 和工具
- [ ] 添加 ACP 快照测试：验证 `NewSessionResponse.config_options` 包含 `agent_role`

**预估代码量**：~100 行新增
**依赖**：无（可与 Phase 1 并行）
**前置验证**：Zed 对 `category="other"` 的渲染行为（go/no-go 决策点）

---

### Phase 3：OpenCode /mode 路由修复（优先级 P1）

**目标**：G4

**范围**：
- [ ] 修改 `config_routes.py` 的 `GET /mode` 动态调用 `agent.get_modes()`
- [ ] 将 `mode` category 的 `available_modes` 转换为 `list[Mode]` 返回
- [ ] 保留 fallback：`get_modes()` 抛出异常时返回 `[Mode(name="default")]`
- [ ] 编写测试：`NativeAgent` 返回 `[always, never, per_tool]` 三个 Mode
- [ ] 编写测试：`agent.get_modes()` 异常时返回 `[default]`

**预估代码量**：~30 行修改
**依赖**：无（最简单，可独立交付）

---

### 里程碑总览

| Phase | 目标 | 工作量 |
|-------|------|--------|
| Phase 0 | ACP Configurable LLM Providers | ~200 行 |
| Phase 1 | 统一 model list 数据来源 + provider 状态过滤 | ~120 行 |
| Phase 2 | Agent role config option | ~100 行 |
| Phase 3 | OpenCode /mode 路由修复 | ~30 行 |
| **合计** | | **~450 行** |

Phase 0 必须最先实施（Phase 1 依赖 `ProviderRouter`）。Phase 1 和 Phase 2 可在 Phase 0 完成后并行。Phase 3 独立，可任意时序交付。

## 开放问题

1. **tokonomics per-agent vs 全局的细微差异**：`build_model_state_for_acp()` 中 tokonomics 来源为 `agent.get_available_models()`（per-agent），而 OpenCode 使用 `get_all_models()`（全局 7 天缓存）。两者实际返回的模型列表是否完全一致？如果某个 agent 配置了特定的 `providers` 参数，per-agent 的返回集合可能是全局的子集。

2. **Agent 切换时的对话历史**：切换 agent role 时，新 agent **不继承**旧 agent 的对话历史。原因如下：
   - 不同 agent 通常具有不同的 system prompt 和工具集，继承历史消息可能导致新 agent 对上下文的理解出现偏差
   - ACP 协议层面没有定义 conversation 迁移的标准方式
   - 若用户需要保持上下文，可通过显式的 session/prompt 将历史摘要传递给新 agent
   **决策**：agent swap 后新 agent 从空对话开始；如需历史继承，将在后续 RFC 中专门设计 `conversation` 迁移协议。

3. **Agent Role 的 `current_value` 初始值**（✅ 已解决）：修正为 `agent.name`（当前 session 实际运行的 agent 实例），而非 `pool.main_agent.name`。这样确保 per-session agent 被切换后，IDE 中显示的是正确的当前 agent。

4. **OpenCode `/mode` 与 config_options 的关系**：`GET /mode` 修复后返回 `NativeAgent` 的 tool permission modes（always/never/per_tool），但 OpenCode TUI 的 mode 切换 UI 与 ACP 的 `config_options` 是独立的。两者是否需要在语义上对齐（即 `PATCH /config` 中修改 `mode` 是否等价于 ACP 的 `session/set_config_option` for `"mode"`）？

5. **agent_role config option 的 display 策略**：若 pool 只有 2 个 agent，是否应显示 agent role 选择器？`len(all_agents) <= 1` 时返回 `None` 的逻辑是否合理？
   **决策**：保持现有逻辑（<=1 不显示），并在文档中建议：对于多 agent 配置，agent_role option 应始终位于 `config_options` 列表的首位（若存在多个 `category="other"` 的 option），以最大化键盘快捷键兼容性。

6. **`providers/set` 对已运行 session 的影响**：ACP 协议规定 Agent MAY 不将变更应用到已运行 session。本 RFC 采用保守策略：`providers/set` 仅影响后续新建 session。但如果用户在 IDE 中 `providers/set` 后立即发送 `session/prompt`，期望新路由立即生效，这可能与保守策略冲突。是否应在 `providers/set` 响应中明确告知客户端变更生效范围？
   **决策**：在 `providers/set` 响应的 `_meta` 字段中添加 `"scope": "new_sessions_only"` 提示（可选字段，不破坏兼容性）。

7. **Provider 路由覆盖与 agent 内部模型初始化的兼容**：agent 的 `model_variants` 中的 `AnyModelConfig` 可能包含 `base_url`、`api_key_env` 等字段。当 `providers/set` 覆盖了 provider 的 `baseUrl` 和 `headers` 后，agent 在创建 LLM client 时应使用覆盖后的配置。这需要在 agent 的 LLM client 初始化路径中注入 `ProviderRouter` 查找逻辑。具体实现可能涉及 pydantic-ai 的 `Provider` 初始化参数注入，需要评估侵入性。
   **状态**：标记为高风险，需在 Phase 0 实施前完成技术预研。

8. **`SessionModelState` 中是否应携带 provider 关联信息**：当前 `ACPModelInfo` 只有 `model_id`、`name`、`description`，不含 provider 信息。客户端无法从 `SessionModelState` 中判断某个模型属于哪个 provider。如果 `providers/list` 和 `SessionModelState` 都可用，客户端可以自行关联（如根据 model 命名约定推断）。但更精确的做法是在 `ACPModelInfo._meta` 中添加 `provider_id` 字段，代价是 `SessionModelState` 结构略有扩展。
   **决策**：暂不添加，保持 `SessionModelState` 格式不变；若未来有强需求，可在不破坏兼容性的前提下通过 `_meta` 扩展。

9. **Zed `category="other"` 的键盘快捷键限制**（✅ 已调研，已知限制）：Zed `first_config_option_id()` 仅返回匹配 category 的第一个 option。若 agent 存在多个 `category="other"` 的 option，只有第一个可通过键盘快捷键访问。不影响鼠标点击。建议在 agent 实现中避免同时启用多个 `category="other"` 的 option。

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-05-26 | 推荐选项 2（三阶段统一） | 技术债务清零，覆盖所有 P0/P1 问题，工作量可接受 |
| 2026-05-26 | ACP 侧保持 `SessionModelState` 扁平结构 | 不改变 ACP 协议格式，只统一数据来源优先级 |
| 2026-05-26 | agent_role 使用 `category="other"` | 不修改 ACP schema，`"other"` category 可承载非标准配置项；Zed 源码验证所有 category 均正确渲染为 UI 按钮 |
| 2026-05-27 | Zed 中多个同 category option 的键盘快捷键冲突为已知限制 | `first_config_option_id()` 仅返回第一个匹配；不影响鼠标点击访问；agent 实现应避免在启用 agent_role 时同时使用多个 `category="other"` option |
| 2026-05-26 | tokonomics 来源保持 per-agent 语义 | ACP per-session 协议与 per-agent model 语义一致，不强制对齐全局缓存 |
| 2026-05-26 | 单一 agent 时不透出 agent_role | 单 agent 配置无需切换，减少无意义 UI 噪声 |
| 2026-05-26 | `_swap_session_agent()` 委托 `session.switch_active_agent()` | 复用已有的完整 session mutation 逻辑（信号、env、input_provider、sys_prompts），避免重复实现和遗漏 |
| 2026-05-26 | agent swap 在 `_session_agent_locks` 保护下执行 | 防止并发 `set_config_option` 请求导致竞态条件和 session corruption |
| 2026-05-26 | agent swap 拒绝在 active prompt 期间执行 | 通过检查 `session._task_lock.locked()` 避免 mid-stream agent 切换导致 crash |
| 2026-05-26 | agent swap 后新 agent 不继承对话历史 | 不同 agent 的 system prompt 和工具集可能不兼容；历史继承需单独 RFC 设计 |
| 2026-05-27 | 新增 Phase 0：实现 ACP Configurable LLM Providers | ACP PR #648 已 MERGED，`providers/*` 为协议已合并特性；Phase 1 的 model list 需感知 provider 禁用状态 |
| 2026-05-27 | `providers/set` 仅影响后续新建 session | 遵循 ACP 协议 MAY 语义，保守策略降低复杂度；已运行 session 的动态路由切换留作后续增强 |
| 2026-05-27 | 从 `model_variants` 派生 `ProviderInfo[]` | 复用已有 `_extract_provider()` 逻辑，`AnyModelConfig` 隐含了 provider 信息，无需额外配置 |
| 2026-05-27 | Phase 1 的 `build_model_state_for_acp()` 接受 `provider_router` 参数 | 解耦 model list 构建与 provider 状态管理；`provider_router=None` 时退化为旧行为，保证向后兼容 |

## 参考

### AgentPool 源码

- `packages/wolfharness/src/wolfharness_server/acp_server/acp_agent.py` — ACP server 核心，`get_session_model_state`、`get_session_config_options`
- `packages/wolfharness/src/wolfharness_server/acp_server/converters.py` — `to_session_config_option()`、`agent_to_mode()`（已有但未使用）
- `packages/wolfharness/src/wolfharness_server/opencode_server/routes/config_routes.py` — OpenCode model list 逻辑、`/mode` 路由
- `packages/wolfharness/src/wolfharness_server/shared/model_utils.py` — 现有共享 model 工具函数
- `packages/wolfharness/src/wolfharness/agents/native_agent/helpers.py` — `get_permission_category()`、`get_model_category()`
- `packages/wolfharness/src/wolfharness/agents/modes.py` — `ModeCategory`、`ModeInfo`、`ModeCategoryId`
- `packages/wolfharness/src/acp/schema/capabilities.py` — `AgentCapabilities`（需新增 `providers` 字段）
- `packages/wolfharness/src/wolfharness/models/manifest.py` — `AgentsManifest.model_variants`（`providers/list` 的数据来源）

### ACP 协议

- `packages/wolfharness/src/acp/schema/session_state.py` — `SessionConfigOption`、`SessionConfigOptionCategory`、`SessionModeState`
- `packages/wolfharness/src/acp/schema/agent_responses.py` — `NewSessionResponse`（`models`、`modes`、`config_options` 字段）
- [ACP 协议官网](https://agentclientprotocol.com/protocol/session-modes)
- [ACP PR #648: Configurable LLM Providers](https://github.com/agentclientprotocol/agent-client-protocol/pull/648) — `providers/*` 方法族定义（已 MERGED）
- `packages/agent-client-protocol/docs/rfds/custom-llm-endpoint.mdx` — ACP RFD 原文

### 相关 RFC

- [RFC-0027: ACP Subagent Zed 兼容性](RFC-0027-acp-subagent-zed-compatibility.md)
- [RFC-0031: ACP Per-Session Agent Isolation](../implemented/RFC-0031-acp-per-session-agent-isolation.md)
- [RFC-0032: ACP Slash Commands Protocol Compliance](RFC-0032-acp-slash-commands-session-update.md)
