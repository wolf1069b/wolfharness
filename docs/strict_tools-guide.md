# Strict Tools Capability 使用指南

## 这是什么

`strict_tools` 是 wolfharness 的一个内置 capability，用于在 tool definition 发送给模型之前强制设置 `strict=True`。

部分 provider（如 LiteLLM 转发到 SGLang）在 `strict=None` 时会忽略该标记，退化为"看起来像 JSON"的松散语法，导致偶发的 tool-call 参数解析 400 错误。`strict_tools` 通过显式设置 `strict=True` 来消除这类问题。

## 前置条件

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| wolfharness | 2.9.5 | `strict_tools` 作为内置 capability 从此版本可用 |
| pydantic-ai | — | wolfharness 自带依赖 |
| logfire | — | wolfharness 自带依赖，用于 span 日志 |

先确认你的 wolfharness 版本：

```bash
pip show wolfharness | grep Version
# 或
uv run python -c "import wolfharness; print(wolfharness.__version__)"
```

---

## 场景一：wolfharness >= 2.9.5（推荐）

如果你的 wolfharness 版本已经 >= 2.9.5，`strict_tools` 已经内置，**不需要拷贝任何文件**，只需在你的 YAML 配置中添加 capability 条目。

### 步骤

在你的 agent YAML 配置文件中加入以下内容：

```yaml
# ── 定义 anchor（放在 .anchors 下或文件顶部任意位置） ──
.anchors:
  strict_tools_caps: &strict_tools_caps
    type: strict_tools
    enabled: true                # 总开关，false 时所有 tool definition 原样透传
    apply_to_output_tools: false # 是否对 structured output 的 tool 也强制 strict

# ── 在每个 agent 的 capabilities 列表中引用 ──
agents:
  my_agent:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are a helpful assistant."
    tools: []
    capabilities:
      - *strict_tools_caps
```

如果你不想用 YAML anchor，也可以直接内联：

```yaml
agents:
  my_agent:
    type: native
    model: openai:gpt-4o
    capabilities:
      - type: strict_tools
        enabled: true
        apply_to_output_tools: false
```

### 验证

启动你的 agent 并观察 logfire 日志，应该能看到 `strict_tools.prepare` span 和类似以下的日志：

```
tool strict status  tool_name=xxx  strict_before=None  strict_after=True  changed=True
```

---

## 场景二：wolfharness < 2.9.5（手动移植）

如果你的 wolfharness 版本低于 2.9.5，或者你使用的是 wolfharness 的私有分支，需要手动拷贝以下文件并做少量修改。

### 需要拷贝的文件清单

| 源文件（在本仓库中的路径） | 目标路径 | 说明 |
|---|---|---|
| `packages/wolfharness/src/wolfharness/capabilities/strict_tools.py` | `<你的wolfharness>/src/wolfharness/capabilities/strict_tools.py` | 核心实现，零外部依赖（仅依赖 pydantic-ai 和 logfire） |
| `packages/wolfharness/tests/capabilities/test_strict_tools.py` | `<你的wolfharness>/tests/capabilities/test_strict_tools.py` | 单元测试（可选但推荐） |

此外，还需要在配置注册文件中追加约 20 行代码（见下方步骤 2）。

### 步骤 1：拷贝核心实现文件

```bash
# 假设你的 wolfharness 源码在 ~/projects/wolfharness
cp packages/wolfharness/src/wolfharness/capabilities/strict_tools.py \
   ~/projects/wolfharness/src/wolfharness/capabilities/strict_tools.py
```

### 步骤 2：在配置注册文件中添加 strict_tools

打开 `<你的wolfharness>/src/wolfharness_config/capabilities.py`，在以下 **4 处** 追加内容：

**(a) `KNOWN_CAPABILITY_TYPES` 集合中添加 `"strict_tools"`：**

```python
KNOWN_CAPABILITY_TYPES: frozenset[str] = frozenset({
    "loop_detection",
    "token_budget",
    # ... 其他已有类型 ...
    "strict_tools",           # ← 添加这一行
})
```

**(b) `IMPORT_MAP` 字典中添加映射：**

```python
IMPORT_MAP: dict[str, str] = {
    # ... 其他已有映射 ...
    "strict_tools": "wolfharness.capabilities.strict_tools.StrictToolsCapability",  # ← 添加
}
```

**(c) 添加 `StrictToolsCapabilityConfig` 类（放在其他 CapabilityConfig 类旁边）：**

```python
class StrictToolsCapabilityConfig(BaseModel):
    """Config for ``StrictToolsCapability``.

    Forces ``strict=True`` on tool definitions before they reach the
    provider, fixing malformed tool-call arguments on providers that
    ignore ``strict=None`` (e.g. LiteLLM → SGLang).
    """

    type: Literal["strict_tools"] = "strict_tools"
    enabled: bool = True
    """Master switch — when ``False``, definitions pass through unchanged."""
    apply_to_output_tools: bool = False
    """Also force ``strict`` on output-tool (structured output) definitions."""
```

**(d) 在 `BuiltinCapabilityConfig` 联合类型中添加：**

```python
BuiltinCapabilityConfig = Annotated[
    LoopDetectionCapabilityConfig
    | TokenBudgetCapabilityConfig
    # ... 其他已有类型 ...
    | StrictToolsCapabilityConfig,        # ← 添加
    Field(discriminator="type"),
]
```

**(e) 在 `build_capability()` 函数的 match 语句中添加分支：**

```python
def build_capability(config: CapabilityConfig) -> Any:
    match config:
        # ... 其他已有分支 ...
        case StrictToolsCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["strict_tools"], config)
        case _ as unreachable:
            from typing import assert_never
            assert_never(unreachable)
```

### 步骤 3：拷贝测试文件（可选但推荐）

```bash
cp packages/wolfharness/tests/capabilities/test_strict_tools.py \
   ~/projects/wolfharness/tests/capabilities/test_strict_tools.py
```

### 步骤 4：验证

```bash
cd ~/projects/wolfharness

# 运行单元测试
uv run pytest tests/capabilities/test_strict_tools.py -v

# 确认 capability 能被正确识别
uv run python -c "
from wolfharness_config.capabilities import is_known_capability_type, build_capability, StrictToolsCapabilityConfig
assert is_known_capability_type('strict_tools')
cap = build_capability(StrictToolsCapabilityConfig())
print(f'OK: {cap}, enabled={cap.enabled}, apply_to_output_tools={cap.apply_to_output_tools}')
"
```

### 步骤 5：在 YAML 配置中启用

参照场景一的 YAML 配置片段，在你的 agent 配置中添加 `type: strict_tools` capability 条目。

---

## 场景三：不使用 wolfharness，仅用 pydantic-ai

如果你不使用 wolfharness，只用 pydantic-ai 原生框架，可以直接将 `strict_tools.py` 作为一个独立 capability 使用。

### 步骤

```bash
# 拷贝文件到你的项目
cp packages/wolfharness/src/wolfharness/capabilities/strict_tools.py \
   ~/my-project/src/my_project/strict_tools.py
```

然后在代码中直接使用：

```python
from my_project.strict_tools import StrictToolsCapability

# 直接传给 pydantic-ai Agent 的 capabilities 参数
agent = Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    capabilities=[StrictToolsCapability(enabled=True, apply_to_output_tools=False)],
)
```

> **注意**：此方式不需要 `wolfharness_config/capabilities.py` 的任何修改，因为绕过了 YAML 配置解析，直接在 Python 代码中实例化。

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `True` | 总开关。`False` 时所有 tool definition 原样透传，不做任何修改 |
| `apply_to_output_tools` | `bool` | `False` | 是否对 structured output 的 tool definition 也强制 `strict=True`。默认 `False`，因为 SGLang 等 provider 的 output 路径不一定支持 strict flag |

## 工作原理

```
Model request 流程:

  Agent.run()
    ↓
  prepare_tools()          ← StrictToolsCapability 在这里介入
    ↓ 遍历所有 ToolDefinition
    ↓ 对 strict is None 的 → replace(td, strict=True)
    ↓ 对 strict 已有值的   → 保持不变
    ↓
  序列化为 JSON 发送给 provider
    ↓
  Provider 看到 strict=True → 使用 constrained grammar
    ↓
  返回合法 JSON tool-call arguments（不再出现 400 错误）
```

关键设计点：

- **只升级 `strict=None` 的定义**：已经显式设置 `strict=True` 或 `strict=False` 的不会被覆盖。
- **`apply_to_output_tools` 默认关闭**：structured output 走的是不同的 provider 路径（如 SGLang 的 output 端），部分 provider 不支持 strict flag，开启可能导致错误。
- **无副作用**：`enabled=False` 时完全透传，等价于没有安装这个 capability。

## 文件清单速查

```
本仓库中 strict_tools 相关的全部文件:

packages/wolfharness/
├── src/wolfharness/capabilities/strict_tools.py          # 核心实现 (115 行)
├── src/wolfharness_config/capabilities.py                 # 配置注册 (4 处修改，共 ~20 行)
└── tests/capabilities/test_strict_tools.py              # 单元测试 (88 行)

消费侧（你的项目）:

your-project/
└── config/your-agents.yaml                              # YAML 配置中引用 *strict_tools_caps
```
