# ToolDisplayCapability

`ToolDisplayCapability` 是 wolfharness 的**全局装饰器能力** (Global Decorator Capability):在不修改任何子能力源码的前提下,对 agent 已装配的全部工具统一起作用 —— **改名** (`rename_mode`) 与 **注入 diff 富信息事件** (`emit_diff`)。它解决的核心问题是:第三方 capability(如 viking)的工具名不在协议客户端(OpenCode TUI / Zed)的渲染白名单内、工具返回不含 diff 富信息,导致客户端无法渲染文件变更的 diff 视图。

模式对齐 `ToolInterceptCapability`(`src/wolfharness/agents/native_agent/tool_intercept.py`):独立 `AbstractCapability` 直接覆写 `get_wrapper_toolset()` 与 `wrap_tool_execute()`,作为全局中间件横切 agent 的全部工具 —— 不组合子能力、无 `capabilities` 字段。

## 五个正交开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `rename_mode: bool` | `true` | 启用工具改名。经 `get_wrapper_toolset()` 返回 pydantic-ai 官方 `RenamedToolset(wrapped=toolset, name_map=...)`,按 `name_map` 重写 `tool_def.name`,执行时自动还原 `ctx.tool_name`。`name_map` 为空或 `rename_mode: false` 时不包装 |
| `emit_diff: bool` | `true` | 启用 diff 事件注入。`wrap_tool_execute()` 在工具真实执行后,对命中的工具注入 `ToolCallProgressEvent.file_edit(...)`(携带 `DiffContentItem`) |
| `emit_diff_for: set[str]` | `set()`(空=不注入) | diff 注入白名单,**按工具名精确过滤**。仅当 `emit_diff: true` 且工具名在名单内时注入 |
| `emit_rich: bool` | `true` | 启用 rich 展示事件注入。对只读/查询类工具,执行前注入 `ToolCallStartEvent`(kind + locations),执行后注入携带内容项的 progress 事件 |
| `emit_rich_for: set[str]` | `set()`(空=不注入) | rich 注入白名单,**按工具名精确过滤**。仅当 `emit_rich: true` 且工具名在名单内时注入 |

`name_map`、`emit_diff_for` 与 `emit_rich_for` 均为空时,退化为**无操作装饰器**:`get_wrapper_toolset` 返回 `None`,`wrap_tool_execute` 直接透传。

## Rich 展示层(emit_rich)

面向 **read/search/glob/find 类只读工具**,让协议客户端获得正确的工具分类、文件锚点与内容展示 —— 与 rename(ACP 场景 `rename_mode: false` 也需 rich)、emit_diff(read/query 工具不需 diff)完全正交。

**数据来源(策略注册表)**:模块级 `_RICH_EXTRACTORS` 按**原始工具名**注册 extractor,接收真实执行结果产出内容项 + locations;未注册的工具退化到 `derive_rich_tool_info` 的 title/kind + 通用参数位置提取,不注入内容。新工具扩展:注册一条 extractor 即可。

**执行前注入** `ToolCallStartEvent(kind, locations)` —— 给客户端正确的工具图标与文件锚点(ACP 转换器原生消费,零协议改动)。

**执行后注入** progress 事件携带 `TextContentItem`(读取内容/搜索结果)—— 经 opencode `_process_tool_progress` 转成 tool output 文本。post 事件**沿用执行前 title**(search/glob 等共享 extractor 的工具不会被显示成通用的 "Read")。

**防重复**:read/query 工具的内容走 rich 通道,不生成 `DiffContentItem`;write/edit 工具的 diff 走 emit_diff 通道 —— 两类工具由 `emit_rich_for`/`emit_diff_for` 白名单天然隔离。

## Diff 数据来源(执行后注入)

`wrap_tool_execute` 先调用 `handler(args)` **拿到真实执行结果**,再从 `args` 解析 diff 字段(模块级 `_parse_diff_fields` 辅助):

- **write 风格** (工具入参含 `content`/`path`|`uri`):`new_text=content`、`old_text=None`(视为新增文件)
- **edit 风格** (工具入参含 `old_string` + `new_string`):`old_text=old_string`、`new_text=new_string`
- **退化兜底**:path 无法从入参解析、或 new_text 为空时,跳过注入且不报错

路径键取自 `path` / `file_path` / `uri` / `filepath` 的任意现值(按序取第一个),因此对 viking 的 `viking://...` URI 与本地文件路径同样有效。

## 事件注入通道

注入的 `ToolCallProgressEvent` + `DiffContentItem` 流经既有管道,零协议改动:

```
wrap_tool_execute → ctx.deps.events.tool_call_progress(title, items=[DiffContentItem(...)])
  → EventBus publish → EventMapper._is_rich_event 原样透传
  → ACP 转换器 DiffContentItem → FileEditToolCallContent + ToolCallLocation
  → 客户端(Zed/OpenCode TUI)渲染 diff
```

- `ctx.deps` 在 wolfharness 中**直接是 `AgentContext`**,携带 `.events` → `StreamEventEmitter`(POC 已验证,同 fsspec 工具集 `wolfharness_toolsets/fsspec_toolset/toolset.py:575` 的先例通道)
- `DiffContentItem(path, old_text, new_text)` 定义于 `src/wolfharness/agents/events/events.py:203`
- **不依赖 metadata 通道**:`ToolReturn.metadata` 在 `process_tool_event`/`event_mapper` 构造 `ToolCallCompleteEvent` 时会被丢弃(仅 `is_error`),本能力刻意绕开该断点,改用事件注入

## 协议区分配置

同一工具集在不同协议客户端下的展示诉求不同,通过**装配期**配置区分(零运行时协议标识改动):

| 场景 | 配置 | 说明 |
|---|---|---|
| **OpenCode TUI** | `rename_mode: true` + `emit_diff: true` + `emit_rich: true` | 改名命中白名单(`viking_write`→`write`, `viking_read`→`read`)+ diff 注入 + rich 展示 |
| **ACP (Zed)** | `rename_mode: false` + `emit_diff: true` + `emit_rich: true` | Zed 展示原名即可,`FileEditToolCallContent` 原生渲染差异、`ToolCallStartEvent` 渲染 kind/locations |
| **子能力已自发射** (fsspec 模式) | `rename_mode: true` + `emit_diff: false` | 子能力已自行 emit `DiffContentItem`,装饰器仅改名,避免重复注入 |

**防重复原则**:子 capability 已自行发射 diff 事件的场景,必须用 `emit_diff: false`,否则同一变更被注入两次。

## 配置与注册

```yaml
# agent YAML capabilities 段 —— 与其它 capability 平级列出即可(全局中间件)
capabilities:
  - type: tool_display
    args:
      rename_mode: true
      name_map:
        viking_write: write
        viking_edit: edit
        viking_read: read
        viking_search: grep
      emit_diff: true
      emit_diff_for: [viking_write, viking_edit]
      emit_rich: true
      emit_rich_for: [viking_read, viking_search, viking_find, viking_glob]
```

- 注册:entry-point 组 `wolfharness.capabilities`,key `tool_display` → `wolfharness.capabilities.tool_display_capability:ToolDisplayCapability`(见 `pyproject.toml`),由 `registry.py` 发现
- 构造:`EntryPointCapabilityConfig(type=..., args={...}).build()` 以 `cls(**args)` 实例化 —— dataclass 字段(`rename_mode`/`name_map`/`emit_diff`/`emit_diff_for`/`emit_rich`/`emit_rich_for`)+ `id` 天然兼容 YAML 装配;`emit_rich_for` 传入的 YAML 列表在 `__post_init__` 转为 `set`

## 已知约束

- **`ctx.tool_name` 双名不一致**:改名后,事件映射层携带新名、工具自发射事件携带原名 —— 注入事件显式构造 `tool_call_id`,不依赖名称匹配
- **只映射语义等价的标准名**:改名可能触发客户端内置行为(如 `write` 触发生成式 diff),只映射语义一致的工具,映射表见配置文档
- **仅上游通道**:本能力不打通 OpenCode TUI 的 last-turn DiffViewer(远端写入无法被本地 git snapshot 捕获)—— 如需另立 change