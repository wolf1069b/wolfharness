# ToolSchemaOverlapCapability

`ToolSchemaOverlapCapability` 是 wolfharness 的**能力级工具 schema 语义适配能力**:在不修改 MCP 服务端与子能力源码的前提下,按声明式配置改写 agent 已装配工具的**模型可见 schema** —— 工具改名、描述改写、参数改名/改类型/改枚举/设默认、增删参数。它填补的空白是:`tool_prefix`(能力级)只做**命名空间级**前缀防冲突、`schema_override`(工具级)只作用于单个工具定义,二者都不提供跨能力、按配置批量改写工具语义的通道。

模式对齐 `ToolDisplayCapability`:独立 `AbstractCapability` 通过 `get_wrapper_toolset()` 返回 pydantic-ai 官方 `WrapperToolset` 子类横切全部工具,运行期经 `wrap_tool_execute()` 还原参数。与 `ToolDisplayCapability` 正交 —— schema 改写在内层,展示改名在外层(`get_ordering()` 声明 `wrapped_by=[ToolDisplayCapability]`)。

## 核心机制:身份标记管道

匹配不依赖工具名(可能带 `tool_prefix` 前缀、可能与其他服务器重名),而是依赖**来源身份 metadata**:

```
McpServerCap._build_toolset 注入 {server_name, original_mcp_tool_name}
  → tools/base.py Patch A:to_pydantic_ai 从 schema 分支携带 metadata
  → tools/base.py Patch B:schema_override prepare 重建 ToolDefinition 时保留 metadata
  → FunctionToolset/PrefixedToolset/RenamedToolset 链路透传(dataclasses.replace 只改 name)
  → SchemaOverrideToolset 按身份查配置,前缀完全不参与匹配
```

配置键**永远是原始 MCP 工具名**:即使服务器配了 `tool_prefix: weather`,配置里仍写 `get_weather` 而非 `weather_get_weather`。

## 配置参考

```yaml
capabilities:
  - type: mcp
    args:
      config:
        type: streamable-http
        name: weather
        url: https://mcp.example.com/weather
      name: weather               # ← servers 键必须与之一致(缺省回退 config 的 client_id)
      tool_prefix: weather        # 可选;不影响本能力配置键
  - type: tool-schema-overlap
    args:
      servers:                    # 按服务器隔离:servers.<服务器名>.<原始工具名>
        weather:
          get_weather:
            name: fetch_weather   # 模型可见工具名
            description: 查询城市天气
            param_names:          # 参数改名 {原名: 新名}
              location: city
            param_descriptions:   # 参数描述改写 {原名: 描述}
              location: 城市名,如 Beijing
            param_removals:       # 对模型隐藏的参数
              - api_key
            param_overrides:      # 按原名逐字段覆写
              api_key:
                default: sk-default   # 删除后运行期注入的默认值
            param_additions:      # 新增参数 {服务端参数名: ParamOverride}
              units:
                type: string
                enum: [celsius, fahrenheit]
                default: celsius
      global_overrides:           # 跨服务器兜底;同名时 servers 优先
        search:
          description: Search the web.
```

**ToolOverride 字段**(所有键均按原始参数名):

| 字段 | 作用 |
|---|---|
| `name` | 改写模型可见工具名 |
| `description` | 改写工具描述 |
| `param_names: {原名: 新名}` | 参数改名 |
| `param_descriptions: {原名: 描述}` | 参数描述改写 |
| `param_overrides: {原名: ParamOverride}` | 逐字段覆写:description/type/enum/required/name/default |
| `param_additions: {服务端名: ParamOverride}` | 新增参数(`name` 字段可另指定模型可见名) |
| `param_removals: [原名...]` | 对模型隐藏参数 |

**`default` 三态语义**:省略 = 不动原 schema;显式 `null` = 删除原 schema 中的 default;给定值 = 写入 schema 且在运行期注入。

**schema 改写顺序(固定)**:removals → param_descriptions → param_overrides → renames → additions → required 去重。改名与改描述互不干扰;被 `param_removals` 删除的参数必须在 `param_overrides.<原名>.default` 配置默认值(若原 schema 为 required),否则**首次列出工具时抛 `ValidationError`,agent 启动失败**。

## 运行期参数还原(desharing)

模型按改写后的 schema 产出参数,执行前 `wrap_tool_execute` 用**同一份配置**做逆映射:参数名还原(`city` → `location`)、被删参数注入配置默认值(`api_key` = `sk-default`)、新增参数缺省时注入默认值(`units` = `celsius`),再以原始参数名调用上游 MCP 工具。schema 改写与运行期还原共享配置,不存在两套映射漂移。

## 校验与降级

- **配置期**:Pydantic 校验配置自洽性(参数名产出唯一性、default∈enum、类型匹配、removal∩rename/addition 冲突、跨工具改名目标冲突)。
- **首次列出工具期**:依赖真实 schema 的校验(删除 required 参数未配默认、改名目标撞现有属性、新增参数撞现有属性)→ `ValidationError`,agent 启动失败。配置键与列出工具对不上(typo)同样报错 —— **拼错的配置不会被静默忽略**。
- **降级安全**:工具缺失身份 metadata(非 McpServerCap 管道注入的工具)时**透传不做任何改写**并记 warning,绝不凭工具名猜测归属,防止跨服务器误伤。

## 与 tool_prefix / ToolDisplayCapability 组合

三层可叠加,各管一层:

```
上游 MCP 工具 get_weather
  → PrefixedToolset(tool_prefix=weather):weather_get_weather   # 命名空间
  → SchemaOverrideToolset:fetch_weather + schema 改写            # 语义层
  → ToolDisplayCapability RenamedToolset:展示名                  # 展示层
```

调用反解按相反顺序自动还原,上游始终收到原始工具名与原始参数。两台服务器有同名工具时**必须**用 `servers.<服务器名>` 分别配置;`global_overrides` 中的同名跨工具改名目标在配置期即被拒绝。

## 文件与职责

| 文件 | 职责 |
|---|---|
| `src/wolfharness/capabilities/tool_schema_overlap_capability.py` | capability + `SchemaOverrideToolset`(WrapperToolset 子类)+ schema 改写/参数还原 |
| `src/wolfharness/capabilities/tool_schema_overlap_config.py` | Pydantic 配置模型 + 身份 metadata 键常量 |
| `src/wolfharness/capabilities/mcp_server_cap.py` | `_build_toolset` 注入来源身份 metadata |
| `src/wolfharness/tools/base.py` | `to_pydantic_ai`/`_generate_schema_override_prepare` 的 metadata 保留补丁 |

## 已知约束

- 身份 metadata 由 wolfharness 的 `McpServerCap` 管道注入;绕过该管道自建工具集的第三方 capability 不参与改写(透传降级)。
- 改写只作用于**模型可见层**,不改动上游服务器;参数增删的合法性以配置期 + 首次列出校验为准。
- `param_additions` 注入的默认值只在本能力配置的服务器工具上生效;上游是否接受新增参数取决于服务端实现。
- 注册方式:entry point `wolfharness.capabilities` 组 `tool-schema-overlap`,YAML 中 `type: tool-schema-overlap`。示例见 `docs/tool-schema-overlap-capability.example.yaml` 与 `tests/config/test_tool_schema_overlap_yaml.py`。
