# ToolSchemaOverlapCapability — capability-level tool schema semantic adaptation

Adds a declarative, capability-level channel for rewriting the model-visible schema
of assembled tools (tool/parameter renames, description rewrites, type/enum/default
overrides, parameter additions/removals) without touching MCP servers or
sub-capability code. Existing mechanisms covered only adjacent ground: `tool_prefix`
handles namespace-level collision avoidance and `schema_override` handles one tool
at a time; neither offered cross-capability semantic adaptation driven by agent
config.

- New `ToolSchemaOverlapCapability` + `SchemaOverrideToolset` (a pydantic-ai
  `WrapperToolset` subclass) in
  `src/wolfharness/capabilities/tool_schema_overlap_capability.py`: identity-driven
  override lookup, schema rewrite (fixed order: removals → descriptions → field
  overrides → renames → additions), first-listing fail-fast, reverse parameter
  desharing in `wrap_tool_execute`, and safe pass-through degradation when source
  identity metadata is absent. Ordering declares `wrapped_by=[ToolDisplayCapability]`
  so display renames sit outside schema renames.
- New Pydantic config models in
  `src/wolfharness/capabilities/tool_schema_overlap_config.py`
  (`ToolSchemaOverlapConfig`, `ToolOverride`, `ParamOverride`) with
  construction-time validation (name-collision checks, enum/default/type
  consistency, tri-state `default` semantics) plus the source-identity metadata
  key constants.
- `McpServerCap._build_toolset` stamps `server_name` and
  `original_mcp_tool_name` metadata on converted tools so matching never depends
  on (possibly prefixed) tool names and remains correct across same-named tools
  from different servers.
- Two metadata-preservation fixes in `src/wolfharness/tools/base.py`:
  `to_pydantic_ai()` now assigns the computed metadata on the `Tool.from_schema`
  branch, and `_generate_schema_override_prepare()` carries
  `metadata=tool_def.metadata` when rebuilding the `ToolDefinition`.
- Registered as entry point `tool-schema-overlap` in the
  `wolfharness.capabilities` group; exported from `wolfharness.capabilities`.
- Docs: `docs/explanation/tool-schema-overlap-capability.md` +
  `docs/tool-schema-overlap-capability.example.yaml`.
- Tests: 100 tests across `tests/capabilities/test_tool_schema_overlap_config.py`,
  `tests/capabilities/test_tool_schema_overlap.py`,
  `tests/capabilities/test_tool_schema_overlap_integration.py`, and
  `tests/config/test_tool_schema_overlap_yaml.py`, covering metadata survival,
  adversarial collisions, `tool_prefix` interop, display-layer composition,
  two-server routing isolation, degradation, and the YAML round-trip.
