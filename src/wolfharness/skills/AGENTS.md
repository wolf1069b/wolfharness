# skills — Skill Discovery and Capability Integration

## Where to Look

| Task | File |
|---|---|
| Skill model (YAML frontmatter, lazy loading) | `skill.py` |
| SkillCapability (AbstractCapability) | `capability.py` |
| SkillsRegistry (auto-discovery) | `registry.py` |
| SkillsManager (pool lifecycle) | `manager.py` |
| SkillMcpManager (session-scoped MCP) | `skill_mcp_manager.py` |
| SkillToolManager (Python tool import) | `skill_tool_manager.py` |
| SkillCommand (slash commands) | `command.py` |
| Skill URI resolver (`skill://`) | `uri_resolver.py` |
| SkillsInstructionProvider (XML injection) | `instruction_provider.py` |
| Skill config models | `wolfharness_config/skills.py` |

## Conventions

- **Skills parse YAML frontmatter**: `Skill` model uses `extra="forbid"` to reject unknown keys. Instructions lazy-load from `SKILL.md`.
- **SkillCapability wraps each Skill**: Provides instructions (`get_instructions`), tools (`get_toolset`), and tool filtering (`get_wrapper_toolset`).
- **Tool prefixing prevents name collisions**: Python tools get `{skill_name}__tool__`, MCP tools get `{skill_name}__mcp__`.
- **Two tool flavors**: Python tools via `tools` field (`SkillToolConfig` with `import_path` like `"os:getcwd"`) imported eagerly by `SkillToolManager`. MCP servers via `mcp_servers` field connected lazily per-run by `SkillMcpManager`.
- **mcp.json companion file**: Claude Desktop format `{"mcpServers": {...}}` in skill directory takes precedence over frontmatter. Env vars (`${VAR}`) expanded. Only filesystem skills (UPath) can have companion files.
- **SkillMcpManager is session-scoped**: Per `(session_id, server_name)` pair, lazy on first access, idle timeout 5 min, exponential backoff retry (3 attempts). `on_run_ended()` triggers cleanup.
- **allowed_tools via FilteredToolset**: `parsed_allowed_tools()` parses space/comma-separated frontmatter. `get_wrapper_toolset()` wraps in `FilteredToolset` dropping unlisted tools.
- **Skill commands are protocol-agnostic**: `SkillCommand` works across ACP, AG-UI, and OpenCode without protocol-specific code.

## Anti-Patterns

- **Creating SkillCapability manually**: Use `SkillsManager` which handles discovery and lifecycle. Manual creation skips MCP/tool manager wiring.
