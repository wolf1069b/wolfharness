# Per-agent configurable resource read truncation + kb_diag_agent config alignment

## ResourceConfig.max_text_chars

`ResourceConfig` now accepts a `max_text_chars` field (default 10 000,
minimum 100) controlling the maximum text length per `read_resource` call
before truncation. Previously this was only configurable programmatically
on `ResourceCapability.__init__`; now it is wired through the agent YAML
config so each agent can set its own limit:

```yaml
agents:
  my_agent:
    resources:
      enabled: true
      max_text_chars: 20000  # allow longer chapter reads for KB agents
```

`NativeAgent` now constructs a per-agent `ResourceCapability` instance
with the agent's `max_text_chars` instead of sharing the pool-level
instance, so different agents can have different truncation limits.

## kb_diag_agent.yaml aligned with live server

The example config is updated to match the knowledge_diag MCP server's
actual tool surface (v3.4.4, 6 tools):

- **Enabled `search_kb`** — removed from `disabled_tools`; it is the
  primary retrieval-first discovery entry point.
- **Added `get_doc_toc` rewrite** — structure navigation tool with
  page-number-aware TOC.
- **Added `read_chapter_page` rewrite** — page-based reading tool
  (`start_page` + `offset`), the server's recommended primary reading path.
- **Updated existing tool descriptions** — `read_resource`/`list_resources`
  now reference the page-based workflow (`search_kb → get_doc_toc →
  read_chapter_page`) as the primary path, with URI-based reading as
  complementary.
- **Added `search_kb` rewrite** — full param descriptions including
  `methods` (FULL/FAST/WIKI), `equipment_model` filter, `dataset_id`.

## Resource subscribe-on-read wiring

`McpServerCap` now best-effort subscribes to resource URIs after a
successful `read_resource()` call, enabling `notifications/resources/updated`
notifications for resources the agent has read. Tracked subscriptions are
automatically re-established after reconnect and cleaned up on disconnect.

This is a no-op for servers that declare `subscribe: false` (like the
current knowledge_diag v3.4.4) — the subscribe call fails silently and
the read proceeds normally. When the server enables subscription support
(FR-5), the wiring activates automatically.
