# Faster resource listing — parallel providers, caching, timeouts

`list_resources`/`list_resource_templates` (the agent-facing
`ResourceCapability` tools) were slow because each provider (MCP server,
Viking capability) was queried **sequentially** with a fresh network round
trip per call, and a dead server re-paid its connection retry backoff on
every invocation.

- **Parallel providers**: `ResourceCapability` now queries all visible
  providers concurrently via `asyncio.gather`, so total latency is bounded
  by the slowest provider instead of the sum of all. Each provider runs
  under a 10s timeout — a hung or unreachable server is skipped with a
  warning rather than blocking the whole listing. Pagination now happens
  before row formatting.
- **Result caching**: `McpServerCap.list_resources()` and
  `list_resource_templates()` cache their results and invalidate them on
  `resources/list_changed` notifications (the notification hook was already
  wired but previously unused for caching). `resource_exists()` reuses the
  cached listing.
- **Connect cooldown**: a failed MCP connection enters a 30s cooldown, so
  repeated calls do not re-pay the 3-attempt exponential-backoff retry for
  a server that is down.
- **Caching invalidation on reconnect**: `McpServerCap` caches are cleared
  on connection drop → reconnect and on exit, so listings never go stale
  after a server restarts (previously only `resources/list_changed`
  notifications invalidated them — many servers never send those).
- **Viking**: per-directory recursive `ls()` calls inside
  `VikingCapability.list_resources()` now run in parallel instead of
  serially.
- **New `web_fetch` tool**: fetches a web page over HTTP(S) and returns its
  text content (50k-char cap). Refuses non-http(s) schemes and hosts that
  resolve to private/loopback/link-local addresses (SSRF guard); TLS
  verification always on; binary responses point at `download_file`. Wired
  into the default filesystem toolset.