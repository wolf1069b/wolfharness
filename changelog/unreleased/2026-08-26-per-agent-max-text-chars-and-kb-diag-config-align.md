# Per-agent configurable resource read truncation + resource subscribe-on-read

## ResourceConfig.max_text_chars

`ResourceConfig` now accepts a `max_text_chars` field (default 10 000,
minimum 100) controlling the maximum text length per `read_resource` call
before truncation. Previously the limit was a hardcoded constant
(``_DEFAULT_READ_TEXT_LIMIT = 10_000``); now it is configurable per agent
via the YAML config, with a programmatic override on
``ResourceCapability.__init__``:

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

## Resource subscribe-on-read wiring

`McpServerCap` now best-effort subscribes to resource URIs after a
successful `read_resource()` call, enabling `notifications/resources/updated`
notifications for resources the agent has read. Tracked subscriptions are
automatically re-established after reconnect and cleaned up on disconnect.

This is a no-op for servers that declare `subscribe: false` — the
subscribe call fails silently and the read proceeds normally. When the
server enables subscription support, the wiring activates automatically.
