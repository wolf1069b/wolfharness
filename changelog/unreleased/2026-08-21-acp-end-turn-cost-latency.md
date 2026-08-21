# Fix ACP end_turn latency caused by blocking token-cost fallback

On turn completion, `TokenCost.from_usage` runs before the
`StreamCompleteEvent` is emitted (cost must be reported in the turn-final
event). For models absent from the local `genai_prices` pricing snapshot, the
tokonomics fallback performs a blocking, untimed download of the LiteLLM
pricing table, delaying `end_turn` by seconds on every turn. The failure was
never cached, so every turn retried the download.

- **Prefetch at startup**: `BaseServer.start` now launches
  `prefetch_token_cost_cache` in the background, seeding the process-wide
  tokonomics pricing cache once so runtime lookups are memory hits.
- **Fallback guard**: the runtime fallback is wrapped in a 0.2s timeout, and
  failed models are negatively cached, degrading cost to 0 instead of
  blocking end_turn. Worst case adds at most 0.2s to a single turn.
- Adds unit tests covering the fast path, negative caching, timeout guard,
  prefetch seeding, and prefetch error suppression.