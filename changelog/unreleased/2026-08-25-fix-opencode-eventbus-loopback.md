# Fix EventBus loopback: duplicate first user message on serve + attach

`wolfharness serve-opencode` + `opencode attach` rendered the first user
message twice in the TUI (issue #380). Root cause was a design flaw: the
OpenCode server republished its own protocol projections
(`MessageUpdatedEvent` / `PartUpdatedEvent`) back into the same EventBus
that carries native agent events, creating a feedback loop. Prior fixes had
patched each crossing of the loop (`60d6fda50` sync clears replay buffer,
`cde629b6d` first-connect `replay=False`, `47baa747b6` C4 consumer skip)
but never removed the loop itself.

## What changed

- **SSE direct-wire projections**: `ServerState.broadcast_event()` now
  delivers OpenCode projections straight to per-connection SSE subscriber
  queues. The `OpenCodeEventBridge` (which republished into the EventBus) is
  deleted; the EventBus carries only native agent events.
- **Loopback isolation**: `EventBus.publish()` accepts a `source_hint` and
  `EventBus.subscribe()` accepts `exclude_source`, so a producer can never
  re-consume its own output. The OpenCode session consumer subscribes with
  `exclude_source={"opencode_event_bridge"}` as belt-and-suspenders.
- **Replay alignment**: the OpenCode session consumer subscribes with
  `replay=False` (matching the SSE endpoint's first-connect policy), so
  replayed native events can no longer be re-converted into duplicate
  projections. Recovery paths opt in to `replay=True` explicitly via a new
  `_get_subscription_replay()` hook.
- **Reconnect replay preserved**: per-session projection buffers in
  `ServerState` keep `Last-Event-ID` conditional replay working on the
  direct-wire path.
- **ACP `_displayed_message_ids` retained**: it guards a separate
  duplication vector (the same native event published twice with one
  `message_id`), which this change does not eliminate; fixing that at the
  publish source is a follow-up.
