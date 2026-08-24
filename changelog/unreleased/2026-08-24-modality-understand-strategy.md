# "Understand" image strategy for the modality filter

The `ModalityFilterCapability` gains a new `understand` image strategy that
replaces unsupported image content with a **real text description** produced by
a vision LLM, instead of the generic placeholder from the `describe` strategy.

Configuration:

```yaml
capabilities:
  - type: modality_filter
    image_strategy: understand
    vision_model: openai:gpt-4o   # variant name or namespaced string
```

`vision_model` resolves through the agent manifest first (model variant names),
falling back to `infer_model` for namespaced strings such as `openai:gpt-4o`
(standalone / no-manifest runs). When `vision_model` is omitted and
`image_strategy: understand` is set, the strategy falls back to `describe` at
runtime. The strategy is image-only: `audio_strategy` / `video_strategy` /
`document_strategy` reject the value at config validation, and setting it
directly on the capability degrades to `describe`.

Safety guarantees:

- The call is bounded by a 30s timeout; on timeout the content falls back to
  `reference` (persisted to disk as `[file: <path>]`).
- Images over 10MB skip the vision call and fall back to `reference`.
- Any other vision-model failure is caught and falls back to `describe` — the
  agent turn can never break because of a vision model error.
- Identical image bytes are deduplicated via a per-instance content-hash cache,
  so repeated tool results cost at most one vision call.
- `before_model_request` history rebuilds never call the vision LLM — history
  is deterministic and degrades to `describe`.