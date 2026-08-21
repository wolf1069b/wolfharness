# Information-preserving degradation for modality filter

The `ModalityFilterCapability` `describe` strategy previously replaced
unsupported multimodal content with a bare MIME placeholder such as
`[image/png]`. That token carried no filename, path, or identifier, so the
model could never retrieve the underlying content — a vision-capable
subagent or file tool had nothing to open, and the placeholder misled the
model into inventing content it never saw.

The placeholder is now **information-preserving** (RFC-0061): binary content
states its media type, that direct model processing is unsupported, and
whether a file identifier is available; control characters in caller-supplied
identifiers are escaped to prevent prompt injection via malformed filenames.
URL-type content keeps its `[image: url]` / `[audio: url]` form since the URL
is already retrievable.

A new opt-in `reference` strategy is added for each modality: instead of a
placeholder, the content bytes are persisted to a per-session scratch
directory under `tempfile.gettempdir()/wolfharness-modality/{session_id}/` and
replaced with a `[file: <path>]` reference a vision-capable subagent or the
agent's `read` tool can open. The directory is removed by
`after_node_run()`. URL and `UploadedFile` content has no local bytes and
falls back to `describe`.

Resolves wolf1069b/wolfharness#377.