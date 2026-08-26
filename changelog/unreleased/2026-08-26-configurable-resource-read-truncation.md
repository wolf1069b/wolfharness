# Configurable resource read truncation limit

`ResourceCapability` now accepts a `max_text_chars` parameter (default 10 000)
controlling the maximum text length per `read_resource` call before
truncation. Previously this limit was hardcoded in
`resolve_resource_content()`, with the tail silently discarded and no way for
the model to recover it — a problem for knowledge-base sources whose chapter
resources can exceed 10k characters.

The truncation suffix now also guides the model to use a narrower resource URI
(e.g. a chapter or chunk URI) or a paginated read tool for full content,
instead of leaving it with an unexplained cut.

Constructors that instantiate `ResourceCapability()` with no arguments are
unaffected (the default is preserved).
