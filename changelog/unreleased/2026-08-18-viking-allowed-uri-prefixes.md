# Restrict Viking knowledge-base access to configured URI prefixes

Adds an `allowed_uri_prefixes` option to `VikingCapabilityConfig` that
scopes shared knowledge-base access to a configured allowlist of
`viking://resources/...` URI prefixes.

- The allowlist applies **only** to the `viking://resources/` namespace:
  all `viking_*` tools and the @-mention flow
  (`list_resources()`/`read_resource()`/`resource_exists()`) reject
  `viking://resources/` URIs outside the listed prefixes.
- Every other namespace — `viking://user/...` (the agent's own memories,
  sessions, skills, and other users' namespaces), `viking://skills/`,
  etc. — is always allowed and governed by its own feature flags.

Previously the agent could access every resource under the whole
`viking://resources/` tree — there was no way to grant read access to a
single subtree such as `viking://resources/wiki/` without also allowing
`viking://resources/raw/`, `viking://resources/814/`, and so on.

Empty list (the default) preserves unrestricted behavior for backward
compatibility. `viking_search`/`viking_find` without a `target_uri`
automatically scope to the configured allowlist when one is set (see below).

## Search scoping covers all allowed prefixes (2026-08-19)

When a `target_uri` is omitted, `viking_search`/`viking_find` previously
scoped to only the **first** allowed prefix, silently dropping results from
the other allowed trees. `target_uri` now accepts a list, and both tools pass
**every** allowed prefix to the SDK — the server searches each tree and the
result set covers the full allowlist. Explicit `str` targets are still
validated against the allowlist as before; explicit `list` targets are
validated element-wise.

Alongside this, `get_instructions()` now renders a dynamic **Allowed URI
Prefixes** section listing the exact prefixes when the allowlist is
configured. The model sees the scoping boundary up front, so it can pass the
most specific `target_uri` directly and skip discovery probing (`viking_ls`),
which also avoids the slower whole-allowlist search.