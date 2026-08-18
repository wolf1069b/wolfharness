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
automatically scope to the first allowed prefix when a allowlist is set.