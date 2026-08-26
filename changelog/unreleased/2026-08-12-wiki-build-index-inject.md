# Wiki build root index injection (B-scheme)

Added config-driven first-turn `<openviking-index>` injection to
`WikiBuildCapability`, parallel to the Viking capability's dynamic index
inject. `WikiBuildConfig` (and the entry-point `__init__` kwargs) gain
`index_enabled`, `index_max_tokens`, `index_limit`; a first-turn handler
lists the config-resolved `wiki`/`raw`/`bom` build roots (no server
enumeration, no viking tools) and injects them as a
`SystemPromptPart` before the latest user message. Disabled by default.
Also root-causes two pre-existing mypy errors: the lazy
`xeno_adp_agentic` import is suppressed via the correct
`xeno_adp_agentic.*` override, and the `RoleFilter.get_wrapper_toolset`
arg-type mismatch is fixed by typing the local toolset as
`FunctionToolset[Any]`.
