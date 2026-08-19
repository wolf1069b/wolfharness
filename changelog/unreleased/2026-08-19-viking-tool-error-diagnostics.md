# Viking tool errors now include the exception type

All 15 `viking_*` tools previously rendered failures as
`viking_search error: {e}`, relying on `str(e)` for the diagnostic text.
Some exceptions carry an empty message — notably `httpx.ReadTimeout('')`
when a slow knowledge-graph search exceeds the configured timeout — which
produced a useless `viking_search error:` with no trailing context and no
indication of what went wrong.

Tool error returns now follow `viking_search error ({ExcType}): {e}`, so
an empty-message timeout renders as `viking_search error (ReadTimeout):`
and the failure class is always identifiable even when the message is
blank. This applies uniformly to all 15 tools (search, find, recall,
grep, glob, ls, read, expand, write, edit, mkdir, add_resource, forget,
link, set_tags).

Also adds a regression test asserting that empty-message exceptions still
surface their exception type in the tool return value.