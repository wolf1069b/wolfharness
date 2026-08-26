# Filter plain-text evidence entries before OPA/OPS/OPL evidence_uris

`_ticket_evidence` blindly merged `cited_references[].uri` with every
`evidence` entry. `evidence` is a free-form user expression: it may carry
provider URIs or plain audit text (e.g. `QuotedText: ...` /
`Matched knowledge snippet: ...`). Plain text reaching the engine's
`_validate_opa_uris` raised

    OPA URI must be a provider resource URI or local wiki/raw URI,
    got 'QuotedText: ...'.

and failed the whole external OPA/OPS ticket submission as
`pending_external_submission`.

Each engine now exposes `is_valid_op_uri` (the same provider/wiki/raw
format rule enforced by `_validate_opa_uris`, extracted so the check is
reusable). The ticket closures capture that predicate and pass it to
`_ticket_evidence`, which keeps `cited_references[].uri` verbatim and only
admits `evidence` entries that pass it. When an engine does not expose the
predicate, the historical blind merge is preserved.