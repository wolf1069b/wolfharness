"""Ticket reference scopes are independent from the ticket write scope."""

from __future__ import annotations

from typing import Any

import pytest

from wolfharness.capabilities.wiki.tickets.opa import OPAMixin
from wolfharness.capabilities.wiki.tickets.ticket import EvalRevision, _ticket_evidence
from wolfharness.capabilities.wiki.tickets.ticket_engine import TicketEngine


pytestmark = pytest.mark.unit


class _FakeRawFs:
    root_uri = "viking://resources/805/raw"


class _FakeStore:
    root_uri = "viking://resources/810test"

    def is_wiki_uri(self, uri: str) -> bool:
        return uri.startswith(self.root_uri + "/")

    def read_text(self, key: str) -> str | None:
        del key
        return None


class _FakeOpaTools(OPAMixin):
    def __init__(self) -> None:
        self.store = _FakeStore()
        self._raw_fs = _FakeRawFs()

    def read_resource(self, uri: str, line_numbers: bool = False) -> str | None:
        del uri, line_numbers
        return None


def _ticket_engine() -> TicketEngine:
    engine = TicketEngine.__new__(TicketEngine)
    engine.store = _FakeStore()
    engine._raw_fs = _FakeRawFs()
    return engine


@pytest.mark.parametrize("factory", [_FakeOpaTools, _ticket_engine])
def test_ticket_references_may_use_read_scope_outside_write_scope(factory: Any) -> None:
    """A ticket written under one scope may cite a resource from another scope."""
    tools = factory()

    tools._validate_opa_uris(
        "viking://resources/730/wikitest/Fault/tmp/sy215c/发动机不启动.draft.md",
        ["viking://resources/730/wikitest/manual.md"],
    )
    tools._validate_op_evidence(
        ["viking://resources/730/wikitest/manual.md"],
        ["viking://resources/730/wikitest/Fault/tmp/sy215c/发动机不启动.draft.md"],
    )


@pytest.mark.parametrize("factory", [_FakeOpaTools, _ticket_engine])
def test_ticket_references_still_reject_non_provider_paths(factory: Any) -> None:
    tools = factory()

    with pytest.raises(ValueError, match="OPA URI must be"):
        tools._validate_opa_uris("not-a-provider-reference", [])


def _mixed_revision() -> EvalRevision:
    """One revision with URI citations plus text and URI evidence entries."""
    return EvalRevision(
        ticket_id="OPA-001",
        kind="OPA",
        cited_references=[
            {
                "ref_id": "knowledge-1",
                "title": "draft",
                "uri": "viking://resources/730/wikitest/Fault/x.md",
            },
        ],
        evidence=[
            "QuotedText: 启动时蓄电池电压 >10V 正常范围",
            "Matched knowledge snippet: 蓄电池电压 SY75C >12V",
            "viking://resources/805/raw/cases/c1.md",
        ],
    )


@pytest.mark.parametrize("factory", [_FakeOpaTools, _ticket_engine])
def test_is_valid_op_uri_accepts_provider_uris_and_rejects_text(factory: Any) -> None:
    """The exposed predicate mirrors ``_validate_opa_uris`` format rules."""
    tools = factory()

    assert tools.is_valid_op_uri("viking://resources/730/wikitest/manual.md")
    assert tools.is_valid_op_uri("viking://resources/805/raw/cases/c.md")
    assert tools.is_valid_op_uri("viking://resources/810test/Fault/f.md")

    assert not tools.is_valid_op_uri("QuotedText: 启动时蓄电池电压 >10V")
    assert not tools.is_valid_op_uri("Matched knowledge snippet: 蓄电池电压 SY75C >12V")
    assert not tools.is_valid_op_uri("not-a-provider-reference")


@pytest.mark.parametrize("factory", [_FakeOpaTools, _ticket_engine])
def test_ticket_evidence_extracts_only_uri_entries_when_predicate_given(
    factory: Any,
) -> None:
    """Plain-text ``evidence`` entries must not leak into evidence URIs.

    Regression: the ticket closures re-derive evidence URIs from the eval
    revision dict (bypassing the xeno-side adapter filter); without the
    predicate the engine's ``_validate_opa_uris`` rejected text entries such
    as ``QuotedText: ...`` with ``OPA URI must be a provider resource URI``.
    """
    tools = factory()
    revision = _mixed_revision()

    evidence = _ticket_evidence(revision, is_uri_valid=tools.is_valid_op_uri)

    assert evidence == [
        "viking://resources/730/wikitest/Fault/x.md",
        "viking://resources/805/raw/cases/c1.md",
    ]


@pytest.mark.parametrize("factory", [_FakeOpaTools, _ticket_engine])
def test_ticket_evidence_without_predicate_keeps_historical_behavior(
    factory: Any,
) -> None:
    """No predicate (third-party engine) preserves the historical blind merge."""
    factory()
    revision = _mixed_revision()

    evidence = _ticket_evidence(revision, is_uri_valid=None)
    assert len(evidence) == 4
    assert any(item.startswith("QuotedText:") for item in evidence)
