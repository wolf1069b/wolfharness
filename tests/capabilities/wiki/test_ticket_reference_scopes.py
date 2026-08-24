"""Ticket reference scopes are independent from the ticket write scope."""

from __future__ import annotations

from typing import Any

import pytest

from wolfharness.capabilities.wiki.tickets.opa import OPAMixin
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
