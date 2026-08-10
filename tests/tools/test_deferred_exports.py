"""Tests for CallDeferred and ApprovalRequired re-exports from wolfharness.tools.

These are pydantic-ai exception classes used by tool authors for:
- CallDeferred: Signal that tool execution should be deferred.
  Takes optional `metadata` dict, stored as `.metadata`.
- ApprovalRequired: Signal that human-in-the-loop approval is needed.
  Takes optional `metadata` dict, stored as `.metadata`.
"""

from __future__ import annotations

import pytest

from wolfharness.tools import ApprovalRequired, CallDeferred


pytestmark = pytest.mark.unit


def test_call_deferred_is_exception():
    """CallDeferred should be a proper exception class."""
    assert issubclass(CallDeferred, Exception)


def test_call_deferred_can_be_raised():
    """CallDeferred should be raise-able without arguments."""
    with pytest.raises(CallDeferred):
        raise CallDeferred


def test_call_deferred_with_metadata():
    """CallDeferred should accept optional metadata dict."""
    exc = CallDeferred({"key": "value"})
    assert exc.metadata == {"key": "value"}


def test_call_deferred_is_pydantic_ai_class():
    """CallDeferred should be the actual pydantic-ai class, not a wrapper."""
    from pydantic_ai.exceptions import CallDeferred as PydanticCallDeferred

    assert CallDeferred is PydanticCallDeferred


def test_approval_required_is_exception():
    """ApprovalRequired should be a proper exception class."""
    assert issubclass(ApprovalRequired, Exception)


def test_approval_required_can_be_raised():
    """ApprovalRequired should be raise-able without arguments."""
    with pytest.raises(ApprovalRequired):
        raise ApprovalRequired


def test_approval_required_with_metadata():
    """ApprovalRequired should accept optional metadata dict."""
    exc = ApprovalRequired({"key": "value"})
    assert exc.metadata == {"key": "value"}


def test_approval_required_is_pydantic_ai_class():
    """ApprovalRequired should be the actual pydantic-ai class, not a wrapper."""
    from pydantic_ai.exceptions import ApprovalRequired as PydanticApprovalRequired

    assert ApprovalRequired is PydanticApprovalRequired


def test_both_in_wolfharness_tools_all():
    """Both exceptions should be listed in wolfharness.tools.__all__."""
    from wolfharness import tools

    assert "CallDeferred" in tools.__all__
    assert "ApprovalRequired" in tools.__all__
