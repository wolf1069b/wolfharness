"""Validate invalid deferred_kind x deferred_strategy combinations on Tool.

These tests enforce the rules documented in
.omo/notepads/durable-execution/decisions.md:
  - unapproved + continue → ToolError
  - unapproved + stream   → ToolError
  - external + block      → valid
  - external + continue   → valid
  - any + stream          → NotImplementedError
"""

import pytest

from wolfharness.tools.base import Tool
from wolfharness.tools.exceptions import ToolError


pytestmark = pytest.mark.unit


class TestInvalidCombinations:
    """Invalid combos must raise at Tool instantiation."""

    def test_unapproved_continue_raises(self):
        """Unapproved + continue is disallowed (would bypass approval)."""
        with pytest.raises(ToolError, match="unapproved"):
            Tool(
                name="test_unapproved_continue",
                deferred=True,
                deferred_kind="unapproved",
                deferred_strategy="continue",
            )

    def test_unapproved_stream_raises(self):
        """Unapproved + stream is disallowed."""
        with pytest.raises(ToolError, match="unapproved"):
            Tool(
                name="test_unapproved_stream",
                deferred=True,
                deferred_kind="unapproved",
                deferred_strategy="stream",
            )


class TestValidCombinations:
    """Valid combos must instantiate without error."""

    def test_external_block_valid(self):
        """External + block is the default and always valid."""
        t = Tool(
            name="test_external_block",
            deferred=True,
            deferred_kind="external",
            deferred_strategy="block",
        )
        assert t.deferred is True
        assert t.deferred_kind == "external"
        assert t.deferred_strategy == "block"

    def test_external_continue_valid(self):
        """External + continue allows non-blocking external tools."""
        t = Tool(
            name="test_external_continue",
            deferred=True,
            deferred_kind="external",
            deferred_strategy="continue",
        )
        assert t.deferred is True
        assert t.deferred_kind == "external"
        assert t.deferred_strategy == "continue"


class TestStreamNotImplemented:
    """Stream strategy is deferred to a follow-up change."""

    def test_external_stream_raises_not_implemented(self):
        """External + stream is not yet implemented."""
        with pytest.raises(NotImplementedError, match="stream"):
            Tool(
                name="test_external_stream",
                deferred=True,
                deferred_kind="external",
                deferred_strategy="stream",
            )

    def test_unapproved_block_stream_raises_not_implemented(self):
        """Unapproved + stream raises ToolError first (kind check wins)."""
        with pytest.raises(ToolError, match="unapproved"):
            Tool(
                name="test_unapproved_block_stream",
                deferred=True,
                deferred_kind="unapproved",
                deferred_strategy="stream",
            )


class TestDeferredFalseSkipsValidation:
    """When deferred=False, no validation should trigger."""

    def test_deferred_false_skips(self):
        """Any combo is accepted when deferred=False."""
        t = Tool(
            name="test_skip",
            deferred=False,
            deferred_kind="unapproved",
            deferred_strategy="continue",
        )
        assert t.deferred is False
        # No error raised — validation skipped
