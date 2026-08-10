"""Tests for _extract_hints() and Command model.

Unit tests for the skill autocomplete fix in AgentPool's OpenCode server.
"""

from __future__ import annotations

import pytest

from wolfharness_server.opencode_server.models.agent import Command
from wolfharness_server.opencode_server.routes.agent_routes import _extract_hints


pytestmark = pytest.mark.integration


# =============================================================================
# _extract_hints() tests
# =============================================================================


def test_extract_hints_no_placeholders() -> None:
    """Template with no placeholders returns empty list."""
    assert _extract_hints("Just a plain string") == []


def test_extract_hints_empty_string() -> None:
    """Empty string returns empty list."""
    assert _extract_hints("") == []


def test_extract_hints_none_input() -> None:
    """None input returns empty list (defensive against provider bugs)."""
    assert _extract_hints(None) == []


def test_extract_hints_single_numbered() -> None:
    r"""Single $1 placeholder returns ["$1"]."""
    assert _extract_hints("Analyze $1") == ["$1"]


def test_extract_hints_multiple_numbered() -> None:
    r"""Multiple numbered placeholders $1 $2 $3 sorted numerically."""
    assert _extract_hints("Analyze $1 and $2 then $3") == ["$1", "$2", "$3"]


def test_extract_hints_arguments_placeholder() -> None:
    r"""$ARGUMENTS placeholder returns ["$ARGUMENTS"]."""
    assert _extract_hints("Process $ARGUMENTS") == ["$ARGUMENTS"]


def test_extract_hints_mixed_numbered_and_arguments() -> None:
    r"""Mix of numbered and $ARGUMENTS: $1 $2 $ARGUMENTS."""
    assert _extract_hints("Analyze $1 with $2 using $ARGUMENTS") == [
        "$1",
        "$2",
        "$ARGUMENTS",
    ]


def test_extract_hints_numeric_sort_not_lexicographic() -> None:
    r"""Numeric sort: $1 $10 $2 → ["$1", "$2", "$10"], NOT lexicographic."""
    result = _extract_hints("$1 $10 $2")
    assert result == ["$1", "$2", "$10"]


def test_extract_hints_deduplicates() -> None:
    r"""Duplicate placeholders are deduplicated: $1 and $1 → ["$1"]."""
    assert _extract_hints("$1 and $1") == ["$1"]


def test_extract_hints_out_of_order_sorted() -> None:
    r"""Out-of-order placeholders are sorted: $3 $1 $2 → ["$1", "$2", "$3"]."""
    assert _extract_hints("$3 $1 $2") == ["$1", "$2", "$3"]


def test_extract_hints_non_placeholder_dollars() -> None:
    r"""Dollar signs that aren't placeholders ($foo, $NOTANUMBER) are ignored."""
    assert _extract_hints("$foo $NOTANUMBER") == []


def test_extract_hints_lone_dollar() -> None:
    r"""Just $ alone is not a placeholder."""
    assert _extract_hints("$") == []


def test_extract_hints_adjacent_placeholders() -> None:
    r"""Adjacent $1$2 are both extracted."""
    assert _extract_hints("$1$2") == ["$1", "$2"]


def test_extract_hints_only_arguments_no_numbered() -> None:
    r"""Template with only $ARGUMENTS and no numbered placeholders."""
    assert _extract_hints("Run with $ARGUMENTS") == ["$ARGUMENTS"]


def test_extract_hints_large_numbers() -> None:
    r"""Large numeric placeholders sorted correctly: $100 $2 $1 → ["$1", "$2", "$100"]."""
    assert _extract_hints("$100 $2 $1") == ["$1", "$2", "$100"]


# =============================================================================
# Command model tests
# =============================================================================


def test_command_default_construction() -> None:
    """Command with only name uses correct defaults."""
    cmd = Command(name="test")
    assert cmd.name == "test"
    assert cmd.description is None
    assert cmd.source == "command"
    assert cmd.template == ""
    assert cmd.subtask is False
    assert cmd.hints == []
    assert cmd.agent is None
    assert cmd.model is None


def test_command_full_construction() -> None:
    """Command with all fields set."""
    cmd = Command(
        name="my-skill",
        description="A great skill",
        agent="coder",
        model="openai:gpt-4o",
        source="skill",
        template="Analyze $1 and $ARGUMENTS",
        subtask=True,
        hints=["$1", "$ARGUMENTS"],
    )
    assert cmd.name == "my-skill"
    assert cmd.description == "A great skill"
    assert cmd.agent == "coder"
    assert cmd.model == "openai:gpt-4o"
    assert cmd.source == "skill"
    assert cmd.template == "Analyze $1 and $ARGUMENTS"
    assert cmd.subtask is True
    assert cmd.hints == ["$1", "$ARGUMENTS"]


def test_command_model_dump_includes_all_fields() -> None:
    """model_dump() includes all fields (None values included)."""
    cmd = Command(name="test")
    data = cmd.model_dump()
    assert "name" in data
    assert "description" in data
    assert "source" in data
    assert "template" in data
    assert "subtask" in data
    assert "hints" in data
    assert "agent" in data
    assert "model" in data


def test_command_model_dump_exclude_none() -> None:
    """model_dump(exclude_none=True) omits fields with None value."""
    cmd = Command(name="test")
    data = cmd.model_dump(exclude_none=True)
    assert "name" in data
    assert "description" not in data
    assert "agent" not in data
    assert "model" not in data
    # Non-None defaults are still present
    assert "source" in data
    assert "template" in data
    assert "subtask" in data
    assert "hints" in data


def test_command_source_literal_command() -> None:
    """Source accepts 'command' literal."""
    cmd = Command(name="test", source="command")
    assert cmd.source == "command"


def test_command_source_literal_mcp() -> None:
    """Source accepts 'mcp' literal."""
    cmd = Command(name="test", source="mcp")
    assert cmd.source == "mcp"


def test_command_source_literal_skill() -> None:
    """Source accepts 'skill' literal."""
    cmd = Command(name="test", source="skill")
    assert cmd.source == "skill"


def test_command_description_optional() -> None:
    """Description is optional and defaults to None."""
    cmd = Command(name="test")
    assert cmd.description is None


def test_command_source_defaults_to_command() -> None:
    """Source defaults to 'command' when not provided."""
    cmd = Command(name="test")
    assert cmd.source == "command"


def test_command_source_none_is_valid() -> None:
    """Source can be explicitly set to None."""
    cmd = Command(name="test", source=None)
    assert cmd.source is None
