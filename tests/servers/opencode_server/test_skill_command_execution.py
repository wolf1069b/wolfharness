"""Tests for OpenCode server skill command execution.

Tests skill command execution including template processing.
"""

from __future__ import annotations

import pytest

from wolfharness_server.opencode_server.routes.session_routes import _process_skill_template


pytestmark = pytest.mark.integration


class TestProcessSkillTemplate:
    """Tests for _process_skill_template helper function."""

    def test_no_placeholders_appends_arguments(self):
        """When template has no placeholders, arguments are wrapped in user_request tag."""
        template = "Please analyze this code"
        arguments = "some file.py"
        result = _process_skill_template(template, arguments)
        assert (
            result
            == "Please analyze this code\n\n<user_request>\n\nsome file.py\n\n</user_request>"
        )

    def test_single_placeholder(self):
        """Single $1 placeholder gets first argument."""
        template = "Analyze $1"
        arguments = "file.py"
        result = _process_skill_template(template, arguments)
        assert result == "Analyze file.py"

    def test_multiple_placeholders(self):
        """Multiple placeholders get respective arguments."""
        template = "Analyze $1 and $2"
        arguments = "file1.py file2.py"
        result = _process_skill_template(template, arguments)
        assert result == "Analyze file1.py and file2.py"

    def test_arguments_placeholder(self):
        """$ARGUMENTS gets all arguments as single string."""
        template = "Analyze: $ARGUMENTS"
        arguments = "file1.py file2.py file3.py"
        result = _process_skill_template(template, arguments)
        assert result == "Analyze: file1.py file2.py file3.py"

    def test_last_placeholder_swallows_remaining(self):
        """Last positional placeholder gets remaining arguments."""
        template = "Analyze $1 with options $2"
        arguments = "file.py --verbose --no-cache"
        result = _process_skill_template(template, arguments)
        assert result == "Analyze file.py with options --verbose --no-cache"

    def test_missing_arguments_empty_string(self):
        """Missing arguments produce empty string."""
        template = "Analyze $1 and $2"
        arguments = "file1.py"
        result = _process_skill_template(template, arguments)
        assert result == "Analyze file1.py and "

    def test_empty_arguments(self):
        """Empty arguments string handles gracefully."""
        template = "Analyze $1"
        arguments = ""
        result = _process_skill_template(template, arguments)
        assert result == "Analyze "

    def test_none_arguments(self):
        """None arguments handles gracefully."""
        template = "Analyze $1"
        arguments = None
        result = _process_skill_template(template, arguments)
        assert result == "Analyze "

    def test_mixed_placeholders(self):
        """Mix of positional and ARGUMENTS placeholders."""
        template = "Analyze $1 with options: $ARGUMENTS"
        arguments = "file.py --verbose"
        result = _process_skill_template(template, arguments)
        # $1 swallows remaining args since it's the last positional placeholder
        assert result == "Analyze file.py --verbose with options: file.py --verbose"

    def test_no_arguments_no_placeholders(self):
        """Template without placeholders and no arguments returns as-is."""
        template = "Please analyze this code"
        arguments = ""
        result = _process_skill_template(template, arguments)
        assert result == "Please analyze this code"

    def test_multiple_placeholders_with_extra_args(self):
        """Multiple placeholders with more args than placeholders."""
        template = "Compare $1 vs $2"
        arguments = "a.py b.py c.py d.py"
        result = _process_skill_template(template, arguments)
        # $2 is last, so it swallows remaining args
        assert result == "Compare a.py vs b.py c.py d.py"
