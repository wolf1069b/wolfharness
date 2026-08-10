"""Import-level smoke tests for all tool_impls submodules.

Verifies that each tool module can be imported without ImportError or
unexpected side effects. Does not call any tool functions.
"""

from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.unit


TOOL_IMPLS_MODULES = [
    "wolfharness.tool_impls.agent_cli",
    "wolfharness.tool_impls.bash",
    "wolfharness.tool_impls.delete_path",
    "wolfharness.tool_impls.download_file",
    "wolfharness.tool_impls.execute_code",
    "wolfharness.tool_impls.grep",
    "wolfharness.tool_impls.list_directory",
    "wolfharness.tool_impls.question",
    "wolfharness.tool_impls.read",
]


@pytest.mark.parametrize("module_name", TOOL_IMPLS_MODULES)
def test_tool_impls_importable(module_name: str) -> None:
    importlib.import_module(module_name)
