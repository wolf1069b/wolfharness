"""Import-level smoke tests for repomap submodules.

Verifies that each repomap module can be imported without ImportError.
Does not call any repomap functions.
"""

from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.unit


REPOMAP_MODULES = [
    "wolfharness.repomap.context",
    "wolfharness.repomap.core",
    "wolfharness.repomap.languages",
    "wolfharness.repomap.outline",
    "wolfharness.repomap.tags",
    "wolfharness.repomap.types",
    "wolfharness.repomap.utils",
]


@pytest.mark.parametrize("module_name", REPOMAP_MODULES)
def test_repomap_importable(module_name: str) -> None:
    importlib.import_module(module_name)
