"""Backward-compat re-export; actual code moved to wolfharness.capabilities.wiki.build."""

from wolfharness.capabilities.wiki.build import *  # noqa: F403
from wolfharness.capabilities.wiki.build import (  # type: ignore[attr-defined]  # noqa: F401
    WikiBuildCapability,
    _build_tool_fns,
)
