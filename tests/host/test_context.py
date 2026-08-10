"""Unit tests for HostContext frozen dataclass.

Covers immutability, construction with required fields,
and default factory values.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import pytest

from wolfharness.host.context import HostContext
from wolfharness.host.stubs import CapabilityCache, ModelCache, ModelRegistry


pytestmark = pytest.mark.unit


def _make_context(**overrides: Any) -> HostContext:
    """Build a HostContext with MagicMock for all required fields."""
    defaults: dict[str, Any] = {
        "manifest": MagicMock(),
        "storage": MagicMock(),
        "vfs_registry": MagicMock(),
        "connection_registry": MagicMock(),
        "mcp": MagicMock(),
        "skills_registry": MagicMock(),
        "prompt_manager": MagicMock(),
        "process_manager": MagicMock(),
        "file_ops": MagicMock(),
        "todos": MagicMock(),
        "session_pool": None,
        "config_file_path": None,
    }
    defaults.update(overrides)
    return HostContext(**defaults)


def test_hostcontext_is_frozen():
    """Given a HostContext, when setting a field, then FrozenInstanceError is raised."""
    ctx = _make_context()
    with pytest.raises(FrozenInstanceError):
        ctx.config_id = "changed"  # type: ignore[misc]


def test_hostcontext_can_be_constructed_with_all_required_fields():
    """Given all required fields, when constructing HostContext, then it succeeds."""
    ctx = _make_context()
    assert ctx.manifest is not None
    assert ctx.storage is not None
    assert ctx.vfs_registry is not None
    assert ctx.connection_registry is not None
    assert ctx.mcp is not None
    assert ctx.skills_registry is not None
    assert ctx.prompt_manager is not None
    assert ctx.process_manager is not None
    assert ctx.file_ops is not None
    assert ctx.todos is not None


def test_default_factory_fields_have_correct_defaults():
    """Given no overrides for stub fields, when constructing HostContext, then defaults are set."""
    ctx = _make_context()
    assert isinstance(ctx.capability_cache, CapabilityCache)
    assert isinstance(ctx.model_registry, ModelRegistry)
    assert isinstance(ctx.model_cache, ModelCache)
    assert ctx.config_id is None
    assert ctx.tenant_id is None


def test_session_pool_defaults_to_none():
    """Given session_pool=None, when constructing HostContext, then session_pool is None."""
    ctx = _make_context(session_pool=None)
    assert ctx.session_pool is None


def test_config_file_path_accepts_string():
    """Given a string config path, when constructing HostContext, then it is stored."""
    ctx = _make_context(config_file_path="/path/to/config.yml")
    assert ctx.config_file_path == "/path/to/config.yml"


def test_main_agent_name_defaults_to_none():
    """Given no main_agent_name, when constructing HostContext, then it defaults to None."""
    ctx = _make_context()
    assert ctx.main_agent_name is None


def test_main_agent_name_can_be_set():
    """Given main_agent_name, when constructing HostContext, then it is stored."""
    ctx = _make_context(main_agent_name="primary")
    assert ctx.main_agent_name == "primary"
