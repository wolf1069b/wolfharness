"""Tests for StorageManager.get_project_provider() capability-based selection."""

from __future__ import annotations

import pytest

from wolfharness.storage.manager import StorageManager
from wolfharness_config.storage import MemoryStorageConfig, StorageConfig
from wolfharness_storage.memory_provider import MemoryStorageProvider
from wolfharness_storage.opencode_provider import OpenCodeStorageProvider


pytestmark = pytest.mark.unit


def _make_manager_with_providers(*capable: bool) -> StorageManager:
    """Create a StorageManager with mock providers.

    Args:
        capable: Whether each provider has can_store_projects=True
    """
    config = StorageConfig()
    manager = StorageManager(config)
    manager.providers = []
    for is_capable in capable:
        provider = MemoryStorageProvider(MemoryStorageConfig())
        provider.can_store_projects = is_capable
        manager.providers.append(provider)
    return manager


def test_get_project_provider_returns_first_capable() -> None:
    """First capable provider is returned."""
    manager = _make_manager_with_providers(False, True, True)
    provider = manager.get_project_provider()
    assert provider is manager.providers[1]
    assert provider.can_store_projects is True


def test_get_project_provider_skips_incapable() -> None:
    """Incapable providers are skipped, capable one selected."""
    manager = _make_manager_with_providers(False, False, True)
    provider = manager.get_project_provider()
    assert provider is manager.providers[2]
    assert provider.can_store_projects is True


def test_get_project_provider_raises_when_none_capable() -> None:
    """RuntimeError raised when no provider supports project storage."""
    manager = _make_manager_with_providers(False, False)
    with pytest.raises(RuntimeError, match="No storage provider supports project storage"):
        manager.get_project_provider()


def test_get_project_provider_raises_when_no_providers() -> None:
    """RuntimeError raised when provider list is empty."""
    manager = _make_manager_with_providers()
    with pytest.raises(RuntimeError, match="No storage provider supports project storage"):
        manager.get_project_provider()


def test_opencode_provider_is_capable() -> None:
    """OpenCodeStorageProvider has can_store_projects=True after implementing project methods."""
    provider = OpenCodeStorageProvider()
    assert provider.can_store_projects is True


def test_memory_provider_is_capable_by_default() -> None:
    """MemoryStorageProvider has can_store_projects=True by default."""
    provider = MemoryStorageProvider(MemoryStorageConfig())
    assert provider.can_store_projects is True
