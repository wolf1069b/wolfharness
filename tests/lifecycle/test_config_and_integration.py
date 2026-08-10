"""Tests for LifecycleConfig, create_dimensions factory, and agent integration."""

from __future__ import annotations

import pytest

from wolfharness.lifecycle.comm_channel import DirectChannel
from wolfharness.lifecycle.event_transport import InProcessTransport
from wolfharness.lifecycle.factory import create_dimensions
from wolfharness.lifecycle.journal import DurableJournal, MemoryJournal
from wolfharness.lifecycle.snapshot_store import (
    DurableSnapshotStore,
    MemorySnapshotStore,
)
from wolfharness_config.lifecycle import LifecycleConfig
from wolfharness_config.nodes import BaseAgentConfig


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# LifecycleConfig tests
# ---------------------------------------------------------------------------


def test_lifecycle_config_defaults() -> None:
    """LifecycleConfig defaults to all-memory with mark_interrupted."""
    config = LifecycleConfig()
    assert config.journal == "memory"
    assert config.snapshot == "memory"
    assert config.recover_strategy == "mark_interrupted"


def test_lifecycle_config_durable_journal() -> None:
    """Can set journal to durable."""
    config = LifecycleConfig(journal="durable")
    assert config.journal == "durable"
    assert config.snapshot == "memory"
    assert config.recover_strategy == "mark_interrupted"


def test_lifecycle_config_durable_snapshot() -> None:
    """Can set snapshot to durable."""
    config = LifecycleConfig(snapshot="durable")
    assert config.journal == "memory"
    assert config.snapshot == "durable"
    assert config.recover_strategy == "mark_interrupted"


def test_lifecycle_config_retry_strategy() -> None:
    """Can set recover_strategy to retry."""
    config = LifecycleConfig(recover_strategy="retry")
    assert config.journal == "memory"
    assert config.snapshot == "memory"
    assert config.recover_strategy == "retry"


def test_lifecycle_config_all_durable() -> None:
    """Can set all fields to durable."""
    config = LifecycleConfig(
        journal="durable",
        snapshot="durable",
        recover_strategy="retry",
    )
    assert config.journal == "durable"
    assert config.snapshot == "durable"
    assert config.recover_strategy == "retry"


def test_lifecycle_config_is_all_defaults_true() -> None:
    """is_all_defaults() returns True for default config."""
    config = LifecycleConfig()
    assert config.is_all_defaults() is True


def test_lifecycle_config_is_all_defaults_false() -> None:
    """is_all_defaults() returns False when any field is non-default."""
    assert not LifecycleConfig(journal="durable").is_all_defaults()
    assert not LifecycleConfig(snapshot="durable").is_all_defaults()
    assert not LifecycleConfig(recover_strategy="retry").is_all_defaults()


def test_lifecycle_config_frozen() -> None:
    """LifecycleConfig is frozen (immutable)."""
    config = LifecycleConfig()
    with pytest.raises((TypeError, ValueError)):
        config.journal = "durable"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# create_dimensions tests
# ---------------------------------------------------------------------------


def test_create_dimensions_none_returns_all_none() -> None:
    """create_dimensions(None, ...) returns all-None tuple."""
    result = create_dimensions(None, "session1")
    assert len(result) == 5
    assert all(item is None for item in result)


def test_create_dimensions_defaults_returns_all_none() -> None:
    """create_dimensions with all-default config returns all-None tuple."""
    config = LifecycleConfig()
    result = create_dimensions(config, "session1")
    assert len(result) == 5
    assert all(item is None for item in result)


def test_create_dimensions_durable_journal() -> None:
    """create_dimensions with journal=durable returns DurableJournal."""
    config = LifecycleConfig(journal="durable")
    trigger, journal, snapshot, comm, transport = create_dimensions(
        config,
        "test_session",
    )
    assert trigger is None  # RunHandle fills in ImmediateTrigger
    assert isinstance(journal, DurableJournal)
    assert isinstance(snapshot, MemorySnapshotStore)  # snapshot still memory
    assert isinstance(comm, DirectChannel)
    assert isinstance(transport, InProcessTransport)


def test_create_dimensions_durable_snapshot() -> None:
    """create_dimensions with snapshot=durable returns DurableSnapshotStore."""
    config = LifecycleConfig(snapshot="durable")
    trigger, journal, snapshot, comm, transport = create_dimensions(
        config,
        "test_session",
    )
    assert trigger is None
    assert isinstance(journal, MemoryJournal)  # journal still memory
    assert isinstance(snapshot, DurableSnapshotStore)
    assert isinstance(comm, DirectChannel)
    assert isinstance(transport, InProcessTransport)


def test_create_dimensions_all_durable() -> None:
    """create_dimensions with all durable returns durable implementations."""
    config = LifecycleConfig(
        journal="durable",
        snapshot="durable",
        recover_strategy="retry",
    )
    trigger, journal, snapshot, comm, transport = create_dimensions(
        config,
        "test_session",
    )
    assert trigger is None
    assert isinstance(journal, DurableJournal)
    assert isinstance(snapshot, DurableSnapshotStore)
    assert isinstance(comm, DirectChannel)
    assert isinstance(transport, InProcessTransport)


def test_create_dimensions_comm_channel_wraps_journal() -> None:
    """DirectChannel wraps the journal created by create_dimensions."""
    config = LifecycleConfig(journal="durable")
    _, journal, _, comm, _ = create_dimensions(config, "test_session")
    assert comm is not None
    # DirectChannel stores journal in _journal
    assert comm._journal is journal  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# BaseAgentConfig integration tests
# ---------------------------------------------------------------------------


def test_base_agent_config_lifecycle_defaults_to_none() -> None:
    """BaseAgentConfig.lifecycle defaults to None."""
    config = BaseAgentConfig(name="test_agent")
    assert config.lifecycle is None


def test_base_agent_config_accepts_lifecycle() -> None:
    """BaseAgentConfig accepts a LifecycleConfig."""
    lifecycle = LifecycleConfig(journal="durable", snapshot="durable")
    config = BaseAgentConfig(
        name="test_agent",
        lifecycle=lifecycle,
    )
    assert config.lifecycle is not None
    assert config.lifecycle.journal == "durable"
    assert config.lifecycle.snapshot == "durable"
    assert config.lifecycle.recover_strategy == "mark_interrupted"


def test_base_agent_config_lifecycle_optional() -> None:
    """BaseAgentConfig.lifecycle is optional in YAML."""
    config = BaseAgentConfig(
        name="test_agent",
        lifecycle=None,
    )
    assert config.lifecycle is None
