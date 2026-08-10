"""Lifecycle package: types, Protocols, and default implementations.

The lifecycle subsystem provides the six dimensions of the RunLoop:
TriggerSource, Journal, SnapshotStore, CommChannel, EventTransport,
and the RunLoop itself.

This module exports the foundational types, Protocols, TriggerSource
implementations, and SnapshotStore implementations. Additional default
implementations (MemoryJournal, CommChannel, etc.) will be added in
subsequent tasks.
"""

from __future__ import annotations

from wolfharness.lifecycle.comm_channel import DirectChannel, ProtocolChannel
from wolfharness.lifecycle.event_transport import InProcessTransport
from wolfharness.lifecycle.factory import create_dimensions
from wolfharness.lifecycle.journal import DurableJournal, MemoryJournal
from wolfharness.lifecycle.protocols import (
    CommChannel,
    EventTransport,
    Journal,
    SnapshotStore,
    TriggerSource,
)
from wolfharness.lifecycle.snapshot_store import (
    DurableSnapshotStore,
    MemorySnapshotStore,
)
from wolfharness.lifecycle.triggers import (
    ChannelTrigger,
    ImmediateTrigger,
    ProtocolTrigger,
    ScheduledTrigger,
)
from wolfharness.lifecycle.types import (
    DeliveryMode,
    EventEnvelope,
    Feedback,
    Prompt,
    ResumeResult,
    RunOutcome,
    RunState,
    ToolExecutionRecord,
)

__all__ = [
    "ChannelTrigger",
    "CommChannel",
    "DeliveryMode",
    "DirectChannel",
    "DurableJournal",
    "DurableSnapshotStore",
    "EventEnvelope",
    "EventTransport",
    "Feedback",
    "ImmediateTrigger",
    "InProcessTransport",
    "Journal",
    "MemoryJournal",
    "MemorySnapshotStore",
    "Prompt",
    "ProtocolChannel",
    "ProtocolTrigger",
    "ResumeResult",
    "RunOutcome",
    "RunState",
    "ScheduledTrigger",
    "SnapshotStore",
    "ToolExecutionRecord",
    "TriggerSource",
    "create_dimensions",
]
