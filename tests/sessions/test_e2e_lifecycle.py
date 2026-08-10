"""E2E test: session lifecycle with SQLModelProvider.

Tests the full flow:
1. Create session and save with SQLModelProvider
2. Save checkpoint
3. Update session status to checkpointed (save_session metadata update)
4. Load session and verify status + pending_deferred_calls restored
5. Verify checkpoint data preserved

This test uses real SQL storage (no mocks) to catch integration bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.orchestrator.core import SessionController
from wolfharness.sessions.models import PendingDeferredCall, SessionData
from wolfharness.storage.manager import StorageManager
from wolfharness_config.storage import SQLStorageConfig, StorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from pathlib import Path

    from wolfharness import AgentPool

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def sql_e2e_storage(tmp_path: Path) -> tuple[StorageManager, SQLModelProvider]:
    """Create real SQL storage with StorageManager + SQLModelProvider."""
    db_path = tmp_path / "test_e2e_lifecycle.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)

    session_store = SQLModelProvider(config)
    storage_config = StorageConfig(providers=[config])
    storage_manager = StorageManager(config=storage_config)

    await session_store.__aenter__()
    await storage_manager.__aenter__()

    yield storage_manager, session_store

    await storage_manager.__aexit__(None, None, None)
    await session_store.__aexit__(None, None, None)


async def test_e2e_session_lifecycle_with_sql_model_provider(
    sql_e2e_storage: tuple[StorageManager, SQLModelProvider],
) -> None:
    """E2E: create → checkpoint → save → load → resume → verify deferred calls.

    Full flow:
    1. Save session with pending_deferred_calls
    2. Save checkpoint via StorageManager
    3. Update session status to checkpointed
    4. Load session — verify status and pending_deferred_calls restored
    5. Load checkpoint — verify data preserved
    """
    storage_manager, session_store = sql_e2e_storage
    session_id = "e2e-lifecycle-001"
    agent_name = "test-agent"

    # --- Step 1: Create and save session ---
    pending_call = PendingDeferredCall(
        tool_call_id="tc-e2e-001",
        tool_name="elicit_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
        elicitation_message="Do you agree?",
    )

    session_data = SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type="native",
        status="active",
        pending_deferred_calls=[pending_call],
    )
    await session_store.save_session(session_data)

    # Verify save
    loaded = await session_store.load_session(session_id)
    assert loaded is not None
    assert loaded.status == "active"
    assert len(loaded.pending_deferred_calls) == 1
    assert loaded.pending_deferred_calls[0].tool_call_id == "tc-e2e-001"

    # --- Step 2: Save checkpoint via StorageManager ---
    from wolfharness.agents.native_agent.checkpoint import CheckpointManager

    checkpoint_mgr = CheckpointManager(storage_manager)
    await checkpoint_mgr.checkpoint(
        session_id=session_id,
        message_history=[],
        pending_calls=[pending_call],
    )

    # Verify checkpoint saved
    checkpoint_data = await checkpoint_mgr.load_checkpoint(session_id)
    assert checkpoint_data is not None
    assert len(checkpoint_data.pending_calls) == 1
    assert checkpoint_data.pending_calls[0].tool_call_id == "tc-e2e-001"

    # --- Step 3: Update session status to checkpointed ---
    updated = session_data.model_copy(
        update={"status": "checkpointed", "pending_deferred_calls": [pending_call]}
    )
    updated.touch()
    await session_store.save_session(updated)

    # --- Step 4: Load session — verify status and pending_deferred_calls ---
    loaded_after = await session_store.load_session(session_id)
    assert loaded_after is not None
    assert loaded_after.status == "checkpointed"
    assert len(loaded_after.pending_deferred_calls) == 1
    assert loaded_after.pending_deferred_calls[0].tool_call_id == "tc-e2e-001"
    assert loaded_after.pending_deferred_calls[0].elicitation_message == "Do you agree?"

    # --- Step 5: Load checkpoint — verify data preserved ---
    checkpoint_after = await session_store.load_checkpoint(session_id)
    assert checkpoint_after is not None

    # --- Step 6: Resume — clear pending calls, set status to active ---
    resumed = loaded_after.model_copy(update={"status": "active", "pending_deferred_calls": []})
    resumed.touch()
    await session_store.save_session(resumed)

    final = await session_store.load_session(session_id)
    assert final is not None
    assert final.status == "active"
    assert len(final.pending_deferred_calls) == 0

    # Checkpoint data should still exist (save_session preserves it)
    final_checkpoint = await session_store.load_checkpoint(session_id)
    assert final_checkpoint is not None


async def test_e2e_session_controller_with_sql_model_provider(
    sql_e2e_storage: tuple[StorageManager, SQLModelProvider],
    minimal_pool: AgentPool,
) -> None:
    """E2E: SessionController uses SQLModelProvider for session persistence.

    Verifies the migrated consumer path:
    - SessionController.store is a SQLModelProvider
    - get_or_create_session saves to SQLModelProvider
    - close_session marks as closed (not deleted)
    """
    _storage_manager, session_store = sql_e2e_storage

    ctrl = SessionController(pool=minimal_pool, store=session_store)
    session_id = "e2e-ctrl-001"

    # Create session
    _state, created = await ctrl.get_or_create_session(session_id, "test-agent")
    assert created is True

    # Verify persisted
    loaded = await session_store.load_session(session_id)
    assert loaded is not None
    assert loaded.agent_name == "test-agent"
    assert loaded.status == "active"

    # Close session — should mark as closed, not delete
    await ctrl.close_session(session_id)

    loaded_after = await session_store.load_session(session_id)
    assert loaded_after is not None
    assert loaded_after.status == "closed"
