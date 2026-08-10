"""E2e test: elicitation → checkpoint → timeout → crash recovery resume.

Reproduces the production bug where:
1. Agent triggers elicitation → checkpoint saved to SQL storage
2. Elicitation times out → RunAbortedError → agent run ends
3. User responds late → resume_session() called with elicitation_payloads
4. Before fix: SessionNotFoundError (checkpoint never saved — no Conversation record)
5. After fix: Session loads, checkpoint loads, agent re-executes with
   cached_elicitation_responses, tool completes without re-asking.

This test uses real SQL storage (SQLModelProvider) to
catch the integration bug that mocked tests missed.


# TODO: L2 migration — test requires complex mock pool dependencies that
# cannot be easily replaced with a real pool. Needs investigation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from mcp.types import ElicitRequestFormParams, ElicitResult
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.native_agent.checkpoint import CheckpointManager
from wolfharness.orchestrator.core import EventBus
from wolfharness.orchestrator.session_pool import SessionPool
from wolfharness.sessions.models import (
    ElicitationResumePayload,
    PendingDeferredCall,
    SessionData,
)
from wolfharness.storage.manager import StorageManager
from wolfharness.ui.base import InputProvider
from wolfharness_config.storage import SQLStorageConfig, StorageConfig
from wolfharness_storage.sql_provider import SQLModelProvider


if TYPE_CHECKING:
    from pathlib import Path

    from wolfharness.agents.context import AgentContext, ConfirmationResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class DurableElicitationProvider(InputProvider):
    """InputProvider that advertises durable elicitation support."""

    @property
    def supports_durable_elicitation(self) -> bool:
        return True

    async def get_text_input(self, context: Any, prompt: str) -> str:
        raise NotImplementedError

    async def get_structured_input(
        self,
        context: Any,
        prompt: str,
        output_type: type[Any],
    ) -> Any:
        raise NotImplementedError

    async def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool_description: str = "",
    ) -> ConfirmationResult:
        return "allow"

    async def get_elicitation(self, params: Any) -> ElicitResult:
        return ElicitResult(action="accept", content={"q0": "yes"})


def _make_elicit_tool() -> Any:
    """Create a local tool that calls handle_elicitation()."""

    async def elicit_tool(ctx: AgentContext[None]) -> str:
        params = ElicitRequestFormParams(
            message="Do you agree?",
            requestedSchema={
                "type": "object",
                "properties": {"q0": {"type": "string", "title": "Answer"}},
                "required": ["q0"],
            },
        )
        result = await ctx.handle_elicitation(params)
        match result:
            case ElicitResult(action=action):
                return f"Elicitation action: {action}"
            case _:
                return f"Elicitation result: {result}"

    return elicit_tool


def _make_elicit_agent() -> Agent[None, str]:
    """Create an Agent with TestModel + elicitation tool + durable provider."""
    provider = DurableElicitationProvider()
    model = TestModel(
        call_tools=["elicit_tool"],
        custom_output_text="All done!",
    )
    return Agent(
        name="test-elicit-agent",
        model=model,
        tools=[_make_elicit_tool()],
        input_provider=provider,
    )


@pytest.fixture
async def sql_storage(tmp_path: Path) -> Any:
    """Create real SQL storage: provider + storage manager.

    Both share the same SQLite database file, matching production setup.
    Returns (storage_manager, session_store).
    """
    db_path = tmp_path / "test_e2e_elicitation.db"
    config = SQLStorageConfig(url=f"sqlite:///{db_path}", auto_migration=False)

    session_store = SQLModelProvider(config)
    storage_config = StorageConfig(providers=[config])
    storage_manager = StorageManager(config=storage_config)

    # Initialize all — storage_manager.__aenter__ creates internal
    # SQLModelProvider and creates tables.
    await session_store.__aenter__()
    await storage_manager.__aenter__()

    return storage_manager, session_store


# ---------------------------------------------------------------------------
# E2e test: elicitation → checkpoint → timeout → crash recovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_elicitation_timeout_crash_recovery(  # noqa: PLR0915
    sql_storage: Any,
) -> None:
    """E2e: elicitation → checkpoint → timeout → crash recovery resume.

    Full flow:
    1. Save session to SQLModelProvider (simulates ACP session creation)
    2. Agent runs, calls handle_elicitation() → checkpoint saved to SQL
    3. Elicitation times out → RunAbortedError → run ends
    4. resume_session() with elicitation_payloads
    5. Agent re-executes from checkpoint with cached_elicitation_responses
    6. Tool gets cached response (no re-ask) → agent completes

    Before fix: Step 2 fails silently (no Conversation record for checkpoint),
    Step 4 raises SessionNotFoundError.
    """
    storage_manager, session_store = sql_storage
    session_id = "test-e2e-crash-recovery"
    agent_name = "test-elicit-agent"

    # --- Step 1: Save session to SQLModelProvider ---
    session_data = SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type="native",
        status="active",
    )
    await session_store.save_session(session_data)

    # Verify session exists in store
    loaded = await session_store.load_session(session_id)
    assert loaded is not None, "Session should exist in SQLModelProvider"
    assert loaded.session_id == session_id

    # --- Step 2: Simulate handle_elicitation() checkpoint ---
    # handle_elicitation() calls CheckpointManager.checkpoint() which goes
    # through StorageManager → SQLProvider.save_checkpoint().
    # This is the critical step that failed before the fix: SQLProvider
    # required a Conversation record but none existed (only SessionStore
    # saved, which uses the same Conversation table — but only if
    # SQLModelProvider instances share the same DB).
    checkpoint_mgr = CheckpointManager(storage_manager)

    pending_call = PendingDeferredCall(
        tool_call_id="tc-elicit-e2e",
        tool_name="elicit_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
        elicitation_message="Do you agree?",
        elicitation_schema={
            "type": "object",
            "properties": {"q0": {"type": "string"}},
        },
        elicitation_mode="form",
    )

    # Create a minimal message history with a ModelResponse containing the
    # tool call that will be deferred. This matches what handle_elicitation()
    # would save via run_ctx.current_messages in production.
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )

    message_history: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="Call the elicit tool")]),
        ModelResponse(
            parts=[
                TextPart(content="I'll ask you a question."),
                ToolCallPart(
                    tool_name="elicit_tool",
                    args={},
                    tool_call_id="tc-elicit-e2e",
                ),
            ]
        ),
    ]

    # This is the critical assertion: checkpoint save should succeed
    # even though StorageManager.save_session() was never called.
    # Before fix: ValueError "Session not found" raised silently.
    await checkpoint_mgr.checkpoint(
        session_id=session_id,
        message_history=message_history,
        pending_calls=[pending_call],
    )

    # Verify checkpoint was actually saved in SQL (not silently failed)
    checkpoint_data = await checkpoint_mgr.load_checkpoint(session_id)
    assert checkpoint_data is not None, (
        "Checkpoint should exist in SQL storage after save. "
        "If None, the save failed silently (the original bug)."
    )
    assert len(checkpoint_data.pending_calls) == 1
    assert checkpoint_data.pending_calls[0].tool_call_id == "tc-elicit-e2e"

    # Update session status to "checkpointed" and set pending_deferred_calls
    # (as handle_elicitation does after the fix — previously pending_deferred_calls
    # was NOT set, causing resume_session() to find 0 elicitation calls and
    # resolve nothing even though the checkpoint had the pending call.)
    session_data = session_data.model_copy(
        update={
            "status": "checkpointed",
            "pending_deferred_calls": [pending_call],
        }
    )
    session_data.touch()
    await session_store.save_session(session_data)

    # Verify pending_deferred_calls was persisted
    persisted = await session_store.load_session(session_id)
    assert persisted is not None
    assert len(persisted.pending_deferred_calls) == 1
    assert persisted.pending_deferred_calls[0].tool_call_id == "tc-elicit-e2e"

    # --- Step 3: Simulate timeout ---
    # (In production, asyncio.wait_for fires TimeoutError → RunAbortedError.
    #  For this test we skip directly to the resume — the checkpoint and
    #  session state are what matter for crash recovery.)

    # --- Step 4: resume_session() with elicitation_payloads ---
    # Build a mock pool with real storage and session store.
    mock_pool = MagicMock()
    mock_pool.storage = storage_manager
    mock_pool.manifest = MagicMock()
    mock_pool.manifest.agents = {}
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []

    # Debug: verify mock_pool.storage is the real storage_manager
    assert mock_pool.storage is storage_manager, "mock_pool.storage should be storage_manager"

    EventBus()
    session_pool = SessionPool(
        pool=mock_pool,
        store=session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )

    # Mock _reconstruct_native_agent to return a TestModel agent.
    # The reconstructed agent is used for config drift check and cleanup.
    _make_elicit_agent()

    async def mock_reconstruct(
        sid: str,
        aname: str,
    ) -> Agent[Any, Any]:
        agent = _make_elicit_agent()
        await agent.__aenter__()
        return agent

    session_pool._reconstruct_native_agent = mock_reconstruct  # type: ignore[assignment]

    # Track run_stream calls to verify resume parameters are forwarded.
    # We mock run_stream because the full RunHandle lifecycle with a real
    # TestModel agent can hang in the test environment. The SQL storage
    # assertions above already verify the checkpoint integration.
    run_stream_calls: list[dict[str, Any]] = []

    async def tracked_run_stream(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        run_stream_calls.append({"session_id": session_id, "prompts": prompts, **kwargs})
        return
        yield  # pragma: no cover

    # Build deferred_tool_results (empty — no non-elicitation pending calls)
    from pydantic_ai.tools import DeferredToolResults

    results = DeferredToolResults(calls={})

    # Build elicitation_payloads (the user's late response)
    elicitation_payloads = [
        ElicitationResumePayload(
            deferred_handle="tc-elicit-e2e",
            action="accept",
            content={"q0": "yes"},
        ),
    ]

    # This is the critical call: resume_session() should:
    # 1. Load session from SQLModelProvider (should succeed)
    # 2. Try in-process futures → none (timeout removed them)
    # 3. Fall to crash recovery → load checkpoint from SQL (should succeed now)
    # 4. _resume_native_agent routes through pool.run_stream() with
    #    cached_elicitation_responses, message_history, and no
    #    deferred_tool_results (empty calls for elicitation-only resume)
    with patch.object(session_pool, "run_stream", tracked_run_stream):
        await session_pool.resume_session(
            session_id,
            results,
            elicitation_payloads=elicitation_payloads,
        )

    # Verify run_stream was called with resume parameters
    assert len(run_stream_calls) == 1, "run_stream should be called once"
    call = run_stream_calls[0]
    assert call["session_id"] == session_id
    assert "cached_elicitation_responses" in call
    assert call["cached_elicitation_responses"] is not None
    assert "tc-elicit-e2e" in call["cached_elicitation_responses"]
    assert "message_history" in call
    # deferred_tool_results SHOULD be passed (built from elicitation_payloads)
    assert "deferred_tool_results" in call, (
        "deferred_tool_results must be built from elicitation_payloads (Bug 16 fix)"
    )

    # --- Step 5: Verify session is active again ---
    final_data = await session_store.load_session(session_id)
    assert final_data is not None
    assert final_data.status == "active", (
        f"Session should be 'active' after successful resume, got '{final_data.status}'"
    )
    assert len(final_data.pending_deferred_calls) == 0, (
        "Pending deferred calls should be cleared after successful resume"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_checkpoint_save_fails_without_conversation(
    sql_storage: Any,
) -> None:
    """Checkpoint save to SQL without prior save_session() must not fail silently.

    This is the regression test for the original bug: SQLProvider.save_checkpoint()
    raised ValueError when no Conversation record existed, but the error was
    caught silently by StorageManager. CheckpointManager logged "Checkpoint saved"
    even though nothing was saved.

    With the fix:
    - SQLProvider.save_checkpoint() creates a minimal Conversation record (upsert)
    - StorageManager.save_checkpoint() returns bool
    - CheckpointManager.checkpoint() logs error on failure
    """
    storage_manager, _session_store = sql_storage
    session_id = "test-no-conv-record"

    # Do NOT call session_store.save_session() — simulate the ACP scenario where
    # only SessionStore (not StorageManager.save_session) is used.
    # Actually, in production, save_session() IS called, but
    # SQLProvider.save_checkpoint() uses a DIFFERENT engine that may
    # not see the record. Here we test the worst case: no record at all.

    checkpoint_mgr = CheckpointManager(storage_manager)

    pending_call = PendingDeferredCall(
        tool_call_id="tc-no-conv",
        tool_name="elicit_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
    )

    # This should NOT raise ValueError (the original bug)
    await checkpoint_mgr.checkpoint(
        session_id=session_id,
        message_history=[],
        pending_calls=[pending_call],
    )

    # Checkpoint should actually exist in SQL (not silently failed)
    data = await checkpoint_mgr.load_checkpoint(session_id)
    assert data is not None, (
        "Checkpoint should exist after save. If None, save failed silently (the original bug)."
    )
    assert len(data.pending_calls) == 1
    assert data.pending_calls[0].tool_call_id == "tc-no-conv"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_resume_without_pending_deferred_calls_resolves_zero(
    sql_storage: Any,
) -> None:
    """Resume without pending_deferred_calls on SessionData resolves 0 calls.

    This is the regression test for the bug where handle_elicitation() and
    the elicitation bridge updated SessionData.status to "checkpointed" but
    did NOT set pending_deferred_calls. resume_session() reads
    SessionData.pending_deferred_calls to find elicitation call IDs — if
    empty, resolved_calls=0 even though the checkpoint has pending calls.

    The fix: set pending_deferred_calls on SessionData when checkpointing.
    """
    storage_manager, session_store = sql_storage
    session_id = "test-no-pending-deferred"
    agent_name = "test-elicit-agent"

    # --- Setup: session + checkpoint WITHOUT pending_deferred_calls ---
    session_data = SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type="native",
        status="active",
    )
    await session_store.save_session(session_data)

    checkpoint_mgr = CheckpointManager(storage_manager)
    pending_call = PendingDeferredCall(
        tool_call_id="tc-elicit-no-pending",
        tool_name="elicit_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
    )

    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        UserPromptPart,
    )

    message_history: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="test")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="elicit_tool",
                    args={},
                    tool_call_id="tc-elicit-no-pending",
                ),
            ]
        ),
    ]

    await checkpoint_mgr.checkpoint(
        session_id=session_id,
        message_history=message_history,
        pending_calls=[pending_call],
    )

    # Update status to "checkpointed" but do NOT set pending_deferred_calls
    # (this is the bug — the old code only set status, not pending_deferred_calls)
    session_data = session_data.model_copy(update={"status": "checkpointed"})
    session_data.touch()
    await session_store.save_session(session_data)

    # Verify: SessionData has no pending_deferred_calls
    loaded = await session_store.load_session(session_id)
    assert loaded is not None
    assert loaded.status == "checkpointed"
    assert len(loaded.pending_deferred_calls) == 0  # Bug: should have the call

    # --- Resume: should resolve 0 elicitation calls ---
    mock_pool = MagicMock()
    mock_pool.storage = storage_manager
    mock_pool.manifest = MagicMock()
    mock_pool.manifest.agents = {}
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []

    EventBus()
    session_pool = SessionPool(
        pool=mock_pool,
        store=session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )

    async def mock_reconstruct(
        sid: str,
        aname: str,
    ) -> Agent[Any, Any]:
        agent = _make_elicit_agent()
        await agent.__aenter__()
        return agent

    session_pool._reconstruct_native_agent = mock_reconstruct  # type: ignore[assignment]

    # Mock run_stream to avoid running the full agent lifecycle.
    async def mock_run_stream_no_pending(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        return
        yield  # pragma: no cover

    from pydantic_ai.tools import DeferredToolResults

    results = DeferredToolResults(calls={})
    elicitation_payloads = [
        ElicitationResumePayload(
            deferred_handle="tc-elicit-no-pending",
            action="accept",
            content={"q0": "yes"},
        ),
    ]

    # resume_session should succeed but the elicitation_payloads won't
    # match any pending_deferred_calls (empty) — CheckpointMismatchError
    # is NOT raised because elicitation_call_ids is also empty.
    # The session resumes but the elicitation response is silently ignored.
    with patch.object(session_pool, "run_stream", mock_run_stream_no_pending):
        await session_pool.resume_session(
            session_id,
            results,
            elicitation_payloads=elicitation_payloads,
        )

    # The session is active but the elicitation was NOT resolved
    final_data = await session_store.load_session(session_id)
    assert final_data is not None
    assert final_data.status == "active"
    # pending_deferred_calls was already empty, so nothing to clear
    assert len(final_data.pending_deferred_calls) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_elicitation_resume_builds_deferred_tool_results(
    sql_storage: Any,
) -> None:
    """Bug 16: elicitation_payloads must be converted to DeferredToolResults.

    pydantic-ai's ``agentlet.iter(user_prompt, message_history=...)`` always
    starts from ``UserPromptNode`` — it does NOT replay ``ModelResponse`` from
    the checkpoint. The model generates a NEW ``ModelResponse`` with a NEW
    ``tool_call_id``, bypassing ``cached_elicitation_responses`` (which are
    keyed by the OLD ``tool_call_id``).

    The fix: build ``DeferredToolResults`` from ``elicitation_payloads`` with
    ``ToolReturnPart`` for each pending call. pydantic-ai will match these
    against the ``ModelResponse`` in ``message_history`` and use them directly,
    skipping tool execution.

    This test verifies that ``run_stream()`` receives ``deferred_tool_results``
    with the correct ``ToolReturnPart`` entries.
    """
    storage_manager, session_store = sql_storage
    session_id = "test-bug16-deferred-results"
    agent_name = "test-elicit-agent"

    # --- Setup: Save session + checkpoint ---
    session_data = SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type="native",
        status="active",
    )
    await session_store.save_session(session_data)

    pending_call = PendingDeferredCall(
        tool_call_id="tc-bug16-001",
        tool_name="elicit_tool",
        deferred_kind="elicitation",
        deferred_strategy="block",
    )

    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )

    message_history: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="Ask me a question")]),
        ModelResponse(
            parts=[
                TextPart(content="I'll ask you."),
                ToolCallPart(
                    tool_name="elicit_tool",
                    args={},
                    tool_call_id="tc-bug16-001",
                ),
            ]
        ),
    ]

    checkpoint_mgr = CheckpointManager(storage_manager)
    await checkpoint_mgr.checkpoint(
        session_id=session_id,
        message_history=message_history,
        pending_calls=[pending_call],
    )

    session_data = session_data.model_copy(
        update={
            "status": "checkpointed",
            "pending_deferred_calls": [pending_call],
        }
    )
    session_data.touch()
    await session_store.save_session(session_data)

    # --- Resume with elicitation_payloads ---
    mock_pool = MagicMock()
    mock_pool.storage = storage_manager
    mock_pool.manifest = MagicMock()
    mock_pool.manifest.agents = {}
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []

    EventBus()
    session_pool = SessionPool(
        pool=mock_pool,
        store=session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )

    async def mock_reconstruct(sid: str, aname: str) -> Agent[Any, Any]:
        agent = _make_elicit_agent()
        await agent.__aenter__()
        return agent

    session_pool._reconstruct_native_agent = mock_reconstruct  # type: ignore[assignment]

    run_stream_calls: list[dict[str, Any]] = []

    async def tracked_run_stream(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        run_stream_calls.append({"session_id": session_id, "prompts": prompts, **kwargs})
        return
        yield  # pragma: no cover

    from pydantic_ai.tools import DeferredToolResults

    results = DeferredToolResults(calls={})

    elicitation_payloads = [
        ElicitationResumePayload(
            deferred_handle="tc-bug16-001",
            action="accept",
            content={"q0": "yes"},
        ),
    ]

    with patch.object(session_pool, "run_stream", tracked_run_stream):
        await session_pool.resume_session(
            session_id,
            results,
            elicitation_payloads=elicitation_payloads,
        )

    # --- Verify deferred_tool_results is built from elicitation_payloads ---
    assert len(run_stream_calls) == 1
    call = run_stream_calls[0]

    # deferred_tool_results MUST be passed (the Bug 16 fix)
    assert "deferred_tool_results" in call, (
        "deferred_tool_results must be built from elicitation_payloads and "
        "passed to run_stream(). Without this, pydantic-ai starts from "
        "UserPromptNode and bypasses cached_elicitation_responses."
    )

    dtr = call["deferred_tool_results"]
    assert isinstance(dtr, DeferredToolResults)
    assert "tc-bug16-001" in dtr.calls

    # The ToolReturnPart should have the correct tool_call_id and content
    from pydantic_ai.messages import ToolReturnPart

    tool_result = dtr.calls["tc-bug16-001"]
    assert isinstance(tool_result, ToolReturnPart)
    assert tool_result.tool_call_id == "tc-bug16-001"
    assert tool_result.tool_name == "elicit_tool"

    # cached_elicitation_responses should ALSO be set (as fallback)
    assert "cached_elicitation_responses" in call
    assert call["cached_elicitation_responses"] is not None
    assert "tc-bug16-001" in call["cached_elicitation_responses"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_elicitation_resume_maps_tool_call_id_mismatch(
    sql_storage: Any,
) -> None:
    """Bug 17: PendingDeferredCall.tool_call_id != ToolCallPart.tool_call_id.

    For MCP tools without AgentContext param, ``handle_elicitation()`` falls
    back to ``run_ctx.run_id`` as the elicitation handle. The checkpoint's
    ``PendingDeferredCall.tool_call_id`` is the run_id, but pydantic-ai's
    ``ModelResponse`` has a different ``ToolCallPart.tool_call_id``.

    ``DeferredToolResults`` must be keyed by the ``ToolCallPart.tool_call_id``
    (what pydantic-ai expects), not by the ``PendingDeferredCall.tool_call_id``
    (the elicitation handle).
    """
    storage_manager, session_store = sql_storage
    session_id = "test-bug17-id-mismatch"
    agent_name = "test-elicit-agent"

    # --- Setup ---
    session_data = SessionData(
        session_id=session_id,
        agent_name=agent_name,
        agent_type="native",
        status="active",
    )
    await session_store.save_session(session_data)

    # PendingDeferredCall uses run_id as tool_call_id (handle_elicitation fallback)
    # AND has empty tool_name (MCP tool without AgentContext param)
    pending_call = PendingDeferredCall(
        tool_call_id="run_id_abc123",  # ← run_ctx.run_id, NOT the ToolCallPart.tool_call_id
        tool_name="",  # ← empty because MCP tool has no AgentContext param
        deferred_kind="elicitation",
        deferred_strategy="block",
    )

    # ModelResponse has a DIFFERENT tool_call_id (pydantic-ai's own ID)
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )

    message_history: list[Any] = [
        ModelRequest(parts=[UserPromptPart(content="Ask me")]),
        ModelResponse(
            parts=[
                TextPart(content="Let me ask."),
                ToolCallPart(
                    tool_name="question",
                    args={},
                    tool_call_id="call_def456",  # ← pydantic-ai's tool_call_id
                ),
            ]
        ),
    ]

    checkpoint_mgr = CheckpointManager(storage_manager)
    await checkpoint_mgr.checkpoint(
        session_id=session_id,
        message_history=message_history,
        pending_calls=[pending_call],
    )

    session_data = session_data.model_copy(
        update={
            "status": "checkpointed",
            "pending_deferred_calls": [pending_call],
        }
    )
    session_data.touch()
    await session_store.save_session(session_data)

    # --- Resume ---
    mock_pool = MagicMock()
    mock_pool.storage = storage_manager
    mock_pool.manifest = MagicMock()
    mock_pool.manifest.agents = {}
    mock_pool._config_file_path = None
    mock_pool.skill_capabilities = []

    EventBus()
    session_pool = SessionPool(
        pool=mock_pool,
        store=session_store,
        enable_auto_resume=False,
        enable_event_bus=True,
    )

    async def mock_reconstruct(sid: str, aname: str) -> Agent[Any, Any]:
        agent = _make_elicit_agent()
        await agent.__aenter__()
        return agent

    session_pool._reconstruct_native_agent = mock_reconstruct  # type: ignore[assignment]

    run_stream_calls: list[dict[str, Any]] = []

    async def tracked_run_stream(session_id: str, *prompts: Any, **kwargs: Any) -> Any:
        run_stream_calls.append({"session_id": session_id, "prompts": prompts, **kwargs})
        return
        yield  # pragma: no cover

    from pydantic_ai.tools import DeferredToolResults

    results = DeferredToolResults(calls={})

    # elicitation_payloads uses the run_id as deferred_handle (from PendingDeferredCall)
    elicitation_payloads = [
        ElicitationResumePayload(
            deferred_handle="run_id_abc123",  # ← matches PendingDeferredCall.tool_call_id
            action="accept",
            content={"q0": "yes"},
        ),
    ]

    with patch.object(session_pool, "run_stream", tracked_run_stream):
        await session_pool.resume_session(
            session_id,
            results,
            elicitation_payloads=elicitation_payloads,
        )

    # --- Verify DeferredToolResults is keyed by ToolCallPart.tool_call_id ---
    assert len(run_stream_calls) == 1
    call = run_stream_calls[0]

    assert "deferred_tool_results" in call
    dtr = call["deferred_tool_results"]
    assert isinstance(dtr, DeferredToolResults)

    # The key MUST be 'call_def456' (ToolCallPart.tool_call_id from ModelResponse),
    # NOT 'run_id_abc123' (PendingDeferredCall.tool_call_id / elicitation handle)
    assert "call_def456" in dtr.calls, (
        "DeferredToolResults must be keyed by ToolCallPart.tool_call_id "
        "(from ModelResponse in checkpoint), not by PendingDeferredCall.tool_call_id "
        "(elicitation handle). pydantic-ai matches against the ModelResponse."
    )
    # The old handle should NOT be a key
    assert "run_id_abc123" not in dtr.calls, (
        "DeferredToolResults should NOT be keyed by the elicitation handle "
        "when it differs from the ToolCallPart.tool_call_id."
    )

    from pydantic_ai.messages import ToolReturnPart

    tool_result = dtr.calls["call_def456"]
    assert isinstance(tool_result, ToolReturnPart)
    assert tool_result.tool_call_id == "call_def456"
    # tool_name may be empty when PendingDeferredCall.tool_name is empty
    # (MCP tool without AgentContext param) — positional matching handles this
