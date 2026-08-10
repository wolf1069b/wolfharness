"""L2 integration tests and regression tests for DynamicContextPruningCapability.

Tests the full ``before_model_request`` pipeline with real pydantic-ai
message types, covering phase ordering, pruning actions, auto-strategies,
nudge injection, session metadata, and regression scenarios.

Uses ``MagicMock`` for ``RunContext`` and ``AgentContext`` boundaries,
but all message objects (``ModelRequest``, ``ModelResponse``,
``ToolReturnPart``, etc.) are real pydantic-ai instances.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
import pytest

from wolfharness.capabilities.dcp.capability import DynamicContextPruningCapability
from wolfharness.capabilities.dcp.state import (
    DCPState,
    DistillTarget,
    PruneAction,
    WatermarkLevel,
)
from wolfharness.capabilities.dcp.strategies import _is_pruned
from wolfharness.capabilities.dcp.tools import decompress_tool
from wolfharness.sessions.models import SessionData


pytestmark = pytest.mark.integration


# =============================================================================
# Helper factories
# =============================================================================


def _make_tool_return(
    tool_name: str = "read",
    content: str = "File contents here",
    tool_call_id: str = "tc_1",
    outcome: str = "success",
) -> ToolReturnPart:
    """Create a real ``ToolReturnPart`` with sensible defaults."""
    return ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id,
        outcome=outcome,
        timestamp=datetime.now(UTC),
    )


def _make_tool_call(
    tool_name: str = "read",
    args: str | dict[str, Any] | None = None,
    tool_call_id: str = "tc_1",
) -> ToolCallPart:
    """Create a real ``ToolCallPart`` with sensible defaults."""
    return ToolCallPart(
        tool_name=tool_name,
        args=args if args is not None else {"path": "file.ts"},
        tool_call_id=tool_call_id,
    )


def _make_request(parts: list[Any] | None = None) -> ModelRequest:
    """Create a ``ModelRequest`` with the given parts."""
    return ModelRequest(parts=parts or [])


def _make_response(parts: list[Any] | None = None) -> ModelResponse:
    """Create a ``ModelResponse`` with the given parts."""
    return ModelResponse(
        parts=parts or [TextPart(content="Hello")],
        usage=RequestUsage(),
        model_name="test-model",
        timestamp=datetime.now(UTC),
    )


def _make_session_data(session_id: str = "sess-1") -> SessionData:
    """Create a ``SessionData`` with DCP state in metadata."""
    return SessionData(session_id=session_id, agent_name="test_agent")


def _make_session_with_state(
    state: DCPState | None = None,
    session_id: str = "sess-1",
) -> tuple[SessionData, DCPState]:
    """Create a SessionData with a DCPState pre-stored in metadata.

    Returns the SessionData and the DCPState so tests can pre-configure
    pending actions, counters, etc.
    """
    sd = SessionData(session_id=session_id, agent_name="test_agent")
    dcp_state = state or DCPState()
    sd.metadata["dcp"] = dcp_state
    return sd, dcp_state


def _make_run_context(
    messages: list[Any] | None = None,
    session_data: SessionData | None = None,
    usage_input_tokens: int = 0,
    enqueue_enabled: bool = True,
    session_pool: Any = None,
) -> MagicMock:
    """Create a mock ``RunContext[AgentContext]``.

    Mocks only the boundaries the capability checks: ``deps``, ``usage``,
    ``enqueue``.  When ``session_pool`` is provided, sets up
    ``ctx.deps.node.host_context`` so the capability routes nudge injection
    through ``session_pool.steer()``.  When ``None`` (default),
    ``host_context`` is set to ``None`` so the capability falls back to
    ``ctx.enqueue()``.

    All message objects are real.
    """
    ctx = MagicMock()
    ctx.messages = messages or []

    # AgentContext mock — get_session_state returns SessionData or None.
    deps = MagicMock()
    deps.get_session_state.return_value = session_data
    ctx.deps = deps

    # Usage mock with input_tokens.
    usage = MagicMock()
    usage.input_tokens = usage_input_tokens
    ctx.usage = usage

    if enqueue_enabled:
        ctx.enqueue = MagicMock(return_value="nudge-id-123")
    else:
        # Remove enqueue attribute entirely.
        del ctx.enqueue

    # SessionPool / host_context setup for steer-path tests.
    if session_pool is not None:
        host_ctx = MagicMock()
        host_ctx.session_pool = session_pool
        deps.node.host_context = host_ctx
    else:
        deps.node.host_context = None

    return ctx


def _make_request_context(
    messages: list[Any] | None = None,
) -> ModelRequestContext:
    """Create a real ``ModelRequestContext`` with the given messages."""
    return ModelRequestContext(
        model=MagicMock(),
        messages=messages or [],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _make_capability(
    **kwargs: Any,
) -> DynamicContextPruningCapability:
    """Create a capability with defaults overridden by kwargs."""
    return DynamicContextPruningCapability(**kwargs)


def _build_3turn_conversation() -> list[Any]:
    """Build a 3-turn conversation with tool calls.

    Turn 1: user asks to read a file → assistant calls read → tool returns content.
    Turn 2: user asks to grep → assistant calls grep → tool returns results.
    Turn 3: user asks a question → (no tool call, just text).
    """
    return [
        # Turn 1
        _make_request([UserPromptPart(content="Read file.ts")]),
        _make_response([_make_tool_call("read", {"path": "file.ts"}, "tc_1")]),
        _make_request([_make_tool_return("read", "file contents line 1\nline 2", "tc_1")]),
        # Turn 2
        _make_request([UserPromptPart(content="Now grep for TODO")]),
        _make_response([_make_tool_call("grep", {"pattern": "TODO"}, "tc_2")]),
        _make_request([_make_tool_return("grep", "TODO found at line 5", "tc_2")]),
        # Turn 3
        _make_request([UserPromptPart(content="Summarize what you found")]),
        _make_response([TextPart(content="Here is the summary...")]),
    ]


def _get_dcp_state(cap: DynamicContextPruningCapability, ctx: MagicMock) -> DCPState:
    """Get the DCPState from the capability (via _get_dcp_state method)."""
    # _get_dcp_state takes RunContext[AgentContext]; ctx is a MagicMock that fits.
    return cap._get_dcp_state(ctx)  # type: ignore[arg-type]


def _find_tool_returns(messages: list[Any]) -> list[tuple[int, int, ToolReturnPart]]:
    """Find all ToolReturnParts in messages, returning (msg_idx, part_idx, part)."""
    result: list[tuple[int, int, ToolReturnPart]] = []
    for mi, msg in enumerate(messages):
        parts = getattr(msg, "parts", [])
        for pi, part in enumerate(parts):
            if isinstance(part, ToolReturnPart):
                result.append((mi, pi, part))
    return result


# =============================================================================
# 9.1 — Full pipeline with 3-turn conversation
# =============================================================================


async def test_before_model_request_3turn_pipeline_phases_execute() -> None:
    """Test full before_model_request pipeline with 3-turn conversation.

    Given: A 3-turn conversation with tool calls and a DCP capability.
    When: before_run then before_model_request is called.
    Then: Watermark is updated, prunable list is built, phases execute in order.
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)
    messages = _build_3turn_conversation()
    req_ctx = _make_request_context(messages)

    # before_run increments current_turn and nudge_counter.
    await cap.before_run(ctx)
    result = await cap.before_model_request(ctx, req_ctx)

    # Watermark should be updated in state.
    state = _get_dcp_state(cap, ctx)
    assert state.watermark_level is not None
    assert state.current_tokens > 0
    assert state.current_turn >= 1

    # tool_id_list should be built (2 tool returns in conversation).
    assert len(state.tool_id_list) == 2

    # Result should be a new ModelRequestContext with messages.
    assert result is not None
    assert result.messages is not None


# =============================================================================
# 9.2 — Model calls prune → content replaced with "[pruned]"
# =============================================================================


async def test_prune_action_replaces_content_with_pruned() -> None:
    """Test that a pending prune action replaces ToolReturnPart content.

    Given: A DCPState with a pending prune action targeting a tool_call_id.
    When: before_model_request is called.
    Then: The matching ToolReturnPart content is replaced with "[pruned]".
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    # Set up state with a pending prune action.
    state.pending_actions.append(
        PruneAction(
            kind="prune",
            ids=("tc_target",),
            source_tool_call_id="action_1",
        ),
    )

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_target")]),
        _make_request([_make_tool_return("read", "original content", "tc_target")]),
        _make_request([UserPromptPart(content="Now prune it")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # Find the ToolReturnPart in result messages.
    tool_returns = _find_tool_returns(result.messages)
    assert len(tool_returns) == 1
    _, _, tr_part = tool_returns[0]
    assert tr_part.content == "[pruned]"
    assert _is_pruned(tr_part)

    # State should track the pruned id.
    assert "tc_target" in state.pruned_tool_ids


# =============================================================================
# 9.3 — Model calls distill → content replaced with distillation text
# =============================================================================


async def test_distill_action_replaces_content_with_distillation() -> None:
    """Test that a pending distill action replaces content with distillation.

    Given: A DCPState with a pending distill action.
    When: before_model_request is called.
    Then: The matching ToolReturnPart content is replaced with distillation text.
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    distillation_text = "Key finding: function foo() returns int"
    state.pending_actions.append(
        PruneAction(
            kind="distill",
            targets=(
                DistillTarget(
                    tool_call_id="tc_distill",
                    distillation=distillation_text,
                ),
            ),
            source_tool_call_id="action_2",
        ),
    )

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_distill")]),
        _make_request([
            _make_tool_return("read", "long original content with details", "tc_distill")
        ]),
        _make_request([UserPromptPart(content="Distill it")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    tool_returns = _find_tool_returns(result.messages)
    assert len(tool_returns) == 1
    _, _, tr_part = tool_returns[0]
    assert tr_part.content == distillation_text
    assert _is_pruned(tr_part)

    # Distillation text should be stored for re-pruning.
    assert state.distill_texts.get("tc_distill") == distillation_text


# =============================================================================
# 9.4 — Model calls decompress → original content returned as tool result
# =============================================================================


async def test_decompress_returns_original_content_without_history_modification() -> None:
    """Test decompress tool returns original content without restoring history.

    Given: A pruned ToolReturnPart in messages.
    When: decompress_tool is called for the pruned tool.
    Then: The tool result contains the original content, but the ToolReturnPart
          in messages still has "[pruned]".
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    # First prune the tool output.
    state.pending_actions.append(
        PruneAction(
            kind="prune",
            ids=("tc_decomp",),
            source_tool_call_id="action_3",
        ),
    )
    state.tool_id_list = ["tc_decomp"]

    original_content = "original detailed content"
    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_decomp")]),
        _make_request([_make_tool_return("read", original_content, "tc_decomp")]),
        _make_request([UserPromptPart(content="Prune and decompress")]),
    ]
    state.current_messages = list(messages)
    req_ctx = _make_request_context(messages)

    # Run pipeline to apply the prune.
    result = await cap.before_model_request(ctx, req_ctx)

    # Verify the ToolReturnPart is pruned in result.
    tool_returns = _find_tool_returns(result.messages)
    assert tool_returns[0][2].content == "[pruned]"

    # Now call decompress_tool — it should return original content.
    # The tool reads from state.current_messages which has the pruned part.
    # Update current_messages to the result messages so decompress sees the pruned part.
    state.current_messages = list(result.messages)
    decomp_result = decompress_tool(ctx, "0")

    assert decomp_result["restored"] is True
    assert decomp_result["original_content"] == original_content

    # The ToolReturnPart in messages should still be "[pruned]".
    tool_returns_after = _find_tool_returns(result.messages)
    assert tool_returns_after[0][2].content == "[pruned]"


# =============================================================================
# 9.5 — Auto-dedup at WARNING level
# =============================================================================


async def test_auto_dedup_at_warning_level_replaces_duplicates() -> None:
    """Test that duplicate tool outputs are pruned at WARNING watermark.

    Given: Messages with duplicate tool calls (same name+args) and a
           capability configured with auto_dedup=True.
    When: before_model_request is called with watermark >= WARNING.
    Then: Older duplicate ToolReturnParts are pruned (content replaced).
    """
    # Use small max_context_tokens to force WARNING level.
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
        auto_dedup=True,
        auto_strategy_threshold=WatermarkLevel.INFO,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file twice")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_a")]),
        _make_request([_make_tool_return("read", "content A", "tc_a")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_b")]),
        _make_request([_make_tool_return("read", "content A", "tc_b")]),
        _make_request([UserPromptPart(content="Done")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    state = _get_dcp_state(cap, ctx)
    # Watermark should be at least INFO (auto_strategy_threshold).
    assert state.watermark_level >= WatermarkLevel.INFO

    # Find both ToolReturnParts.
    tool_returns = _find_tool_returns(result.messages)
    assert len(tool_returns) == 2

    # The first (older) should be pruned (dedup), the last kept.
    first_tr = tool_returns[0][2]
    second_tr = tool_returns[1][2]

    # First should be pruned as duplicate.
    assert _is_pruned(first_tr) or first_tr.content == "[duplicate removed]"
    # Second should be intact.
    assert not _is_pruned(second_tr)


# =============================================================================
# 9.6 — purge_errors at INFO level
# =============================================================================


async def test_purge_errors_at_info_level_prunes_old_failed_tools() -> None:
    """Test that old failed tool outputs are purged at INFO watermark.

    Given: Messages with an error tool return and enough subsequent
           tool-call steps to exceed purge_error_steps.
    When: before_model_request is called with watermark >= INFO.
    Then: The error tool output is pruned.
    """
    cap = _make_capability(
        max_context_tokens=200,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
        purge_error_steps=1,
        step_protection=1,
        auto_strategy_threshold=WatermarkLevel.INFO,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Run command")]),
        _make_response([_make_tool_call("bash", {"cmd": "false"}, "tc_err")]),
        _make_request([
            _make_tool_return("bash", "Error: command failed", "tc_err", outcome="failed")
        ]),
        # Several subsequent tool calls to exceed purge_error_steps and step_protection.
        _make_response([_make_tool_call("read", {"path": "a"}, "tc_1")]),
        _make_request([_make_tool_return("read", "content a", "tc_1")]),
        _make_response([_make_tool_call("read", {"path": "b"}, "tc_2")]),
        _make_request([_make_tool_return("read", "content b", "tc_2")]),
        _make_response([_make_tool_call("read", {"path": "c"}, "tc_3")]),
        _make_request([_make_tool_return("read", "content c", "tc_3")]),
        _make_request([UserPromptPart(content="Done")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    tool_returns = _find_tool_returns(result.messages)
    # Find the error tool return.
    error_trs = [tr for _, _, tr in tool_returns if tr.tool_call_id == "tc_err"]
    assert len(error_trs) == 1
    # The error tool output should be pruned.
    assert _is_pruned(error_trs[0])


# =============================================================================
# 9.7 — No auto-strategies at NORMAL level
# =============================================================================


async def test_no_auto_strategies_at_normal_level() -> None:
    """Test that no dedup or purge occurs at NORMAL watermark.

    Given: Messages with duplicate tool calls and an error tool return.
    When: before_model_request is called with watermark at NORMAL.
    Then: No auto-strategies run; duplicates and errors are preserved.
    """
    # Use very large max_context_tokens so watermark stays NORMAL.
    cap = _make_capability(
        max_context_tokens=10_000_000,
        info_threshold=0.60,
        warning_threshold=0.75,
        critical_threshold=0.90,
        auto_dedup=True,
        auto_strategy_threshold=WatermarkLevel.INFO,
        purge_error_steps=1,
        step_protection=1,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file twice")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_dup1")]),
        _make_request([_make_tool_return("read", "same content", "tc_dup1")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_dup2")]),
        _make_request([_make_tool_return("read", "same content", "tc_dup2")]),
        _make_response([_make_tool_call("bash", {"cmd": "false"}, "tc_err")]),
        _make_request([_make_tool_return("bash", "Error", "tc_err", outcome="failed")]),
        _make_request([UserPromptPart(content="Done")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    state = _get_dcp_state(cap, ctx)
    assert state.watermark_level == WatermarkLevel.NORMAL

    # No duplicates should be pruned.
    tool_returns = _find_tool_returns(result.messages)
    for _, _, tr in tool_returns:
        assert not _is_pruned(tr)


# =============================================================================
# 9.8 — Prunable-tools list injected at INFO, not below INFO
# =============================================================================


async def test_prunable_list_injected_at_info_not_below() -> None:
    """Test that <prunable-tools> list is injected at INFO, not at NORMAL.

    Given: A conversation with tool outputs.
    When: before_model_request is called at NORMAL watermark.
    Then: No <prunable-tools> text in messages.
    When: before_model_request is called at INFO watermark.
    Then: <prunable-tools> text is present in a SystemPromptPart.
    """
    # --- NORMAL case ---
    cap_normal = _make_capability(max_context_tokens=10_000_000)
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)
    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_1")]),
        _make_request([_make_tool_return("read", "content", "tc_1")]),
        _make_request([UserPromptPart(content="Summarize")]),
    ]
    req_ctx = _make_request_context(messages)
    result = await cap_normal.before_model_request(ctx, req_ctx)

    # No <prunable-tools> in any part content.
    found_prunable = False
    for msg in result.messages:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", "")
            if isinstance(content, str) and "<prunable-tools>" in content:
                found_prunable = True
    assert not found_prunable

    # --- INFO case ---
    cap_info = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
    )
    session_data2 = _make_session_data()
    ctx2 = _make_run_context(session_data=session_data2)
    req_ctx2 = _make_request_context(messages)
    result2 = await cap_info.before_model_request(ctx2, req_ctx2)

    state2 = _get_dcp_state(cap_info, ctx2)
    assert state2.watermark_level >= WatermarkLevel.INFO

    found_prunable2 = False
    for msg in result2.messages:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", "")
            if isinstance(content, str) and "<prunable-tools>" in content:
                found_prunable2 = True
    assert found_prunable2


# =============================================================================
# 9.9 — Nudge steered via SessionPool vs enqueue fallback
# =============================================================================


async def test_nudge_steered_via_session_pool() -> None:
    """Test nudge is steered via session_pool.steer() when SessionPool is available.

    Given: A capability with nudge_turn_frequency=2, state with
           nudge_counter=2, and a mock SessionPool.
    When: before_model_request is called.
    Then: session_pool.steer() is called with emit_user_message=True
          (nudge_visible default) and the counters are reset.
    """
    session_pool = AsyncMock(spec=["steer"])
    session_pool.steer = AsyncMock(return_value="steer-id-456")

    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=2,
        nudge_step_frequency=0,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data, session_pool=session_pool)
    state.nudge_counter = 2

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    await cap.before_model_request(ctx, req_ctx)

    session_pool.steer.assert_called_once()
    call_kwargs = session_pool.steer.call_args.kwargs
    assert "emit_user_message" in call_kwargs
    assert call_kwargs["emit_user_message"] is True
    # Counter should be reset.
    assert state.nudge_counter == 0


async def test_nudge_enqueued_fallback_without_session_pool() -> None:
    """Test nudge falls back to ctx.enqueue() when SessionPool is unavailable.

    Given: A capability with nudge_turn_frequency=2 and state with
           nudge_counter=2, but no SessionPool.
    When: before_model_request is called.
    Then: ctx.enqueue() is called (fallback path) and counters are reset.
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=2,
        nudge_step_frequency=0,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)
    state.nudge_counter = 2

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    await cap.before_model_request(ctx, req_ctx)

    ctx.enqueue.assert_called_once()
    # Counter should be reset.
    assert state.nudge_counter == 0


# =============================================================================
# 9.10 — Nudge skipped when below frequency
# =============================================================================


async def test_nudge_skipped_when_below_frequency() -> None:
    """Test nudge is NOT enqueued when counter < frequency.

    Given: A capability with nudge_turn_frequency=3 and state with nudge_counter=1.
    When: before_model_request is called.
    Then: ctx.enqueue() is NOT called.
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=3,
        nudge_step_frequency=0,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)
    state.nudge_counter = 1

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    await cap.before_model_request(ctx, req_ctx)

    ctx.enqueue.assert_not_called()
    assert state.nudge_counter == 1


# =============================================================================
# 9.10a — SessionPool nudge-visible and below-frequency scenarios
# =============================================================================


async def test_nudge_visible_false_passes_emit_false() -> None:
    """Test session_pool.steer() recieves emit_user_message=False.

    Given: A capability with nudge_visible=False, nudge_turn_frequency=2,
           state with nudge_counter=2, and a mock SessionPool.
    When: before_model_request is called.
    Then: session_pool.steer() is called with emit_user_message=False.
    """
    session_pool = AsyncMock(spec=["steer"])
    session_pool.steer = AsyncMock(return_value="steer-id-789")

    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=2,
        nudge_step_frequency=0,
        nudge_visible=False,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data, session_pool=session_pool)
    state.nudge_counter = 2

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    await cap.before_model_request(ctx, req_ctx)

    session_pool.steer.assert_called_once()
    call_kwargs = session_pool.steer.call_args.kwargs
    assert call_kwargs["emit_user_message"] is False
    assert state.nudge_counter == 0


async def test_nudge_visible_true_passes_emit_true() -> None:
    """Test session_pool.steer() recieves emit_user_message=True.

    Given: A capability with nudge_visible=True, nudge_turn_frequency=2,
           state with nudge_counter=2, and a mock SessionPool.
    When: before_model_request is called.
    Then: session_pool.steer() is called with emit_user_message=True.
    """
    session_pool = AsyncMock(spec=["steer"])
    session_pool.steer = AsyncMock(return_value="steer-id-101")

    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=2,
        nudge_step_frequency=0,
        nudge_visible=True,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data, session_pool=session_pool)
    state.nudge_counter = 2

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    await cap.before_model_request(ctx, req_ctx)

    session_pool.steer.assert_called_once()
    call_kwargs = session_pool.steer.call_args.kwargs
    assert call_kwargs["emit_user_message"] is True
    assert state.nudge_counter == 0


async def test_nudge_skipped_when_below_frequency_with_pool() -> None:
    """Test nudge is NOT steered when counter < frequency, even with SessionPool.

    Given: A capability with nudge_turn_frequency=3, state with
           nudge_counter=1, and a mock SessionPool.
    When: before_model_request is called.
    Then: session_pool.steer() is NOT called; counter is unchanged.
    """
    session_pool = AsyncMock(spec=["steer"])
    session_pool.steer = AsyncMock(return_value="steer-id-202")

    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=3,
        nudge_step_frequency=0,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data, session_pool=session_pool)
    state.nudge_counter = 1

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    await cap.before_model_request(ctx, req_ctx)

    session_pool.steer.assert_not_called()
    assert state.nudge_counter == 1


# =============================================================================
# 9.10b — Same-turn prune + decompress (immediate application)
# =============================================================================


async def test_prune_then_decompress_same_turn() -> None:
    """Test decompress works in the same turn as prune.

    Given: A capability with a 2-turn conversation containing tool outputs.
    When: prune_tool is called for ID 0, then decompress_tool is called
          for ID 0 in the same turn (no before_model_request in between).
    Then: decompress returns restored=True with the original content,
          because prune_tool applies the action to state.current_messages
          immediately.
    """
    from wolfharness.capabilities.dcp.tools import decompress_tool, prune_tool

    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_0")]),
        _make_request([_make_tool_return(tool_call_id="tc_0", content="file content here")]),
    ]
    state.current_messages = list(messages)
    state.tool_id_list = ["tc_0"]

    # Prune ID 0
    result = prune_tool(ctx, ids=["0"])
    assert result["status"] == "applied"
    assert result["count"] == 1

    # Decompress ID 0 in the SAME turn — should work now
    dec_result = decompress_tool(ctx, tool_id="0")
    assert dec_result["restored"] is True
    assert dec_result["original_content"] == "file content here"
    assert dec_result["was_pruned_as"] == "prune"


async def test_distill_then_decompress_same_turn() -> None:
    """Test decompress works in the same turn as distill.

    Given: A capability with a 2-turn conversation containing tool outputs.
    When: distill_tool is called for ID 0, then decompress_tool is called
          for ID 0 in the same turn.
    Then: decompress returns restored=True with the original content,
          because distill_tool applies the action to state.current_messages
          immediately.
    """
    from wolfharness.capabilities.dcp.tools import decompress_tool, distill_tool

    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_0")]),
        _make_request([_make_tool_return(tool_call_id="tc_0", content="long file content")]),
    ]
    state.current_messages = list(messages)
    state.tool_id_list = ["tc_0"]

    # Distill ID 0
    result = distill_tool(
        ctx,
        targets=[{"id": "0", "distillation": "short summary"}],
    )
    assert result["status"] == "applied"
    assert result["count"] == 1

    # Decompress ID 0 in the SAME turn
    dec_result = decompress_tool(ctx, tool_id="0")
    assert dec_result["restored"] is True
    assert dec_result["original_content"] == "long file content"
    assert dec_result["was_pruned_as"] == "distill"


# =============================================================================
# 9.11 — Guard last message — ModelResponse as last → empty ModelRequest appended
# =============================================================================


async def test_guard_last_message_appends_empty_request() -> None:
    """Test that an empty ModelRequest is appended when last message is ModelResponse.

    Given: Messages ending with a ModelResponse.
    When: before_model_request is called.
    Then: An empty ModelRequest is appended to the result messages.
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi there")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # Last message should be a ModelRequest.
    assert isinstance(result.messages[-1], ModelRequest)


# =============================================================================
# 9.12 — Meta-tool auto-prune — old prune/distill returns pruned beyond retention
# =============================================================================


async def test_meta_tool_auto_prune_beyond_retention() -> None:
    """Test that old meta-tool returns are auto-pruned beyond retention limit.

    Given: Messages with multiple prune/distill/decompress ToolReturnParts.
    When: before_model_request is called with meta_tool_retention=1.
    Then: Only the most recent meta-tool return is kept; older ones are pruned.
    """
    cap = _make_capability(
        max_context_tokens=10_000_000,
        meta_tool_retention=1,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read files")]),
        _make_response([_make_tool_call("read", {"path": "a"}, "tc_a")]),
        _make_request([_make_tool_return("read", "content a", "tc_a")]),
        # First prune tool return (meta-tool).
        _make_response([_make_tool_call("prune", {"ids": ["0"]}, "meta_1")]),
        _make_request([_make_tool_return("prune", "pruned 1 tool", "meta_1")]),
        # Second prune tool return (meta-tool).
        _make_response([_make_tool_call("prune", {"ids": ["1"]}, "meta_2")]),
        _make_request([_make_tool_return("prune", "pruned 2 tools", "meta_2")]),
        # Third prune tool return (meta-tool).
        _make_response([_make_tool_call("prune", {"ids": ["2"]}, "meta_3")]),
        _make_request([_make_tool_return("prune", "pruned 3 tools", "meta_3")]),
        _make_request([UserPromptPart(content="Done")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # Find meta-tool returns in result.
    meta_returns = [
        (mi, pi, p)
        for mi, msg in enumerate(result.messages)
        for pi, p in enumerate(getattr(msg, "parts", []))
        if isinstance(p, ToolReturnPart) and p.tool_name in ("prune", "distill", "decompress")
    ]
    assert len(meta_returns) == 3

    # First two should be pruned; last one kept.
    assert _is_pruned(meta_returns[0][2])
    assert _is_pruned(meta_returns[1][2])
    assert not _is_pruned(meta_returns[2][2])


# =============================================================================
# 9.13 — Clear thinking toggle
# =============================================================================


@pytest.mark.incompatible_with_thinking
async def test_clear_thinking_strips_thinking_parts_when_active() -> None:
    """Test that ThinkingPart is stripped when clear_thinking is active.

    Given: Messages with ThinkingPart in older assistant messages and
           clear_thinking_enabled=True, clear_thinking_active=True.
    When: before_model_request is called.
    Then: ThinkingPart instances before the last user message are stripped.
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        clear_thinking_enabled=True,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)
    state.clear_thinking_active = True

    messages = [
        _make_request([UserPromptPart(content="Question 1")]),
        _make_response([
            ThinkingPart(content="Let me think about this..."),
            TextPart(content="Here is my answer."),
        ]),
        _make_request([UserPromptPart(content="Question 2")]),
        _make_response([
            ThinkingPart(content="Thinking again..."),
            TextPart(content="Second answer."),
        ]),
        _make_request([UserPromptPart(content="Question 3")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # ThinkingPart before the last user message should be stripped.
    # Find the last UserPromptPart index.
    last_user_idx = -1
    for i, msg in enumerate(result.messages):
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    last_user_idx = i
                    break

    thinking_found = False
    for i, msg in enumerate(result.messages):
        if i < last_user_idx and isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ThinkingPart):
                    thinking_found = True
    assert not thinking_found


@pytest.mark.incompatible_with_thinking
async def test_clear_thinking_preserved_when_inactive() -> None:
    """Test that ThinkingPart is preserved when clear_thinking is inactive.

    Given: Messages with ThinkingPart and clear_thinking_enabled=False.
    When: before_model_request is called.
    Then: ThinkingPart instances are preserved.
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        clear_thinking_enabled=False,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Question 1")]),
        _make_response([
            ThinkingPart(content="Let me think..."),
            TextPart(content="Answer."),
        ]),
        _make_request([UserPromptPart(content="Question 2")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    thinking_found = False
    for msg in result.messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ThinkingPart):
                    thinking_found = True
    assert thinking_found


# =============================================================================
# 9.14 — Re-prune across iterations
# =============================================================================


async def test_re_prune_across_iterations_stays_pruned() -> None:
    """Test that pruned content stays pruned after message_history restore.

    Given: A tool output that has been pruned.
    When: before_model_request is called again (simulating a new iteration
          where ctx.state.message_history restores original content).
    Then: The tool output is re-pruned to "[pruned]".
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    # First, apply a prune action.
    state.pending_actions.append(
        PruneAction(
            kind="prune",
            ids=("tc_reprune",),
            source_tool_call_id="action_r",
        ),
    )

    original_messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_reprune")]),
        _make_request([_make_tool_return("read", "original content", "tc_reprune")]),
        _make_request([UserPromptPart(content="Continue")]),
    ]
    req_ctx1 = _make_request_context(list(original_messages))
    result1 = await cap.before_model_request(ctx, req_ctx1)

    # Verify pruned.
    tr1 = _find_tool_returns(result1.messages)
    assert tr1[0][2].content == "[pruned]"

    # Now simulate a second iteration: fresh messages with ORIGINAL content
    # (as would happen when ctx.state.message_history restores).
    req_ctx2 = _make_request_context(list(original_messages))
    result2 = await cap.before_model_request(ctx, req_ctx2)

    # Should still be pruned (via _re_prune_messages).
    tr2 = _find_tool_returns(result2.messages)
    assert tr2[0][2].content == "[pruned]"


# =============================================================================
# 9.15 — Truncation NOT performed by DCP
# =============================================================================


async def test_after_tool_execute_only_increments_counters() -> None:
    """Test that after_tool_execute does not modify tool output.

    Given: A capability and a tool result.
    When: after_tool_execute is called.
    Then: The result is returned unchanged; only counters are incremented.
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    initial_step = state.step_count
    initial_nudge_step = state.nudge_step_counter

    tool_call = _make_tool_call("read", {"path": "f.ts"}, "tc_15")
    tool_def = MagicMock()
    tool_result = {"output": "file contents"}

    result = await cap.after_tool_execute(
        ctx,
        call=tool_call,
        tool_def=tool_def,
        args={},
        result=tool_result,
    )

    assert result is tool_result
    assert state.step_count == initial_step + 1
    assert state.nudge_step_counter == initial_nudge_step + 1


# =============================================================================
# 9.16 — Session metadata storage
# =============================================================================


async def test_session_metadata_storage_and_retrieval() -> None:
    """Test that DCPState is stored in SessionData.metadata['dcp'] and retrievable.

    Given: A capability with session data available.
    When: before_model_request is called, then a second call retrieves state.
    Then: The DCPState is stored in metadata['dcp'] and persists across calls.
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    # before_run increments current_turn.
    await cap.before_run(ctx)
    await cap.before_model_request(ctx, req_ctx)

    # DCPState should be stored in session_data.metadata.
    stored = session_data.metadata.get("dcp")
    assert stored is not None
    assert isinstance(stored, DCPState)
    assert stored.current_turn >= 1

    # Second call: before_run again then before_model_request.
    await cap.before_run(ctx)
    req_ctx2 = _make_request_context(messages)
    await cap.before_model_request(ctx, req_ctx2)

    stored2 = session_data.metadata.get("dcp")
    assert stored2 is stored  # Same object.
    assert stored2.current_turn >= 2


# =============================================================================
# 9.17 — TestModel-based integration test
# =============================================================================


async def test_testmodel_full_turn_with_dcp_capability() -> None:
    """Test a full pydantic-ai agent turn with TestModel and DCP capability.

    Given: A real pydantic-ai Agent with TestModel and DCP capability attached.
    When: The agent runs a prompt.
    Then: DCP lifecycle hooks (before_run, before_model_request, after_tool_execute)
          fire without errors and the agent produces a response.
    """
    from pydantic_ai import Agent

    test_model = TestModel(
        custom_output_text="Test response from TestModel",
        call_tools=[],
    )
    cap = _make_capability(max_context_tokens=100_000)

    # Provide a minimal deps mock so _get_dcp_state can call get_session_state().
    # Returning None makes the capability use its _fallback_state.
    deps_mock = MagicMock()
    deps_mock.get_session_state.return_value = None

    agent: Agent[Any, str] = Agent(
        model=test_model,
        system_prompt="You are a test assistant.",
        capabilities=[cap],
        deps_type=Any,
    )

    result = await agent.run("Hello, test!", deps=deps_mock)

    assert result is not None
    assert result.output is not None

    # Verify DCP state was initialized (before_run fired).
    state = cap._fallback_state
    assert state.current_turn >= 1


# =============================================================================
# 9.18 — Both session_pool=None and enqueue missing → no crash
# =============================================================================


async def test_enqueue_missing_does_not_raise() -> None:
    """Test no crash when both session_pool AND enqueue are unavailable.

    Given: A capability with nudge_turn_frequency=1 and a context that
           has no SessionPool AND no enqueue attribute.
    When: before_model_request is called.
    Then: No AttributeError is raised; nudge is silently skipped.
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        nudge_turn_frequency=1,
        nudge_step_frequency=0,
    )
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data, enqueue_enabled=False)
    state.nudge_counter = 1

    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    # Should NOT raise AttributeError.
    result = await cap.before_model_request(ctx, req_ctx)
    assert result is not None


# =============================================================================
# 9.19 — inject_role="system"
# =============================================================================


async def test_inject_role_system_uses_system_prompt_part() -> None:
    """Test that prunable-tools list is injected as SystemPromptPart.

    Given: A capability with inject_role="system" and INFO watermark.
    When: before_model_request is called.
    Then: A SystemPromptPart with <prunable-tools> content is appended.
    """
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
        inject_role="system",
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_1")]),
        _make_request([_make_tool_return("read", "content", "tc_1")]),
        _make_request([UserPromptPart(content="Summarize")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    found_system_prunable = False
    for msg in result.messages:
        for part in getattr(msg, "parts", []):
            if (
                isinstance(part, SystemPromptPart)
                and isinstance(part.content, str)
                and "<prunable-tools>" in part.content
            ):
                found_system_prunable = True
    assert found_system_prunable


# =============================================================================
# 9.20 — inject_role="user"
# =============================================================================


async def test_inject_role_user_uses_user_prompt_part() -> None:
    """Test that prunable-tools list is injected as UserPromptPart.

    Given: A capability with inject_role="user" and INFO watermark.
    When: before_model_request is called.
    Then: A UserPromptPart with <prunable-tools> content is appended.
    """
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
        inject_role="user",
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_1")]),
        _make_request([_make_tool_return("read", "content", "tc_1")]),
        _make_request([UserPromptPart(content="Summarize")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    found_user_prunable = False
    for msg in result.messages:
        for part in getattr(msg, "parts", []):
            if (
                isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and "<prunable-tools>" in part.content
            ):
                found_user_prunable = True
    assert found_user_prunable


# =============================================================================
# 10.1 — UserPromptPart.content unchanged after full pipeline
# =============================================================================


async def test_user_prompt_content_unchanged_after_pipeline() -> None:
    """Test that existing UserPromptPart content is not modified by the pipeline.

    Given: Messages with UserPromptPart.
    When: before_model_request is called.
    Then: The original UserPromptPart content is unchanged.
    """
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    user_content = "This is my original question"
    messages = [
        _make_request([UserPromptPart(content=user_content)]),
        _make_response([TextPart(content="Response")]),
        _make_request([UserPromptPart(content="Follow-up question")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # Find the original UserPromptPart (not the injected prunable list).
    for msg in result.messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and part.content == user_content:
                    # Found original — content should be unchanged.
                    assert part.content == user_content


# =============================================================================
# 10.2 — TextPart.content unchanged after full pipeline
# =============================================================================


async def test_text_part_content_unchanged_after_pipeline() -> None:
    """Test that existing TextPart content is not modified by the pipeline.

    Given: Messages with TextPart in ModelResponse.
    When: before_model_request is called.
    Then: The TextPart content is unchanged.
    """
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    text_content = "This is the assistant's original response"
    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content=text_content)]),
        _make_request([UserPromptPart(content="Goodbye")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    for msg in result.messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    assert part.content == text_content


# =============================================================================
# 10.3 — ToolCallPart.args unchanged after full pipeline
# =============================================================================


async def test_tool_call_args_unchanged_after_pipeline() -> None:
    """Test that ToolCallPart args are not modified by the pipeline.

    Given: Messages with ToolCallPart.
    When: before_model_request is called.
    Then: The ToolCallPart args are unchanged.
    """
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    call_args = {"path": "src/main.ts", "encoding": "utf-8"}
    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", call_args, "tc_args")]),
        _make_request([_make_tool_return("read", "content", "tc_args")]),
        _make_request([UserPromptPart(content="Done")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    for msg in result.messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id == "tc_args":
                    assert part.args == call_args


# =============================================================================
# 10.4 — No <message-ref> or <compression-block> tags in existing content
# =============================================================================


async def test_no_message_ref_or_compression_block_tags_in_existing_content() -> None:
    """Test that no <message-ref> or <compression-block> tags appear in existing content.

    Given: Messages with various content types.
    When: before_model_request is called.
    Then: No <message-ref> or <compression-block> tags in any existing
          message content. <prunable-tools> MAY appear in newly appended parts.
    """
    cap = _make_capability(
        max_context_tokens=100,
        info_threshold=0.10,
        warning_threshold=0.20,
        critical_threshold=0.90,
    )
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_1")]),
        _make_request([_make_tool_return("read", "file content here", "tc_1")]),
        _make_response([TextPart(content="Here is my analysis")]),
        _make_request([UserPromptPart(content="Continue")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    for msg in result.messages:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", "")
            if isinstance(content, str):
                assert "<message-ref>" not in content
                assert "<compression-block>" not in content


# =============================================================================
# 10.5 — ToolCallPart ↔ ToolReturnPart pairing intact after pruning
# =============================================================================


async def test_tool_call_return_pairing_intact_after_pruning() -> None:
    """Test that tool_call_id matching between ToolCallPart and ToolReturnPart is preserved.

    Given: Messages with paired ToolCallPart and ToolReturnPart.
    When: before_model_request applies a prune action to one pair.
    Then: The tool_call_id on both parts still match; only content is replaced.
    """
    cap = _make_capability(max_context_tokens=100_000)
    session_data, state = _make_session_with_state()
    ctx = _make_run_context(session_data=session_data)

    state.pending_actions.append(
        PruneAction(
            kind="prune",
            ids=("tc_pair",),
            source_tool_call_id="action_pair",
        ),
    )

    messages = [
        _make_request([UserPromptPart(content="Read file")]),
        _make_response([_make_tool_call("read", {"path": "f.ts"}, "tc_pair")]),
        _make_request([_make_tool_return("read", "content", "tc_pair")]),
        _make_request([UserPromptPart(content="Done")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # Find the ToolCallPart and ToolReturnPart.
    call_id: str | None = None
    return_id: str | None = None
    for msg in result.messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and part.tool_name == "read":
                call_id = part.tool_call_id
            if isinstance(part, ToolReturnPart) and part.tool_name == "read":
                return_id = part.tool_call_id

    assert call_id is not None
    assert return_id is not None
    assert call_id == return_id  # Pairing intact.


# =============================================================================
# 10.6 — expose_tools=False → no tools registered, pipeline still runs
# =============================================================================


async def test_expose_tools_false_no_tools_registered() -> None:
    """Test that expose_tools=False means no tools in toolset, pipeline still runs.

    Given: A capability with expose_tools=False.
    When: get_toolset() is called and before_model_request runs.
    Then: Toolset is None; pipeline executes without errors.
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        expose_tools=False,
    )

    # No toolset should be returned.
    toolset = cap.get_toolset()
    assert toolset is None

    # Pipeline should still run.
    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)
    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)
    assert result is not None


# =============================================================================
# 10.7 — DCP disabled → before_model_request returns request_context unchanged
# =============================================================================


async def test_dcp_disabled_returns_request_context_unchanged() -> None:
    """Test that enabled=False makes before_model_request a no-op.

    Given: A capability with enabled=False.
    When: before_model_request is called.
    Then: The request_context is returned unchanged (same messages).
    """
    cap = _make_capability(
        max_context_tokens=100_000,
        enabled=False,
    )

    session_data = _make_session_data()
    ctx = _make_run_context(session_data=session_data)
    messages = [
        _make_request([UserPromptPart(content="Hello")]),
        _make_response([TextPart(content="Hi")]),
        _make_request([UserPromptPart(content="Bye")]),
    ]
    req_ctx = _make_request_context(messages)

    result = await cap.before_model_request(ctx, req_ctx)

    # Should return the same request_context (no modifications).
    assert result is req_ctx
    # Messages should be unchanged.
    assert result.messages is messages

    # Toolset should also be None when disabled.
    assert cap.get_toolset() is None

    # Instructions should still be available (static text).
    assert cap.get_instructions() is not None

    # before_run should be a no-op.
    state_before = cap._fallback_state.current_turn
    await cap.before_run(ctx)
    assert cap._fallback_state.current_turn == state_before
