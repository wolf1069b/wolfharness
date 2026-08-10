"""L1 unit tests for the Dynamic Context Pruning (DCP) capability.

Covers token_utils, watermark, block_store, state, config, strategies,
tools, nudge, and prunable_list modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
import pytest

from wolfharness.capabilities.dcp.block_store import CompressionBlockStore
from wolfharness.capabilities.dcp.config import DCPConfig
from wolfharness.capabilities.dcp.nudge import build_nudge_text
from wolfharness.capabilities.dcp.prunable_list import (
    META_TOOL_NAMES,
    build_prunable_list,
    inject_prunable_list,
)
from wolfharness.capabilities.dcp.state import (
    CompressionBlock,
    DCPState,
    DistillTarget,
    PruneAction,
    WatermarkLevel,
)
from wolfharness.capabilities.dcp.strategies import (
    PrunableState,
    _dedup_exact,
    _is_pruned,
    _prune_part,
    _strip_thinking_content,
    purge_failed_tool_inputs,
)
from wolfharness.capabilities.dcp.token_utils import (
    calculate_context_pressure,
    estimate_tokens,
)
from wolfharness.capabilities.dcp.tools import (
    decompress_tool,
    distill_tool,
    prune_tool,
)
from wolfharness.capabilities.dcp.watermark import WatermarkStateMachine
from wolfharness.sessions.models import SessionData


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_ctx(state: DCPState) -> MagicMock:
    """Create a MagicMock RunContext with a DCPState stored in session metadata."""
    session_data = SessionData(
        session_id="test-session",
        agent_name="test-agent",
        pool_id="test-pool",
        project_id="test-project",
        parent_id=None,
        version="1",
        cwd="/tmp",
        agent_type="native",
    )
    session_data.metadata["dcp"] = state
    ctx = MagicMock()
    ctx.deps.get_session_state.return_value = session_data
    return ctx


def _make_tool_return(
    tool_name: str = "search",
    content: str = "result data",
    tool_call_id: str | None = None,
    outcome: str = "success",
) -> ToolReturnPart:
    """Create a ToolReturnPart with a deterministic tool_call_id."""
    return ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id or f"call_{uuid4().hex[:8]}",
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# 8.1 — token_utils.py
# ---------------------------------------------------------------------------


def test_estimate_tokens_with_text_content() -> None:
    """estimate_tokens returns a positive count for a ModelRequest with text."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="Hello world, this is a test prompt.")]),
    ]
    result = estimate_tokens(messages)
    assert result > 0


def test_estimate_tokens_empty_messages() -> None:
    """estimate_tokens returns 0 for an empty message list."""
    assert estimate_tokens([]) == 0


def test_estimate_tokens_cjk_content() -> None:
    """estimate_tokens produces a higher count for CJK content than equivalent-length ASCII."""
    cjk_text = "你好世界这是一段中文文本" * 4
    ascii_text = "a" * len(cjk_text)
    cjk_messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content=cjk_text)]),
    ]
    ascii_messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content=ascii_text)]),
    ]
    cjk_tokens = estimate_tokens(cjk_messages)
    ascii_tokens = estimate_tokens(ascii_messages)
    assert cjk_tokens > ascii_tokens


def test_estimate_tokens_none_content() -> None:
    """estimate_tokens handles None-like content gracefully (empty string)."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="")]),
    ]
    assert estimate_tokens(messages) == 0


def test_calculate_context_pressure_below_max() -> None:
    """calculate_context_pressure returns ratio < 1.0 when current < max."""
    result = calculate_context_pressure(500, 1000)
    assert result == 0.5


def test_calculate_context_pressure_equal_max() -> None:
    """calculate_context_pressure returns 1.0 when current == max."""
    result = calculate_context_pressure(1000, 1000)
    assert result == 1.0


def test_calculate_context_pressure_above_max() -> None:
    """calculate_context_pressure returns ratio > 1.0 when current > max."""
    result = calculate_context_pressure(1500, 1000)
    assert result == 1.5


def test_calculate_context_pressure_max_zero() -> None:
    """calculate_context_pressure returns 0.0 when max is 0 (division guard)."""
    result = calculate_context_pressure(500, 0)
    assert result == 0.0


# ---------------------------------------------------------------------------
# 8.1b — token_utils.py tiktoken fallback
# ---------------------------------------------------------------------------


def test_estimate_tokens_tiktoken_fallback_no_import_error() -> None:
    """estimate_tokens works without ImportError when tiktoken is unavailable."""
    with patch.dict("sys.modules", {"tiktoken": None}):
        messages: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[UserPromptPart(content="Hello world test")]),
        ]
        result = estimate_tokens(messages)
        assert result > 0


def test_estimate_tokens_tiktoken_fallback_cjk_heuristic() -> None:
    """When tiktoken is unavailable, CJK gets higher token count than ASCII."""
    cjk_text = "你好世界这是中文" * 5
    ascii_text = "a" * len(cjk_text)
    with patch.dict("sys.modules", {"tiktoken": None}):
        cjk_messages: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[UserPromptPart(content=cjk_text)]),
        ]
        ascii_messages: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[UserPromptPart(content=ascii_text)]),
        ]
        cjk_tokens = estimate_tokens(cjk_messages)
        ascii_tokens = estimate_tokens(ascii_messages)
        assert cjk_tokens > ascii_tokens


# ---------------------------------------------------------------------------
# 8.2 — watermark.py
# ---------------------------------------------------------------------------


def test_watermark_normal_to_info_at_60_percent() -> None:
    """Watermark transitions from NORMAL to INFO at exactly 0.60 pressure ratio."""
    wsm = WatermarkStateMachine()
    level = wsm.update_with_tokens(60, 100)
    assert level == WatermarkLevel.INFO


def test_watermark_info_to_warning_at_75_percent() -> None:
    """Watermark transitions from INFO to WARNING at exactly 0.75 pressure ratio."""
    wsm = WatermarkStateMachine()
    level = wsm.update_with_tokens(75, 100)
    assert level == WatermarkLevel.WARNING


def test_watermark_warning_to_critical_at_90_percent() -> None:
    """Watermark transitions from WARNING to CRITICAL at exactly 0.90 pressure ratio."""
    wsm = WatermarkStateMachine()
    level = wsm.update_with_tokens(90, 100)
    assert level == WatermarkLevel.CRITICAL


def test_watermark_ratio_zero_returns_normal() -> None:
    """Watermark returns NORMAL when pressure ratio is 0.0."""
    wsm = WatermarkStateMachine()
    level = wsm.update_with_tokens(0, 100)
    assert level == WatermarkLevel.NORMAL


def test_watermark_ratio_above_one_returns_critical() -> None:
    """Watermark returns CRITICAL when pressure ratio exceeds 1.0."""
    wsm = WatermarkStateMachine()
    level = wsm.update_with_tokens(150, 100)
    assert level == WatermarkLevel.CRITICAL


def test_watermark_max_tokens_zero_raises_value_error() -> None:
    """WatermarkStateMachine.update raises ValueError when max_tokens is 0."""
    wsm = WatermarkStateMachine()
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        wsm.update([], 0)


def test_watermark_max_tokens_negative_raises_value_error() -> None:
    """WatermarkStateMachine.update raises ValueError when max_tokens is negative."""
    wsm = WatermarkStateMachine()
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        wsm.update([], -100)


def test_watermark_update_with_tokens_zero_raises() -> None:
    """WatermarkStateMachine.update_with_tokens raises ValueError when max_tokens is 0."""
    wsm = WatermarkStateMachine()
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        wsm.update_with_tokens(100, 0)


# ---------------------------------------------------------------------------
# 8.3 — block_store.py
# ---------------------------------------------------------------------------


def test_block_store_put_get_round_trip() -> None:
    """put() then get() returns the same block for the same session."""
    store = CompressionBlockStore()
    block = CompressionBlock(
        block_id="blk_001",
        original_tool_call_ids=("call_1",),
        compressed_content="[pruned]",
        kind="prune",
    )
    block_id = store.put("session-a", block)
    assert block_id == "blk_001"
    retrieved = store.get("session-a", "blk_001")
    assert retrieved is not None
    assert retrieved.block_id == "blk_001"
    assert retrieved.compressed_content == "[pruned]"


def test_block_store_session_isolation() -> None:
    """Blocks stored under one session are not visible to another session."""
    store = CompressionBlockStore()
    block_a = CompressionBlock(
        block_id="blk_a",
        original_tool_call_ids=("call_1",),
        compressed_content="content_a",
        kind="prune",
    )
    store.put("session-a", block_a)
    # Session B should not see session A's blocks.
    assert store.get("session-b", "blk_a") is None
    assert store.get_all("session-b") == []


def test_block_store_get_chain_caps_at_10() -> None:
    """get_chain() stops traversing after 10 levels even with a deeper chain."""
    store = CompressionBlockStore()
    # Create 15 blocks with parent chain: blk_14 -> blk_13 -> ... -> blk_0
    for i in range(15):
        parent = f"blk_{i - 1}" if i > 0 else None
        block = CompressionBlock(
            block_id=f"blk_{i}",
            original_tool_call_ids=(f"call_{i}",),
            compressed_content=f"content_{i}",
            kind="prune",
            parent_block_id=parent,
        )
        store.put("session-chain", block)

    chain = store.get_chain("session-chain", "blk_14")
    assert len(chain) == 10
    assert chain[0].block_id == "blk_14"
    # The 10th element is blk_5 (14 - 9 = 5).
    assert chain[9].block_id == "blk_5"


def test_block_store_get_chain_nonexistent_block() -> None:
    """get_chain() returns empty list for a non-existent block_id."""
    store = CompressionBlockStore()
    assert store.get_chain("session-x", "nonexistent") == []


def test_block_store_get_chain_nonexistent_session() -> None:
    """get_chain() returns empty list for a non-existent session."""
    store = CompressionBlockStore()
    assert store.get_chain("no-such-session", "blk_0") == []


def test_block_store_max_blocks_overflow_eviction() -> None:
    """Adding MAX_BLOCKS_PER_SESSION + 1 blocks evicts the oldest."""
    store = CompressionBlockStore()
    original_max = CompressionBlockStore.MAX_BLOCKS_PER_SESSION
    # Use a small max for test speed.
    CompressionBlockStore.MAX_BLOCKS_PER_SESSION = 5
    try:
        for i in range(6):
            block = CompressionBlock(
                block_id=f"blk_{i}",
                original_tool_call_ids=(f"call_{i}",),
                compressed_content=f"content_{i}",
                kind="prune",
            )
            store.put("session-overflow", block)
        all_blocks = store.get_all("session-overflow")
        assert len(all_blocks) == 5
        # The first block (blk_0) should have been evicted.
        assert store.get("session-overflow", "blk_0") is None
        # The last block (blk_5) should still be there.
        assert store.get("session-overflow", "blk_5") is not None
    finally:
        CompressionBlockStore.MAX_BLOCKS_PER_SESSION = original_max


def test_block_store_get_stats_returns_correct_counts() -> None:
    """get_stats() returns correct total_blocks, algorithms_used, and average_ratio."""
    store = CompressionBlockStore()
    for i in range(3):
        block = CompressionBlock(
            block_id=f"blk_{i}",
            original_tool_call_ids=(f"call_{i}",),
            compressed_content="x" * (10 + i),
            kind="prune",
        )
        store.put("session-stats", block)
    for i in range(2):
        block = CompressionBlock(
            block_id=f"dblk_{i}",
            original_tool_call_ids=(f"dcall_{i}",),
            compressed_content="y" * 5,
            kind="dedup",
        )
        store.put("session-stats", block)

    stats = store.get_stats("session-stats")
    assert stats.total_blocks == 5
    assert stats.algorithms_used == {"prune": 3, "dedup": 2}
    # average_ratio = (10+11+12+5+5) / 5 = 43 / 5 = 8.6
    assert stats.average_ratio == 8.6


def test_block_store_get_stats_empty_session() -> None:
    """get_stats() returns zeroed stats for a session with no blocks."""
    store = CompressionBlockStore()
    stats = store.get_stats("empty-session")
    assert stats.total_blocks == 0
    assert stats.algorithms_used == {}
    assert stats.average_ratio == 0.0


# ---------------------------------------------------------------------------
# 8.4 — state.py
# ---------------------------------------------------------------------------


def test_prune_action_creation_kind_prune() -> None:
    """PruneAction with kind='prune' stores ids correctly."""
    action = PruneAction(kind="prune", ids=("call_1", "call_2"))
    assert action.kind == "prune"
    assert action.ids == ("call_1", "call_2")
    assert action.targets == ()


def test_prune_action_creation_kind_distill() -> None:
    """PruneAction with kind='distill' stores targets correctly."""
    targets = (
        DistillTarget(tool_call_id="call_1", distillation="summary"),
        DistillTarget(tool_call_id="call_2", distillation="another summary"),
    )
    action = PruneAction(kind="distill", targets=targets)
    assert action.kind == "distill"
    assert len(action.targets) == 2
    assert action.targets[0].tool_call_id == "call_1"
    assert action.targets[0].distillation == "summary"


def test_dcp_state_from_dict_valid_data() -> None:
    """DCPState.from_dict() reconstructs state from a valid dict."""
    data = {
        "current_turn": 5,
        "step_count": 3,
        "pending_actions": [
            {"kind": "prune", "ids": ["call_1"], "source_tool_call_id": "act_1"},
        ],
        "pruned_tool_ids": ["call_1", "call_2"],
        "watermark_level": 2,
    }
    state = DCPState.from_dict(data)
    assert state.current_turn == 5
    assert state.step_count == 3
    assert len(state.pending_actions) == 1
    action = state.pending_actions[0]
    assert action.kind == "prune"
    assert action.ids == ("call_1",)
    assert state.pruned_tool_ids == {"call_1", "call_2"}
    assert state.watermark_level == WatermarkLevel.WARNING


def test_dcp_state_from_dict_missing_fields_graceful_defaults() -> None:
    """DCPState.from_dict() with empty dict returns default state."""
    state = DCPState.from_dict({})
    assert state.current_turn == 0
    assert state.step_count == 0
    assert len(state.pending_actions) == 0
    assert state.watermark_level == WatermarkLevel.NORMAL
    assert state.pruned_tool_ids == set()


def test_watermark_level_intenum_ordering() -> None:
    """WatermarkLevel IntEnum values are ordered NORMAL < INFO < WARNING < CRITICAL."""
    assert WatermarkLevel.NORMAL < WatermarkLevel.INFO
    assert WatermarkLevel.INFO < WatermarkLevel.WARNING
    assert WatermarkLevel.WARNING < WatermarkLevel.CRITICAL
    assert int(WatermarkLevel.NORMAL) == 0
    assert int(WatermarkLevel.INFO) == 1
    assert int(WatermarkLevel.WARNING) == 2
    assert int(WatermarkLevel.CRITICAL) == 3


# ---------------------------------------------------------------------------
# 8.5 — config.py
# ---------------------------------------------------------------------------


def test_dcp_config_default_values() -> None:
    """DCPConfig defaults: enabled, inject_role, nudge_role, nudge_turn_frequency, nudge_visible."""
    config = DCPConfig()
    assert config.enabled is True
    assert config.inject_role == "user"
    assert config.nudge_role == "user"
    assert config.nudge_turn_frequency == 3
    assert config.nudge_visible is True


def test_dcp_config_threshold_ordering_validation() -> None:
    """DCPConfig rejects configs where info >= warning threshold."""
    with pytest.raises(ValidationError, match=r"info_threshold.*must be less than"):
        DCPConfig(info_threshold=0.80, warning_threshold=0.75)


def test_dcp_config_threshold_warning_ge_critical_raises() -> None:
    """DCPConfig rejects configs where warning >= critical threshold."""
    with pytest.raises(ValidationError, match=r"warning_threshold.*must be less than"):
        DCPConfig(warning_threshold=0.95, critical_threshold=0.90)


def test_dcp_config_max_context_tokens_negative_raises() -> None:
    """DCPConfig rejects negative max_context_tokens."""
    with pytest.raises(ValidationError, match=r"max_context_tokens.*must be positive"):
        DCPConfig(max_context_tokens=-1)


def test_dcp_config_max_context_tokens_zero_raises() -> None:
    """DCPConfig rejects zero max_context_tokens."""
    with pytest.raises(ValidationError, match=r"max_context_tokens.*must be positive"):
        DCPConfig(max_context_tokens=0)


def test_dcp_config_glob_pattern_expansion() -> None:
    """DCPConfig expands protected_tool_patterns into protected_tools set."""
    config = DCPConfig(protected_tool_patterns=("search_*", "ask"))
    assert "search_*" in config.protected_tools
    assert "ask" in config.protected_tools


def test_dcp_config_default_protected_tool_patterns() -> None:
    """DCPConfig default protected_tool_patterns are expanded into protected_tools."""
    config = DCPConfig()
    assert "ask" in config.protected_tools
    assert "confirm" in config.protected_tools
    assert "approval_*" in config.protected_tools


def test_dcp_config_disabled_is_valid() -> None:
    """DCPConfig with enabled=False is valid."""
    config = DCPConfig(enabled=False)
    assert config.enabled is False


# ---------------------------------------------------------------------------
# 8.6 — strategies.py
# ---------------------------------------------------------------------------


def test_prune_part_stores_prune_metadata() -> None:
    """_prune_part() stores _prune_original, _prune_kind, and _prune_summary in part.metadata."""
    original = _make_tool_return(content="original content")
    pruned = _prune_part(original, "[pruned]", "prune", summary="test summary")
    assert pruned.content == "[pruned]"
    assert isinstance(pruned.metadata, dict)
    assert pruned.metadata["_prune_original"] == "original content"
    assert pruned.metadata["_prune_kind"] == "prune"
    assert pruned.metadata["_prune_summary"] == "test summary"


def test_prune_part_is_idempotent() -> None:
    """_prune_part() on an already-pruned part returns it unchanged."""
    original = _make_tool_return(content="original content")
    pruned = _prune_part(original, "[pruned]", "prune")
    re_pruned = _prune_part(pruned, "[re-pruned]", "prune")
    assert re_pruned.content == "[pruned]"
    assert re_pruned is pruned


def test_is_pruned_returns_true_for_pruned_part() -> None:
    """_is_pruned() returns True for a part that has been pruned."""
    original = _make_tool_return(content="original")
    pruned = _prune_part(original, "[pruned]", "prune")
    assert _is_pruned(pruned) is True


def test_is_pruned_returns_false_for_unpruned_part() -> None:
    """_is_pruned() returns False for a part that has not been pruned."""
    part = _make_tool_return(content="not pruned")
    assert _is_pruned(part) is False


def test_dedup_exact_keeps_newest_prunes_older() -> None:
    """_dedup_exact() keeps the last duplicate and prunes earlier ones."""
    call_id_1 = "call_dup_1"
    call_id_2 = "call_dup_2"
    messages: list[ModelRequest | ModelResponse] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="search", args='{"q": "test"}', tool_call_id=call_id_1)]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="search", content="result 1", tool_call_id=call_id_1)]
        ),
        ModelResponse(
            parts=[ToolCallPart(tool_name="search", args='{"q": "test"}', tool_call_id=call_id_2)]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="search", content="result 2", tool_call_id=call_id_2)]
        ),
    ]
    state = DCPState()
    config = DCPConfig()
    _dedup_exact(messages, state, config)

    # The first ToolReturnPart should be pruned (content replaced).
    first_return = messages[1].parts[0]
    assert isinstance(first_return, ToolReturnPart)
    assert _is_pruned(first_return) is True
    assert first_return.content == "[duplicate removed]"

    # The second ToolReturnPart should be untouched.
    second_return = messages[3].parts[0]
    assert isinstance(second_return, ToolReturnPart)
    assert _is_pruned(second_return) is False
    assert second_return.content == "result 2"


def test_strip_thinking_content_removes_thinking_before_last_user() -> None:
    """_strip_thinking_content() removes ThinkingPart from messages before the last user message."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="first prompt")]),
        ModelResponse(parts=[ThinkingPart(content="thinking 1"), TextPart(content="response 1")]),
        ModelRequest(parts=[UserPromptPart(content="second prompt")]),
        ModelResponse(parts=[ThinkingPart(content="thinking 2"), TextPart(content="response 2")]),
    ]
    result, stripped_count = _strip_thinking_content(messages)
    assert stripped_count == 1
    # The first ModelResponse should have ThinkingPart removed.
    first_response = result[1]
    assert isinstance(first_response, ModelResponse)
    thinking_parts = [p for p in first_response.parts if isinstance(p, ThinkingPart)]
    assert len(thinking_parts) == 0
    text_parts = [p for p in first_response.parts if isinstance(p, TextPart)]
    assert len(text_parts) == 1
    # The second ModelResponse (after last user) should retain ThinkingPart.
    second_response = result[3]
    assert isinstance(second_response, ModelResponse)
    thinking_parts_after = [p for p in second_response.parts if isinstance(p, ThinkingPart)]
    assert len(thinking_parts_after) == 1


def test_strip_thinking_content_no_user_message() -> None:
    """_strip_thinking_content() returns messages unchanged when no user message exists."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelResponse(parts=[ThinkingPart(content="thinking"), TextPart(content="response")]),
    ]
    result, stripped_count = _strip_thinking_content(messages)
    assert stripped_count == 0
    assert result is messages


def test_purge_failed_tool_inputs_old_errors_pruned() -> None:
    """purge_failed_tool_inputs() prunes outputs of failed tools that are old enough."""
    failed_id = "call_failed_1"
    # Create messages with enough tool-call iterations so the error is beyond protection.
    messages: list[ModelRequest | ModelResponse] = []
    # Add a failed tool call + retry prompt early in history.
    messages.append(
        ModelResponse(
            parts=[ToolCallPart(tool_name="failing_tool", args="{}", tool_call_id=failed_id)]
        ),
    )
    messages.append(
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content="Error: something failed",
                    tool_name="failing_tool",
                    tool_call_id=failed_id,
                )
            ]
        ),
    )
    # Add several more tool-call iterations to push the error beyond protection window.
    for i in range(5):
        cid = f"call_ok_{i}"
        messages.append(
            ModelResponse(parts=[ToolCallPart(tool_name="ok_tool", args="{}", tool_call_id=cid)])
        )
        messages.append(
            ModelRequest(
                parts=[ToolReturnPart(tool_name="ok_tool", content="ok", tool_call_id=cid)]
            )
        )

    state = MagicMock(spec=PrunableState)
    state.pruned_tools = set()
    state.current_turn = 10

    purge_failed_tool_inputs(
        messages,
        state,
        purge_error_iterations=2,
        iteration_protection=2,
        protected_tools=set(),
    )
    assert failed_id in state.pruned_tools


def test_purge_failed_tool_inputs_recent_errors_preserved() -> None:
    """purge_failed_tool_inputs() preserves recent errors within the protection window."""
    failed_id = "call_recent_fail"
    messages: list[ModelRequest | ModelResponse] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="failing_tool", args="{}", tool_call_id=failed_id)]
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content="Error: failed", tool_name="failing_tool", tool_call_id=failed_id
                )
            ]
        ),
        # Only 1 additional iteration — within protection window of 3.
        ModelResponse(parts=[ToolCallPart(tool_name="ok_tool", args="{}", tool_call_id="call_ok")]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="ok_tool", content="ok", tool_call_id="call_ok")]
        ),
    ]

    state = MagicMock(spec=PrunableState)
    state.pruned_tools = set()
    state.current_turn = 1

    purge_failed_tool_inputs(
        messages,
        state,
        purge_error_iterations=5,
        iteration_protection=3,
        protected_tools=set(),
    )
    assert failed_id not in state.pruned_tools


def test_purge_failed_tool_inputs_prunes_outputs_not_inputs() -> None:
    """purge_failed_tool_inputs() adds to pruned_tools (outputs), does not modify messages."""
    failed_id = "call_output_fail"
    messages: list[ModelRequest | ModelResponse] = []
    messages.append(
        ModelResponse(
            parts=[ToolCallPart(tool_name="bad_tool", args="{}", tool_call_id=failed_id)]
        ),
    )
    messages.append(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="bad_tool",
                    content="error output",
                    tool_call_id=failed_id,
                    outcome="failed",
                ),
            ],
        ),
    )
    # Add iterations to push beyond protection.
    for i in range(5):
        cid = f"call_extra_{i}"
        messages.append(
            ModelResponse(parts=[ToolCallPart(tool_name="x", args="{}", tool_call_id=cid)])
        )
        messages.append(
            ModelRequest(parts=[ToolReturnPart(tool_name="x", content="ok", tool_call_id=cid)])
        )

    original_len = len(messages)
    state = MagicMock(spec=PrunableState)
    state.pruned_tools = set()
    state.current_turn = 10

    purge_failed_tool_inputs(
        messages,
        state,
        purge_error_iterations=2,
        iteration_protection=2,
        protected_tools=set(),
    )
    # The function only marks IDs — it does NOT modify messages in place.
    assert len(messages) == original_len
    assert failed_id in state.pruned_tools


def test_purge_failed_tool_inputs_protected_tools_exempt() -> None:
    """purge_failed_tool_inputs() does not prune tools listed in protected_tools."""
    failed_id = "call_protected_fail"
    messages: list[ModelRequest | ModelResponse] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="protected_tool", args="{}", tool_call_id=failed_id)]
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(content="Error", tool_name="protected_tool", tool_call_id=failed_id)
            ]
        ),
    ]
    for i in range(5):
        cid = f"call_x_{i}"
        messages.append(
            ModelResponse(parts=[ToolCallPart(tool_name="x", args="{}", tool_call_id=cid)])
        )
        messages.append(
            ModelRequest(parts=[ToolReturnPart(tool_name="x", content="ok", tool_call_id=cid)])
        )

    state = MagicMock(spec=PrunableState)
    state.pruned_tools = set()
    state.current_turn = 10

    purge_failed_tool_inputs(
        messages,
        state,
        purge_error_iterations=1,
        iteration_protection=1,
        protected_tools={"protected_tool"},
    )
    assert failed_id not in state.pruned_tools


# ---------------------------------------------------------------------------
# 8.7 — tools.py
# ---------------------------------------------------------------------------


def test_prune_tool_valid_id_returns_success() -> None:
    """prune_tool() with a valid numeric ID returns a success dict with 'applied' status."""
    state = DCPState()
    state.tool_id_list = ["call_1", "call_2"]
    ctx = _make_run_ctx(state)

    result = prune_tool(ctx, ids=["0"])
    assert result["status"] == "applied"
    assert result["count"] == 1
    assert result["pruned_ids"] == ["0"]
    assert len(state.pending_actions) == 1
    action = state.pending_actions[0]
    assert action.kind == "prune"
    assert action.ids == ("call_1",)


def test_prune_tool_invalid_id_raises_model_retry() -> None:
    """prune_tool() with an out-of-range ID raises ModelRetry."""
    from pydantic_ai import ModelRetry

    state = DCPState()
    state.tool_id_list = ["call_1"]
    ctx = _make_run_ctx(state)

    with pytest.raises(ModelRetry, match="out of range"):
        prune_tool(ctx, ids=["5"])


def test_prune_tool_non_numeric_id_raises_model_retry() -> None:
    """prune_tool() with a non-numeric ID raises ModelRetry."""
    from pydantic_ai import ModelRetry

    state = DCPState()
    state.tool_id_list = ["call_1"]
    ctx = _make_run_ctx(state)

    with pytest.raises(ModelRetry, match="not a valid number"):
        prune_tool(ctx, ids=["abc"])


def test_distill_tool_valid_targets_returns_success() -> None:
    """distill_tool() with valid targets returns a success dict."""
    state = DCPState()
    state.tool_id_list = ["call_1", "call_2"]
    ctx = _make_run_ctx(state)

    result = distill_tool(
        ctx,
        targets=[
            {"id": "0", "distillation": "summary of tool 1"},
            {"id": "1", "distillation": "summary of tool 2"},
        ],
    )
    assert result["status"] == "applied"
    assert result["count"] == 2
    assert result["distilled_ids"] == ["0", "1"]
    assert len(state.pending_actions) == 1
    action = state.pending_actions[0]
    assert action.kind == "distill"
    assert len(action.targets) == 2
    assert action.targets[0].tool_call_id == "call_1"
    assert action.targets[0].distillation == "summary of tool 1"


def test_distill_tool_empty_targets_raises_model_retry() -> None:
    """distill_tool() with empty targets raises ModelRetry."""
    from pydantic_ai import ModelRetry

    state = DCPState()
    ctx = _make_run_ctx(state)

    with pytest.raises(ModelRetry, match="targets cannot be empty"):
        distill_tool(ctx, targets=[])


def test_decompress_tool_on_pruned_returns_original_content() -> None:
    """decompress_tool() on a pruned tool returns original content from metadata."""
    original_content = "the original output"
    part = _make_tool_return(
        tool_name="search", content=original_content, tool_call_id="call_pruned"
    )
    pruned_part = _prune_part(part, "[pruned]", "prune")

    state = DCPState()
    state.tool_id_list = ["call_pruned"]
    state.current_messages = [
        ModelRequest(parts=[pruned_part]),
    ]
    ctx = _make_run_ctx(state)

    result = decompress_tool(ctx, tool_id="0")
    assert result["restored"] is True
    assert result["original_content"] == original_content
    assert result["was_pruned_as"] == "prune"


def test_decompress_tool_on_non_pruned_returns_error() -> None:
    """decompress_tool() on a non-pruned tool returns restored=False."""
    part = _make_tool_return(tool_name="search", content="live content", tool_call_id="call_live")

    state = DCPState()
    state.tool_id_list = ["call_live"]
    state.current_messages = [
        ModelRequest(parts=[part]),
    ]
    ctx = _make_run_ctx(state)

    result = decompress_tool(ctx, tool_id="0")
    assert result["restored"] is False
    assert "not pruned" in result["reason"]


def test_decompress_tool_does_not_modify_history() -> None:
    """decompress_tool() returns original content without modifying message history."""
    original_content = "secret data"
    part = _make_tool_return(tool_name="search", content=original_content, tool_call_id="call_d")
    pruned_part = _prune_part(part, "[pruned]", "prune")

    state = DCPState()
    state.tool_id_list = ["call_d"]
    state.current_messages = [ModelRequest(parts=[pruned_part])]
    ctx = _make_run_ctx(state)

    decompress_tool(ctx, tool_id="0")
    # The message in history should still have "[pruned]" as content.
    history_part = state.current_messages[0].parts[0]
    assert isinstance(history_part, ToolReturnPart)
    assert history_part.content == "[pruned]"


def test_decompress_tool_not_found_returns_error() -> None:
    """decompress_tool() with an ID not in messages returns restored=False."""
    state = DCPState()
    state.tool_id_list = ["call_missing"]
    state.current_messages = []
    ctx = _make_run_ctx(state)

    result = decompress_tool(ctx, tool_id="0")
    assert result["restored"] is False


def test_prune_tool_clear_thinking_strips_all() -> None:
    """prune_tool() with clear_thinking=True strips all ThinkingPart immediately."""
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    state = DCPState()
    state.tool_id_list = ["call_1"]
    ctx = _make_run_ctx(state)

    # Set up messages with ThinkingPart in multiple ModelResponses
    state.current_messages = [
        ModelResponse(
            parts=[ThinkingPart(content="old reasoning"), TextPart(content="r1")],
        ),
        ModelResponse(
            parts=[ThinkingPart(content="more reasoning"), TextPart(content="r2")],
        ),
    ]

    result = prune_tool(ctx, clear_thinking=True)
    assert result["status"] == "applied"
    assert "cleared" in result["action"]
    # All ThinkingPart should be removed from current_messages
    for msg in state.current_messages:
        for part in msg.parts:
            assert not isinstance(part, ThinkingPart)


def test_prune_tool_both_ids_and_clear_thinking() -> None:
    """prune_tool() accepts both ids and clear_thinking in the same call."""
    state = DCPState()
    state.tool_id_list = ["call_1"]
    ctx = _make_run_ctx(state)

    # clear_thinking=False is a no-op, ids still work
    result = prune_tool(ctx, ids=["0"], clear_thinking=False)
    assert result["status"] == "applied"
    assert len(state.pending_actions) == 1


# ---------------------------------------------------------------------------
# 8.8 — nudge.py
# ---------------------------------------------------------------------------


def test_build_nudge_text_normal_level() -> None:
    """build_nudge_text() for NORMAL level includes 'low' and no action items."""
    state = DCPState()
    state.watermark_level = WatermarkLevel.NORMAL
    state.current_tokens = 1000
    config = DCPConfig(max_context_tokens=128_000)

    text = build_nudge_text(state, config)
    assert "<system-reminder>" in text
    assert "low" in text
    assert "No context management action needed" in text


def test_build_nudge_text_info_level() -> None:
    """build_nudge_text() for INFO level includes 'moderate' and suggested actions."""
    state = DCPState()
    state.watermark_level = WatermarkLevel.INFO
    state.current_tokens = 80000
    config = DCPConfig(max_context_tokens=128_000)

    text = build_nudge_text(state, config)
    assert "moderate" in text
    assert "SUGGESTED ACTIONS" in text


def test_build_nudge_text_warning_level() -> None:
    """build_nudge_text() for WARNING level includes 'filling up' and recommended actions."""
    state = DCPState()
    state.watermark_level = WatermarkLevel.WARNING
    state.current_tokens = 100000
    config = DCPConfig(max_context_tokens=128_000)

    text = build_nudge_text(state, config)
    assert "filling up" in text
    assert "RECOMMENDED ACTIONS" in text


def test_build_nudge_text_critical_level() -> None:
    """build_nudge_text() for CRITICAL level includes 'CRITICAL' and immediate action."""
    state = DCPState()
    state.watermark_level = WatermarkLevel.CRITICAL
    state.current_tokens = 120000
    config = DCPConfig(max_context_tokens=128_000)

    text = build_nudge_text(state, config)
    assert "CRITICAL" in text
    assert "IMMEDIATE ACTION REQUIRED" in text


def test_build_nudge_text_includes_pressure_percentage() -> None:
    """build_nudge_text() includes the pressure percentage (e.g., '62%')."""
    state = DCPState()
    state.watermark_level = WatermarkLevel.INFO
    state.current_tokens = 80000
    config = DCPConfig(max_context_tokens=128_000)

    text = build_nudge_text(state, config)
    pct = round(80000 / 128_000 * 100)
    assert f"{pct}%" in text


def test_build_nudge_text_higher_level_more_urgent() -> None:
    """Higher watermark levels produce more urgent messaging than lower levels."""
    config = DCPConfig(max_context_tokens=1000)

    state_normal = DCPState()
    state_normal.watermark_level = WatermarkLevel.NORMAL
    state_normal.current_tokens = 100
    text_normal = build_nudge_text(state_normal, config)

    state_critical = DCPState()
    state_critical.watermark_level = WatermarkLevel.CRITICAL
    state_critical.current_tokens = 950
    text_critical = build_nudge_text(state_critical, config)

    # Critical should have "IMMEDIATE ACTION REQUIRED" while normal says "No action needed".
    assert "IMMEDIATE ACTION REQUIRED" in text_critical
    assert "No context management action needed" in text_normal


# ---------------------------------------------------------------------------
# 8.9 — prunable_list.py
# ---------------------------------------------------------------------------


def test_build_prunable_list_with_three_tool_returns() -> None:
    """build_prunable_list() with 3 ToolReturnParts produces a numbered list."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="search", content="result 1", tool_call_id="c1"),
                ToolReturnPart(tool_name="read", content="result 2", tool_call_id="c2"),
                ToolReturnPart(tool_name="bash", content="result 3", tool_call_id="c3"),
            ],
        ),
    ]
    state = DCPState()
    config = DCPConfig()

    text = build_prunable_list(messages, state, config)
    assert "<prunable-tools>" in text
    assert "</prunable-tools>" in text
    assert "0: search" in text
    assert "1: read" in text
    assert "2: bash" in text
    assert len(state.tool_id_list) == 3
    assert state.tool_id_list == ["c1", "c2", "c3"]


def test_build_prunable_list_no_tools_returns_empty() -> None:
    """build_prunable_list() with no ToolReturnParts returns an empty string."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
    ]
    state = DCPState()
    config = DCPConfig()

    text = build_prunable_list(messages, state, config)
    assert text == ""
    assert state.tool_id_list == []


def test_build_prunable_list_filters_meta_tool_names() -> None:
    """build_prunable_list() excludes prune/distill/decompress tool returns."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="search", content="real result", tool_call_id="c1"),
                ToolReturnPart(
                    tool_name="prune", content='{"status":"applied"}', tool_call_id="c2"
                ),
                ToolReturnPart(
                    tool_name="distill", content='{"status":"applied"}', tool_call_id="c3"
                ),
                ToolReturnPart(
                    tool_name="decompress", content='{"restored":true}', tool_call_id="c4"
                ),
            ],
        ),
    ]
    state = DCPState()
    config = DCPConfig()

    text = build_prunable_list(messages, state, config)
    assert "0: search" in text
    # Meta tools should not appear as numbered entries.
    assert "1:" not in text
    assert "2:" not in text
    assert "3:" not in text
    assert len(state.tool_id_list) == 1
    assert state.tool_id_list == ["c1"]


def test_meta_tool_names_contains_prune_distill_decompress() -> None:
    """META_TOOL_NAMES contains exactly 'prune', 'distill', and 'decompress'."""
    assert "prune" in META_TOOL_NAMES
    assert "distill" in META_TOOL_NAMES
    assert "decompress" in META_TOOL_NAMES
    assert len(META_TOOL_NAMES) == 3


def test_inject_prunable_list_system_role() -> None:
    """inject_prunable_list() with role='system' appends SystemPromptPart to last ModelRequest."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="response")]),
        ModelRequest(parts=[UserPromptPart(content="follow up")]),
    ]
    text = "<prunable-tools>\n0: search\n</prunable-tools>"

    result = inject_prunable_list(messages, text, role="system")
    # The last ModelRequest should now have a SystemPromptPart appended.
    last_request = result[-1]
    assert isinstance(last_request, ModelRequest)
    system_parts = [p for p in last_request.parts if isinstance(p, SystemPromptPart)]
    assert len(system_parts) == 1
    assert system_parts[0].content == text


def test_inject_prunable_list_user_role() -> None:
    """inject_prunable_list() with role='user' appends UserPromptPart."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
    ]
    text = "<prunable-tools>\n0: search\n</prunable-tools>"

    result = inject_prunable_list(messages, text, role="user")
    last_request = result[-1]
    assert isinstance(last_request, ModelRequest)
    user_parts = [p for p in last_request.parts if isinstance(p, UserPromptPart)]
    # There should be 2 UserPromptParts: the original + the injected one.
    assert len(user_parts) == 2
    assert user_parts[1].content == text


def test_inject_prunable_list_empty_text_returns_unchanged() -> None:
    """inject_prunable_list() with empty text returns messages unchanged."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
    ]
    result = inject_prunable_list(messages, "")
    assert result is messages


def test_inject_prunable_list_strips_old_injections() -> None:
    """inject_prunable_list() removes old <prunable-tools> injections before adding new one."""
    old_text = "<prunable-tools>\n0: old\n</prunable-tools>"
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[UserPromptPart(content="hello"), SystemPromptPart(content=old_text)]),
        ModelResponse(parts=[TextPart(content="response")]),
        ModelRequest(parts=[UserPromptPart(content="new prompt")]),
    ]
    new_text = "<prunable-tools>\n0: new\n</prunable-tools>"

    result = inject_prunable_list(messages, new_text, role="system")
    # The old injection should be stripped from the first ModelRequest.
    first_request = result[0]
    assert isinstance(first_request, ModelRequest)
    old_system_parts = [p for p in first_request.parts if isinstance(p, SystemPromptPart)]
    assert len(old_system_parts) == 0
    # The new injection should be in the last ModelRequest.
    last_request = result[-1]
    assert isinstance(last_request, ModelRequest)
    new_system_parts = [p for p in last_request.parts if isinstance(p, SystemPromptPart)]
    assert len(new_system_parts) == 1
    assert new_system_parts[0].content == new_text


def test_inject_prunable_list_no_model_request_appends_one() -> None:
    """inject_prunable_list() appends a new ModelRequest when none exists."""
    messages: list[ModelRequest | ModelResponse] = [
        ModelResponse(parts=[TextPart(content="response")]),
    ]
    text = "<prunable-tools>\n0: search\n</prunable-tools>"

    result = inject_prunable_list(messages, text, role="system")
    assert len(result) == 2
    new_request = result[-1]
    assert isinstance(new_request, ModelRequest)
    system_parts = [p for p in new_request.parts if isinstance(p, SystemPromptPart)]
    assert len(system_parts) == 1
