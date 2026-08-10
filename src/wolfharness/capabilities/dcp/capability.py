"""Dynamic Context Pruning capability — V2 pipeline with id-based actions.

Provides the ``DynamicContextPruningCapability`` class that integrates the
context pruning subsystem into the pydantic-ai capability lifecycle.  The
capability exposes prune/distill/decompress tools, system-prompt
instructions, and lifecycle hooks for context management.

The ``before_model_request`` hook implements a V2 pipeline:

- **Phase 0**: Watermark update — estimate tokens and escalate pressure.
- **Phase 0.5**: Auto-prune old meta-tool returns (prune/distill/decompress)
  beyond ``meta_tool_retention`` count.  Then build ``tool_id_list`` always;
  inject ``<prunable-tools>`` numbered list into the last ``ModelRequest``
  only when watermark >= INFO.
- **Phase 1**: Apply pending id-based actions (prune/distill)
  deferred from tool calls.
- **Phase 1.5**: Clear thinking — strip ``ThinkingPart`` from assistant
  messages before the last user message (persistent flag).
- **Phase 2**: Auto-strategies (exact dedup, purge errors) gated by
  watermark level (WARNING+).  Uses ``_PrunableStateAdapter`` for
  ``purge_failed_tool_inputs`` compatibility.
- **Phase 3c**: Guard last message — ensure the model receives a
  ``ModelRequest`` it can respond to.
- **Phase 4**: Counter-based nudge injection via ``ctx.enqueue()``
  when ``nudge_counter >= nudge_turn_frequency`` (turn-based) or
  ``nudge_step_counter >= nudge_step_frequency`` (step-based).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic_ai import RunContext, Tool
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    NativeTool,
    ProcessHistory,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolReturnPart,
)
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.agents.context import (
    AgentContext,  # noqa: TC001 — needed at runtime for get_type_hints()
)
from wolfharness.capabilities.dcp.block_store import (
    CompressionBlockStore,
)
from wolfharness.capabilities.dcp.config import DCPConfig
from wolfharness.capabilities.dcp.nudge import (
    build_nudge_text,
)
from wolfharness.capabilities.dcp.prunable_list import (
    META_TOOL_NAMES,
    build_prunable_list,
    inject_prunable_list,
)
from wolfharness.capabilities.dcp.state import (
    CompressionBlock,
    DCPState,
    PruneAction,
    WatermarkLevel,
)
from wolfharness.capabilities.dcp.strategies import (
    _apply_pruned_tools,
    _dedup_exact,
    _is_pruned,
    _PrunableStateAdapter,
    _prune_part,
    _StrategyConfigAdapter,
    _strip_thinking_content,
    purge_failed_tool_inputs,
)
from wolfharness.capabilities.dcp.token_utils import estimate_tokens
from wolfharness.capabilities.dcp.tools import (
    DistillTargetInput,
    decompress_tool,
    distill_tool,
    prune_tool,
)
from wolfharness.capabilities.dcp.watermark import (
    WatermarkStateMachine,
)


if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition

_DCP_METADATA_KEY = "dcp"

logger = logging.getLogger(__name__)

_INSTRUCTIONS_TEXT = """\
You operate a context-constrained environment and MUST PROACTIVELY MANAGE IT TO AVOID CONTEXT ROT.

AVAILABLE TOOLS FOR CONTEXT MANAGEMENT:
- `prune`: remove individual tool calls that are noise, irrelevant, or superseded. No preservation \
of content. DO NOT let irrelevant tool calls accumulate. DO NOT PRUNE TOOL OUTPUTS THAT YOU MAY \
NEED LATER.
- `distill`: condense key findings from tool calls into high-fidelity distillation to preserve \
gained insights. Use to extract valuable knowledge. BE THOROUGH, your distillation MUST be \
high-signal, low noise and complete.
- `decompress`: restore original content from a pruned or distilled tool output when you realize \
you need it back.
- `prune` with `clear_thinking`: toggle stripping of thinking/reasoning content from assistant \
messages before the last user message. Pass `clear_thinking: true` to enable persistent stripping \
(saves tokens by removing old reasoning that is no longer needed), `clear_thinking: false` to \
disable. This is orthogonal to `ids` — both can be used in the same call.

THE DISTILL TOOL
`distill` is the favored way to target specific tools and crystalize their value into high-signal \
low-noise knowledge nuggets. Your distillation must be comprehensive, capturing technical details \
(symbols, signatures, logic, constraints) such that the raw output is no longer needed. \
THINK complete technical substitute.

THE PRUNE TOOL
`prune` is your last resort for context management. It is a blunt instrument that removes tool \
outputs entirely, without ANY preservation. It is best used to eliminate noise, irrelevant \
information, or superseded outputs that no longer add value to the conversation.

CLEAR THINKING
When `clear_thinking` is enabled (via `prune(clear_thinking=True)`), reasoning/thinking content \
from previous assistant turns is automatically stripped from the context on every subsequent \
request. This reduces token usage significantly when the model produces long reasoning chains. \
The most recent thinking (after the last user message) is always preserved. Toggle it off with \
`prune(clear_thinking=False)` when you need full reasoning history.

TIMING
Prefer managing context at the START of a new agentic loop (after receiving a user message) \
rather than at the END of your previous turn.

EVALUATE YOUR CONTEXT AND MANAGE REGULARLY TO AVOID CONTEXT ROT. AVOID USING MANAGEMENT TOOLS \
AS THE ONLY TOOL CALLS IN YOUR RESPONSE, PARALLELIZE WITH OTHER RELEVANT TOOLS TO TASK \
CONTINUATION.
"""


class DynamicContextPruningCapability(AbstractCapability[Any]):
    """Dynamic context pruning capability with V2 pipeline.

    Integrates the context pruning subsystem into the agent capability
    lifecycle.  Provides prune/distill/decompress tools, system-prompt
    instructions, and lifecycle hooks for context management.

    The capability maintains per-session ``DCPState`` in
    ``SessionData.metadata['dcp']`` and falls back to an in-memory
    ``DCPState`` when session state is unavailable.

    Ordering: wraps ``ProcessHistory`` (runs before it), and is wrapped
    by ``NativeTool`` (NativeTool runs outermost).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        expose_tools: bool = True,
        info_threshold: float = 0.60,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
        max_context_tokens: int = 128_000,
        inject_role: str = "user",
        nudge_role: str = "user",
        nudge_visible: bool = True,
        nudge_turn_frequency: int = 3,
        nudge_step_frequency: int = 50,
        auto_dedup: bool = True,
        auto_strategy_threshold: WatermarkLevel = WatermarkLevel.INFO,
        purge_error_steps: int = 3,
        step_protection: int = 2,
        protected_tool_patterns: tuple[str, ...] = ("ask", "confirm", "approval_*"),
        protected_tools: set[str] | None = None,
        meta_tool_retention: int = 1,
        clear_thinking_enabled: bool = False,
    ) -> None:
        """Initialize the dynamic context pruning capability.

        Args:
            enabled: Whether the capability is active.  When ``False``,
                all hooks are no-ops.
            expose_tools: Whether to expose prune/distill/decompress tools.
            info_threshold: Context pressure ratio for INFO watermark.
            warning_threshold: Context pressure ratio for WARNING.
            critical_threshold: Context pressure ratio for CRITICAL.
            max_context_tokens: Maximum context window size in tokens.
            inject_role: Role for ``<prunable-tools>`` list injection
                (``"system"`` or ``"user"``).
            nudge_role: Role for nudge injection (``"system"`` or ``"user"``).
            nudge_visible: Whether nudge messages are visible to the frontend
                via EventBus (``True``) or hidden (``False``).
            nudge_turn_frequency: Number of turns between nudge injections.
            nudge_step_frequency: Number of tool-call steps between nudge
                injections (0 to disable step-based nudges).
            auto_dedup: Whether to auto-deduplicate at the strategy threshold.
            auto_strategy_threshold: Watermark level at which auto-strategies
                (dedup, purge errors) begin running.
            purge_error_steps: Tool-call steps before error tool calls are purged.
            step_protection: Recent tool-call steps protected from pruning.
            protected_tool_patterns: Glob patterns for protected tools.
            protected_tools: Additional protected tool names.
            meta_tool_retention: How many recent meta-tool (prune/distill/
                decompress) returns to keep before auto-pruning older ones.
            clear_thinking_enabled: Whether the ``clear_thinking`` parameter
                on the prune tool is active.  When ``True``, the model can
                toggle persistent stripping of ``ThinkingPart`` content from
                assistant messages before the last user message.
        """
        self._config: DCPConfig = DCPConfig(
            enabled=enabled,
            expose_tools=expose_tools,
            info_threshold=info_threshold,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            max_context_tokens=max_context_tokens,
            inject_role=inject_role if inject_role in ("system", "user") else "user",
            nudge_role=nudge_role if nudge_role in ("system", "user") else "user",
            nudge_visible=nudge_visible,
            nudge_turn_frequency=nudge_turn_frequency,
            nudge_step_frequency=nudge_step_frequency,
            auto_dedup=auto_dedup,
            auto_strategy_threshold=auto_strategy_threshold,
            purge_error_steps=purge_error_steps,
            step_protection=step_protection,
            protected_tool_patterns=protected_tool_patterns,
            protected_tools=protected_tools if protected_tools is not None else set(),
            meta_tool_retention=meta_tool_retention,
            clear_thinking_enabled=clear_thinking_enabled,
        )
        self._watermark: WatermarkStateMachine = WatermarkStateMachine(
            info_threshold=self._config.info_threshold,
            warning_threshold=self._config.warning_threshold,
            critical_threshold=self._config.critical_threshold,
        )
        self._block_store: CompressionBlockStore = CompressionBlockStore()
        self._fallback_state: DCPState = DCPState()
        self._fallback_state.block_store = self._block_store

        logger.info(
            "DynamicContextPruning capability initialized: enabled=%s max_context_tokens=%d "
            "auto_dedup=%s nudge_turn_frequency=%d nudge_step_frequency=%d meta_tool_retention=%d",
            self._config.enabled,
            self._config.max_context_tokens,
            self._config.auto_dedup,
            self._config.nudge_turn_frequency,
            self._config.nudge_step_frequency,
            self._config.meta_tool_retention,
        )

    @classmethod
    def from_config(cls, config: DCPConfig) -> DynamicContextPruningCapability:
        """Create a capability from a pre-built ``DCPConfig``.

        Args:
            config: Configuration object with all parameters set.

        Returns:
            A new ``DynamicContextPruningCapability`` instance.
        """
        return cls(
            enabled=config.enabled,
            expose_tools=config.expose_tools,
            info_threshold=config.info_threshold,
            warning_threshold=config.warning_threshold,
            critical_threshold=config.critical_threshold,
            max_context_tokens=config.max_context_tokens,
            inject_role=config.inject_role,
            nudge_role=config.nudge_role,
            nudge_visible=config.nudge_visible,
            nudge_turn_frequency=config.nudge_turn_frequency,
            nudge_step_frequency=config.nudge_step_frequency,
            auto_dedup=config.auto_dedup,
            auto_strategy_threshold=config.auto_strategy_threshold,
            purge_error_steps=config.purge_error_steps,
            step_protection=config.step_protection,
            protected_tool_patterns=config.protected_tool_patterns,
            protected_tools=config.protected_tools,
            meta_tool_retention=config.meta_tool_retention,
            clear_thinking_enabled=config.clear_thinking_enabled,
        )

    # ---- DCPState access ----

    def _get_dcp_state(self, ctx: RunContext[AgentContext]) -> DCPState:
        """Get the ``DCPState`` for the current session, creating it if needed.

        Tries to read from ``SessionData.metadata['dcp']``.  If
        session state is unavailable (no pool, no session, etc.), falls
        back to the per-capability ``_fallback_state``.

        Args:
            ctx: The run context with agent dependencies.

        Returns:
            The mutable ``DCPState`` for this session.
        """
        session_data = ctx.deps.get_session_state()
        if session_data is None:
            return self._fallback_state

        metadata = session_data.metadata
        state = metadata.get(_DCP_METADATA_KEY)
        if not isinstance(state, DCPState):
            state = DCPState.from_dict(state) if isinstance(state, dict) else DCPState()
            state.block_store = self._block_store
            metadata[_DCP_METADATA_KEY] = state
        return state

    def _get_session_id(self, ctx: RunContext[AgentContext]) -> str:
        """Get the session ID for block store namespace isolation.

        Args:
            ctx: The run context with agent dependencies.

        Returns:
            The session ID string, or ``"default"`` if unavailable.
        """
        session_data = ctx.deps.get_session_state()
        if session_data is None:
            return "default"
        return session_data.session_id if hasattr(session_data, "session_id") else "default"

    # ---- Static configuration ----

    def get_instructions(self) -> AgentInstructions[Any] | None:
        """Return system-prompt instructions describing pruning tools.

        Returns a static string that documents the available pruning
        tools, the numbered ID system, and the decompress tool.
        """
        return _INSTRUCTIONS_TEXT

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Return a ``FunctionToolset`` with prune/distill/decompress tools.

        Returns ``None`` when ``config.expose_tools`` is ``False`` or
        ``config.enabled`` is ``False``.
        """
        if not self._config.enabled or not self._config.expose_tools:
            return None

        return FunctionToolset(
            tools=[
                Tool(
                    self._prune_tool_handler,
                    name="prune",
                    description=(
                        "Use this tool to remove tool outputs from context entirely. "
                        "No preservation - pure deletion.\n\n"
                        "THE PRUNABLE TOOLS LIST\n"
                        "A `<prunable-tools>` section surfaces in context showing outputs "
                        "eligible for removal. Each line reads "
                        "`ID: tool, parameter (~token usage)` (e.g., "
                        "`20: read, /path/to/file.ts (~1500 tokens)`). Reference outputs "
                        "by their numeric ID - these are your ONLY valid targets for pruning.\n\n"
                        "THE WAYS OF PRUNE\n"
                        "`prune` is surgical deletion - eliminating noise (irrelevant or "
                        "unhelpful outputs), superseded information (older outputs replaced "
                        "by newer data), or wrong targets (you accessed something that "
                        "turned out to be irrelevant). Use it to keep your context lean "
                        "and focused.\n\n"
                        "BATCH WISELY! Pruning is most effective when consolidated. Don't "
                        "prune a single tiny output - accumulate several candidates before "
                        "acting.\n\n"
                        "Do NOT prune when:\n"
                        "NEEDED LATER: You plan to edit the file or reference this context "
                        "for implementation.\n"
                        "UNCERTAINTY: If you might need to re-examine the original, keep it.\n\n"
                        'Before pruning, ask: "Is this noise, or will it serve me?" If the '
                        "latter, keep it. Pruning that forces re-fetching is a net loss.\n\n"
                        "THE FORMAT OF PRUNE\n"
                        "`ids`: Array of numeric IDs (as strings) from the `<prunable-tools>` "
                        "list\n\n"
                        "DISCOVERING IDS\n"
                        "If the `<prunable-tools>` list is not visible in your context, "
                        'pass `"999"` or `"-1"` as an ID to retrieve the full list of '
                        "valid prunable-tool IDs and their summaries.\n\n"
                        "CLEAR THINKING\n"
                        "Pass `clear_thinking: true` to enable persistent stripping of "
                        "thinking/reasoning content from assistant messages before the "
                        "last user message. This saves tokens when old reasoning chains "
                        "are no longer needed. Pass `clear_thinking: false` to disable. "
                        "Both `ids` and `clear_thinking` can be used in the same call."
                    ),
                ),
                Tool(
                    self._distill_tool_handler,
                    name="distill",
                    description=(
                        "Use this tool to distill relevant findings from a selection of "
                        "raw tool outputs into preserved knowledge, in order to denoise "
                        "key bits and parts of context.\n\n"
                        "THE PRUNABLE TOOLS LIST\n"
                        "A <prunable-tools> will show in context when outputs are available "
                        "for distillation (you don't need to look for it). Each entry follows "
                        "the format `ID: tool, parameter (~token usage)` (e.g., "
                        "`20: read, /path/to/file.ts (~1500 tokens)`). You MUST select outputs "
                        "by their numeric ID. THESE ARE YOUR ONLY VALID TARGETS.\n\n"
                        "THE PHILOSOPHY OF DISTILLATION\n"
                        "`distill` is your favored instrument for transforming raw tool outputs "
                        "into preserved knowledge. This is not mere summarization; it is "
                        "high-fidelity extraction that makes the original output obsolete.\n\n"
                        "Your distillation must be COMPLETE. Capture function signatures, type "
                        "definitions, business logic, constraints, configuration values... "
                        "EVERYTHING essential. Think of it as creating a high signal technical "
                        "substitute so faithful that re-fetching the original would yield no "
                        "additional value. Be thorough; be comprehensive; leave no ambiguity, "
                        "ensure that your distillation stands alone, and is designed for easy "
                        "retrieval and comprehension.\n\n"
                        "AIM FOR IMPACT. Distillation is most powerful when applied to outputs "
                        "that contain signal buried in noise. A single line requires no "
                        "distillation; a hundred lines of API documentation do. Make sure the "
                        "distillation is meaningful.\n\n"
                        "THE WAYS OF DISTILL\n"
                        "`distill` when you have extracted the essence from tool outputs and "
                        "the raw form has served its purpose.\n"
                        "Here are some examples:\n"
                        "EXPLORATION: You've read extensively and grasp the architecture. The "
                        "original file contents are no longer needed; your understanding, "
                        "synthesized, is sufficient.\n"
                        "PRESERVATION: Valuable technical details (signatures, logic, "
                        "constraints) coexist with noise. Preserve the former; discard the "
                        "latter.\n\n"
                        "Not everything should be distilled. Prefer keeping raw outputs when:\n"
                        "PRECISION MATTERS: You will edit the file, grep for exact strings, or "
                        "need line-accurate references. Distillation sacrifices precision for "
                        "essence.\n"
                        "UNCERTAINTY REMAINS: If you might need to re-examine the original, defer. "
                        "Distillation is irreversible; be certain before you commit.\n\n"
                        'Before distilling, ask yourself: "Will I need the raw output for upcoming '
                        'work?" If you plan to edit a file you just read, keep it intact. '
                        "Distillation is for completed exploration, not active work.\n\n"
                        "THE FORMAT OF DISTILL\n"
                        "`targets`: Array of objects, each containing:\n"
                        "`id`: Numeric ID (as string) from the `<prunable-tools>` list\n"
                        "`distillation`: Complete technical substitute for that tool output\n\n"
                        "DISCOVERING IDS\n"
                        "If the `<prunable-tools>` list is not visible in your context, "
                        'pass an ID of `"999"` or `"-1"` to retrieve the full list of '
                        "valid prunable-tool IDs and their summaries."
                    ),
                ),
                Tool(
                    self._decompress_tool_handler,
                    name="decompress",
                    description=(
                        "Restore original content from a pruned or distilled tool output.\n\n"
                        "Use this when you realize you need the full original output that was "
                        "previously pruned or distilled. Look for entries marked [pruned] in "
                        "the <prunable-tools> list — their numeric IDs are valid targets.\n\n"
                        "THE FORMAT OF DECOMPRESS\n"
                        "`id`: Numeric ID (as string) of a [pruned] entry from the "
                        "<prunable-tools> list"
                    ),
                ),
            ],
        )

    def get_ordering(self) -> CapabilityOrdering | None:
        """Declare middleware chain position.

        Dynamic Context Pruning wraps ``ProcessHistory`` (runs before it),
        and is wrapped by ``NativeTool`` (NativeTool is outermost).
        """
        return CapabilityOrdering(
            wraps=[ProcessHistory],
            wrapped_by=[NativeTool],
        )

    # ---- Run lifecycle hooks ----

    async def before_run(self, ctx: RunContext[AgentContext]) -> None:
        """Initialize ``DCPState`` for the new run.

        Increments ``current_turn``, resets ``step_count`` to
        zero, and increments ``nudge_counter`` for the new turn.

        Early-returns as a no-op when ``config.enabled=False``.
        """
        if not self._config.enabled:
            return

        state = self._get_dcp_state(ctx)
        state.current_turn += 1
        state.step_count = 0
        state.nudge_counter += 1
        logger.debug(
            "DynamicContextPruning before_run: turn=%d pending=%d nudge_counter=%d",
            state.current_turn,
            len(state.pending_actions),
            state.nudge_counter,
        )

    # ---- Pre-request pipeline ----

    async def before_model_request(  # noqa: PLR0915
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Pre-request pipeline: Phase 0-4 context management.

        Pipeline phases:

        1. **Phase 0**: Watermark update — estimate tokens, compute
           pressure ratio, store watermark level and token count in state.
        2. **Phase 0.5**: Auto-prune old meta-tool returns beyond
           ``meta_tool_retention``.  Then build ``tool_id_list`` always;
           inject ``<prunable-tools>`` numbered list into the last
           ``ModelRequest`` only when watermark >= INFO.
        3. **Phase 1**: Apply pending id-based actions (prune/distill)
           from the deferred action queue.
        4. **Phase 1.5**: Clear thinking — strip ``ThinkingPart`` from
           assistant messages before the last user message (persistent flag).
        5. **Phase 2**: Auto-strategies (exact dedup, purge errors)
           gated by ``auto_strategy_threshold`` watermark level.
           Uses a temporary ``_PrunableStateAdapter`` for
           ``purge_failed_tool_inputs``.
        6. **Phase 3c**: Guard last message — append an empty
           ``ModelRequest`` if the last message is not one.
        7. **Phase 4**: Counter-based nudge injection when
           ``nudge_counter >= nudge_turn_frequency`` (turn-based) or
           ``nudge_step_counter >= nudge_step_frequency`` (step-based).

        Early-returns as a no-op when ``config.enabled=False``.

        Returns a new ``ModelRequestContext`` via
        ``dataclasses.replace()`` — never mutates the original.
        """
        if not self._config.enabled:
            return request_context

        state = self._get_dcp_state(ctx)
        session_id = self._get_session_id(ctx)
        messages = list(request_context.messages)
        original_msg_count = len(messages)
        state.current_messages = messages

        # --- Phase 0: Watermark Update (exact base + incremental delta) ---
        # Track message count at pipeline start for delta calculation.
        prev_msg_count = state.last_pipeline_msg_count
        state.last_pipeline_msg_count = len(messages)

        estimated_tokens = estimate_tokens(messages)
        current_cumulative = getattr(ctx, "usage", None)
        current_cumulative = current_cumulative.input_tokens if current_cumulative else 0

        last_request_actual = current_cumulative - state.last_cumulative_input_tokens

        if last_request_actual > 0 and prev_msg_count > 0 and len(messages) > prev_msg_count:
            # We have an exact per-request count from the API and know
            # which messages are new since the last pipeline run.  Base
            # is exact from API; only the incremental tool results need
            # estimation — minimising error.
            new_msgs = messages[prev_msg_count:]
            delta = estimate_tokens(new_msgs)
            total_tokens = last_request_actual + delta
        elif last_request_actual > 0 and prev_msg_count > 0 and len(messages) == prev_msg_count:
            # No new messages — reuse the exact last count.
            total_tokens = last_request_actual
            delta = 0
        else:
            # First request: no previous API data available; fall back
            # to raw estimate_tokens (will undercount but first request
            # watermark is not critical).
            total_tokens = estimated_tokens
            delta = 0

        level = self._watermark.update_with_tokens(
            total_tokens,
            self._config.max_context_tokens,
        )
        state.watermark_level = level
        state.current_tokens = total_tokens

        logger.info(
            "DynamicContextPruning Phase 0: turn=%d msgs=%d tokens=%d/%d (%.1f%%) "
            "watermark=%s (estimated=%d last_real=%d new_msgs=%d delta=%d)",
            state.current_turn,
            original_msg_count,
            total_tokens,
            self._config.max_context_tokens,
            (
                (total_tokens / self._config.max_context_tokens * 100)
                if self._config.max_context_tokens > 0
                else 0.0
            ),
            level.name,
            estimated_tokens,
            last_request_actual,
            len(messages) - prev_msg_count,
            delta,
        )

        # --- Phase 0.5: Auto-prune meta-tools + Build prunable list ---
        # Auto-prune old prune/distill/decompress returns so they don't
        # accumulate in context.  Done before building the prunable list
        # so meta-tool returns are already cleaned up.
        messages = self._auto_prune_meta_tools(messages, state)

        # Build tool_id_list always; inject <prunable-tools> list (INFO-gated).
        prunable_text = build_prunable_list(messages, state, self._config)
        if level >= WatermarkLevel.INFO and prunable_text:
            messages = inject_prunable_list(messages, prunable_text, self._config.inject_role)
            logger.debug(
                "DynamicContextPruning Phase 0.5: injected prunable list with "
                "%d entries (watermark=%s)",
                len(state.tool_id_list),
                level.name,
            )
        elif state.tool_id_list:
            logger.debug(
                "DynamicContextPruning Phase 0.5: built tool_id_list (%d entries), "
                "list injection skipped (watermark=%s < INFO)",
                len(state.tool_id_list),
                level.name,
            )

        else:
            logger.debug(
                "DynamicContextPruning Phase 0.5: no prunable tools found (watermark=%s)",
                level.name,
            )

        # --- Phase 1: Apply Pending Actions (id-based) + Re-prune ---
        # Pending actions from prune/distill tool calls are applied here.
        # Then _re_prune_messages re-applies previously-recorded pruning to
        # parts that ctx.state.message_history restored to original content
        # (our modifications to request_context.messages are ephemeral since
        # _clean_message_history returns a new list each iteration).
        pre_prune_estimate = estimate_tokens(messages)
        pending_count = len(state.pending_actions)
        if pending_count > 0:
            logger.debug(
                "DynamicContextPruning Phase 1: applying %d pending action(s) (kinds=%s)",
                pending_count,
                [a.kind for a in state.pending_actions],
            )
        messages = self._apply_pending_actions(messages, state, session_id)
        messages = self._re_prune_messages(messages, state)

        # --- Phase 1.5: Clear Thinking (persistent flag) ---
        # Strip ThinkingPart from assistant messages before the last user
        # message.  Re-applied every iteration because ctx.state.message_history
        # restores original content.  Gated by config + state flag.
        if self._config.clear_thinking_enabled:
            messages, stripped = _strip_thinking_content(messages)
            if stripped > 0:
                logger.info(
                    "DynamicContextPruning Phase 1.5: stripped %d ThinkingPart(s) "
                    "from history (turn=%d)",
                    stripped,
                    state.current_turn,
                )
            else:
                logger.debug(
                    "DynamicContextPruning Phase 1.5: no ThinkingPart found to strip (turn=%d)",
                    state.current_turn,
                )

        # --- Phase 2: Auto-Strategies (watermark-gated) ---
        if state.watermark_level >= self._config.auto_strategy_threshold:
            adapter = _PrunableStateAdapter(
                pruned_tools=set(),
                current_turn=state.current_turn,
                applied_action_ids=state.applied_action_ids,
                block_store=state.block_store,
            )

            if self._config.auto_dedup:
                _dedup_exact(messages, state, self._config)
                logger.debug(
                    "DynamicContextPruning Phase 2 dedup_exact: completed step=%d protected=%s",
                    state.step_count,
                    sorted(self._config.protected_tools),
                )

            cfg_adapter = _StrategyConfigAdapter(self._config)
            purge_failed_tool_inputs(
                messages,
                adapter,
                purge_error_iterations=cfg_adapter.purge_error_steps,
                iteration_protection=cfg_adapter.step_protection,
                protected_tools=cfg_adapter.protected_tools,
            )

            _apply_pruned_tools(messages, adapter, session_id=session_id)
            logger.debug(
                "DynamicContextPruning Phase 2 purge: applied pruned_tools from adapter",
            )

        # Log approximate savings from Phase 1 and Phase 2.
        # Uses tiktoken estimates — not exact, for logging/debugging only.
        post_prune_estimate = estimate_tokens(messages)
        saved_estimate = pre_prune_estimate - post_prune_estimate
        if saved_estimate > 0:
            logger.info(
                "DynamicContextPruning savings (approx): pre=%d post=%d saved~%d tokens",
                pre_prune_estimate,
                post_prune_estimate,
                saved_estimate,
            )

        # --- Phase 3c: Guard Last Message ---
        if messages and not isinstance(messages[-1], ModelRequest):
            messages.append(ModelRequest(parts=[]))
            logger.debug(
                "DynamicContextPruning Phase 3c: appended empty ModelRequest (last was %s)",
                type(messages[-2]).__name__ if len(messages) > 1 else "none",
            )

        # --- Phase 4: Nudge (counter-based, via session_pool.steer) ---
        # Route the nudge through SessionPool.steer() so that:
        # 1. The message is enqueued into the active agent run (model sees it)
        # 2. A UserMessageInsertedEvent is published to the EventBus when
        #    ``nudge_visible`` is True, allowing protocol frontends to
        #    display the nudge to the user.
        # When no SessionPool is available (standalone execution), fall
        # back to ``ctx.enqueue()`` which injects the message without
        # publishing an event.
        turn_trigger = state.nudge_counter >= self._config.nudge_turn_frequency
        step_trigger = (
            self._config.nudge_step_frequency > 0
            and state.nudge_step_counter >= self._config.nudge_step_frequency
        )
        if turn_trigger or step_trigger:
            nudge_text = build_nudge_text(state, self._config)
            session_id = self._get_session_id(ctx)
            steer_id: str | None = None
            host_ctx = ctx.deps.node.host_context
            session_pool = host_ctx.session_pool if host_ctx is not None else None
            if session_pool is not None:
                steer_id = await session_pool.steer(
                    session_id,
                    nudge_text,
                    emit_user_message=self._config.nudge_visible,
                )
                trigger = "turn" if turn_trigger else "step"
                state.nudge_counter = 0
                state.nudge_step_counter = 0
                logger.debug(
                    "DynamicContextPruning Phase 4: nudge steered via %s "
                    "counter (visible=%s steer_id=%s turn=%d tokens=%d/%d) "
                    "counters reset",
                    trigger,
                    self._config.nudge_visible,
                    steer_id,
                    state.current_turn,
                    state.current_tokens,
                    self._config.max_context_tokens,
                )
            elif hasattr(ctx, "enqueue"):
                # Fallback: standalone execution without SessionPool.
                if self._config.nudge_role == "system":
                    steer_id = ctx.enqueue(
                        SystemPromptPart(content=nudge_text),
                        priority="asap",
                    )
                else:
                    steer_id = ctx.enqueue(nudge_text, priority="asap")
                trigger = "turn" if turn_trigger else "step"
                state.nudge_counter = 0
                state.nudge_step_counter = 0
                logger.debug(
                    "DynamicContextPruning Phase 4: nudge enqueued (no "
                    "session_pool) via %s counter (role=%s enqueue_id=%s "
                    "turn=%d tokens=%d/%d) counters reset",
                    trigger,
                    self._config.nudge_role,
                    steer_id,
                    state.current_turn,
                    state.current_tokens,
                    self._config.max_context_tokens,
                )
            else:
                logger.debug(
                    "DynamicContextPruning Phase 4: no session_pool and "
                    "ctx.enqueue unavailable, skipping nudge injection "
                    "(turn=%d tokens=%d/%d)",
                    state.current_turn,
                    state.current_tokens,
                    self._config.max_context_tokens,
                )
        else:
            logger.debug(
                "DynamicContextPruning Phase 4: nudge skipped "
                "(nudge_counter=%d < frequency=%d, nudge_step_counter=%d < step_frequency=%d)",
                state.nudge_counter,
                self._config.nudge_turn_frequency,
                state.nudge_step_counter,
                self._config.nudge_step_frequency,
            )

        # --- Pipeline summary ---
        # Store cumulative for next iteration's delta calculation.
        state.last_cumulative_input_tokens = current_cumulative

        logger.info(
            "DynamicContextPruning pipeline: turn=%d msgs=%d→%d watermark=%s tokens=%d/%d (%.1f%%) "
            "pending=%d applied=%d nudge_counter=%d nudge_step_counter=%d step=%d "
            "(estimated=%d last_real=%d cum=%d)",
            state.current_turn,
            original_msg_count,
            len(messages),
            state.watermark_level.name,
            state.current_tokens,
            self._config.max_context_tokens,
            (
                (state.current_tokens / self._config.max_context_tokens * 100)
                if self._config.max_context_tokens > 0
                else 0.0
            ),
            pending_count,
            len(state.applied_action_ids),
            state.nudge_counter,
            state.nudge_step_counter,
            state.step_count,
            estimated_tokens,
            last_request_actual,
            current_cumulative,
        )

        return dataclasses.replace(request_context, messages=messages)

    # ---- Post-tool hook ----

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentContext],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: object,
    ) -> object:
        """Increment ``step_count`` and ``nudge_step_counter``.

        No truncation is performed here — truncation is handled by
        ``ToolOutputBudgetCapability``.  Pending prune/distill actions
        are NOT applied here — they are deferred to Phase 1 of the next
        ``before_model_request`` cycle.

        Early-returns as a no-op when ``config.enabled=False``.
        """
        if not self._config.enabled:
            return result

        state = self._get_dcp_state(ctx)
        state.step_count += 1
        state.nudge_step_counter += 1

        logger.debug(
            "DynamicContextPruning after_tool_execute: tool=%s step=%d turn=%d",
            call.tool_name,
            state.step_count,
            state.current_turn,
        )
        return result

    # ---- Tool handlers (wrappers for type-hint resolution) ----

    def _prune_tool_handler(
        self,
        ctx: RunContext[AgentContext],
        ids: list[str] | None = None,
        reason: str | None = None,
        clear_thinking: bool | None = None,
    ) -> dict[str, object]:
        """Handle a ``prune`` tool call from the model.

        ``tool_id_list`` is always populated by Phase 0.5 in
        ``before_model_request``, so no lazy rebuild is needed.

        When ``clear_thinking`` is ``True``, immediately strips all
        ``ThinkingPart`` from ``state.current_messages`` (one-shot).
        When the feature is disabled in config, the ``clear_thinking``
        parameter is ignored and a note is included in the return value.
        """
        if clear_thinking is True and not self._config.clear_thinking_enabled:
            # Feature disabled — report but don't strip.
            result = prune_tool(ctx, ids, reason, clear_thinking=None)  # type: ignore[arg-type]
            result["clear_thinking"] = "disabled (feature not enabled in config)"
            return result
        return prune_tool(ctx, ids, reason, clear_thinking)  # type: ignore[arg-type]

    def _distill_tool_handler(
        self,
        ctx: RunContext[AgentContext],
        targets: list[DistillTargetInput],
    ) -> dict[str, object]:
        """Handle a ``distill`` tool call from the model.

        ``tool_id_list`` is always populated by Phase 0.5 in
        ``before_model_request``, so no lazy rebuild is needed.
        """
        return distill_tool(ctx, targets)

    def _decompress_tool_handler(
        self,
        ctx: RunContext[AgentContext],
        tool_id: str,
    ) -> dict[str, object]:
        """Handle a ``decompress`` tool call from the model.

        ``tool_id_list`` is always populated by Phase 0.5 in
        ``before_model_request``, so no lazy rebuild is needed.
        """
        return decompress_tool(ctx, tool_id)

    # ---- Meta-tool auto-prune ----

    def _auto_prune_meta_tools(
        self,
        messages: list[ModelMessage],
        state: DCPState,
    ) -> list[ModelMessage]:
        """Auto-prune old meta-tool (prune/distill/decompress) returns.

        Keeps only the most recent ``config.meta_tool_retention`` meta-tool
        returns; older ones are replaced with ``"[pruned]"`` via
        ``_prune_part``.  This prevents prune/distill/decompress results
        from accumulating indefinitely in context.

        Args:
            messages: The conversation message list (not mutated).
            state: The DCP state (unused but kept for API consistency).

        Returns:
            A new list of messages with old meta-tool returns pruned.
        """
        retention = self._config.meta_tool_retention
        if retention < 0:
            return messages

        # Collect indices of all unpruned meta-tool ToolReturnParts.
        meta_parts: list[tuple[int, int, ToolReturnPart]] = []  # (msg_idx, part_idx, part)
        for msg_idx, msg in enumerate(messages):
            parts = getattr(msg, "parts", [])
            for part_idx, part in enumerate(parts):
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_name in META_TOOL_NAMES
                    and not _is_pruned(part)
                ):
                    meta_parts.append((msg_idx, part_idx, part))

        # If within retention, nothing to do.
        if len(meta_parts) <= retention:
            return messages

        # Determine which to prune: all except the most recent `retention`.
        to_prune = meta_parts[: len(meta_parts) - retention]
        prune_set = {(msg_idx, part_idx) for msg_idx, part_idx, _ in to_prune}

        result = list(messages)
        for msg_idx, part_idx in prune_set:
            msg = result[msg_idx]
            parts = getattr(msg, "parts", [])
            if part_idx >= len(parts):
                continue
            part = parts[part_idx]
            if not isinstance(part, ToolReturnPart):
                continue
            new_part = _prune_part(part, "[pruned]", "auto")
            new_parts = [*parts[:part_idx], new_part, *parts[part_idx + 1 :]]
            result[msg_idx] = dataclasses.replace(msg, parts=new_parts)  # type: ignore[arg-type]

        logger.info(
            "DynamicContextPruning auto-prune meta-tools: pruned %d/%d "
            "meta-tool returns (retention=%d)",
            len(to_prune),
            len(meta_parts),
            retention,
        )
        return result

    def _re_prune_messages(
        self,
        messages: list[ModelMessage],
        state: DCPState,
    ) -> list[ModelMessage]:
        """Re-apply previously-recorded pruning to restored messages.

        Each ``before_model_request`` receives a fresh copy from
        ``ctx.state.message_history`` (via ``_clean_message_history``)
        which has ORIGINAL content — our pruning from previous iterations
        is ephemeral. This method scans for ``ToolReturnPart`` instances
        whose ``tool_call_id`` is in ``state.pruned_tool_ids`` and
        replaces their content if they haven't been pruned yet in this
        iteration.

        For distill actions, the original distillation text from
        ``state.distill_texts`` is used. For prune actions, content is
        replaced with ``"[pruned]"``.

        Args:
            messages: The conversation message list (not mutated).
            state: The DCP state with ``pruned_tool_ids`` and
                ``distill_texts``.

        Returns:
            A new list of messages with previously-pruned parts re-pruned.
        """
        if not state.pruned_tool_ids:
            return messages

        result: list[ModelMessage] = []
        re_pruned = 0

        for msg in messages:
            if not isinstance(msg, (ModelRequest, ModelResponse)):
                result.append(msg)
                continue

            new_parts: list[Any] = []
            modified = False
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_call_id in state.pruned_tool_ids
                    and not _is_pruned(part)
                ):
                    # Re-prune this part
                    if part.tool_call_id in state.distill_texts:
                        distillation = state.distill_texts[part.tool_call_id]
                        new_parts.append(
                            _prune_part(part, distillation, "distill", summary=distillation),
                        )
                    else:
                        new_parts.append(_prune_part(part, "[pruned]", "prune"))
                    modified = True
                    re_pruned += 1
                else:
                    new_parts.append(part)

            if modified:
                result.append(dataclasses.replace(msg, parts=new_parts))  # type: ignore[arg-type]
            else:
                result.append(msg)

        if re_pruned > 0:
            logger.debug(
                "DynamicContextPruning re-prune: re-applied pruning to %d part(s) "
                "(total tracked=%d)",
                re_pruned,
                len(state.pruned_tool_ids),
            )

        return result

    # ---- Phase 1 implementation ----

    def _apply_pending_actions(
        self,
        messages: list[ModelMessage],
        state: DCPState,
        session_id: str = "default",
    ) -> list[ModelMessage]:
        """Apply all pending pruning actions from the deferred queue.

        Processes actions in FIFO order.  For each action:

        - ``prune``: Replace matching ``ToolReturnPart`` content with
          ``"[pruned]"`` via ``_prune_part``, located by ``action.ids``.
        - ``distill``: Replace matching ``ToolReturnPart`` content with
          the target's distillation text via ``_prune_part``, located by
          ``DistillTarget.tool_call_id``.

        Creates a ``CompressionBlock`` for each applied action and stores
        it in the block store using ``session_id`` for namespace isolation.

        Uses ``dataclasses.replace()`` for all message/part modifications
        — never mutates in-place.

        Args:
            messages: The conversation message list (not mutated).
            state: The DCP state with ``pending_actions``.
            session_id: The session ID for block store namespace isolation.

        Returns:
            A new list of messages with actions applied.
        """
        if not state.pending_actions:
            return messages

        result = list(messages)

        while state.pending_actions:
            action = state.pending_actions.popleft()
            result = self._apply_single_action(result, state, action, session_id)

        return result

    def _apply_single_action(
        self,
        messages: list[ModelMessage],
        state: DCPState,
        action: PruneAction,
        session_id: str = "default",
    ) -> list[ModelMessage]:
        """Apply a single pruning action to the message list.

        Args:
            messages: The conversation message list (not mutated).
            state: The DCP state for recording compression blocks.
            action: The action to apply.
            session_id: The session ID for block store namespace isolation.

        Returns:
            A new list of messages with the action applied.
        """
        if action.kind == "distill":
            return self._apply_distill(messages, state, action, session_id)
        return self._apply_id_action(messages, state, action, session_id)

    def _apply_id_action(
        self,
        messages: list[ModelMessage],
        state: DCPState,
        action: PruneAction,
        session_id: str = "default",
    ) -> list[ModelMessage]:
        """Apply a prune action by locating parts via ``action.ids``.

        For each ``tool_call_id`` in ``action.ids``, finds the matching
        ``ToolReturnPart`` and replaces its content with ``"[pruned]"``
        using ``_prune_part``.

        Args:
            messages: The conversation message list (not mutated).
            state: The DCP state for recording compression blocks.
            action: The prune action with ``ids``.
            session_id: The session ID for block store namespace isolation.

        Returns:
            A new list of messages with matching content replaced.
        """
        replacement = "[pruned]"
        target_ids = set(action.ids)
        matched_ids: list[str] = []

        result: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, (ModelRequest, ModelResponse)):
                result.append(msg)
                continue

            new_parts: list[Any] = []
            modified = False
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id in target_ids:
                    new_parts.append(_prune_part(part, replacement, "prune"))
                    modified = True
                    if part.tool_call_id:
                        matched_ids.append(part.tool_call_id)
                else:
                    new_parts.append(part)

            if modified:
                result.append(dataclasses.replace(msg, parts=new_parts))  # type: ignore[arg-type]
            else:
                result.append(msg)

        # Record compression block and mark as applied.
        if matched_ids:
            block = CompressionBlock(
                block_id=f"cb_{uuid4().hex[:12]}",
                original_tool_call_ids=tuple(matched_ids),
                compressed_content=replacement,
                kind="prune",
            )
            if state.block_store is not None:
                state.block_store.put(session_id, block)
            state.applied_action_ids.add(action.source_tool_call_id)
            # Record for re-pruning across iterations
            state.pruned_tool_ids.update(matched_ids)
            logger.debug(
                "DynamicContextPruning Phase 1 prune: matched %d tool output(s) ids=%s",
                len(matched_ids),
                action.ids,
            )
        else:
            logger.debug(
                "DynamicContextPruning Phase 1 prune: NO MATCH ids=%s "
                "(action recorded but no content changed)",
                action.ids,
            )

        return result

    def _apply_distill(
        self,
        messages: list[ModelMessage],
        state: DCPState,
        action: PruneAction,
        session_id: str = "default",
    ) -> list[ModelMessage]:
        """Apply a distill action using ``action.targets``.

        For each ``DistillTarget`` in ``action.targets``, finds the
        ``ToolReturnPart`` with matching ``tool_call_id`` and replaces
        its content with ``target.distillation`` using ``_prune_part``.

        Args:
            messages: The conversation message list (not mutated).
            state: The DCP state for recording compression blocks.
            action: The distill action with ``targets``.
            session_id: The session ID for block store namespace isolation.

        Returns:
            A new list of messages with matching content replaced.
        """
        # Build a mapping of tool_call_id -> distillation text.
        target_map: dict[str, str] = {t.tool_call_id: t.distillation for t in action.targets}
        matched_ids: list[str] = []

        result: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, (ModelRequest, ModelResponse)):
                result.append(msg)
                continue

            new_parts: list[Any] = []
            modified = False
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id in target_map:
                    distillation = target_map[part.tool_call_id]
                    new_parts.append(
                        _prune_part(part, distillation, "distill", summary=distillation),
                    )
                    modified = True
                    if part.tool_call_id:
                        matched_ids.append(part.tool_call_id)
                else:
                    new_parts.append(part)

            if modified:
                result.append(dataclasses.replace(msg, parts=new_parts))  # type: ignore[arg-type]
            else:
                result.append(msg)

        # Record compression block and mark as applied.
        if matched_ids:
            block = CompressionBlock(
                block_id=f"cb_{uuid4().hex[:12]}",
                original_tool_call_ids=tuple(matched_ids),
                compressed_content=action.summary or "[distilled]",
                kind="distill",
            )
            if state.block_store is not None:
                state.block_store.put(session_id, block)
            state.applied_action_ids.add(action.source_tool_call_id)
            # Record for re-pruning across iterations
            state.pruned_tool_ids.update(matched_ids)
            for tid in matched_ids:
                state.distill_texts[tid] = target_map.get(tid, "")
            logger.debug(
                "DynamicContextPruning Phase 1 distill: matched %d tool output(s) targets=%d",
                len(matched_ids),
                len(action.targets),
            )
        else:
            logger.debug(
                "DynamicContextPruning Phase 1 distill: NO MATCH targets=%d "
                "(action recorded but no content changed)",
                len(action.targets),
            )

        return result
