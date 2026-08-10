"""Base hook classes and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from exxec import ExecutionEnvironment

logger = get_logger(__name__)

HookEvent = Literal["pre_turn", "post_turn", "pre_tool_use", "post_tool_use"]


class HookInput(TypedDict, total=False):
    """Input data passed to hooks."""

    # Common fields
    event: HookEvent
    agent_name: str
    session_id: str | None

    # Tool-related fields (pre_tool_use, post_tool_use)
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any
    duration_ms: float

    # Run-related fields (pre_turn, post_turn)
    prompt: str
    result: Any

    # Agent context (available in tool-use hooks when running inside an agent)
    agent_context: Any


class HookResult(TypedDict, total=False):
    """Result returned from hook execution."""

    decision: Literal["allow", "deny", "ask"]
    """Decision for pre_* hooks: allow, deny, or ask user."""

    reason: str
    """Explanation for the decision."""

    modified_input: dict[str, Any]
    """Modified input for pre_* hooks (e.g., modified tool_input)."""

    additional_context: str
    """Context to inject into conversation."""

    modified_output: Any
    """Replacement for tool output in post_tool_use hooks.

    This is optional. When omitted, the original tool output is preserved.
    When set, the tool's return value is replaced entirely rather than merged
    or appended.
    Takes precedence over ``additional_context``.
    """

    continue_: bool
    """Whether to continue execution. False = stop."""


class Hook(ABC):
    """Base class for runtime hooks."""

    def __init__(
        self,
        event: HookEvent,
        matcher: str | None = None,
        timeout: float = 60.0,
        enabled: bool = True,
        input_match: dict[str, str] | None = None,
    ) -> None:
        """Initialize hook.

        Args:
            event: The lifecycle event this hook handles.
            matcher: Regex pattern for matching (e.g., tool names). None matches all.
            timeout: Maximum execution time in seconds.
            enabled: Whether this hook is active.
            input_match: Optional regex patterns to match against ``tool_input``
                fields.  Every pattern must match for the hook to trigger.
        """
        self.event = event
        self.matcher = matcher
        self.timeout = timeout
        self.enabled = enabled
        self._pattern = re.compile(matcher) if matcher and matcher != "*" else None
        self._input_matchers: dict[str, re.Pattern[str]] | None = (
            {k: re.compile(v) for k, v in input_match.items()} if input_match else None
        )

    def matches(self, input_data: HookInput) -> bool:
        """Check if this hook should run for the given input.

        Args:
            input_data: The hook input data.

        Returns:
            True if the hook should execute.
        """
        if not self.enabled:
            return False

        # For tool events, match against tool_name
        if self._pattern is not None and self.event in ("pre_tool_use", "post_tool_use"):
            tool_name = input_data.get("tool_name", "")
            if not self._pattern.search(tool_name):
                return False

        # Match against tool_input fields (all patterns must match)
        if self._input_matchers is not None:
            tool_input = input_data.get("tool_input") or {}
            for key, pattern in self._input_matchers.items():
                value = str(tool_input.get(key, ""))
                if not pattern.search(value):
                    return False

        return True

    @abstractmethod
    async def execute(
        self,
        input_data: HookInput,
        env: ExecutionEnvironment | None = None,
    ) -> HookResult:
        """Execute the hook.

        Args:
            input_data: The hook input data.
            env: Optional execution environment from the agent. Command hooks
                use this to run in the same environment as the agent's tools.

        Returns:
            Hook result with decision and optional modifications.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"event={self.event!r}, matcher={self.matcher!r}, enabled={self.enabled})"
        )
