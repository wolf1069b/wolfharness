"""Tests for CodeModeCapability — wraps all agent tools into a single execute_code meta-tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
import pytest

from wolfharness.capabilities.code_mode_capability import CodeModeCapability
from wolfharness.tools.base import FunctionTool


pytestmark = pytest.mark.unit


# =============================================================================
# Test fixtures
# =============================================================================


async def _add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


async def _greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: Name to greet.
    """
    return f"Hello, {name}!"


def _make_tools() -> list[FunctionTool[Any]]:
    """Create test tools for the code mode capability."""
    return [
        FunctionTool.from_callable(_add),
        FunctionTool.from_callable(_greet),
    ]


def _make_ctx() -> Any:
    """Create a minimal RunContext-like mock for tool execution."""
    ctx = MagicMock()
    ctx.internal_fs = MagicMock()
    return ctx


# =============================================================================
# Tests — isinstance and basic structure
# =============================================================================


def test_is_abstract_capability() -> None:
    """CodeModeCapability is an instance of AbstractCapability."""
    cap = CodeModeCapability[Any](tools=[])
    assert isinstance(cap, AbstractCapability)


def test_get_toolset_returns_function_toolset() -> None:
    """get_toolset() returns a FunctionToolset."""
    cap = CodeModeCapability[Any](tools=_make_tools())
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)


def test_single_execute_code_tool_exposed() -> None:
    """Toolset contains exactly one tool named 'execute_code'."""
    cap = CodeModeCapability[Any](tools=_make_tools())
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert len(toolset.tools) == 1
    assert "_execute_code" in toolset.tools


def test_toolset_id_customizable() -> None:
    """Toolset ID can be customized via constructor."""
    cap = CodeModeCapability[Any](tools=[], toolset_id="custom_code")
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert toolset.id == "custom_code"


def test_default_toolset_id() -> None:
    """Default toolset ID is 'code_mode'."""
    cap = CodeModeCapability[Any](tools=[])
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert toolset.id == "code_mode"


# =============================================================================
# Tests — instructions
# =============================================================================


def test_get_instructions_returns_code_mode_prompt() -> None:
    """get_instructions() returns a non-None string with code mode guidance."""
    cap = CodeModeCapability[Any](tools=_make_tools())
    instructions = cap.get_instructions()
    assert instructions is not None
    assert "execute_code" in instructions
    assert "async def main" in instructions


def test_get_instructions_includes_usage_notes() -> None:
    """Instructions include the standard USAGE text from default_prompt."""
    cap = CodeModeCapability[Any](tools=[])
    instructions = cap.get_instructions()
    assert instructions is not None
    assert "main" in instructions


# =============================================================================
# Tests — on_change
# =============================================================================


def test_on_change_returns_none() -> None:
    """on_change() returns None — tools are static once wrapped."""
    cap = CodeModeCapability[Any](tools=_make_tools())
    assert cap.on_change() is None


# =============================================================================
# Tests — lifecycle
# =============================================================================


async def test_aenter_returns_self() -> None:
    """__aenter__ returns the capability itself (no-op lifecycle)."""
    cap = CodeModeCapability[Any](tools=[])
    result = await cap.__aenter__()
    assert result is cap


async def test_aexit_is_noop() -> None:
    """__aexit__ is a no-op that returns None."""
    cap = CodeModeCapability[Any](tools=[])
    result = await cap.__aexit__(None, None, None)
    assert result is None


# =============================================================================
# Tests — code execution with inner tools
# =============================================================================


async def test_inner_tools_callable_via_meta_tool() -> None:
    """execute_code can call wrapped tools from within the generated namespace."""
    tools = _make_tools()
    cap = CodeModeCapability[Any](tools=tools)
    ctx = _make_ctx()

    code = """
async def main():
    result = await _add(a=3, b=5)
    return result
"""
    result = await cap._execute_code(ctx, code, "test_add")

    assert result == 8


async def test_multiple_inner_tools_callable() -> None:
    """Multiple wrapped tools can be called from the same code block."""
    tools = _make_tools()
    cap = CodeModeCapability[Any](tools=tools)
    ctx = _make_ctx()

    code = """
async def main():
    sum_result = await _add(a=10, b=20)
    greeting = await _greet(name="World")
    return f"{greeting} Sum is {sum_result}"
"""
    result = await cap._execute_code(ctx, code, "multi_tool")

    assert "Hello, World" in result
    assert "30" in result


async def test_code_execution_returns_success_message_on_falsy() -> None:
    """When main() returns a falsy value, a success message is returned."""
    tools = _make_tools()
    cap = CodeModeCapability[Any](tools=tools)
    ctx = _make_ctx()

    code = """
async def main():
    await _add(a=1, b=2)
    return None
"""
    result = await cap._execute_code(ctx, code, "falsy_return")

    assert result == "Code executed successfully"


async def test_code_execution_error_captured() -> None:
    """Errors during code execution are captured and returned as string."""
    tools = _make_tools()
    cap = CodeModeCapability[Any](tools=tools)
    ctx = _make_ctx()

    code = """
async def main():
    raise ValueError("boom")
    return "unreachable"
"""
    result = await cap._execute_code(ctx, code, "error_test")

    assert isinstance(result, str)
    assert "Error executing code" in result
    assert "boom" in result


# =============================================================================
# Tests — empty tools list
# =============================================================================


def test_empty_tools_toolset_still_has_execute_code() -> None:
    """Empty tools list still produces a toolset with the execute_code tool."""
    cap = CodeModeCapability[Any](tools=[])
    toolset = cap.get_toolset()
    assert isinstance(toolset, FunctionToolset)
    assert len(toolset.tools) == 1
    assert "_execute_code" in toolset.tools


def test_empty_tools_instructions_still_returned() -> None:
    """Empty tools list still returns instructions."""
    cap = CodeModeCapability[Any](tools=[])
    instructions = cap.get_instructions()
    assert instructions is not None


async def test_empty_tools_code_execution_without_tool_calls() -> None:
    """Code can be executed even with no wrapped tools — just pure Python."""
    cap = CodeModeCapability[Any](tools=[])
    ctx = _make_ctx()

    code = """
async def main():
    x = 2 + 3
    return f"Result: {x}"
"""
    result = await cap._execute_code(ctx, code, "no_tools")

    assert result == "Result: 5"


# =============================================================================
# Tests — invalid code validation
# =============================================================================


async def test_invalid_syntax_raises_model_retry() -> None:
    """Invalid Python syntax raises ModelRetry from validate_code."""
    from pydantic_ai import ModelRetry

    cap = CodeModeCapability[Any](tools=_make_tools())
    ctx = _make_ctx()

    with pytest.raises(ModelRetry, match="Invalid code syntax"):
        await cap._execute_code(ctx, "def broken(:", "syntax_error")


async def test_missing_main_function_raises_model_retry() -> None:
    """Code without 'async def main(' raises ModelRetry."""
    from pydantic_ai import ModelRetry

    cap = CodeModeCapability[Any](tools=_make_tools())
    ctx = _make_ctx()

    with pytest.raises(ModelRetry, match="async def main"):
        await cap._execute_code(ctx, "x = 1", "no_main")


async def test_missing_return_raises_model_retry() -> None:
    """Code with main() but no return statement raises ModelRetry."""
    from pydantic_ai import ModelRetry

    cap = CodeModeCapability[Any](tools=_make_tools())
    ctx = _make_ctx()

    code = """
async def main():
    x = 1
"""
    with pytest.raises(ModelRetry, match="return"):
        await cap._execute_code(ctx, code, "no_return")


# =============================================================================
# Tests — toolset caching
# =============================================================================


def test_toolset_cached_on_repeated_calls() -> None:
    """get_toolset() returns the same FunctionToolset on repeated calls."""
    cap = CodeModeCapability[Any](tools=_make_tools())
    toolset1 = cap.get_toolset()
    toolset2 = cap.get_toolset()
    assert toolset1 is toolset2
