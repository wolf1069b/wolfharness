"""Tests for deferred execution fields on Tool."""

from datetime import timedelta

import pytest

from wolfharness.tools.base import FunctionTool, Tool


pytestmark = pytest.mark.unit


def test_tool_default_deferred_fields():
    """Tool() defaults to deferred=False with correct default kind/strategy."""
    t = Tool(name="test")
    assert t.deferred is False
    assert t.deferred_kind == "external"
    assert t.deferred_strategy == "block"
    assert t.deferred_placeholder == "This tool is processing in the background."
    assert t.deferred_timeout is None


def test_tool_deferred_true_external():
    """Tool(deferred=True, deferred_kind='external') instantiates without error."""
    t = Tool(name="test", deferred=True, deferred_kind="external")
    assert t.deferred is True
    assert t.deferred_kind == "external"


def test_tool_deferred_true_unapproved():
    """Tool(deferred=True, deferred_kind='unapproved') stores the value."""
    t = Tool(name="test", deferred=True, deferred_kind="unapproved")
    assert t.deferred is True
    assert t.deferred_kind == "unapproved"


def test_tool_deferred_strategy_continue():
    """deferred_strategy='continue' is accepted."""
    t = Tool(name="test", deferred=True, deferred_strategy="continue")
    assert t.deferred_strategy == "continue"


def test_tool_deferred_strategy_stream():
    """deferred_strategy='stream' raises NotImplementedError (deferred to follow-up)."""
    with pytest.raises(NotImplementedError, match="stream"):
        Tool(name="test", deferred=True, deferred_strategy="stream")


def test_tool_deferred_timeout():
    """Tool(deferred_timeout=timedelta(minutes=30)) stores timeout correctly."""
    t = Tool(name="test", deferred_timeout=timedelta(minutes=30))
    assert t.deferred_timeout == timedelta(minutes=30)


def test_tool_deferred_placeholder_custom():
    """deferred_placeholder can be overridden."""
    t = Tool(name="test", deferred_placeholder="Please wait...")
    assert t.deferred_placeholder == "Please wait..."


# ── to_pydantic_ai() deferred mapping tests ──


async def _simple_tool_func(x: int) -> str:
    """A simple test tool function."""
    return str(x)


def _make_tool(**kwargs: object) -> FunctionTool[str]:
    """Create a FunctionTool for testing to_pydantic_ai()."""
    return FunctionTool.from_callable(_simple_tool_func, **kwargs)  # type: ignore[arg-type]


def test_to_pydantic_ai_not_deferred():
    """Tool(deferred=False).to_pydantic_ai() produces standard tool (kind='function')."""
    t = _make_tool()
    pydantic_tool = t.to_pydantic_ai()
    assert pydantic_tool.requires_approval is False
    assert pydantic_tool.tool_def.kind == "function"


def test_to_pydantic_ai_deferred_unapproved():
    """Deferred unapproved tool maps to requires_approval=True.

    The deferred_kind 'unapproved' should set requires_approval=True on the
    pydantic-ai tool.
    """
    t = _make_tool(deferred=True, deferred_kind="unapproved")
    pydantic_tool = t.to_pydantic_ai()
    assert pydantic_tool.requires_approval is True
    assert pydantic_tool.tool_def.kind == "unapproved"


def test_to_pydantic_ai_deferred_external_kind():
    """Deferred external tool produces ToolDefinition with kind='external'.

    The kind override is applied via a prepare function during prepare_tool_def().
    """
    t = _make_tool(deferred=True, deferred_kind="external")
    pydantic_tool = t.to_pydantic_ai()
    # The raw tool_def property will still show 'function' because the kind
    # override is applied via a prepare function during prepare_tool_def().
    # Verify that a prepare function is set to handle the external kind mapping.
    assert pydantic_tool.prepare is not None, (
        "A prepare function must be set to map deferred_kind='external' "
        "to ToolDefinition.kind='external'"
    )


@pytest.mark.asyncio
async def test_to_pydantic_ai_deferred_external_prepare_sets_kind():
    """The prepare function on an external deferred tool sets kind='external'.

    The prepare function on the ToolDefinition should override the kind.
    """
    from pydantic_ai.tools import ToolDefinition

    t = _make_tool(deferred=True, deferred_kind="external")
    pydantic_tool = t.to_pydantic_ai()
    assert pydantic_tool.prepare is not None

    # Create a minimal ToolDefinition and call the prepare function with a mock context.
    # The prepare function should override the kind to 'external'.
    base_tool_def = pydantic_tool.tool_def
    # Simulate passing a dummy context (prepare functions receive context + tool_def)
    # We use None for ctx since the prepare only needs to modify the tool_def.
    result = pydantic_tool.prepare(None, base_tool_def)  # type: ignore[arg-type]
    import inspect

    if inspect.isawaitable(result):
        result = await result  # type: ignore[assignment]

    assert isinstance(result, ToolDefinition)
    assert result.kind == "external", f"Expected kind='external', got kind='{result.kind}'"


def test_to_pydantic_ai_deferred_does_not_affect_requires_confirmation_flag():
    """Tool(requires_confirmation=True, deferred=False) preserves requires_confirmation.

    The pydantic-ai tool should keep requires_approval=True.
    """
    t = _make_tool(requires_confirmation=True, deferred=False)
    pydantic_tool = t.to_pydantic_ai()
    assert pydantic_tool.requires_approval is True
    assert pydantic_tool.tool_def.kind == "unapproved"


def test_to_pydantic_ai_deferred_unapproved_overrides_requires_confirmation():
    """Deferred unapproved overrides requires_confirmation=False.

    Tool with requires_confirmation=False, deferred=True, deferred_kind='unapproved'
    still sets requires_approval.
    """
    t = _make_tool(requires_confirmation=False, deferred=True, deferred_kind="unapproved")
    pydantic_tool = t.to_pydantic_ai()
    assert pydantic_tool.requires_approval is True
    assert pydantic_tool.tool_def.kind == "unapproved"
