"""Tests for QuestionTool error handling.

Regression test for issue #87: ACP tool_call status shows "completed" when
question_for_user fails with ErrorData.

Root cause: ``QuestionTool._execute`` returns a ``ToolResult`` with error
content when ``handle_elicitation`` yields ``ErrorData``, instead of raising
``ModelRetry``.  This causes pydantic-ai to wrap the result as a
``ToolReturnPart`` (success) rather than a ``RetryPromptPart`` (failure),
so the ACP event converter reports ``completion_status = "completed"``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from mcp.types import ElicitResult, ErrorData
from pydantic_ai import ModelRetry
import pytest

from wolfharness.tasks.exceptions import RunAbortedError
from wolfharness.tool_impls.question.tool import QuestionTool
from wolfharness.tools.base import ToolResult


@pytest.mark.unit
async def test_question_tool_raises_model_retry_on_error_data() -> None:
    """Raise ModelRetry when elicitation returns ErrorData.

    Given: handle_elicitation returns ErrorData.
    When: QuestionTool._execute is called.
    Then: ModelRetry is raised so pydantic-ai produces RetryPromptPart (failure).
    """
    # Given
    tool = QuestionTool(name="question")
    ctx = AsyncMock()
    ctx.handle_elicitation = AsyncMock(
        return_value=ErrorData(code=400, message="XML parse error: invalid token"),
    )

    # When / Then
    with pytest.raises(ModelRetry, match="Elicitation failed"):
        await tool._execute(ctx=ctx, prompt="What temperature?")


@pytest.mark.unit
async def test_question_tool_error_data_does_not_return_tool_result() -> None:
    """Never return ToolResult for ErrorData.

    Given: handle_elicitation returns ErrorData.
    When: QuestionTool._execute is called.
    Then: The result must NOT be a ToolResult (which would be treated as success).
    """
    # Given
    tool = QuestionTool(name="question")
    ctx = AsyncMock()
    ctx.handle_elicitation = AsyncMock(
        return_value=ErrorData(code=400, message="Schema validation failed"),
    )

    # When
    try:
        result = await tool._execute(ctx=ctx, prompt="Pick an option")
    except ModelRetry:
        # Expected — this is the correct behavior
        return

    # Then: if we reach here, the tool returned a result instead of raising
    pytest.fail(
        f"Expected ModelRetry but got {type(result).__name__}: {result}. "
        "ErrorData must raise ModelRetry, not return a ToolResult. "
        "Returning ToolResult causes ACP to report 'completed' status for failed tool calls."
    )


@pytest.mark.unit
async def test_question_tool_cancel_still_raises_run_aborted() -> None:
    """Raise RunAbortedError on cancel action.

    Given: handle_elicitation returns ElicitResult(action='cancel').
    When: QuestionTool._execute is called.
    Then: RunAbortedError is raised (user cancellation, not tool failure).
    """
    # Given
    tool = QuestionTool(name="question")
    ctx = AsyncMock()
    ctx.handle_elicitation = AsyncMock(
        return_value=ElicitResult(action="cancel"),
    )

    # When / Then
    with pytest.raises(RunAbortedError, match="User cancelled"):
        await tool._execute(ctx=ctx, prompt="Question?")


@pytest.mark.unit
async def test_question_tool_accept_returns_tool_result() -> None:
    """Return ToolResult on accept action.

    Given: handle_elicitation returns ElicitResult(action='accept').
    When: QuestionTool._execute is called.
    Then: A ToolResult is returned with the answer (happy path, regression guard).
    """
    # Given
    tool = QuestionTool(name="question")
    ctx = AsyncMock()
    ctx.handle_elicitation = AsyncMock(
        return_value=ElicitResult(
            action="accept",
            content={"value": "42°C"},
        ),
    )

    # When
    result = await tool._execute(ctx=ctx, prompt="What temperature?")

    # Then
    assert isinstance(result, ToolResult)
    assert "42°C" in result.content
