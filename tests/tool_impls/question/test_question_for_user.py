"""Unit tests for the question tool."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from mcp.types import ErrorData
from pydantic_ai import ModelRetry
import pytest

from wolfharness.tasks.exceptions import RunAbortedError
from wolfharness_toolsets.builtin.question_tools import (
    Question,
    QuestionTools,
    Suggest,
    _build_acp_schema,
    _format_question_response,
    parse_questionnaire,
)


# =============================================================================
# XML Parsing Tests
# =============================================================================


@pytest.mark.unit
def test_parse_enum_question_with_questions_wrapper() -> None:
    """Test parsing enum question with explicit <questions> wrapper."""
    xml = (
        '<questions><question header="Model" type="enum">'
        "<text>What is the equipment model?</text>"
        "<suggest>SY215C</suggest><suggest>SY235C</suggest>"
        "</question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "Model"
    assert result[0].type == "enum"
    assert result[0].text == "What is the equipment model?"
    assert len(result[0].options) == 2
    assert result[0].options[0].label == "SY215C"
    assert result[0].options[1].label == "SY235C"


@pytest.mark.unit
def test_parse_enum_question_backward_compatible() -> None:
    """Test parsing bare enum question (backward compatibility - auto-wraps)."""
    xml = (
        '<question header="Model" type="enum">'
        "<text>What is the equipment model?</text>"
        "<suggest>SY215C</suggest><suggest>SY235C</suggest></question>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "Model"
    assert result[0].type == "enum"
    assert result[0].text == "What is the equipment model?"
    assert len(result[0].options) == 2
    assert result[0].options[0].label == "SY215C"
    assert result[0].options[1].label == "SY235C"


@pytest.mark.unit
def test_parse_multi_question_with_wrapper() -> None:
    """Test parsing multi-select with <questions> wrapper."""
    xml = (
        '<questions><question header="Symptoms" type="multi">'
        "<text>Select the observed symptoms</text>"
        "<suggest>Black smoke</suggest><suggest>Low power</suggest>"
        "<suggest>Abnormal noise</suggest></question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "Symptoms"
    assert result[0].type == "multi"
    assert result[0].text == "Select the observed symptoms"
    assert len(result[0].options) == 3
    assert result[0].options[0].label == "Black smoke"
    assert result[0].options[2].label == "Abnormal noise"


@pytest.mark.unit
def test_parse_input_question_with_wrapper() -> None:
    """Test parsing input type question with <questions> wrapper."""
    xml = (
        '<questions><question header="Notes" type="input">'
        "<text>Enter additional notes</text></question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "Notes"
    assert result[0].type == "input"
    assert result[0].text == "Enter additional notes"
    assert len(result[0].options) == 0


@pytest.mark.unit
def test_parse_multiple_questions_with_wrapper() -> None:
    """Test parsing multiple questions with explicit <questions> wrapper."""
    xml = (
        "<questions>"
        '<question header="First" type="enum"><text>Question 1</text>'
        "<suggest>Option A</suggest></question>"
        '<question header="Second" type="input"><text>Question 2</text></question>'
        "</questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 2
    assert result[0].header == "First"
    assert result[0].type == "enum"
    assert result[1].header == "Second"
    assert result[1].type == "input"


@pytest.mark.unit
def test_parse_question_with_suggest_attributes_and_wrapper() -> None:
    """Test parsing questions with suggest attributes using <questions> wrapper."""
    xml = (
        '<questions><question header="Test" type="enum"><text>Select option</text>'
        '<suggest type="input" description="Custom option" next_action="next">Custom</suggest>'
        "</question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert len(result[0].options) == 1
    option = result[0].options[0]
    assert option.label == "Custom"
    assert option.type == "input"
    assert option.description == "Custom option"
    assert option.next_action == "next"


@pytest.mark.unit
def test_parse_question_optional_with_wrapper() -> None:
    """Test parsing optional questions with <questions> wrapper."""
    xml = (
        '<questions><question header="Optional" type="input" required="false">'
        "<text>Optional question</text></question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "Optional"
    assert not result[0].required


@pytest.mark.unit
def test_parse_enum_with_text_instead_of_suggest() -> None:
    """Test parsing when LLM incorrectly uses <text> instead of <suggest> for options.

    With extra_texts removed, additional <text> elements beyond the first
    are ignored by the parser. Options must use <suggest> tags.
    """
    xml = (
        '<questions><question header="诊断选择" type="enum">'
        "<text>挖掘机无法启动，我可以为您提供系统化的故障诊断流程</text>"
        "<text>1. 收集故障信息</text>"
        "<text>2. 分析可能的故障原因</text>"
        "<text>3. 指导您逐步排查检查</text>"
        "</question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "诊断选择"
    assert result[0].type == "enum"
    assert result[0].text == "挖掘机无法启动，我可以为您提供系统化的故障诊断流程"
    # No <suggest> tags means empty options
    assert len(result[0].options) == 0

    # Schema generation: no options means fallback to free text input
    schema = _build_acp_schema(result)
    prop = schema["properties"]["q0"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["1. 收集故障信息", "2. 分析可能的故障原因", "3. 指导您逐步排查检查"]
    assert "oneOf" not in prop
    assert "minLength" not in prop  # Not an input fallback


@pytest.mark.unit
def test_parse_enum_with_multiple_text_and_suggest() -> None:
    """Test parsing when multiple <text> elements precede <suggest> elements.

    This reproduces the actual user issue: LLM generates explanatory bullets as
    <text> tags, then real options as <suggest> tags. Without extra_texts,
    pydantic_xml fails to parse the <suggest> elements.
    """
    xml = (
        '<questions><question header="诊断选择" type="enum" required="true">'
        "<text>挖掘机无法启动，我可以为您提供系统化的故障诊断流程，包括：</text>"
        "<text>1. 收集详细的故障信息</text>"
        "<text>2. 分析可能的故障原因</text>"
        "<text>3. 指导您逐步排查检查</text>"
        "<text>4. 定位根本原因并给出解决方案</text>"
        '<suggest description="立即开始系统化的故障诊断流程">开始诊断</suggest>'
        '<suggest description="了解诊断流程后再决定">先了解流程</suggest>'
        '<suggest description="自己尝试简单解决">自行处理</suggest>'
        "</question></questions>"
    )
    result = parse_questionnaire(xml)

    assert len(result) == 1
    assert result[0].header == "诊断选择"
    assert result[0].type == "enum"
    # Primary text is the first <text>
    assert result[0].text == "挖掘机无法启动，我可以为您提供系统化的故障诊断流程，包括："
    assert result[0].options[2].label == "自行处理"

    # Schema should use <suggest> options (not extra_texts fallback)
    schema = _build_acp_schema(result)
    prop = schema["properties"]["q0"]
    assert prop["type"] == "string"
    assert "oneOf" in prop  # Descriptions present → oneOf
    one_of = prop["oneOf"]
    assert len(one_of) == 3
    # XML suggest description maps to schema title (human-readable display)
    assert one_of[0] == {
        "const": "开始诊断",
        "title": "立即开始系统化的故障诊断流程",
    }
    assert one_of[1] == {
        "const": "先了解流程",
        "title": "了解诊断流程后再决定",
    }
    assert one_of[2] == {
        "const": "自行处理",
        "title": "自己尝试简单解决",
    }


# =============================================================================
# Schema Generation Tests
# =============================================================================


@pytest.mark.unit
def test_build_enum_schema() -> None:
    """Test JSON schema generation for enum type question (no descriptions → simple enum)."""
    questions = [
        Question(
            header="Model",
            type="enum",
            text="What model?",
            required=True,
            options=[Suggest(label="SY215C"), Suggest(label="SY235C")],
        ),
    ]
    schema = _build_acp_schema(questions)

    assert schema["type"] == "object"
    assert "properties" in schema
    assert schema["required"] == ["q0"]
    assert "q0" in schema["properties"]
    assert schema["properties"]["q0"]["type"] == "string"
    assert schema["properties"]["q0"]["title"] == "Model"
    assert schema["properties"]["q0"]["description"] == "What model?"
    # When no descriptions, use simple enum for maximum ACP compatibility
    assert "enum" in schema["properties"]["q0"]
    assert schema["properties"]["q0"]["enum"] == ["SY215C", "SY235C"]
    assert "oneOf" not in schema["properties"]["q0"]


@pytest.mark.unit
def test_build_enum_schema_with_descriptions() -> None:
    """Test JSON schema generation for enum with option descriptions (→ oneOf)."""
    questions = [
        Question(
            header="Model",
            type="enum",
            text="What model?",
            required=True,
            options=[
                Suggest(label="SY215C", description="21.5 ton excavator"),
                Suggest(label="SY365H", description="36.5 ton excavator"),
            ],
        ),
    ]
    schema = _build_acp_schema(questions)

    prop = schema["properties"]["q0"]
    assert prop["type"] == "string"
    assert "oneOf" in prop
    assert "enum" not in prop
    one_of = prop["oneOf"]
    assert len(one_of) == 2
    # XML suggest description maps to schema title (human-readable display)
    assert one_of[0] == {
        "const": "SY215C",
        "title": "21.5 ton excavator",
    }
    assert one_of[1] == {
        "const": "SY365H",
        "title": "36.5 ton excavator",
    }


@pytest.mark.unit
def test_build_enum_schema_mixed_descriptions() -> None:
    """Test enum where some options have descriptions and some don't."""
    questions = [
        Question(
            header="Model",
            type="enum",
            text="What model?",
            required=True,
            options=[
                Suggest(label="SY215C", description="21.5 ton"),
                Suggest(label="SY365H"),  # No description
            ],
        ),
    ]
    schema = _build_acp_schema(questions)

    prop = schema["properties"]["q0"]
    assert "oneOf" in prop
    one_of = prop["oneOf"]
    assert len(one_of) == 2
    # Entry with description: title comes from description
    assert one_of[0] == {
        "const": "SY215C",
        "title": "21.5 ton",
    }
    # Entry without description: title falls back to label
    assert one_of[1] == {"const": "SY365H", "title": "SY365H"}
    assert "description" not in one_of[1]


@pytest.mark.unit
def test_build_enum_schema_empty_options_fallback() -> None:
    """Test enum with no options falls back to input (e.g., LLM used <text> not <suggest>)."""
    questions = [
        Question(
            header="诊断选择",
            type="enum",
            text="挖掘机无法启动，我可以为您提供系统化的故障诊断流程",
            required=True,
            options=[],  # Empty — LLM generated <text> instead of <suggest>
        ),
    ]
    schema = _build_acp_schema(questions)

    prop = schema["properties"]["q0"]
    # Should fall back to string input instead of empty enum/oneOf
    assert prop["type"] == "string"
    assert prop["title"] == "诊断选择"
    assert prop["description"] == "挖掘机无法启动，我可以为您提供系统化的故障诊断流程"
    assert prop["minLength"] == 1
    assert "enum" not in prop
    assert "oneOf" not in prop


@pytest.mark.unit
def test_build_enum_schema_extra_texts_fallback() -> None:
    """Test enum with empty options falls back to free text input."""
    questions = [
        Question(
            header="诊断选择",
            type="enum",
            text="挖掘机无法启动，我可以为您提供系统化的故障诊断流程",
            required=True,
            options=[],  # Empty <suggest>
        ),
    ]
    schema = _build_acp_schema(questions)

    prop = schema["properties"]["q0"]
    # Empty options → fall back to free text input
    assert prop["type"] == "string"
    assert "enum" not in prop
    assert "oneOf" not in prop
    assert "minLength" in prop


@pytest.mark.unit
def test_build_multi_schema_extra_texts_fallback() -> None:
    """Test multi-select with empty options falls back to free text input."""
    questions = [
        Question(
            header="Symptoms",
            type="multi",
            text="Select symptoms",
            required=True,
            options=[],  # Empty <suggest>
        ),
    ]
    schema = _build_acp_schema(questions)

    prop = schema["properties"]["q0"]
    # Empty options → fall back to free text input
    assert prop["type"] == "string"
    assert "minLength" in prop


@pytest.mark.unit
def test_build_multi_schema_empty_options_fallback() -> None:
    """Test multi-select with no options falls back to input."""
    questions = [
        Question(
            header="Symptoms",
            type="multi",
            text="Select symptoms",
            required=True,
            options=[],  # Empty
        ),
    ]
    schema = _build_acp_schema(questions)

    prop = schema["properties"]["q0"]
    # Should fall back to string input
    assert prop["type"] == "string"
    assert prop["title"] == "Symptoms"
    assert prop["minLength"] == 1
    assert "enum" not in prop
    assert "items" not in prop


@pytest.mark.unit
def test_build_multi_schema() -> None:
    """Test JSON schema generation for multi-select type question."""
    questions = [
        Question(
            header="Symptoms",
            type="multi",
            text="Select symptoms",
            required=True,
            options=[Suggest(label="A"), Suggest(label="B"), Suggest(label="C")],
        ),
    ]
    schema = _build_acp_schema(questions)

    assert schema["properties"]["q0"]["type"] == "array"
    assert schema["properties"]["q0"]["title"] == "Symptoms"
    assert schema["properties"]["q0"]["description"] == "Select symptoms"
    assert "items" in schema["properties"]["q0"]
    assert schema["properties"]["q0"]["items"]["type"] == "string"
    assert schema["properties"]["q0"]["items"]["enum"] == ["A", "B", "C"]
    assert schema["properties"]["q0"]["uniqueItems"]


@pytest.mark.unit
def test_build_input_schema() -> None:
    """Test JSON schema generation for input type question."""
    questions = [
        Question(
            header="Notes",
            type="input",
            text="Enter notes",
            required=True,
            options=[],
        ),
    ]
    schema = _build_acp_schema(questions)

    assert schema["properties"]["q0"]["type"] == "string"
    assert schema["properties"]["q0"]["title"] == "Notes"
    assert schema["properties"]["q0"]["description"] == "Enter notes"
    assert schema["properties"]["q0"]["minLength"] == 1


@pytest.mark.unit
def test_build_acp_schema_multiple_questions() -> None:
    """Test JSON schema generation with multiple questions of different types."""
    questions = [
        Question(
            header="Enum Q",
            type="enum",
            text="Enum question",
            required=True,
            options=[Suggest(label="Option1")],
        ),
        Question(
            header="Multi Q",
            type="multi",
            text="Multi question",
            required=True,
            options=[Suggest(label="Opt1"), Suggest(label="Opt2")],
        ),
        Question(
            header="Input Q",
            type="input",
            text="Input question",
            required=False,
            options=[],
        ),
    ]
    schema = _build_acp_schema(questions)

    assert "q0" in schema["properties"]
    assert "q1" in schema["properties"]
    assert "q2" in schema["properties"]
    # Only q0 and q1 should be required
    assert "required" in schema
    assert sorted(schema["required"]) == ["q0", "q1"]


@pytest.mark.unit
def test_build_acp_schema_not_required() -> None:
    """Test schema without required fields."""
    questions = [
        Question(
            header="Optional",
            type="input",
            text="Optional question",
            required=False,
            options=[],
        ),
    ]
    schema = _build_acp_schema(questions)

    # No required key if all questions are optional
    assert "required" not in schema or len(schema.get("required", [])) == 0


# =============================================================================
# Response Formatting Tests
# =============================================================================


@pytest.mark.unit
def test_format_response_accept_enum() -> None:
    """Test formatting accept response for enum question."""
    questions = [
        Question(
            header="Model",
            type="enum",
            text="What model?",
            required=True,
            options=[Suggest(label="SY215C")],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": "SY215C"}

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert content == "Model: SY215C"
    assert metadata["answers"] == [["SY215C"]]


@pytest.mark.unit
def test_format_response_accept_multi() -> None:
    """Test formatting accept response for multi-select question."""
    questions = [
        Question(
            header="Symptoms",
            type="multi",
            text="Select symptoms",
            required=True,
            options=[Suggest(label="A"), Suggest(label="B")],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": ["A", "B"]}

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert content == "Symptoms: A, B"
    assert metadata["answers"] == [["A", "B"]]


@pytest.mark.unit
def test_format_response_accept_input() -> None:
    """Test formatting accept response for input question."""
    questions = [
        Question(
            header="Notes",
            type="input",
            text="Enter notes",
            required=True,
            options=[],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": "My notes"}

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert content == "Notes: My notes"
    assert metadata["answers"] == [["My notes"]]


@pytest.mark.unit
def test_format_response_accept_multiple_questions() -> None:
    """Test formatting accept response for multiple questions."""
    questions = [
        Question(
            header="Q1",
            type="enum",
            text="Question 1",
            required=True,
            options=[Suggest(label="A")],
        ),
        Question(
            header="Q2",
            type="input",
            text="Question 2",
            required=True,
            options=[],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": "A", "q1": "Response"}

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert content == "Q1: A\nQ2: Response"
    assert metadata["answers"] == [["A"], ["Response"]]


@pytest.mark.unit
def test_format_response_cancel() -> None:
    """Test formatting cancel action response raises RunAbortedError."""
    questions = [
        Question(
            header="Test",
            type="enum",
            text="Test question",
            required=True,
            options=[Suggest(label="A")],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "cancel"

    with pytest.raises(RunAbortedError, match="cancelled"):
        _format_question_response(questions, mock_result)


@pytest.mark.unit
def test_format_response_decline() -> None:
    """Test formatting decline action response returns ToolResult."""
    questions = [
        Question(
            header="Test",
            type="enum",
            text="Test question",
            required=True,
            options=[Suggest(label="A")],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "decline"

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert "declined" in content.lower()
    assert metadata["answers"] == []


@pytest.mark.unit
def test_format_response_error_data() -> None:
    """Test _format_question_response raises ModelRetry on ErrorData."""
    questions = [
        Question(
            header="Test",
            type="enum",
            text="Test question",
            required=True,
            options=[Suggest(label="A")],
        ),
    ]
    error_result = ErrorData(code=500, message="Server error occurred")

    with pytest.raises(ModelRetry, match="Server error occurred"):
        _format_question_response(questions, error_result)


@pytest.mark.unit
def test_format_response_unknown_action_raises() -> None:
    """Test that unknown action raises RuntimeError."""
    questions = [
        Question(
            header="Test",
            type="enum",
            text="Test question",
            required=True,
            options=[Suggest(label="A")],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "unknown_action"

    with pytest.raises(RuntimeError) as exc_info:
        _format_question_response(questions, mock_result)

    assert "Unknown action: unknown_action" in str(exc_info.value)


@pytest.mark.unit
def test_format_response_multi_empty() -> None:
    """Test formatting empty multi-select response."""
    questions = [
        Question(
            header="Symptoms",
            type="multi",
            text="Select symptoms",
            required=True,
            options=[Suggest(label="A")],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": []}

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    # Empty multi-select returns empty list (not list with empty string)
    assert content == "Symptoms: "
    assert metadata["answers"] == [[]]


@pytest.mark.unit
def test_format_response_input_empty() -> None:
    """Test formatting empty input response."""
    questions = [
        Question(
            header="Notes",
            type="input",
            text="Enter notes",
            required=True,
            options=[],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": ""}

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert content == "Notes: "
    assert metadata["answers"] == [[""]]


@pytest.mark.unit
def test_format_response_non_dict_content() -> None:
    """Test handling non-dict content."""
    questions = [
        Question(
            header="Notes",
            type="input",
            text="Enter notes",
            required=True,
            options=[],
        ),
    ]
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = "Direct string content"

    result = _format_question_response(questions, mock_result)
    content = cast(str, result.content)

    assert content == "Notes: "


# =============================================================================
# Integration Tests (question with mocked handle_elicitation)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
async def test_accept_response_single() -> None:
    """Test question with single enum accept response."""
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": "SY215C"}
    mock_ctx.handle_elicitation.return_value = mock_result

    xml = (
        '<question header="Model" type="enum"><text>What model?</text>'
        "<suggest>SY215C</suggest><suggest>SY235C</suggest></question>"
    )
    result = await QuestionTools().question(mock_ctx, xml)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert metadata["answers"] == [["SY215C"]]
    content = cast(str, result.content)
    assert "Model: SY215C" in content


@pytest.mark.asyncio
@pytest.mark.unit
async def test_accept_response_multi() -> None:
    """Test question with multi-select accept response."""
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": ["A", "B", "C"]}
    mock_ctx.handle_elicitation.return_value = mock_result

    xml = (
        '<question header="Symptoms" type="multi"><text>Select symptoms</text>'
        "<suggest>A</suggest><suggest>B</suggest><suggest>C</suggest></question>"
    )
    result = await QuestionTools().question(mock_ctx, xml)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert metadata["answers"] == [["A", "B", "C"]]
    content = cast(str, result.content)
    assert "Symptoms: A, B, C" in content


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancel_response() -> None:
    """Test question with cancel action raises RunAbortedError."""
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    mock_result = MagicMock()
    mock_result.action = "cancel"
    mock_ctx.handle_elicitation.return_value = mock_result

    xml = '<question header="Test"><text>Q</text><suggest>A</suggest></question>'
    with pytest.raises(RunAbortedError, match="cancelled"):
        await QuestionTools().question(mock_ctx, xml)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_decline_response() -> None:
    """Test question with decline action returns ToolResult."""
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    mock_result = MagicMock()
    mock_result.action = "decline"
    mock_ctx.handle_elicitation.return_value = mock_result

    xml = '<question header="Test"><text>Q</text><suggest>A</suggest></question>'
    result = await QuestionTools().question(mock_ctx, xml)
    content = cast(str, result.content)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert "declined" in content.lower()
    assert metadata["answers"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_error_data_response() -> None:
    """Test question raises ModelRetry on ErrorData from elicitation.

    Previously this returned a ToolResult (which pydantic-ai treated as success,
    causing ACP to report status="completed" instead of "failed").
    """
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    error_data = ErrorData(code=500, message="Server error")
    mock_ctx.handle_elicitation.return_value = error_data

    xml = '<question header="Test"><text>Q</text><suggest>A</suggest></question>'
    with pytest.raises(ModelRetry, match="Server error"):
        await QuestionTools().question(mock_ctx, xml)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_error_data_response_via_format_response() -> None:
    """Test _format_question_response raises ModelRetry on ErrorData."""
    xml = '<question header="Test"><text>Q</text><suggest>A</suggest></question>'
    questions = parse_questionnaire(xml)
    error_data = ErrorData(code=500, message="Elicitation failed")

    with pytest.raises(ModelRetry, match="Elicitation failed"):
        _format_question_response(questions, error_data)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_xml_parse_error_raises_model_retry() -> None:
    """Test question raises ModelRetry on invalid XML.

    Previously this returned a ToolResult (success), but XML parse errors
    should signal failure to the model so it can retry with corrected XML.
    """
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    invalid_xml = "<<<not valid xml>>>"

    with pytest.raises(ModelRetry, match="Error parsing questions"):
        await QuestionTools().question(mock_ctx, invalid_xml)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_question_multiple() -> None:
    """Test question with multiple questions."""
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": "SY215C", "q1": "Notes here"}
    mock_ctx.handle_elicitation.return_value = mock_result

    xml = (
        '<question header="Model" type="enum"><text>Model?</text>'
        "<suggest>SY215C</suggest></question>"
        '<question header="Notes" type="input"><text>Notes?</text></question>'
    )
    result = await QuestionTools().question(mock_ctx, xml)
    metadata: dict[str, list[list[str]]] = cast(dict[str, Any], result.metadata)

    assert metadata["answers"] == [["SY215C"], ["Notes here"]]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_question_params_check() -> None:
    """Verify question calls handle_elicitation with correct params."""
    mock_ctx = MagicMock()
    mock_ctx.handle_elicitation = AsyncMock()

    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.content = {"q0": "A"}
    mock_ctx.handle_elicitation.return_value = mock_result

    xml = (
        '<question header="Test Question" type="enum"><text>Select option</text>'
        "<suggest>A</suggest><suggest>B</suggest></question>"
    )
    await QuestionTools().question(mock_ctx, xml)

    mock_ctx.handle_elicitation.assert_called_once()
    params = mock_ctx.handle_elicitation.call_args[0][0]
    assert params.message == "Test Question"  # Uses first question header as message
    assert "requestedSchema" in vars(params) or hasattr(params, "requestedSchema")
