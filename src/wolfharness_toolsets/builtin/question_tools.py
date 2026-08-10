"""User interaction tools for asking questions via MCP Elicit.

Provides a single ``question`` tool that supports rich multi-question XML
questionnaires with enum/multi/input types.  Uses
``AgentContext.handle_elicitation()`` to present forms to the user through the
MCP Elicit protocol and collect structured responses.
"""

from __future__ import annotations

from typing import Any

from mcp.types import ElicitRequestFormParams, ErrorData
from pydantic_ai import ModelRetry
from pydantic_xml import BaseXmlModel, attr, element

from wolfharness.agents.context import AgentContext  # noqa: TC001
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.tasks.exceptions import RunAbortedError
from wolfharness.tools.base import ToolResult


class Suggest(BaseXmlModel, tag="suggest"):
    """A suggestion option within a question."""

    type: str = attr(default="choice")
    description: str | None = attr(default=None)
    next_action: str | None = attr(default=None)
    label: str  # Element text


class Question(BaseXmlModel, tag="question"):
    """A single question with its options.

    Supports multiple ``<text>`` elements via ``extra_texts`` — when LLM generates
    explanatory bullets as separate ``<text>`` tags, they are captured here so
    that ``options`` (``<suggest>`` tags) are still parsed correctly.
    """

    header: str = attr()
    type: str = attr(default="enum")
    required: bool = attr(default=True)
    text: str = element(tag="text")
    """Primary question text (first ``<text>`` element)."""

    extra_texts: list[str] = element(tag="text", default=[])
    """Additional ``<text>`` elements beyond the first (explanatory bullets, etc.)."""

    options: list[Suggest] = element(tag="suggest", default=[])
    """Selectable options parsed from ``<suggest>`` tags."""


class Questions(BaseXmlModel, tag="questions"):
    """Wrapper for multiple questions."""

    questions: list[Question] = element(tag="question")


def parse_questionnaire(xml: str) -> list[Question]:
    """Parse XML questionnaire into Question objects.

    Expects XML wrapped in <questions> tag containing one or more <question> elements.
    For backward compatibility, also accepts bare <question> tags (auto-wraps them).

    Args:
        xml: XML string containing <questions> wrapper with <question> tags inside,
             or bare <question> tags for backward compatibility.

    Returns:
        List of parsed Question objects.

    Examples:
        >>> xml = '''<questions>
        ...   <question header="Model" type="enum">
        ...     <text>What model?</text>
        ...     <suggest>Option A</suggest>
        ...   </question>
        ... </questions>'''
        >>> questions = parse_questionnaire(xml)
    """
    # Normalize whitespace and strip
    xml = xml.strip()

    # Check if already wrapped in <questions> tag
    if not xml.startswith("<questions"):
        # Backward compatibility: wrap bare <question> tags
        xml = f"<questions>{xml}</questions>"

    return Questions.from_xml(xml).questions


def _build_acp_schema(questions: list[Question]) -> dict[str, Any]:
    """Build ACP JSON schema from questions.

    Maps question types to appropriate JSON schema constructs:
    - enum -> string with oneOf for options
    - multi -> array with enum items
    - input -> string with minLength=1

    Args:
        questions: List of Question objects.

    Returns:
        JSON schema dictionary for ACP form.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for i, q in enumerate(questions):
        key = f"q{i}"
        if q.required:
            required.append(key)

        if q.type == "enum":
            # Build effective option list: prefer <suggest>, fallback to extra <text> elements
            effective_options = (
                q.options if q.options else [Suggest(label=t) for t in q.extra_texts]
            )
            if not effective_options:
                # No options at all — fall back to free text input
                properties[key] = {
                    "type": "string",
                    "title": q.header,
                    "description": q.text,
                    "minLength": 1,
                }
            elif any(o.description for o in effective_options):
                # Use oneOf with title (required by ACP clients).
                # XML suggest description maps to schema title (human-readable label).
                properties[key] = {
                    "type": "string",
                    "title": q.header,
                    "description": q.text,
                    "oneOf": [
                        {
                            "const": o.label,
                            "title": o.description if o.description else o.label,
                        }
                        for o in effective_options
                    ],
                }
            else:
                # Simple enum for maximum compatibility with ACP clients
                properties[key] = {
                    "type": "string",
                    "title": q.header,
                    "description": q.text,
                    "enum": [o.label for o in effective_options],
                }
        elif q.type == "multi":
            # Build effective option list: prefer <suggest>, fallback to extra <text> elements
            effective_options = (
                q.options if q.options else [Suggest(label=t) for t in q.extra_texts]
            )
            if not effective_options:
                # No options at all — fall back to free text input
                properties[key] = {
                    "type": "string",
                    "title": q.header,
                    "description": q.text,
                    "minLength": 1,
                }
            else:
                # Build options and x-option-descriptions mapping
                option_labels = [o.label for o in effective_options]
                descriptions = {o.label: o.description for o in effective_options if o.description}
                items_schema: dict[str, Any] = {
                    "type": "string",
                    "enum": option_labels,
                }
                if descriptions:
                    items_schema["x-option-descriptions"] = descriptions
                multi_schema: dict[str, Any] = {
                    "type": "array",
                    "title": q.header,
                    "description": q.text,
                    "items": items_schema,
                    "uniqueItems": True,
                }
                properties[key] = multi_schema
        elif q.type == "input":
            properties[key] = {
                "type": "string",
                "title": q.header,
                "description": q.text,
                "minLength": 1,
            }

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _format_question_response(questions: list[Question], result: Any) -> ToolResult:
    """Format the elicitation result into a ToolResult.

    Args:
        questions: List of Question objects (for structure reference).
        result: Result from ctx.handle_elicitation (ElicitResult or ErrorData).

    Returns:
        ToolResult with formatted content and metadata.
    """
    if isinstance(result, ErrorData):
        raise ModelRetry(f"Elicitation failed: {result.message}")

    action = result.action
    if action == "accept":
        content = result.content
        # Content is a dict with q0, q1... keys per schema
        answers: list[list[Any]] = []
        content_parts: list[str] = []

        for i, q in enumerate(questions):
            key = f"q{i}"
            value = content.get(key) if isinstance(content, dict) else None

            if q.type == "multi":
                # Multi-select returns list
                answer_list = value if isinstance(value, list) else [value] if value else []
                answers.append(answer_list)
                content_parts.append(f"{q.header}: {', '.join(str(a) for a in answer_list)}")
            else:
                # enum or input returns single value
                answer = str(value) if value else ""
                answers.append([answer])
                content_parts.append(f"{q.header}: {answer}")

        return ToolResult(
            content="\n".join(content_parts),
            metadata={"answers": answers},
        )
    if action == "cancel":
        raise RunAbortedError("User cancelled the questionnaire.")
    if action == "decline":
        return ToolResult(
            content="User declined to complete the questionnaire.",
            metadata={"answers": []},
        )
    raise RuntimeError(f"Unknown action: {action}")


class QuestionTools(FunctionToolsetCapability):
    """Built-in toolset providing the user interaction ``question`` tool.

    Exposes a single ``question`` tool via ``create_tool`` that accepts a
    multi-question XML questionnaire with enum/multi/input types.
    """

    def __init__(
        self,
        name: str = "question_tools",
    ) -> None:
        super().__init__(name=name)
        self.create_tool(self.question, category="other")

    async def question(  # noqa: D417
        self,
        ctx: AgentContext,
        questions: str,
    ) -> ToolResult:
        """Ask the user one or more questions and collect their responses.

        Parses questions from XML format, presents them as a form via MCP Elicit,
        and returns the user's answers.

        XML Format (use SINGLE QUOTES for attributes to avoid JSON encoding issues):
            <questions>
                <question header='...' type='enum|multi|input' required='true'>
                    <text>Question text</text>
                    <suggest type='choice'>Option 1</suggest>
                    <suggest type='choice'>Option 2</suggest>
                </question>
            </questions>

        For a single question you may omit the <questions> wrapper:

            <question header='Confirm' type='enum'>
                <text>Proceed?</text>
                <suggest>Yes</suggest>
                <suggest>No</suggest>
            </question>

        Args:
            questions: XML string with a <questions> root element containing
                one or more <question> tags (bare <question> tags also accepted).
        """
        try:
            parsed = parse_questionnaire(questions)
        except Exception as e:
            raise ModelRetry(f"Error parsing questions: {e!s}") from e
        schema = _build_acp_schema(parsed)
        message = "Please answer the following questions:"
        if parsed:
            message = parsed[0].header
        params = ElicitRequestFormParams(message=message, requestedSchema=schema)
        result = await ctx.handle_elicitation(params)
        return _format_question_response(parsed, result)
