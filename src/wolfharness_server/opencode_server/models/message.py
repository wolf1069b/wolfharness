"""Message related models."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from wolfharness.utils import identifiers as identifier
from wolfharness_server.opencode_server.models.base import OpenCodeBaseModel
from wolfharness_server.opencode_server.models.common import (
    FileDiff,
    ModelRef,
    TextSpan,
    TimeCreated,
    Tokens,
)
from wolfharness_server.opencode_server.models.parts import (
    AgentPart,
    APIErrorInfo,
    FilePart,
    FilePartSource,
    Part,
    RetryPart,
    StepFinishPart,
    StepStartPart,
    SubtaskPart,
    TextPart,
    TimeStartEndOptional,
    ToolPart,
    ToolState,
)


class MessageSummary(OpenCodeBaseModel):
    """Summary information for a message."""

    title: str | None = None
    body: str | None = None
    diffs: list[FileDiff] = Field(default_factory=list)


class MessagePath(OpenCodeBaseModel):
    """Path context for a message."""

    cwd: str
    root: str


class MessageTime(OpenCodeBaseModel):
    """Time information for a message (milliseconds)."""

    created: int
    completed: int | None = None


class OutputFormatText(OpenCodeBaseModel):
    """Text output format."""

    type: Literal["text"] = Field(default="text", init=False)


class OutputFormatJsonSchema(OpenCodeBaseModel):
    """JSON schema output format."""

    type: Literal["json_schema"] = Field(default="json_schema", init=False)
    schema_: dict[str, Any] = Field(alias="schema")
    retry_count: int = 2


OutputFormat = OutputFormatText | OutputFormatJsonSchema


def _migrate_variant_into_model_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Backward-compat: move top-level ``variant`` into ``model.variant``.

    OpenCode v1.4.0+ nests ``variant`` inside the ``model`` object.
    Older clients may still send ``variant`` at the top level.
    """
    variant = data.get("variant")
    if variant is None:
        return data
    model = data.get("model")
    if model is None:
        # No model object — create one with just the variant
        data["model"] = {"variant": variant}
    elif isinstance(model, dict) and "variant" not in model:
        # Model exists but has no variant — add it
        model["variant"] = variant
    # Remove top-level variant so it doesn't shadow the nested one
    data.pop("variant", None)
    return data


class UserMessage(OpenCodeBaseModel):
    """User message."""

    id: str
    role: Literal["user"] = "user"
    session_id: str
    time: TimeCreated
    agent: str = "default"
    model: ModelRef | None = None
    format: OutputFormat | None = None
    summary: MessageSummary | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_variant_to_model(cls, data: Any) -> Any:
        """Backward-compat: move top-level ``variant`` into ``model.variant``."""
        if not isinstance(data, dict):
            return data
        return _migrate_variant_into_model_dict(data)


# --- Assistant message error types ---
# These match the NamedError pattern from upstream OpenCode:
# Each error is { name: Literal["..."], data: { ... } }


class ProviderAuthErrorData(OpenCodeBaseModel):
    """Data for provider authentication errors."""

    provider_id: str
    message: str


class ProviderAuthError(OpenCodeBaseModel):
    """Provider authentication error."""

    name: Literal["ProviderAuthError"] = Field(default="ProviderAuthError", init=False)
    data: ProviderAuthErrorData


class UnknownErrorData(OpenCodeBaseModel):
    """Data for unknown errors."""

    message: str


class UnknownError(OpenCodeBaseModel):
    """Unknown error."""

    name: Literal["UnknownError"] = Field(default="UnknownError", init=False)
    data: UnknownErrorData


class MessageOutputLengthErrorData(OpenCodeBaseModel):
    """Data for output length errors (empty)."""


class MessageOutputLengthError(OpenCodeBaseModel):
    """Message output length exceeded error."""

    name: Literal["MessageOutputLengthError"] = Field(
        default="MessageOutputLengthError", init=False
    )
    data: MessageOutputLengthErrorData = Field(default_factory=MessageOutputLengthErrorData)


class MessageAbortedErrorData(OpenCodeBaseModel):
    """Data for aborted message errors."""

    message: str


class MessageAbortedError(OpenCodeBaseModel):
    """Message was aborted."""

    name: Literal["MessageAbortedError"] = Field(default="MessageAbortedError", init=False)
    data: MessageAbortedErrorData


class APIErrorData(OpenCodeBaseModel):
    """Data for API errors."""

    message: str
    status_code: int | None = None
    is_retryable: bool = False
    response_headers: dict[str, str] | None = None
    response_body: str | None = None
    metadata: dict[str, str] | None = None


class APIError(OpenCodeBaseModel):
    """API error."""

    name: Literal["APIError"] = Field(default="APIError", init=False)
    data: APIErrorData


class StructuredOutputErrorData(OpenCodeBaseModel):
    """Data for structured output errors."""

    message: str
    retries: int


class StructuredOutputError(OpenCodeBaseModel):
    """Structured output validation error."""

    name: Literal["StructuredOutputError"] = Field(default="StructuredOutputError", init=False)
    data: StructuredOutputErrorData


class ContextOverflowErrorData(OpenCodeBaseModel):
    """Data for context overflow errors."""

    message: str
    response_body: str | None = None


class ContextOverflowError(OpenCodeBaseModel):
    """Context window overflow error."""

    name: Literal["ContextOverflowError"] = Field(default="ContextOverflowError", init=False)
    data: ContextOverflowErrorData


MessageError = (
    ProviderAuthError
    | UnknownError
    | MessageOutputLengthError
    | MessageAbortedError
    | APIError
    | StructuredOutputError
    | ContextOverflowError
)


class AssistantMessage(OpenCodeBaseModel):
    """Assistant message."""

    id: str
    role: Literal["assistant"] = "assistant"
    session_id: str
    parent_id: str  # Required - links to user message
    model_id: str
    provider_id: str
    mode: str = "default"
    agent: str = "default"
    path: MessagePath
    time: MessageTime
    tokens: Tokens = Field(default_factory=Tokens)
    """Context window usage from the latest step.

    Replaced (not accumulated) on each step. The TUI shows this from the
    last assistant message as the session "Context" indicator.
    """
    cost: float = 0.0
    """Per-message cost in USD.

    The TUI sums this across all assistant messages for the session total,
    so this must be per-message, not cumulative.
    """
    error: MessageError | None = None
    summary: bool | None = None
    finish: str | None = None
    structured: Any | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_variant_to_model(cls, data: Any) -> Any:
        """Backward-compat: move top-level ``variant`` into model context.

        For AssistantMessage, variant is informational only (no model object).
        We keep the variant in a private field for backward compat but the
        canonical location is on the associated UserMessage's model.variant.
        """
        if not isinstance(data, dict):
            return data
        # For assistant messages, variant was informational only.
        # Remove from top-level to match OpenCode v1.4.0+ schema.
        # We don't have a model object on AssistantMessage to nest it in.
        data.pop("variant", None)
        return data


class MessageWithParts(OpenCodeBaseModel):
    """Message with its parts."""

    info: MessageInfo
    parts: list[Part] = Field(default_factory=list)

    @property
    def role(self) -> Literal["user", "assistant"]:
        """Return the role of the message (user or assistant)."""
        return self.info.role

    @classmethod
    def user(
        cls,
        message_id: str,
        session_id: str,
        time: TimeCreated,
        agent_name: str,
        model: ModelRef | None = None,
    ) -> Self:
        user_msg = UserMessage(
            id=message_id,
            session_id=session_id,
            time=time,
            agent=agent_name,
            model=model,
        )
        return cls(info=user_msg)

    @classmethod
    def assistant(
        cls,
        message_id: str,
        session_id: str,
        time: MessageTime,
        agent_name: str,
        model_id: str,
        parent_id: str,
        provider_id: str,
        path: MessagePath,
        mode: str = "default",
        cost: float = 0.0,
        summary: bool | None = None,
        finish: str | None = None,
        error: MessageError | None = None,
        tokens: Tokens | None = None,
    ) -> Self:
        user_msg = AssistantMessage(
            id=message_id,
            session_id=session_id,
            time=time,
            agent=agent_name,
            model_id=model_id,
            parent_id=parent_id,
            provider_id=provider_id,
            path=path,
            mode=mode,
            cost=cost,
            error=error,
            summary=summary,
            finish=finish,
            tokens=tokens or Tokens(),
        )
        return cls(info=user_msg)

    def update_part(self, updated: Part) -> None:
        """Replace a part in the assistant message's parts list by ID."""
        for i, p in enumerate(self.parts):
            if isinstance(p, type(updated)) and p.id == updated.id:
                self.parts[i] = updated
                break

    def add_text_part(
        self,
        text: str,
        synthetic: bool | None = None,
        ignored: bool | None = None,
        time: TimeStartEndOptional | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TextPart:
        """Create and append a text part."""
        part = TextPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            text=text,
            synthetic=synthetic,
            ignored=ignored,
            time=time,
            metadata=metadata,
        )
        self.parts.append(part)
        return part

    def add_file_part(
        self,
        mime: str,
        url: str,
        filename: str | None = None,
        source: FilePartSource | None = None,
    ) -> FilePart:
        """Create and append a file part."""
        part = FilePart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            mime=mime,
            url=url,
            filename=filename,
            source=source,
        )
        self.parts.append(part)
        return part

    def add_agent_part(
        self,
        name: str,
        source: TextSpan | None = None,
    ) -> AgentPart:
        """Create and append an agent mention part."""
        part = AgentPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            name=name,
            source=source,
        )
        self.parts.append(part)
        return part

    def add_subtask_part(
        self,
        prompt: str,
        description: str,
        agent: str,
        model: ModelRef | None = None,
    ) -> SubtaskPart:
        """Create and append a subtask part."""
        part = SubtaskPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            prompt=prompt,
            description=description,
            agent=agent,
            model=model,
        )
        self.parts.append(part)
        return part

    def add_step_start_part(self, snapshot: str | None = None) -> StepStartPart:
        """Create and append a step start marker."""
        part = StepStartPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            snapshot=snapshot,
        )
        self.parts.append(part)
        return part

    def add_step_finish_part(
        self,
        reason: str = "stop",
        cost: float = 0.0,
        tokens: Tokens | None = None,
        snapshot: str | None = None,
    ) -> StepFinishPart:
        """Create and append a step finish marker."""
        part = StepFinishPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            reason=reason,
            cost=cost,
            tokens=tokens or Tokens(),
            snapshot=snapshot,
        )
        self.parts.append(part)
        return part

    def add_tool_part(
        self,
        tool: str,
        call_id: str,
        state: ToolState,
    ) -> ToolPart:
        """Create and append a tool call part."""
        part = ToolPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            tool=tool,
            call_id=call_id,
            state=state,
        )
        self.parts.append(part)
        return part

    def add_retry_part(
        self,
        attempt: int,
        message: str,
        created: int,
        is_retryable: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> RetryPart:
        """Create and append a retry part."""
        part = RetryPart(
            id=identifier.ascending("part"),
            message_id=self.info.id,
            session_id=self.info.session_id,
            attempt=attempt,
            error=APIErrorInfo(
                message=message,
                is_retryable=is_retryable,
                metadata=metadata,
            ),
            time=TimeCreated(created=created),
        )
        self.parts.append(part)
        return part


class TextPartInput(OpenCodeBaseModel):
    """Text part for input."""

    type: Literal["text"] = Field(default="text", init=False)
    text: str


class FilePartInput(OpenCodeBaseModel):
    """File part for input (image, document, etc.)."""

    type: Literal["file"] = Field(default="file", init=False)
    mime: str
    filename: str | None = None
    url: str  # Can be data: URI or file path
    source: FilePartSource | None = None


class AgentPartInput(OpenCodeBaseModel):
    """Agent mention part for input - references a sub-agent to delegate to.

    When a user types @agent-name in the prompt, this part is created.
    """

    type: Literal["agent"] = Field(default="agent", init=False)
    name: str
    """Name of the agent to delegate to."""
    source: TextSpan | None = None
    """Source location in the original prompt text."""


class SubtaskPartInput(OpenCodeBaseModel):
    """Subtask part for input - spawns a subtask to another agent."""

    type: Literal["subtask"] = Field(default="subtask", init=False)
    prompt: str
    """The prompt for the subtask."""
    description: str
    """Description of what the subtask does."""
    agent: str
    """The agent to handle this subtask."""
    model: ModelRef | None = None
    """Optional model to use for the subtask."""


PartInput = TextPartInput | FilePartInput | AgentPartInput | SubtaskPartInput


class MessageRequest(OpenCodeBaseModel):
    """Request body for sending a message."""

    parts: list[PartInput]
    message_id: str | None = None
    delivery: str | None = None
    model: ModelRef | None = None
    agent: str | None = None
    no_reply: bool | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_variant_to_model(cls, data: Any) -> Any:
        """Backward-compat: move top-level ``variant`` into ``model.variant``."""
        if not isinstance(data, dict):
            return data
        return _migrate_variant_into_model_dict(data)


class ShellRequest(OpenCodeBaseModel):
    """Request body for running a shell command."""

    agent: str
    command: str
    model: ModelRef | None = None


class CommandRequest(OpenCodeBaseModel):
    """Request body for executing a slash command."""

    command: str
    arguments: str | None = None
    agent: str | None = None
    model: str | None = None  # Format: "providerID/modelID"
    message_id: str | None = None


# Type unions

MessageInfo = UserMessage | AssistantMessage
