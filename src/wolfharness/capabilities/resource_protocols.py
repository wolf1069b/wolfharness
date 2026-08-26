"""Domain-specific Resource Protocols for unified extension access.

Defines Protocol interfaces for skill, MCP, and command resource access.
Each protocol is ``@runtime_checkable`` so capabilities can be queried via
``isinstance(cap, SkillResource)`` etc.

The deprecated ``McpResource`` Protocol has been split into three focused
protocols: ``ToolAccess``, ``ResourceAccess``, and ``ResourceTemplateAccess``.
``McpResource`` remains as a ``DeprecatedAlias`` that emits ``DeprecationWarning``
on ``isinstance()`` checks.

ChangeEvent is imported from ``wolfharness.capabilities.change_event`` —
this module does NOT define a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import warnings

from wolfharness.common_types import JsonObject, JsonValue  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import PurePosixPath

    from upath import UPath

    from wolfharness.capabilities.change_event import ChangeEvent


def resource_catalog_key(server: str, uri: str) -> str:
    """Build the OpenCode-compatible escaped ``server:uri`` catalog key."""
    escaped_server = server.replace("%", "%25").replace(":", "%3A")
    return f"{escaped_server}:{uri}"


def normalize_mcp_json_object(value: object) -> JsonObject | None:
    """Convert MCP metadata models into the repository JSON value types."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not isinstance(value, dict):
        return None
    return {
        key: _normalize_mcp_json_value(item) for key, item in value.items() if isinstance(key, str)
    }


def _normalize_mcp_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {
            key: _normalize_mcp_json_value(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, list):
        return [_normalize_mcp_json_value(item) for item in value]
    return str(value)


# ---- Dataclasses ----


@dataclass(frozen=True, slots=True)
class SkillEntry:
    """A skill descriptor returned by ``SkillResource.list_skills()``.

    Attributes:
        name: Skill name (e.g., ``"ponytail"``).
        description: Short human-readable description.
        uri: Canonical URI (e.g., ``"skill://ponytail/SKILL.md"``).
        source: Where the skill comes from — ``"local"`` or ``"remote"``.
        skill_path: Real filesystem path for local skills (``UPath``) or
            ``None`` for virtual/MCP skills.
    """

    name: str
    description: str = ""
    uri: str = ""
    source: str = "local"
    skill_path: UPath | PurePosixPath | None = None


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """A tool descriptor returned by ``McpResource.list_tools()``.

    Attributes:
        name: Tool name as known to the MCP server.
        description: Tool description from the server.
        schema: JSON schema dict for the tool's input parameters.
    """

    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of an MCP tool call.

    Attributes:
        content: The tool output content as text.
        is_error: Whether the tool returned an error.
    """

    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ResourceEntry:
    """A resource descriptor returned by ``McpResource.list_resources()``.

    Attributes:
        uri: Resource URI (e.g., ``"file:///path/to/resource"``).
        name: Human-readable resource name.
        title: Optional display title.
        description: Optional description.
        mime_type: MIME type of the resource content.
        size: Optional byte size reported by the provider.
        annotations: Optional MCP annotations.
        meta: Optional provider metadata.
        server: Configured MCP server/client name.
    """

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""
    title: str = ""
    size: int | None = None
    annotations: JsonObject | None = None
    meta: JsonObject | None = None
    server: str = ""


@dataclass(frozen=True, slots=True)
class CommandEntry:
    """A command descriptor returned by ``CommandResource.list_commands()``.

    Attributes:
        name: Command name (e.g., ``"ponytail"``).
        description: Short description of what the command does.
        skill_uri: URI of the skill backing this command, if any.
        source: Where the command comes from — ``"local"`` or ``"remote"``.
    """

    name: str
    description: str = ""
    skill_uri: str = ""
    source: str = "local"


# ---- Resource Content Types (MCP-aligned) ----


@dataclass(frozen=True, slots=True)
class ResourceContent:
    """Base type for resource content returned by ``read_resource()``.

    Mirrors MCP's ``ResourceContents`` structure.

    Attributes:
        uri: The URI of the resource that was read.
        mime_type: MIME type of the resource content, or ``None``.
        meta: Optional metadata dict from the MCP server.
    """

    uri: str
    mime_type: str | None = None
    meta: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class TextResourceContent(ResourceContent):
    """Text resource content returned by ``read_resource()``.

    Mirrors MCP's ``TextResourceContents``.

    Attributes:
        text: The text content of the resource.
    """

    text: str = ""


@dataclass(frozen=True, slots=True)
class BlobResourceContent(ResourceContent):
    """Binary resource content returned by ``read_resource()``.

    Mirrors MCP's ``BlobResourceContents``. The ``blob`` field contains
    base64-encoded bytes — decoding happens at the agent tool layer.

    Attributes:
        blob: Base64-encoded string of the binary content.
    """

    blob: str = ""


# ---- Resource Template & Completion Types ----


@dataclass(frozen=True, slots=True)
class ResourceTemplateEntry:
    """A resource template descriptor returned by ``list_resource_templates()``.

    Mirrors MCP's ``ResourceTemplate`` structure.

    Attributes:
        uri_template: URI template pattern (e.g., ``"file:///{path}"``).
        name: Human-readable template name.
        title: Display title (may be empty).
        description: Optional description.
        mime_type: MIME type of expanded resources.
        annotations: Optional MCP annotations dict.
        meta: Optional provider metadata.
        server: Configured MCP server/client name.
    """

    uri_template: str
    name: str = ""
    title: str = ""
    description: str = ""
    mime_type: str = ""
    annotations: JsonObject | None = None
    meta: JsonObject | None = None
    server: str = ""


@dataclass(frozen=True, slots=True)
class ResourceError:
    """Structured error returned by a resource provider."""

    code: str
    message: str
    retryable: bool = False
    suggestion: str = ""


@dataclass(frozen=True, slots=True)
class McpResourceListResult:
    """One host-aggregated page of MCP resources."""

    summary: str
    resources: list[ResourceEntry] = field(default_factory=list)
    next_cursor: str | None = None
    errors: list[ResourceError] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class McpResourceTemplateListResult:
    """One host-aggregated page of MCP resource templates."""

    summary: str
    templates: list[ResourceTemplateEntry] = field(default_factory=list)
    next_cursor: str | None = None
    errors: list[ResourceError] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class McpResourceReadResult:
    """Structured result for a single MCP resource read."""

    summary: str
    uri: str
    contents: list[JsonObject] = field(default_factory=list)
    truncated: bool = False
    original_char_count: int | None = None
    errors: list[ResourceError] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class McpResourceListPage:
    """Single upstream MCP resources/list page."""

    entries: list[ResourceEntry] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class McpResourceTemplateListPage:
    """Single upstream MCP resources/templates/list page."""

    entries: list[ResourceTemplateEntry] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionArgument:
    """Argument for a resource template completion request.

    Mirrors MCP's ``CompletionArgument``.

    Attributes:
        name: The parameter name being completed.
        value: The current value of the parameter.
    """

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Result of a resource template completion request.

    Mirrors MCP's ``Completion`` type.

    Attributes:
        values: List of completion suggestions.
        total: Total number of completions available (may exceed ``len(values)``).
        has_more: Whether more completions exist beyond those returned.
    """

    values: list[str]
    total: int | None = None
    has_more: bool | None = None


# ---- Protocols ----


@runtime_checkable
class SkillResource(Protocol):
    """Protocol for accessing skill resources."""

    async def list_skills(self) -> Sequence[SkillEntry]:
        """Return all available skills.

        Returns:
            Sequence of ``SkillEntry`` descriptors.
        """
        ...

    async def read_skill(self, name: str) -> str | None:
        """Read skill content by name.

        Args:
            name: Skill name to read.

        Returns:
            Skill content as string, or ``None`` if not found.
        """
        ...

    async def skill_exists(self, name: str) -> bool:
        """Check if a skill exists without reading it.

        Args:
            name: Skill name to check.

        Returns:
            ``True`` if the skill exists, ``False`` otherwise.
        """
        ...


@runtime_checkable
class ToolAccess(Protocol):
    """Protocol for accessing MCP tools.

    This is the tool-access portion of the deprecated ``McpResource``.
    Capabilities implementing this protocol provide tool listing and invocation.
    """

    async def list_tools(self) -> Sequence[ToolEntry]:
        """List available MCP tools.

        Returns:
            Sequence of ``ToolEntry`` descriptors.
        """
        ...

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call an MCP tool.

        Args:
            name: Tool name to call.
            args: Arguments to pass to the tool.

        Returns:
            ``ToolResult`` with the tool output.
        """
        ...


class UriSchemeMismatchError(ValueError):
    """Raised when a provider receives a URI whose scheme it does not own.

    Attributes:
        scheme: The URI scheme that was not recognized.
        provider_name: The name of the provider that rejected the URI.
        uri: The full URI that was rejected.
    """

    def __init__(self, scheme: str, provider_name: str, uri: str) -> None:
        self.scheme = scheme
        self.provider_name = provider_name
        self.uri = uri
        super().__init__(
            f"Provider '{provider_name}' does not own URI scheme '{scheme}' for URI: {uri}"
        )


class UriSchemeConflictError(ValueError):
    """Raised when two providers claim the same URI scheme.

    Attributes:
        scheme: The URI scheme with conflicting claims.
        existing_provider: The provider already registered for this scheme.
        conflicting_provider: The provider attempting to register.
    """

    def __init__(self, scheme: str, existing_provider: str, conflicting_provider: str) -> None:
        self.scheme = scheme
        self.existing_provider = existing_provider
        self.conflicting_provider = conflicting_provider
        super().__init__(
            f"URI scheme '{scheme}' is already claimed by "
            f"'{existing_provider}'; cannot register '{conflicting_provider}'"
        )


@runtime_checkable
class ResourceAccess(Protocol):
    """Protocol for accessing MCP resources.

    This is the resource-access portion of the deprecated ``McpResource``.
    Capabilities implementing this protocol provide resource listing, reading,
    and existence checking.
    """

    @property
    def owned_schemes(self) -> frozenset[str]:
        """URI schemes this provider authoritatively handles.

        Returns:
            ``frozenset`` of URI scheme strings (e.g., ``{"viking"}``).
            An empty set (default) means the provider handles opaque
            URIs and is consulted for unregistered schemes.
        """
        return frozenset()

    async def list_resources(self) -> Sequence[ResourceEntry]:
        """List available MCP resources.

        Returns:
            Sequence of ``ResourceEntry`` descriptors.
        """
        ...

    async def read_resource(
        self, uri: str
    ) -> list[TextResourceContent | BlobResourceContent] | None:
        """Read an MCP resource by URI.

        Args:
            uri: Resource URI to read.

        Returns:
            List of ``TextResourceContent`` and/or ``BlobResourceContent``
            instances, or ``None`` if the resource is not found.
        """
        ...

    async def resource_exists(self, uri: str) -> bool:
        """Check if an MCP resource exists.

        Args:
            uri: Resource URI to check.

        Returns:
            ``True`` if the resource exists, ``False`` otherwise.
        """
        ...


@runtime_checkable
class ResourceTemplateAccess(Protocol):
    """Protocol for accessing MCP resource templates.

    Capabilities implementing this protocol provide resource template listing
    and parameter completion. Capabilities that do not support completion
    SHALL raise ``NotImplementedError`` from ``complete_resource_template()``.
    """

    async def list_resource_templates(self) -> Sequence[ResourceTemplateEntry]:
        """List available resource templates.

        Returns:
            Sequence of ``ResourceTemplateEntry`` descriptors.
        """
        ...

    async def complete_resource_template(
        self,
        uri_template: str,
        argument: CompletionArgument,
        context: dict[str, str] | None = None,
    ) -> CompletionResult:
        """Complete a resource template parameter."""
        ...


@runtime_checkable
class McpResourceProvider(Protocol):
    """Protocol for host-side MCP Resource catalog and reads."""

    @property
    def server_name(self) -> str:
        """Return the stable configured MCP server identifier."""
        ...

    async def supports_resources(self) -> bool:
        """Return whether the upstream server declared Resource capability."""
        ...

    async def list_resources(self) -> Sequence[ResourceEntry]:
        """Read all resources for compatibility Host catalog consumers."""
        ...

    async def list_resources_page(self, cursor: str | None = None) -> McpResourceListPage:
        """Read one upstream resources/list page."""
        ...

    async def list_resource_templates_page(
        self, cursor: str | None = None
    ) -> McpResourceTemplateListPage:
        """Read one upstream resources/templates/list page."""
        ...

    async def read_mcp_resource(
        self, uri: str
    ) -> list[TextResourceContent | BlobResourceContent] | None:
        """Read an MCP resource by opaque URI."""
        ...


# ---- Deprecated McpResource alias ----


class _McpResourceMeta(type):
    """Metaclass for the deprecated ``McpResource`` alias.

    Implements ``__instancecheck__`` to check against both ``ToolAccess``
    and ``ResourceAccess`` protocols, emitting a ``DeprecationWarning`` on
    each check. This is necessary because Python's ``@runtime_checkable``
    Protocol does not support custom ``__instancecheck__`` — a plain
    ``Union[ToolAccess, ResourceAccess]`` type alias cannot emit warnings
    or be used with ``isinstance()``.
    """

    def __instancecheck__(cls, obj: object) -> bool:
        """Check if obj implements both ToolAccess and ResourceAccess.

        Emits ``DeprecationWarning`` on each call.
        """
        warnings.warn(
            "McpResource is deprecated. Use ToolAccess and ResourceAccess "
            "instead for isinstance() checks.",
            DeprecationWarning,
            stacklevel=2,
        )
        return isinstance(obj, (ToolAccess, ResourceAccess))


class McpResource(metaclass=_McpResourceMeta):
    """Deprecated alias for ``ToolAccess`` + ``ResourceAccess``.

    This class has been replaced by three focused protocols:
        - ``ToolAccess``: tool listing and invocation
        - ``ResourceAccess``: resource listing, reading, existence checking
        - ``ResourceTemplateAccess``: resource template listing and completion

    ``isinstance(obj, McpResource)`` still works but emits a
    ``DeprecationWarning``. Migrate to explicit ``ToolAccess`` /
    ``ResourceAccess`` checks.

    Note:
        ``wolfharness_server.opencode_server.models.mcp.McpResource`` (Pydantic
        model) is unrelated to this deprecated Protocol and is NOT affected.
    """


@runtime_checkable
class CommandResource(Protocol):
    """Protocol for accessing commands (slash commands)."""

    async def list_commands(self) -> Sequence[CommandEntry]:
        """List available commands.

        Returns:
            Sequence of ``CommandEntry`` descriptors.
        """
        ...

    async def get_command(self, name: str) -> CommandEntry | None:
        """Get a specific command by name.

        Args:
            name: Command name to retrieve.

        Returns:
            ``CommandEntry`` if found, ``None`` otherwise.
        """
        ...


@runtime_checkable
class ChangeObservable(Protocol):
    """Protocol for capabilities that emit change notifications."""

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        """Return an async iterator of change events, or ``None``.

        Returns:
            An ``AsyncIterator[ChangeEvent]`` if the capability supports
            change notifications, ``None`` otherwise.
        """
        ...
