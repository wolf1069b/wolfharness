"""ToolDisplayCapability — global decorator for tool display names and diff-rich events.

A configurable ``AbstractCapability`` that decorates the agent's fully
assembled toolset without modifying any tool or capability:

- **Rename layer** (``rename_mode``): maps selected tool names to
  display names via :class:`~pydantic_ai.toolsets.RenamedToolset`, so
  protocol clients (e.g. the OpenCode TUI) that dispatch on a whitelist
  of standard tool names render the tools properly.
- **Rich-info layer** (``emit_diff``): injects a
  :class:`~wolfharness.agents.events.DiffContentItem` progress event after
  a matching tool executes, so ACP clients (e.g. Zed) render a file
  diff.

The two layers are orthogonal: an OpenCode-facing deployment uses
``rename_mode=True + emit_diff=True``; an ACP-facing deployment uses
``rename_mode=False + emit_diff=True`` (original names displayed with
diffs); a child capability that already emits its own
``DiffContentItem`` uses ``rename_mode=True + emit_diff=False`` (rename
only, no duplicate diff).

Modeled on :class:`~wolfharness.agents.native_agent.tool_intercept.ToolInterceptCapability`
— a standalone ``AbstractCapability`` overriding ``get_wrapper_toolset``
and ``wrap_tool_execute`` as a global middleware over all assembled
tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import logfire
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, RenamedToolset

from wolfharness.agents.events import DiffContentItem


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic_ai._run_context import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

    from wolfharness.agents.events import ToolCallContentItem
    from wolfharness.tools.base import ToolKind


@dataclass(kw_only=True)
class RichDisplayInfo:
    """Rich display info derived for a tool call.

    Attributes:
        title: Human-readable title describing the operation.
        kind: Tool kind (read, edit, search, …, other).
        locations: Target paths/URIs affected by the tool call.
        items: Rich content items (text/file content) for the client.
    """

    title: str
    kind: ToolKind = "other"
    locations: list[str] = field(default_factory=list)
    items: list[ToolCallContentItem] = field(default_factory=list)


class RichExtractor(Protocol):
    """Strategy for extracting rich display info from a tool result.

    Implementations receive the validated arguments and the real tool
    result, and return rich content items plus any locations that could
    not be derived from arguments alone.
    """

    def __call__(
        self, args: Mapping[str, Any], result: Any
    ) -> tuple[list[ToolCallContentItem], list[str]]: ...


def _parse_diff_fields(
    args: Mapping[str, Any], result: Any
) -> tuple[str | None, str | None, str | None]:
    """Extract (path, old_text, new_text) from tool call arguments and result.

    Recognizes common parameter shapes across file-writing tools:

    - ``path``/``file_path``/``uri`` → target path
    - ``content`` (write-style) → new text, old text ``None`` (new file)
    - ``old_string``/``new_string`` (edit-style) → old/new text pair

    ``result`` is inspected as a fallback when ``new_text`` cannot be
    derived from arguments (e.g. a tool that returns the written content
    as a string).

    Args:
        args: The validated tool call arguments.
        result: The tool execution result.

    Returns:
        A ``(path, old_text, new_text)`` tuple with ``None`` values for
        fields that could not be derived.
    """
    path = next(
        (
            str(args[k])
            for k in ("path", "file_path", "uri", "filepath")
            if isinstance(args.get(k), str) and args[k]
        ),
        None,
    )
    if path is None:
        return (None, None, None)

    old_text: str | None = None
    new_text: str | None = None
    if isinstance(args.get("old_string"), str) and isinstance(args.get("new_string"), str):
        old_text = args["old_string"]
        new_text = args["new_string"]
    elif isinstance(args.get("content"), str):
        new_text = args["content"]
    elif isinstance(result, str) and result:
        new_text = result

    return (path, old_text, new_text)


# Path-like keys recognized across write/edit/read/query tools. Includes
# ``target_uri`` (subtree restriction on viking search/find).
_PATH_KEYS = ("path", "file_path", "uri", "uris", "filepath", "target_uri")


def _parse_locations(args: Mapping[str, Any]) -> list[str]:
    """Extract target paths/URIs from tool arguments.

    Recognizes ``path``/``file_path``/``uri``/``uris``/``filepath``/
    ``target_uri`` keys. List-valued keys (e.g. viking ``uris``) expand
    to one location per element; scalar strings are kept as a single
    location.

    Args:
        args: The validated tool call arguments.

    Returns:
        A list of location strings, possibly empty.
    """
    locations: list[str] = []
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str):
            if value:
                locations.append(value)
        elif isinstance(value, (list, tuple)):
            locations.extend(str(item) for item in value if item)
    return locations


def _unwrap_result(result: Any) -> Any:
    """Unwrap a pydantic-ai ``ToolReturn`` to its ``return_value``.

    Native capability tools (viking, …) return ``ToolReturn`` objects;
    the real tool result seen by the wrapper may be either the object or
    its ``return_value`` depending on the call path. Content extraction
    targets the plain value.

    Args:
        result: The raw tool result.

    Returns:
        The ``return_value`` for a ``ToolReturn``; the result unchanged
        otherwise.
    """
    from pydantic_ai.messages import ToolReturn

    if isinstance(result, ToolReturn):
        return result.return_value
    return result


# Registry of per-tool rich content extractors (Tool-kind strategies).
# Maps original tool name → extractor producing content items + locations
# from the real execution result. New tools extend this registry.
_RICH_EXTRACTORS: dict[str, RichExtractor] = {}

# Known tool-name prefixes whose suffix maps to a standard tool kind
# (e.g. ``viking_read`` → kind ``read``). Used by rich-title derivation
# when ``derive_rich_tool_info`` does not recognize the prefixed name.
_PREFIXED_KIND_TOOLS = (
    "viking_read",
    "viking_search",
    "viking_find",
    "viking_glob",
    "fsspec_read",
    "fsspec_write",
    "fsspec_edit",
)


def _derive_kind(original_name: str, args: Mapping[str, Any]) -> tuple[str, ToolKind]:
    """Derive (title, kind) for a possibly-prefixed tool name.

    Strips a known prefix (``viking_``, ``fsspec_``) and re-derives via
    ``derive_rich_tool_info`` so ``viking_read`` is classified as a read
    tool rather than ``other``. Falls back to the raw name when no prefix
    applies.

    Args:
        original_name: The original tool name.
        args: The validated tool call arguments.

    Returns:
        A ``(title, kind)`` pair with a protocol-safe kind.
    """
    from wolfharness.agents.events.infer_info import derive_rich_tool_info

    kind_map: dict[str, ToolKind] = {
        "read": "read",
        "search": "search",
        "find": "search",
        "glob": "search",
        "write": "edit",
        "edit": "edit",
    }
    for tool in _PREFIXED_KIND_TOOLS:
        if original_name == tool and "_" in tool:
            suffix = tool.partition("_")[2]
            if suffix in kind_map:
                rich = derive_rich_tool_info(suffix, args)
                return (rich.title, kind_map[suffix])
    rich = derive_rich_tool_info(original_name, args)
    return (rich.title, rich.kind)


def _text_extractor(
    items_builder: Callable[[str], list[ToolCallContentItem]],
) -> RichExtractor:
    """Build an extractor producing content items from an unwrapped result string."""

    def extract(
        args: Mapping[str, Any], result: Any
    ) -> tuple[list[ToolCallContentItem], list[str]]:
        locations = _parse_locations(args)
        text = _unwrap_result(result)
        if not isinstance(text, str) or not text:
            return ([], locations)
        return (items_builder(text), locations)

    return extract


def _register_viking_extractors() -> None:
    """Register rich content extractors for the built-in viking tools.

    Read/query tools return formatted text (line-numbered content, search
    results). The extractors package that text into a ``TextContentItem``
    for protocol clients and reuse argument-derived locations.

    Note:
        These are display-only enhancements applied by ``emit_rich``;
        they do not modify the viking tools themselves.
    """
    from wolfharness.agents.events import TextContentItem

    def text_content_items(text: str) -> list[ToolCallContentItem]:
        return [TextContentItem(text=text)]

    for tool in ("viking_read", "viking_search", "viking_find", "viking_glob"):
        _RICH_EXTRACTORS[tool] = _text_extractor(text_content_items)


_register_viking_extractors()


@dataclass(kw_only=True)
class ToolDisplayCapability(AbstractCapability[Any]):
    """Global tool display decorator: rename tools + inject diff events.

    Attributes:
        rename_mode: Enable tool name mapping via ``name_map``. When
            ``False``, tools keep their native names (ACP-style display).
        name_map: Mapping of **original** tool name to **display** name
            (what the user writes in YAML). Internally inverted before
            passing to ``RenamedToolset``, which expects ``{new: original}``.
        emit_diff: Enable diff event injection after tool execution.
            When ``False``, rely on tools' own diff emission.
        emit_diff_for: Set of **original** tool names eligible for diff
            event injection. Empty means no injection. When rename is
            active, display names are resolved back to originals before
            matching.
        emit_rich: Enable rich display event injection (kind,
            locations, content) for read/query tools.
        emit_rich_for: Set of **original** tool names eligible for rich
            display injection. Empty means no injection.
        id: Optional capability id.
    """

    id: str | None = None
    rename_mode: bool = True
    name_map: Mapping[str, str] = field(default_factory=dict)
    emit_diff: bool = True
    emit_diff_for: set[str] = field(default_factory=set)
    emit_rich: bool = True
    emit_rich_for: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Coerce target sets to ``set`` if lists were provided via YAML."""
        if isinstance(self.emit_diff_for, list):
            self.emit_diff_for = set(self.emit_diff_for)
        if isinstance(self.emit_rich_for, list):
            self.emit_rich_for = set(self.emit_rich_for)

    @property
    def _reverse_name_map(self) -> dict[str, str]:
        """Display → original lookup, derived from ``name_map`` (original → display)."""
        return {v: k for k, v in self.name_map.items()}

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
        """Wrap the assembled toolset with ``RenamedToolset`` when enabled.

        ``name_map`` is stored as ``{original: display}`` (user-facing
        convention) but ``RenamedToolset`` expects ``{new: original}``,
        so we invert before construction.

        Args:
            toolset: The agent's fully assembled toolset.

        Returns:
            A ``RenamedToolset`` applying ``name_map``, or ``None`` when
            renaming is disabled or the map is empty (toolset unchanged).
        """
        if not self.rename_mode or not self.name_map:
            return None
        # RenamedToolset.name_map is {new_name: original_name}.
        # Our name_map is {original: display} → invert to {display: original}.
        inverted = {v: k for k, v in self.name_map.items()}
        return RenamedToolset(wrapped=toolset, name_map=inverted)

    def _get_original_name(self, call_name: str) -> str:
        """Resolve the display (renamed) tool name to its original name.

        When rename is active, ``call.tool_name`` is the display name;
        emit-diff/emit-rich filters contain original names.

        Args:
            call_name: The tool name as seen by the wrapper.

        Returns:
            The original tool name.
        """
        return self._reverse_name_map.get(call_name, call_name)

    def _prepare_emitter(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        original_name: str,
    ) -> Any:
        """Populate ``ctx.deps.tool_call_id`` / ``tool_name`` and return the emitter.

        Capability tools bypass ``tool_wrapping.py``, leaving these fields
        ``None``. ``StreamEventEmitter`` reads them to tag emitted events;
        without a ``tool_call_id`` the ACP converter drops the event.

        Args:
            ctx: The pydantic-ai run context (``deps`` is the agentpool
                ``AgentContext``).
            call: The tool call part (carries ``tool_call_id``).
            original_name: The resolved original tool name.

        Returns:
            The ``events`` emitter, or ``None`` when unavailable.
        """
        deps = ctx.deps
        if hasattr(deps, "tool_call_id"):
            deps.tool_call_id = call.tool_call_id
        if hasattr(deps, "tool_name"):
            deps.tool_name = original_name
        events = getattr(deps, "events", None)
        return events if events is not None else None

    def _derive_rich_pre(self, original_name: str, args: dict[str, Any]) -> RichDisplayInfo:
        """Derive pre-execution rich display info (title, kind, locations).

        Uses the registry extractor when available; otherwise falls back
        to ``derive_rich_tool_info`` for title/kind plus generic location
        extraction from arguments.

        Args:
            original_name: The original tool name.
            args: The validated tool call arguments.

        Returns:
            Rich display info with title, kind and locations.
        """
        title, kind = _derive_kind(original_name, args)
        locations = _parse_locations(args)
        return RichDisplayInfo(
            title=title,
            kind=kind,
            locations=locations,
        )

    def _derive_rich_post(
        self, original_name: str, args: dict[str, Any], result: Any
    ) -> RichDisplayInfo | None:
        """Derive post-execution rich content (content items + locations).

        Applies the registry extractor for the tool; returns ``None`` when
        no extractor is registered (title/kind/locations already provided
        by the pre-execution event).

        The returned info carries **no title**: the post event is a content
        update for an already-titled tool call. Protocol clients fall back
        to the pre-execution title (opencode ``_process_tool_progress`` uses
        ``title or existing_title``; ACP keeps its state title), so a
        shared read/query extractor never clobbers the start title (e.g.
        ``viking_search`` must stay "Search for '<query>'").

        Args:
            original_name: The original tool name.
            args: The validated tool call arguments.
            result: The real tool result (may be a ``ToolReturn``).

        Returns:
            Rich content info, or ``None`` when no extractor applies.
        """
        extractor = _RICH_EXTRACTORS.get(original_name)
        if extractor is None:
            return None
        items, extra_locations = extractor(args, result)
        return RichDisplayInfo(
            title="",
            kind="read",
            locations=extra_locations,
            items=items,
        )

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """Execute the tool, then inject diff/rich display events when enabled.

        Orchestrates the three orthogonal layers:

        1. **Rich (emit_rich)**: for read/query tools, emits a
           ``ToolCallStartEvent`` (kind + locations) *before* execution
           and a progress event carrying content items *after* execution.
        2. **Diff (emit_diff)**: for write/edit tools, emits a
           ``ToolCallProgressEvent`` carrying a ``DiffContentItem`` after
           execution — the same channel fsspec tools use, which reaches
           ACP converters as ``FileEditToolCallContent``.
        3. **Rename**: handled by ``get_wrapper_toolset``; here we only
           resolve display → original names for both filters above.

        Args:
            ctx: The pydantic-ai run context (carries ``deps`` → wolfharness
                ``AgentContext`` with the ``events`` emitter).
            call: The tool call part.
            tool_def: The tool definition.
            args: The validated tool call arguments.
            handler: The wrapped tool execution callable.

        Returns:
            The tool execution result, unchanged.
        """
        with logfire.span(
            "capability.tool_display.wrap_tool_execute",
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            args=args,
        ):
            original_name = self._get_original_name(call.tool_name)
            rich_target = self.emit_rich and original_name in self.emit_rich_for

            # Pre-execution: emit rich tool-start event for read/query tools.
            pre_title: str | None = None
            if rich_target:
                pre = self._derive_rich_pre(original_name, args)
                pre_title = pre.title
                events = self._prepare_emitter(ctx, call=call, original_name=original_name)
                if events is not None:
                    await events.tool_call_start(
                        title=pre.title,
                        kind=pre.kind,
                        locations=pre.locations,
                    )

            result = await handler(args)

            # Post-execution: rich content for read/query tools. Only
            # short-circuits when the rich layer actually produced content;
            # otherwise (e.g. a write tool also listed in emit_rich_for)
            # we fall through to diff injection. The post event reuses the
            # pre-execution title so search/glob tools don't get clobbered
            # to a generic "Read".
            if rich_target:
                post = self._derive_rich_post(original_name, args, result)
                if post is not None and post.items:
                    events = self._prepare_emitter(ctx, call=call, original_name=original_name)
                    if events is not None:
                        await events.tool_call_progress(
                            title=pre_title or post.title,
                            items=[*post.items],
                        )
                    return result

            # Diff injection for write/edit tools (existing layer).
            if not self.emit_diff or not self.emit_diff_for:
                return result
            if original_name not in self.emit_diff_for:
                return result

            path, old_text, new_text = _parse_diff_fields(args, result)
            if path is None or new_text is None:
                return result

            events = self._prepare_emitter(ctx, call=call, original_name=original_name)
            if events is None:
                return result

            await events.tool_call_progress(
                title=f"Modified: {path}",
                items=[
                    DiffContentItem(
                        path=path,
                        old_text=old_text,
                        new_text=new_text,
                    )
                ],
            )
            return result
