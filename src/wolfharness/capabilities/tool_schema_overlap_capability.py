"""Capability for semantically adapting MCP tool schemas at runtime.

`ToolSchemaOverlapCapability` rewrites the model-facing schema of MCP tools —
tool descriptions, parameter names and descriptions, types, enums, required
flags, defaults — without touching the upstream MCP server. Overrides are
declared per server (`servers`) or globally (`global_overrides`) and resolved
against live tools via source-identity metadata stamped by `McpServerCap`,
never by parsing (possibly prefixed) tool names. Config keys are always raw
MCP tool names, so the same configuration works identically with or without
`McpServerCap.tool_prefix`.

The schema rewrite (applied by `SchemaOverrideToolset` in `get_tools`) and
the runtime parameter desharing (applied in `wrap_tool_execute`) derive from
the same configuration instance keyed by
`(server_name, original_mcp_tool_name)` — a single source of truth.

Validation is two-stage:

- Construction time: pure-config self-consistency (rename collisions,
  enum/default mismatches) — see `tool_schema_overlap_config`.
- First tool listing (agent startup): schema-dependent checks that need the
  live `required` lists (required-param removal without a default, unmatched
  server/tool keys). The agent fails to start on such misconfiguration.

Degradation: tools without identity metadata pass through unchanged. The
capability never infers identity from tool names — name-guessing would
misroute overrides across prefixed/unprefixed servers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import logging
from typing import TYPE_CHECKING, Any, Final

import logfire
from pydantic import ValidationError
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import WrapperToolset
from pydantic_core import InitErrorDetails

from wolfharness.capabilities.tool_display_capability import ToolDisplayCapability
from wolfharness.capabilities.tool_schema_overlap_config import (
    ORIGINAL_TOOL_NAME_METADATA_KEY,
    SERVER_NAME_METADATA_KEY,
    UNDEFINED,
    ParamOverride,
    ToolOverride,
    ToolSchemaOverlapConfig,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

logger = logging.getLogger(__name__)

# Marker written onto rewritten ToolDefinitions so that supplementary
# `prepare_tools` passes never apply the same override twice.
APPLIED_METADATA_KEY: Final[str] = "tool_schema_overlap_applied"

_LOCATION_UNMATCHED_SERVER: Final[str] = "a configured server name matched no listed tool"
_LOCATION_UNMATCHED_TOOL: Final[str] = "a configured tool name matched no listed tool"


def _read_identity(tool_def: ToolDefinition) -> tuple[str, str] | None:
    """Extract stamped source identity from a tool definition.

    Args:
        tool_def: The tool definition to inspect.

    Returns:
        `(server_name, original_mcp_tool_name)` when both keys are present
        and non-empty, otherwise `None`. Identity is never inferred from the
        tool name.
    """
    metadata = tool_def.metadata
    if not metadata:
        return None
    server_name = metadata.get(SERVER_NAME_METADATA_KEY)
    original_name = metadata.get(ORIGINAL_TOOL_NAME_METADATA_KEY)
    if not isinstance(server_name, str) or not isinstance(original_name, str):
        return None
    if not server_name or not original_name:
        return None
    return server_name, original_name


def resolve_override(
    config: ToolSchemaOverlapConfig, server_name: str, original_name: str
) -> ToolOverride | None:
    """Resolve the override for a tool identified by its source metadata.

    Server-scoped overrides take precedence over `global_overrides` for the
    same tool name on that server.

    Args:
        config: The validated overlap configuration.
        server_name: The server identity stamped on the tool.
        original_name: The raw MCP tool name stamped on the tool.

    Returns:
        The matching `ToolOverride`, or `None` when nothing is configured.
    """
    server_tools = config.servers.get(server_name)
    if server_tools is not None:
        override = server_tools.get(original_name)
        if override is not None:
            return override
    return config.global_overrides.get(original_name)


def _error_entry(loc: tuple[int | str, ...], msg: str) -> InitErrorDetails:
    """Build one first-listing line-error entry for fail-fast reporting.

    Args:
        loc: Config-style location of the error.
        msg: Human-readable error description.

    Returns:
        A pydantic-core line-error entry; the message is carried by the
        wrapped `ValueError` and rendered by `ValidationError`.
    """
    return InitErrorDetails(type="value_error", loc=loc, input=None, ctx={"error": ValueError(msg)})


def _listing_validation_error(entries: list[InitErrorDetails]) -> ValidationError:
    """Bundle first-listing validation entries into a single `ValidationError`.

    Args:
        entries: Line-error entries produced by the schema-dependent checks.

    Returns:
        A `ValidationError` carrying all entries, raised at agent startup.
    """
    return ValidationError.from_exception_data(
        "ToolSchemaOverlap", entries, input_type="python", hide_input=False
    )


def _original_location(server_name: str, original_name: str) -> tuple[int | str, ...]:
    """Build the config-style location prefix for error messages."""
    return ("servers", server_name, original_name)


def validate_schema_dependent(
    location: tuple[int | str, ...], schema: dict[str, Any], override: ToolOverride
) -> list[InitErrorDetails]:
    """Validate override rules that require the live schema.

    Runs at the first tool listing (agent startup), when the MCP `required`
    lists are known.

    Args:
        location: Config location of the override, used in error locations.
        schema: The live `parameters_json_schema` of the tool.
        override: The override configured for this tool.

    Returns:
        Line-error entries; empty when the override is safe to apply.
    """
    entries: list[InitErrorDetails] = []
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required_list = schema.get("required")
    required = required_list if isinstance(required_list, list) else []

    # Removing a required parameter is only safe when a configured default
    # is injected at call time; otherwise the MCP server would receive an
    # incomplete call.
    for removed in sorted(override.param_removals):
        if removed not in required:
            continue
        param_override = override.param_overrides.get(removed)
        if param_override is None or param_override.default is UNDEFINED:
            msg = (
                f"removes required parameter {removed!r} without a configured "
                f"default; configure param_overrides.{removed}.default or keep the parameter"
            )
            entries.append(_error_entry((*location, "param_removals", removed), msg))

    # A parameter renamed onto a name that still exists afterwards produces a
    # duplicate property (ambiguous schema and ambiguous desharing).
    renamed_originals = set(override.param_names)
    renamed_originals.update(
        original for original, po in override.param_overrides.items() if po.name is not None
    )
    rename_targets = dict(override.param_names)
    rename_targets.update(
        (original, po.name)
        for original, po in override.param_overrides.items()
        if po.name is not None
    )
    for original, target in sorted(rename_targets.items()):
        if target in properties and target not in renamed_originals:
            msg = f"renames parameter {original!r} onto existing parameter {target!r}"
            entries.append(_error_entry((*location, "param_names", original), msg))

    # Adding a parameter that already exists would silently clobber the
    # upstream schema.
    for added in sorted(override.param_additions):
        if added in properties:
            msg = f"adds parameter {added!r} which already exists in the tool schema"
            entries.append(_error_entry((*location, "param_additions", added), msg))

    return entries


def _apply_param_renames(
    properties: dict[str, Any], required: list[str], renames: dict[str, str]
) -> None:
    """Move renamed properties and swap their `required` entries.

    Args:
        properties: The schema `properties` mapping being rewritten.
        required: The schema `required` list kept in sync with renames.
        renames: Mapping of original parameter name to renamed parameter.
    """
    for original, new in renames.items():
        moved = properties.pop(original, None)
        if moved is not None:
            properties[new] = moved
        if original in required:
            required.remove(original)
            if new not in required:
                required.append(new)


def _apply_param_additions(
    properties: dict[str, Any],
    required: list[str],
    additions: dict[str, ParamOverride],
) -> None:
    """Append added parameters with their schema fields and defaults.

    The visible name is `ParamOverride.name` when set, otherwise the addition
    key (which is the server-side parameter name).

    Args:
        properties: The schema `properties` mapping being rewritten.
        required: The schema `required` list kept in sync with additions.
        additions: Mapping of added parameter name to its override.
    """
    for added, param_override in additions.items():
        visible = param_override.name if param_override.name is not None else added
        new_prop: dict[str, Any] = {}
        if param_override.type is not None:
            new_prop["type"] = param_override.type
        if param_override.enum is not None:
            new_prop["enum"] = list(param_override.enum)
        if param_override.description is not None:
            new_prop["description"] = param_override.description
        if param_override.default is not UNDEFINED and param_override.default is not None:
            new_prop["default"] = param_override.default
        properties[visible] = new_prop
        if param_override.required is True and visible not in required:
            required.append(visible)


def _apply_override_to_schema(schema: dict[str, Any], override: ToolOverride) -> None:
    """Apply one override to a parameter schema, mutating it in place.

    Application order is fixed: removals → description rewrites → field
    overrides → renames → additions. The caller passes a deep copy; the
    original schema is never mutated.

    Args:
        schema: The (copied) `parameters_json_schema` to rewrite.
        override: The override to apply.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        schema["properties"] = properties
    required_value = schema.get("required")
    required: list[str] = (
        [name for name in required_value if isinstance(name, str)]
        if isinstance(required_value, list)
        else []
    )
    for name in list(properties):
        if not isinstance(name, str):
            del properties[name]

    # 1. Removals — hide parameters from the model entirely.
    for removed in override.param_removals:
        properties.pop(removed, None)
        if removed in required:
            required.remove(removed)

    # 2. Description rewrites, keyed by original parameter name.
    for original, description in override.param_descriptions.items():
        prop_desc = properties.get(original)
        if isinstance(prop_desc, dict):
            prop_desc["description"] = description

    # 3. Per-parameter field overrides (description/type/enum/default/required).
    for original, param_override in override.param_overrides.items():
        prop_field = properties.get(original)
        if not isinstance(prop_field, dict):
            continue
        if param_override.description is not None:
            prop_field["description"] = param_override.description
        if param_override.type is not None:
            prop_field["type"] = param_override.type
        if param_override.enum is not None:
            prop_field["enum"] = list(param_override.enum)
        if param_override.default is None:
            prop_field.pop("default", None)
        elif param_override.default is not UNDEFINED:
            prop_field["default"] = param_override.default
        if param_override.required is True and original not in required:
            required.append(original)
        elif param_override.required is False and original in required:
            required.remove(original)

    # 4. Renames — move properties and swap required entries. `param_names`
    #    and `param_overrides[].name` are merged; the config validator
    #    guarantees they never disagree for the same parameter.
    renames = dict(override.param_names)
    for original, param_override in override.param_overrides.items():
        if param_override.name is not None:
            renames.setdefault(original, param_override.name)
    _apply_param_renames(properties, required, renames)

    # 5. Additions — append new parameters, optionally with a default used
    #    for runtime injection.
    _apply_param_additions(properties, required, override.param_additions)

    # Deduplicate while preserving order.
    schema["required"] = list(dict.fromkeys(required))


def apply_override_to_def(tool_def: ToolDefinition, override: ToolOverride) -> ToolDefinition:
    """Build the rewritten `ToolDefinition` for one override.

    Idempotent by construction: every call rewrites a fresh deep copy of the
    live schema; repeated applications produce identical output.

    Args:
        tool_def: The live tool definition (never mutated).
        override: The override to apply.

    Returns:
        A new `ToolDefinition` carrying the rewritten schema and an
        application marker in its metadata.
    """
    new_schema = copy.deepcopy(tool_def.parameters_json_schema)
    _apply_override_to_schema(new_schema, override)
    metadata = dict(tool_def.metadata or {})
    metadata[APPLIED_METADATA_KEY] = True
    return replace(
        tool_def,
        name=override.name if override.name is not None else tool_def.name,
        description=override.description
        if override.description is not None
        else tool_def.description,
        parameters_json_schema=new_schema,
        metadata=metadata,
    )


def deshare_args(override: ToolOverride, args: dict[str, Any]) -> dict[str, Any]:
    """Map model-visible call arguments back to upstream parameter names.

    Uses the same configuration as the schema rewrite: parameter renames are
    reversed, removed parameters receive their configured default (or are
    dropped), and added parameters get their default injected when the model
    omitted them.

    Args:
        override: The override that produced the model-visible schema.
        args: The validated arguments produced by the model.

    Returns:
        New argument dict ready for the upstream MCP call.
    """
    model_to_original = {new: original for original, new in override.param_names.items()}
    for original, param_override in override.param_overrides.items():
        if param_override.name is not None:
            model_to_original[param_override.name] = original
    for added, param_override in override.param_additions.items():
        visible = param_override.name if param_override.name is not None else added
        model_to_original[visible] = added

    deshared: dict[str, Any] = {
        model_to_original.get(key, key): value for key, value in args.items()
    }

    for removed in override.param_removals:
        removed_override = override.param_overrides.get(removed)
        if removed_override is not None and removed_override.default is not UNDEFINED:
            deshared[removed] = removed_override.default
        else:
            deshared.pop(removed, None)

    for added, param_override in override.param_additions.items():
        if param_override.default is not UNDEFINED and added not in deshared:
            deshared[added] = param_override.default

    return deshared


@dataclass
class SchemaOverrideToolset(WrapperToolset[AgentDepsT]):
    """Wrapper applying identity-resolved schema overrides to a toolset.

    Both tool renames and schema mutations are resolved per tool inside
    `get_tools` from source-identity metadata, because a configured
    `tool_prefix` makes inner visible names unknowable when the capability is
    constructed. Conflict semantics mirror `RenamedToolset`: a rename
    colliding with an existing tool raises `UserError`.
    """

    config: ToolSchemaOverlapConfig
    routing: dict[str, str] = field(default_factory=dict, repr=False)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """List tools with overrides applied.

        Performs the schema-dependent fail-fast validation on every listing;
        misconfiguration therefore fails agent startup rather than surfacing
        mid-conversation.

        Args:
            ctx: The current run context.

        Returns:
            Tools keyed by model-visible name.

        Raises:
            UserError: When a rename collides with an existing tool name.
            ValidationError: When the override is unsafe for the live schema
                or config keys match no listed tool.
        """
        with logfire.span("schema_override_toolset.get_tools"):
            inner_tools = await super().get_tools(ctx)
            result: dict[str, ToolsetTool[AgentDepsT]] = {}
            routing: dict[str, str] = {}
            errors: list[InitErrorDetails] = []
            identified_servers: dict[str, set[str]] = {}

            for inner_name, tool in inner_tools.items():
                identity = _read_identity(tool.tool_def)
                if identity is None:
                    # Degradation: never apply server-scoped overrides by
                    # name-guessing. Pass the tool through unchanged.
                    self._insert(result, inner_name, tool, renamed=False)
                    continue

                server_name, original_name = identity
                identified_servers.setdefault(server_name, set()).add(original_name)
                override = resolve_override(self.config, server_name, original_name)
                if override is None:
                    self._insert(result, inner_name, tool, renamed=False)
                    continue

                location = _original_location(server_name, original_name)
                errors.extend(
                    validate_schema_dependent(
                        location, tool.tool_def.parameters_json_schema, override
                    )
                )
                new_def = apply_override_to_def(tool.tool_def, override)
                final_name = new_def.name
                self.routing_check(result, final_name, original_name)
                result[final_name] = replace(tool, toolset=self, tool_def=new_def)
                routing[final_name] = inner_name

            self._check_unmatched_keys(identified_servers, errors)
            if errors:
                raise _listing_validation_error(errors)
            self.routing = routing
            return result

    @staticmethod
    def _insert(
        result: dict[str, ToolsetTool[AgentDepsT]],
        name: str,
        tool: ToolsetTool[AgentDepsT],
        *,
        renamed: bool,
    ) -> None:
        """Insert a tool into the result, mirroring `RenamedToolset` conflicts.

        Args:
            result: The tool dict being built.
            name: The visible name to insert under.
            tool: The tool to insert.
            renamed: Whether the name was produced by a rename.

        Raises:
            UserError: When the name is already taken.
        """
        if name in result:
            if renamed:
                msg = f"Renaming tool to {name!r} conflicts with an existing tool."
            else:
                msg = f"Tool name conflicts with previously renamed tool: {name!r}."
            raise UserError(msg)
        result[name] = tool

    @staticmethod
    def routing_check(
        result: dict[str, ToolsetTool[AgentDepsT]], final_name: str, original_name: str
    ) -> None:
        """Raise `UserError` when a renamed tool name collides.

        Args:
            result: The tool dict being built.
            final_name: The model-visible name after renaming.
            original_name: The raw MCP tool name, used in the error message.

        Raises:
            UserError: When the final name is already taken.
        """
        if final_name in result:
            msg = (
                f"Renaming tool {original_name!r} to {final_name!r} "
                "conflicts with an existing tool."
            )
            raise UserError(msg)

    def _check_unmatched_keys(
        self, identified_servers: dict[str, set[str]], errors: list[InitErrorDetails]
    ) -> None:
        """Fail fast on config keys that matched no listed tool.

        A healthy pipeline (at least one identified tool) treats unmatched
        keys as errors — typos must not be silently ignored. When no tool
        carries identity metadata (degraded stamping), unmatched keys are
        logged as warnings and the agent still starts.

        Args:
            identified_servers: Raw tool names seen per identified server.
            errors: Error entries to extend in the strict case.
        """
        pipeline_healthy = bool(identified_servers)
        for server_name, configured in sorted(self.config.servers.items()):
            listed = identified_servers.get(server_name)
            if listed is None:
                msg = (
                    f"configured server {server_name!r} matched no listed tool "
                    f"(servers listed: {sorted(identified_servers) or 'none'})"
                )
                if pipeline_healthy:
                    errors.append(
                        _error_entry(
                            ("servers", server_name), f"{_LOCATION_UNMATCHED_SERVER}: {msg}"
                        )
                    )
                else:
                    logger.warning("tool-schema-overlap: %s", msg)
                continue
            for tool_name in sorted(configured):
                if tool_name not in listed:
                    msg = (
                        f"configured tool {tool_name!r} does not exist on server {server_name!r} "
                        f"(tools listed: {sorted(listed)})"
                    )
                    errors.append(
                        _error_entry(
                            ("servers", server_name, tool_name),
                            f"{_LOCATION_UNMATCHED_TOOL}: {msg}",
                        )
                    )
        for tool_name in sorted(self.config.global_overrides):
            if not any(tool_name in listed for listed in identified_servers.values()):
                msg = f"global override for {tool_name!r} matched no listed tool on any server"
                if pipeline_healthy:
                    errors.append(
                        _error_entry(
                            ("global_overrides", tool_name), f"{_LOCATION_UNMATCHED_TOOL}: {msg}"
                        )
                    )
                else:
                    logger.warning("tool-schema-overlap: %s", msg)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Route a model-visible tool call back to the inner visible name.

        Parameter-level desharing happens in the capability's
        `wrap_tool_execute` (which runs before toolset dispatch); here only
        the tool name is reversed.

        Args:
            name: The model-visible tool name.
            tool_args: Validated arguments (still model-visible names).
            ctx: The current run context.
            tool: The tool as listed by `get_tools`.

        Returns:
            The upstream tool result.
        """
        inner_name = self.routing.get(name, name)
        if inner_name != name:
            ctx = replace(ctx, tool_name=inner_name)
            tool = replace(tool, tool_def=replace(tool.tool_def, name=inner_name))
        return await super().call_tool(inner_name, tool_args, ctx, tool)


@dataclass(kw_only=True)
class ToolSchemaOverlapCapability(AbstractCapability[Any]):
    """Adapts MCP tool schemas for the model without touching the server.

    Constructed from raw YAML kwargs by the entry-point capability registry:
    `type: tool-schema-overlap` with `args: {servers: ..., global_overrides: ...}`.

    Attributes:
        servers: Per-server overrides,
            `{server_name: {raw_mcp_tool_name: override}}`.
        global_overrides: Overrides applied to tools of any server, keyed by
            raw MCP tool name.
    """

    servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_overrides: dict[str, Any] = field(default_factory=dict)
    _parsed_config: ToolSchemaOverlapConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the raw config eagerly.

        Raises:
            ValidationError: When the configuration is internally
                inconsistent (rename collisions, enum/default mismatches).
        """
        self._parsed_config = ToolSchemaOverlapConfig.model_validate({
            "servers": self.servers,
            "global_overrides": self.global_overrides,
        })

    @property
    def config(self) -> ToolSchemaOverlapConfig:
        """The validated configuration (single source of truth)."""
        return self._parsed_config

    def get_ordering(self) -> CapabilityOrdering | None:
        """Order schema adaptation before the display layer.

        Returns:
            Ordering declaring that `ToolDisplayCapability` wraps this
            capability, so display renames see model-visible tool names.
        """
        return CapabilityOrdering(wrapped_by=[ToolDisplayCapability])

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
        """Wrap the assembled toolset with the schema-override wrapper.

        Args:
            toolset: The fully assembled agent toolset.

        Returns:
            A `SchemaOverrideToolset` when any override is configured,
            otherwise `None` (zero overhead when unused).
        """
        if not self._parsed_config.servers and not self._parsed_config.global_overrides:
            return None
        return SchemaOverrideToolset[Any](wrapped=toolset, config=self._parsed_config)

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Supplementary per-step schema application.

        The `SchemaOverrideToolset` wrapper is the primary application point
        and marks rewritten definitions; this hook re-applies overrides only
        to definitions that bypassed the wrapper, keeping repeated passes
        idempotent.

        Args:
            ctx: The current run context.
            tool_defs: Function tool definitions for this step.

        Returns:
            The (possibly rewritten) tool definitions.
        """
        if not self._parsed_config.servers and not self._parsed_config.global_overrides:
            return tool_defs
        result: list[ToolDefinition] = []
        for tool_def in tool_defs:
            metadata = tool_def.metadata
            if metadata and metadata.get(APPLIED_METADATA_KEY):
                result.append(tool_def)
                continue
            identity = _read_identity(tool_def)
            if identity is None:
                result.append(tool_def)
                continue
            override = resolve_override(self._parsed_config, identity[0], identity[1])
            if override is None:
                result.append(tool_def)
                continue
            result.append(apply_override_to_def(tool_def, override))
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """Reverse-map model args to upstream names before execution.

        Args:
            ctx: The current run context.
            call: The model's tool call part.
            tool_def: The tool definition as presented to the model.
            args: Validated model arguments (model-visible names).
            handler: Runs the tool with the (potentially remapped) args.

        Returns:
            The upstream tool result.
        """
        identity = _read_identity(tool_def)
        override = (
            resolve_override(self._parsed_config, identity[0], identity[1]) if identity else None
        )
        if override is not None:
            args = deshare_args(override, args)
        with logfire.span("tool_schema_overlap.wrap_tool_execute", tool_name=tool_def.name):
            return await handler(args)
