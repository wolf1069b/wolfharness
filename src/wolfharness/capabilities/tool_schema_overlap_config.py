"""Configuration model for the tool schema overlap capability.

Defines the declarative configuration used to adapt MCP tool schemas at the
capability level: tool and parameter renames, description rewrites, and
parameter schema mutations (additions, removals, type/enum/required/default
overrides).

Config keys are always raw MCP tool names, exactly as returned by the MCP
server's `list_tools` — never prefixed (`tool_prefix`) or display names.
Resolution against live tools happens via source identity metadata, so the
same configuration works identically with or without a `tool_prefix`.

Validation is split in two stages:

- Construction time (this module): pure-config self-consistency checks that
  require no live tool schema (rename collisions, enum/default mismatches).
- First tool listing (agent startup): schema-dependent checks that need the
  live `required` lists, performed by the wrapping toolset.
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyStr = Annotated[str, Field(min_length=1)]

# Best-effort Python-type expectations for JSON schema types. Unknown types
# (custom formats, vendor extensions) are left to the MCP server to validate.
_JSON_TYPE_EXPECTATIONS: Final[dict[str, tuple[type[Any], ...]]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

# Keys used to stamp/resolve tool source identity on ToolDefinition.metadata.
SERVER_NAME_METADATA_KEY: Final[str] = "server_name"
ORIGINAL_TOOL_NAME_METADATA_KEY: Final[str] = "original_mcp_tool_name"


class _UndefinedType:
    """Sentinel distinguishing "no value configured" from an explicit `None`.

    Used by `ParamOverride.default` so that an explicit `default: null` in
    YAML means "remove the existing default" while omitting `default` means
    "leave the schema unchanged".
    """

    _instance: _UndefinedType | None = None

    def __new__(cls) -> _UndefinedType:  # noqa: PYI034 - cached singleton returns the concrete class
        if _UndefinedType._instance is None:
            _UndefinedType._instance = super().__new__(cls)
        return _UndefinedType._instance

    def __copy__(self) -> _UndefinedType:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _UndefinedType:
        return self

    def __eq__(self, other: object) -> bool:
        return other is self

    def __hash__(self) -> int:
        return hash(_UndefinedType)

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNDEFINED"


UNDEFINED: Final[_UndefinedType] = _UndefinedType()


def _default_matches_json_type(value: Any, json_type: str) -> bool:
    """Check whether a Python value is plausible for a JSON schema type.

    Args:
        value: The configured default or enum value.
        json_type: The JSON schema type name (e.g. "string", "integer").

    Returns:
        True when the value fits the type, when the type is unknown, or when
        the value would need schema-level validation this helper cannot do.
    """
    expected = _JSON_TYPE_EXPECTATIONS.get(json_type)
    if expected is None:
        return True
    if json_type in ("integer", "number") and isinstance(value, bool):
        # bool subclasses int in Python, but JSON booleans are not numbers.
        return False
    return isinstance(value, expected)


class ParamOverride(BaseModel):
    """Schema adaptation for a single tool parameter.

    Attributes:
        name: New parameter name. The model sees this name; the runtime
            desharing maps it back to the original before the MCP call.
        description: New parameter description.
        type: Override the JSON schema type (e.g. "string", "integer").
        enum: Constrain the parameter to these values.
        required: Override required/optional status.
        default: Default value. `None` removes an existing default;
            `UNDEFINED` (the default) leaves the schema unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr | None = None
    description: str | None = None
    type: str | None = None
    enum: list[Any] | None = None
    required: bool | None = None
    default: Any = UNDEFINED

    @model_validator(mode="after")
    def _validate_enum_default_consistency(self) -> Self:
        # An explicit `default: None` is the removal marker (strip an existing
        # schema default at rewrite time), not a value sent to the server, so
        # it is exempt from the enum/type consistency checks.
        if (
            self.enum is not None
            and self.default is not UNDEFINED
            and self.default is not None
            and self.default not in self.enum
        ):
            msg = (
                f"default value {self.default!r} is not one of the configured"
                f" enum values {self.enum!r}"
            )
            raise ValueError(msg)
        if (
            self.type is not None
            and self.default is not UNDEFINED
            and self.default is not None
            and not _default_matches_json_type(self.default, self.type)
        ):
            msg = f"default value {self.default!r} does not match the configured type {self.type!r}"
            raise ValueError(msg)
        if self.type is not None and self.enum is not None:
            for value in self.enum:
                if not _default_matches_json_type(value, self.type):
                    msg = f"enum value {value!r} does not match the configured type {self.type!r}"
                    raise ValueError(msg)
        return self


class ToolOverride(BaseModel):
    """Schema adaptation for one MCP tool, keyed by raw MCP tool name.

    Attributes:
        name: New tool name. The model calls the tool by this name; the
            framework resolves it back to the raw MCP tool name.
        description: New tool description.
        param_names: Parameter renames, `{original_name: new_name}`.
        param_descriptions: Parameter description rewrites,
            `{param_name: new_description}`, keyed by original name.
        param_overrides: Per-parameter schema overrides keyed by original
            parameter name. Also applies to removed parameters (e.g. to supply
            a default that is injected when a required parameter is removed).
        param_additions: New parameters to add, `{new_param_name: ParamOverride}`.
        param_removals: Parameters to hide from the model. Optional parameters
            are always removable; required parameters need a configured default.
    """

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr | None = None
    description: str | None = None
    param_names: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    param_descriptions: dict[NonEmptyStr, str] = Field(default_factory=dict)
    param_overrides: dict[NonEmptyStr, ParamOverride] = Field(default_factory=dict)
    param_additions: dict[NonEmptyStr, ParamOverride] = Field(default_factory=dict)
    param_removals: set[NonEmptyStr] = Field(default_factory=set)

    @model_validator(mode="after")
    def _validate_param_consistency(self) -> Self:
        # Every configured source of a model-visible parameter name must
        # produce a unique name, otherwise the rewritten schema (and the
        # reverse desharing at call time) would be ambiguous.
        produced: dict[str, str] = {}

        def _record(new_name: str, origin: str) -> None:
            existing = produced.get(new_name)
            if existing is not None and existing != origin:
                msg = f"parameter name {new_name!r} is produced by both {existing} and {origin}"
                raise ValueError(msg)
            produced[new_name] = origin

        for original, new in self.param_names.items():
            _record(new, f"param_names[{original!r}]")
        for original, override in self.param_overrides.items():
            if override.name is not None:
                _record(override.name, f"param_overrides[{original!r}].name")
        for added, addition in self.param_additions.items():
            # The model sees the visible name (po.name if set, else the
            # addition key), so uniqueness is checked on the visible name.
            visible = addition.name if addition.name is not None else added
            _record(visible, f"param_additions[{added!r}]")

        # The same original parameter must not receive two different renames.
        for original, new in self.param_names.items():
            renamed_override = self.param_overrides.get(original)
            if (
                renamed_override is not None
                and renamed_override.name is not None
                and renamed_override.name != new
            ):
                msg = (
                    f"parameter {original!r} is renamed to both {new!r} (param_names) "
                    f"and {renamed_override.name!r} (param_overrides)"
                )
                raise ValueError(msg)

        # Removal contradicts renaming or adding the same parameter. Removal
        # combined with param_overrides is intentional (default injection).
        renamed_removed = self.param_removals & self.param_names.keys()
        if renamed_removed:
            msg = f"parameters {sorted(renamed_removed)!r} are both removed and renamed"
            raise ValueError(msg)
        added_removed = self.param_removals & self.param_additions.keys()
        if added_removed:
            msg = f"parameters {sorted(added_removed)!r} are both removed and added"
            raise ValueError(msg)

        return self


class ToolSchemaOverlapConfig(BaseModel):
    """Top-level configuration for `ToolSchemaOverlapCapability`.

    Attributes:
        servers: Per-server overrides,
            `{server_name: {raw_mcp_tool_name: ToolOverride}}`. Server-scoped
            entries take precedence over `global_overrides` for that server.
        global_overrides: Overrides applied to tools of any server, keyed by
            raw MCP tool name.
    """

    model_config = ConfigDict(extra="forbid")

    servers: dict[NonEmptyStr, dict[NonEmptyStr, ToolOverride]] = Field(default_factory=dict)
    global_overrides: dict[NonEmptyStr, ToolOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_rename_uniqueness(self) -> Self:
        # Two different source tools renamed to the same target collide
        # whenever both tools are listed. Same-tool shadowing (a
        # server-scoped entry overriding the global entry for the same tool
        # name) is not a config error; multi-server collisions between
        # identically named tools surface as UserError at the first tool
        # listing.
        targets: dict[str, tuple[str, str]] = {}

        def _record(target: str, origin: str, tool_name: str) -> None:
            existing = targets.get(target)
            if existing is not None and existing[0] != tool_name:
                msg = f"tool rename target {target!r} is used by both {existing[1]} and {origin}"
                raise ValueError(msg)
            targets[target] = (tool_name, origin)

        for server, tools in self.servers.items():
            for tool_name, override in tools.items():
                if override.name is not None:
                    _record(override.name, f"servers.{server}.{tool_name}", tool_name)
        for tool_name, override in self.global_overrides.items():
            if override.name is not None:
                _record(override.name, f"global_overrides.{tool_name}", tool_name)

        return self
