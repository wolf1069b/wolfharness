"""ACP-based input provider for agent interactions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, assert_never
import webbrowser

from mcp import types

from acp import AllowedOutcome, DeniedOutcome, PermissionOption
from acp.utils import DEFAULT_PERMISSION_OPTIONS
from wolfharness.log import get_logger
from wolfharness.ui.base import InputProvider
from wolfharness.ui.elicitation import normalize_elicit_content


if TYPE_CHECKING:
    from acp import RequestPermissionResponse
    from acp.schema.elicitation import ElicitationCreateResponse
    from wolfharness import AgentContext
    from wolfharness.agents.context import ConfirmationResult
    from wolfharness_server.acp_server.session import ACPSession

logger = get_logger(__name__)


def _create_enum_elicitation_options(schema: dict[str, Any]) -> list[PermissionOption] | None:
    """Create permission options for enum elicitation.

    max 4 options (not more ids available)
    """
    enum_values = schema.get("enum", [])
    enum_names = schema.get("enumNames", enum_values)  # Use enumNames if available
    # Limit to 3 choices + cancel to keep ACP UI reasonable
    if len(enum_values) > 3:  # noqa: PLR2004
        logger.warning("Enum truncated for UI", enum_count=len(enum_values), showing_count=3)
        enum_values = enum_values[:3]
        enum_names = enum_names[:3]

    if not enum_values:
        return None

    options = []
    for i, (value, name) in enumerate(zip(enum_values, enum_names, strict=False)):
        opt_id = f"enum_{i}_{value}"
        option = PermissionOption(option_id=opt_id, name=str(name), kind="allow_once")
        options.append(option)
    # Add cancel option
    options.append(PermissionOption(option_id="cancel", name="Cancel", kind="reject_always"))

    return options


def _is_boolean_schema(schema: dict[str, Any]) -> bool:
    """Check if the elicitation schema is requesting a boolean value."""
    # Direct boolean type
    match schema:
        case {"type": "boolean"}:
            return True
        case {"type": "object", "properties": dict() as props} if len(props) == 1:
            prop_schema = next(iter(props.values()))
            return bool(prop_schema.get("type") == "boolean")
        case _:
            return False


def _is_enum_schema(schema: dict[str, Any]) -> bool:
    """Check if the elicitation schema is requesting an enum value."""
    return schema.get("type") == "string" and "enum" in schema


def _is_oneof_schema(schema: dict[str, Any]) -> bool:
    """Check if the elicitation schema uses oneOf with const entries."""
    if schema.get("type") != "string":
        return False
    one_of = schema.get("oneOf")
    if not isinstance(one_of, list) or not one_of:
        return False
    # At least one entry must have a const field to be considered valid
    return any(isinstance(entry, dict) and "const" in entry for entry in one_of)


def _is_array_enum_schema(schema: dict[str, Any]) -> bool:
    """Check if the elicitation schema is an array with enum items."""
    if schema.get("type") != "array":
        return False
    items = schema.get("items")
    if not isinstance(items, dict):
        return False
    return "enum" in items and isinstance(items.get("enum"), list) and bool(items["enum"])


def _unwrap_object_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """If schema is an object wrapper, extract the first elicitable property.

    xeno-agent generates ``{"type": "object", "properties": {"q0": {...}}}``
    for multi-question schemas.  ACP ``request_permission`` can only present
    a single question, so we pick the first property that looks like an
    enum/oneOf/array-enum and use its schema for option generation.
    """
    if schema.get("type") != "object":
        return None
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    for prop_schema in props.values():
        if not isinstance(prop_schema, dict):
            continue
        if (
            _is_enum_schema(prop_schema)
            or _is_oneof_schema(prop_schema)
            or _is_array_enum_schema(prop_schema)
        ):
            return prop_schema
    return None


def _create_boolean_elicitation_options() -> list[PermissionOption]:
    """Create permission options for boolean elicitation (Yes/No)."""
    return [
        PermissionOption(option_id="true", name="Yes", kind="allow_once"),
        PermissionOption(option_id="false", name="No", kind="reject_once"),
        PermissionOption(option_id="cancel", name="Cancel", kind="reject_always"),
    ]


def _create_oneof_elicitation_options(schema: dict[str, Any]) -> list[PermissionOption] | None:
    """Create permission options from a oneOf schema.

    Uses ``const`` as the option value and ``title`` as the display label.
    Entries without ``const`` are skipped. Falls back to generic options
    if no valid entries are found.
    """
    one_of = schema.get("oneOf", [])
    if not isinstance(one_of, list):
        return None

    options: list[PermissionOption] = []
    for entry in one_of:
        if not isinstance(entry, dict):
            continue
        const_val = entry.get("const")
        if const_val is None:
            continue
        label = entry.get("title") or str(const_val)
        opt_id = f"oneof_{len(options)}_{const_val}"
        options.append(PermissionOption(option_id=opt_id, name=str(label), kind="allow_once"))

    if not options:
        return None

    options.append(PermissionOption(option_id="cancel", name="Cancel", kind="reject_always"))
    return options


def _create_array_enum_elicitation_options(schema: dict[str, Any]) -> list[PermissionOption] | None:
    """Create permission options from an array schema with enum items.

    Uses ``items.enum`` values as options and ``items.x-option-descriptions``
    for display labels when available.
    """
    items = schema.get("items", {})
    if not isinstance(items, dict):
        return None

    enum_values = items.get("enum", [])
    if not isinstance(enum_values, list) or not enum_values:
        return None

    descriptions = items.get("x-option-descriptions", {})
    if not isinstance(descriptions, dict):
        descriptions = {}

    options: list[PermissionOption] = []
    for value in enum_values:
        label = descriptions.get(str(value), str(value))
        opt_id = f"array_enum_{len(options)}_{value}"
        options.append(PermissionOption(option_id=opt_id, name=str(label), kind="allow_once"))

    if not options:
        return None

    options.append(PermissionOption(option_id="cancel", name="Cancel", kind="reject_always"))
    return options


class ACPInputProvider(InputProvider):
    """Input provider that uses ACP session for user interactions.

    This provider enables tool confirmation and elicitation requests
    through the ACP protocol, allowing clients to interact with agents
    for permission requests and additional input.
    """

    def __init__(self, session: ACPSession) -> None:
        """Initialize ACP input provider.

        Args:
            session: Active ACP session for handling requests
        """
        self.session = session
        self._tool_approvals: dict[str, Literal["allow_always", "reject_always"]] = {}

    @property
    def supports_durable_elicitation(self) -> bool:
        """Whether this provider supports durable (checkpointable) elicitation.

        Checks the session's ``checkpoint_enabled`` flag at runtime.
        Returns False when checkpointing is not enabled for this session.
        """
        return self.session.checkpoint_enabled

    async def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool_description: str = "",
    ) -> ConfirmationResult:
        """Get tool execution confirmation via ACP request permission.

        Uses the ACP session's request_permission mechanism to ask
        the client for confirmation before executing the tool.

        Args:
            context: Current node context with tool_name, tool_call_id, tool_input set
            tool_description: Human-readable description of the tool

        Returns:
            Confirmation result indicating whether to allow, skip, or abort
        """
        tool_name = context.tool_name or "unknown"
        args = context.tool_input
        from acp.utils import generate_tool_title

        try:
            # Check if we have a standing approval/rejection for this tool
            if decision := self._tool_approvals.get(tool_name):
                logger.debug("Get tool confirmation", tool_name=tool_name, reason=decision)
                match decision:
                    case "allow_always":
                        return "allow"
                    case "reject_always":
                        return "skip"
                    case _ as unreachable:
                        assert_never(unreachable)

            # Create a descriptive title for the permission request
            # args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            # Use the tool_call_id from context - this must match the UI tool call
            tc_id = context.tool_call_id
            if not tc_id:
                msg = f"No tool_call_id in context for tool {tool_name!r}. "
                logger.error(msg)
                raise RuntimeError(msg)  # noqa: TRY301
            logger.debug("Requesting permission", tool_name=tool_name, tool_call_id=tc_id)
            # Note: We no longer send tool_call_start/progress notifications here.
            # The streaming loop (via ACPEventConverter) already sends ToolCallStart
            # before the permission callback is invoked, so Zed already knows about
            # the tool call. Sending duplicates was causing sync issues.

            title = generate_tool_title(tool_name, args)
            response = await self.session.requests.request_permission(
                tool_call_id=tc_id,
                title=title,
                raw_input=args,
                options=DEFAULT_PERMISSION_OPTIONS,
            )
            logger.info(
                "Permission response received",
                tool_name=tool_name,
                outcome=response.outcome.outcome,
            )
            # Map ACP permission response to our confirmation result
            match response.outcome:
                case AllowedOutcome(option_id=option_id):
                    return self._handle_permission_response(option_id, tool_name)
                case DeniedOutcome():
                    return "skip"
                case _ as outcome:
                    logger.warning("Unexpected permission outcome", outcome=outcome)

        except Exception:
            logger.exception("Failed to get tool confirmation")
            # Default to abort on error to be safe
            return "abort_run"
        else:
            return "skip"  # Default to skip for unknown outcomes

    def _handle_permission_response(self, option_id: str, tool_name: str) -> ConfirmationResult:
        """Handle permission response and update tool approval state."""
        # Normalize to hyphen format for matching
        normalized = option_id.replace("_", "-")
        logger.info(
            "Handling permission response",
            option_id=option_id,
            normalized=normalized,
            tool_name=tool_name,
        )
        match normalized:
            case "allow-once":
                return "allow"
            case "allow-always":
                self._tool_approvals[tool_name] = "allow_always"
                logger.info("Tool approval set", tool_name=tool_name, approval="allow_always")
                return "allow"
            case "reject-once" | "deny-once":
                return "skip"
            case "reject-always" | "deny-always":
                self._tool_approvals[tool_name] = "reject_always"
                logger.info("Tool approval set", tool_name=tool_name, approval="reject_always")
                return "skip"
            case _:
                logger.warning("Unknown permission option", option_id=option_id)
                return "abort_run"

    def _client_supports_elicitation(self, mode: Literal["form", "url"]) -> bool:
        """Check if the client supports the given elicitation mode."""
        caps = self.session.client_capabilities
        if caps.elicitation is None:
            return False
        logger.info("Checking elicitation capability", mode=mode, capabilities=caps.elicitation)
        match mode:
            case "form":
                return bool(caps.elicitation.form)
            case "url":
                return bool(caps.elicitation.url)

    @staticmethod
    def _map_elicitation_create_response(
        response: ElicitationCreateResponse,
        mode: Literal["form", "url"] = "form",
    ) -> types.ElicitResult:
        """Map elicitation/create response action to ElicitResult.

        For URL mode, ``content`` is omitted per the MCP elicitation spec.
        For form mode, ``content`` is normalized to the primitive-only schema
        required by ``mcp.types.ElicitResult``.
        """
        match response.action:
            case "accept":
                if mode == "url":
                    return types.ElicitResult(action="accept")
                return types.ElicitResult(
                    action="accept",
                    content=normalize_elicit_content(response.content) or {},
                )
            case "decline":
                return types.ElicitResult(action="decline")
            case "cancel":
                return types.ElicitResult(action="cancel")
            case _ as unreachable:
                assert_never(unreachable)  # ty:ignore[type-assertion-failure]

    async def get_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request with capability-gated dual path.

        When the client declares the ``elicitation`` capability (form or url mode), uses the
        native ``elicitation/create`` protocol method.  Otherwise falls back to
        the legacy ``request_permission`` approach for backward compatibility.

        Args:
            params: MCP elicit request parameters

        Returns:
            Elicit result with user's response or error data
        """
        try:
            if isinstance(params, types.ElicitRequestURLParams):
                return await self._get_url_elicitation(params)
            return await self._get_form_elicitation(params)
        except asyncio.CancelledError:
            logger.debug("Elicitation cancelled by user")
            return types.ElicitResult(action="cancel")
        except Exception as e:
            logger.exception("Failed to handle elicitation")
            return types.ErrorData(code=types.INTERNAL_ERROR, message=f"Elicitation failed: {e}")

    async def _get_url_elicitation(
        self,
        params: types.ElicitRequestURLParams,
    ) -> types.ElicitResult:
        """Handle URL-mode elicitation (OAuth, credentials, payments).

        Uses ``elicitation/create`` when the client supports it, otherwise
        falls back to ``request_permission``.
        """
        elicit_id = params.elicitationId
        logger.info(
            "URL elicitation request",
            message=params.message,
            url=params.url,
            elicitation_id=elicit_id,
        )

        if self._client_supports_elicitation("url"):
            # TODO: URL-mode elicitation currently returns the immediate response
            # from ``elicitation/create``. For full URL flows where the user
            # completes an external action (OAuth, payments), the result arrives
            # asynchronously via ``ElicitationCompleteNotification``. Implementing
            # the async Future + notification registry to wait for completion is
            # deferred to a future PR.
            response = await self.session.requests.elicitation_create(
                message=params.message,
                mode="url",
                requested_schema={"type": "object"},
                url=params.url,
                elicitation_id=elicit_id,
            )
            return self._map_elicitation_create_response(response, mode="url")

        # Fallback: request_permission
        tool_call_id = f"elicit_url_{elicit_id}"
        title = f"URL Authorization: {params.message}"
        url_options = [
            PermissionOption(option_id="accept", name="Open URL", kind="allow_once"),
            PermissionOption(option_id="decline", name="Decline", kind="reject_once"),
        ]
        perm_response = await self.session.requests.request_permission(
            tool_call_id=tool_call_id,
            title=title,
            options=url_options,
        )
        match perm_response.outcome:
            case AllowedOutcome(option_id="accept"):
                webbrowser.open(params.url)
                return types.ElicitResult(action="accept")
            case AllowedOutcome():
                return types.ElicitResult(action="decline")
            case DeniedOutcome():
                return types.ElicitResult(action="cancel")
            case _ as unreachable:
                assert_never(unreachable)  # ty:ignore[type-assertion-failure]

    async def _get_form_elicitation(
        self,
        params: types.ElicitRequestFormParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Handle form-mode elicitation with schema support.

        Uses ``elicitation/create`` when the client supports it, otherwise
        falls back to ``request_permission`` with boolean/enum/generic handling.
        """
        schema = params.requestedSchema
        logger.info("Elicitation request", message=params.message, schema=schema)

        if self._client_supports_elicitation("form"):
            response = await self.session.requests.elicitation_create(
                message=params.message,
                mode="form",
                requested_schema=schema,
            )
            return self._map_elicitation_create_response(response, mode="form")
        return await self._get_form_elicitation_fallback(params, schema)

    async def _get_form_elicitation_fallback(
        self,
        params: types.ElicitRequestFormParams,
        schema: dict[str, Any],
    ) -> types.ElicitResult | types.ErrorData:
        """Fallback path using request_permission for form elicitation."""
        tool_call_id = f"elicit_{hash(params.message)}"
        title = f"Elicitation: {params.message}"

        # xeno-agent wraps questions in {"type": "object", "properties": {"q0": {...}}}.
        # request_permission can only present a single question, so unwrap the first
        # elicitable property and remember the original key to wrap the result back.
        effective_schema, object_key = self._resolve_effective_schema(schema)
        result = await self._dispatch_elicitation_by_schema(effective_schema, tool_call_id, title)
        return self._wrap_object_result(result, object_key)

    def _resolve_effective_schema(
        self, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """If schema is an object wrapper, extract the first elicitable property."""
        if (
            schema.get("type") == "object"
            and not _is_boolean_schema(schema)
            and not _is_enum_schema(schema)
            and not _is_oneof_schema(schema)
            and not _is_array_enum_schema(schema)
            and (unwrapped := _unwrap_object_schema(schema))
        ):
            for key, prop in schema.get("properties", {}).items():
                if prop is unwrapped:
                    return unwrapped, key
        return schema, None

    async def _dispatch_elicitation_by_schema(
        self,
        schema: dict[str, Any],
        tool_call_id: str,
        title: str,
    ) -> types.ElicitResult | types.ErrorData:
        """Route to the appropriate handler based on schema type."""
        result: types.ElicitResult | types.ErrorData

        if _is_boolean_schema(schema):
            options = _create_boolean_elicitation_options()
            perm_response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=options,
            )
            result = self._handle_boolean_elicitation_response(perm_response, schema)
        elif _is_enum_schema(schema) and (enum_options := _create_enum_elicitation_options(schema)):
            perm_response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=enum_options,
            )
            result = _handle_enum_elicitation_response(perm_response, schema)
        elif _is_oneof_schema(schema) and (
            oneof_options := _create_oneof_elicitation_options(schema)
        ):
            perm_response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=oneof_options,
            )
            result = _handle_oneof_elicitation_response(perm_response, schema)
        elif _is_array_enum_schema(schema) and (
            array_options := _create_array_enum_elicitation_options(schema)
        ):
            perm_response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=array_options,
            )
            result = _handle_array_enum_elicitation_response(perm_response, schema)
        else:
            # Generic fallback: Accept/Decline
            generic_options = [
                PermissionOption(option_id="accept", name="Accept", kind="allow_once"),
                PermissionOption(option_id="decline", name="Decline", kind="reject_once"),
            ]
            perm_response = await self.session.requests.request_permission(
                tool_call_id=tool_call_id,
                title=title,
                options=generic_options,
            )

            match perm_response.outcome:
                case AllowedOutcome(option_id="accept"):
                    result = types.ElicitResult(action="accept", content={})
                case AllowedOutcome():
                    result = types.ElicitResult(action="decline")
                case DeniedOutcome():
                    result = types.ElicitResult(action="cancel")
                case _ as unreachable:
                    assert_never(unreachable)  # ty:ignore[type-assertion-failure]

        return result

    @staticmethod
    def _wrap_object_result(
        result: types.ElicitResult | types.ErrorData,
        object_key: str | None,
    ) -> types.ElicitResult | types.ErrorData:
        """If we unwrapped an object property, wrap the result content back."""
        if (
            object_key is not None
            and isinstance(result, types.ElicitResult)
            and result.action == "accept"
        ):
            inner_content = result.content if isinstance(result.content, dict) else {}
            value = inner_content.get("value")
            return types.ElicitResult(action="accept", content={object_key: value})
        return result

    def _handle_boolean_elicitation_response(
        self, response: RequestPermissionResponse, schema: dict[str, Any]
    ) -> types.ElicitResult | types.ErrorData:
        """Handle ACP response for boolean elicitation."""
        match response.outcome:
            case AllowedOutcome(option_id="false" | "true" as option_id):
                # Check if we need to wrap in object structure
                if schema.get("type") == "object":
                    properties = schema.get("properties", {})
                    if len(properties) == 1:
                        prop_name = next(iter(properties.keys()))
                        val = option_id == "true"
                        return types.ElicitResult(action="accept", content={prop_name: val})
                return types.ElicitResult(action="accept", content={"value": True})
            case AllowedOutcome(option_id="cancel"):
                return types.ElicitResult(action="cancel")
            case DeniedOutcome():
                return types.ElicitResult(action="cancel")
            case _:
                return types.ElicitResult(action="cancel")

    def clear_tool_approvals(self) -> None:
        """Clear all stored tool approval decisions.

        This resets the "allow_always" and "reject_always" states,
        so tools will ask for permission again.
        """
        approval_count = len(self._tool_approvals)
        self._tool_approvals.clear()
        logger.info("Cleared tool approval decisions", count=approval_count)

    def get_tool_approval_state(self) -> dict[str, Literal["allow_always", "reject_always"]]:
        """Get current tool approval state for debugging/inspection.

        Returns:
            Dictionary mapping tool names to their approval state
            ("allow_always" or "reject_always")
        """
        return self._tool_approvals.copy()


def _handle_enum_elicitation_response(
    response: RequestPermissionResponse, schema: dict[str, Any]
) -> types.ElicitResult | types.ErrorData:
    """Handle ACP response for enum elicitation."""
    from mcp import types

    match response.outcome:
        case AllowedOutcome(option_id="cancel"):
            return types.ElicitResult(action="cancel")
        case AllowedOutcome(option_id=option_id) if option_id.startswith("enum_"):
            # Extract enum value from option_id format: "enum_{index}_{value}"
            try:
                parts = option_id.split("_", 2)  # Split into ["enum", index, value]
                if len(parts) >= 3:  # noqa: PLR2004
                    enum_index = int(parts[1])
                    enum_values = schema.get("enum", [])
                    if 0 <= enum_index < len(enum_values):
                        selected_value = enum_values[enum_index]
                        return types.ElicitResult(
                            action="accept", content={"value": selected_value}
                        )
            except (ValueError, IndexError):
                pass
            logger.warning("Failed to parse enum option_id", option_id=option_id)
            return types.ElicitResult(action="cancel")
        case AllowedOutcome(option_id=option_id):
            logger.warning("Failed to parse enum option_id", option_id=option_id)
            return types.ElicitResult(action="cancel")
        case DeniedOutcome():
            return types.ElicitResult(action="cancel")
        case _ as unreachable:
            assert_never(unreachable)


def _handle_oneof_elicitation_response(
    response: RequestPermissionResponse, schema: dict[str, Any]
) -> types.ElicitResult | types.ErrorData:
    """Handle ACP response for oneOf elicitation.

    Maps the selected option_id back to the ``const`` value from the
    corresponding oneOf entry.
    """
    from mcp import types

    match response.outcome:
        case AllowedOutcome(option_id="cancel"):
            return types.ElicitResult(action="cancel")
        case AllowedOutcome(option_id=option_id) if option_id.startswith("oneof_"):
            try:
                parts = option_id.split("_", 2)  # ["oneof", index, value]
                if len(parts) >= 3:  # noqa: PLR2004
                    oneof_index = int(parts[1])
                    one_of = schema.get("oneOf", [])
                    if isinstance(one_of, list) and 0 <= oneof_index < len(one_of):
                        entry = one_of[oneof_index]
                        if isinstance(entry, dict) and "const" in entry:
                            selected_value = entry["const"]
                            return types.ElicitResult(
                                action="accept", content={"value": selected_value}
                            )
            except (ValueError, IndexError):
                pass
            logger.warning("Failed to parse oneof option_id", option_id=option_id)
            return types.ElicitResult(action="cancel")
        case AllowedOutcome(option_id=option_id):
            logger.warning("Failed to parse oneof option_id", option_id=option_id)
            return types.ElicitResult(action="cancel")
        case DeniedOutcome():
            return types.ElicitResult(action="cancel")
        case _ as unreachable:
            assert_never(unreachable)


def _handle_array_enum_elicitation_response(
    response: RequestPermissionResponse, schema: dict[str, Any]
) -> types.ElicitResult | types.ErrorData:
    """Handle ACP response for array-enum elicitation.

    Returns the selected value as a single-element list since
    request_permission only supports single selection.
    """
    from mcp import types

    match response.outcome:
        case AllowedOutcome(option_id="cancel"):
            return types.ElicitResult(action="cancel")
        case AllowedOutcome(option_id=option_id) if option_id.startswith("array_enum_"):
            try:
                parts = option_id.split("_", 3)  # ["array", "enum", index, value]
                if len(parts) >= 4:  # noqa: PLR2004
                    enum_index = int(parts[2])
                    items = schema.get("items", {})
                    if isinstance(items, dict):
                        enum_values = items.get("enum", [])
                        if isinstance(enum_values, list) and 0 <= enum_index < len(enum_values):
                            selected_value = enum_values[enum_index]
                            return types.ElicitResult(
                                action="accept", content={"value": [selected_value]}
                            )
            except (ValueError, IndexError):
                pass
            logger.warning("Failed to parse array_enum option_id", option_id=option_id)
            return types.ElicitResult(action="cancel")
        case AllowedOutcome(option_id=option_id):
            logger.warning("Failed to parse array_enum option_id", option_id=option_id)
            return types.ElicitResult(action="cancel")
        case DeniedOutcome():
            return types.ElicitResult(action="cancel")
        case _ as unreachable:
            assert_never(unreachable)
