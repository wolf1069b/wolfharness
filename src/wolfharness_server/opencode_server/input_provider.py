"""OpenCode-based input provider for agent interactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp import types

from wolfharness.log import get_logger
from wolfharness.ui.base import InputProvider
from wolfharness_server.opencode_server.models import (
    PermissionAskedProperties,
    PermissionRequestEvent,
    PermissionToolInfo,
)


if TYPE_CHECKING:
    from wolfharness.agents.context import AgentContext, ConfirmationResult
    from wolfharness_server.opencode_server.models import PermissionReply
    from wolfharness_server.opencode_server.models.question import QuestionInfo, QuestionRequest
    from wolfharness_server.opencode_server.state import ServerState

logger = get_logger(__name__)

#: Default timeout (in seconds) for elicitation questions.
#: If the user does not respond within this period, the elicitation
#: is automatically cancelled and the agent receives a "cancel" result.
ELICITATION_TIMEOUT_SECONDS: int = 300


@dataclass
class PendingPermission:
    """A pending permission request awaiting user response."""

    permission_id: str
    tool_name: str
    args: dict[str, Any]
    future: asyncio.Future[PermissionReply]
    created_at: float = field(default_factory=lambda: __import__("time").time())


class OpenCodeInputProvider(InputProvider):
    """Input provider that uses OpenCode SSE/REST for user interactions.

    This provider enables tool confirmation and elicitation requests
    through the OpenCode protocol. When a tool needs confirmation:
    1. A permission request is created and stored
    2. An SSE event is broadcast to notify clients
    3. The provider awaits a response via the REST endpoint
    4. The client POSTs to /session/{id}/permissions/{permissionID} to respond
    """

    def __init__(self, state: ServerState, session_id: str) -> None:
        """Initialize OpenCode input provider.

        Args:
            state: Server state for broadcasting events
            session_id: The session ID for this provider
        """
        self.state = state
        self.session_id = session_id
        self._pending_permissions: dict[str, PendingPermission] = {}
        self._tool_approvals: dict[str, str] = {}  # tool_name -> "always" | "reject"
        self._id_counter = 0
        self._fallback_pending_questions: dict[str, Any] = {}

    @property
    def supports_durable_elicitation(self) -> bool:
        """Whether this provider supports durable (checkpointable) elicitation.

        Checks the session's checkpoint configuration at runtime via the
        session controller. Returns False when the session controller is
        not available, the session does not exist, or checkpointing is
        not enabled for the session.
        """
        if self.state.session_controller is not None:
            session = self.state.session_controller.get_session(self.session_id)
            if session is not None:
                return session.checkpoint_enabled
        return False

    @property
    def _pending_questions_dict(self) -> dict[str, Any]:
        """Get the pending questions dict for this session.

        Returns SessionState.pending_questions for per-session isolation.
        Falls back to an instance-level dict if no session_controller or
        session not found, ensuring writes persist across accesses.
        """
        if self.state.session_controller is not None:
            session = self.state.session_controller.get_session(self.session_id)
            if session is not None:
                return session.pending_questions
        return self._fallback_pending_questions

    def _generate_permission_id(self) -> str:
        """Generate a unique permission ID."""
        self._id_counter += 1
        return f"perm_{self._id_counter}_{int(__import__('time').time() * 1000)}"

    def _generate_question_id(self) -> str:
        """Generate a unique question ID."""
        self._id_counter += 1
        return f"que_{self._id_counter}_{int(__import__('time').time() * 1000)}"

    async def get_tool_confirmation(
        self,
        context: AgentContext[Any],
        tool_description: str = "",
    ) -> ConfirmationResult:
        """Get tool execution confirmation via OpenCode permission request.

        Creates a pending permission, broadcasts an SSE event, and waits
        for the client to respond via POST /session/{id}/permissions/{permissionID}.

        Args:
            context: Current node context with tool_name, tool_call_id, tool_input set
            tool_description: Human-readable description of the tool

        Returns:
            Confirmation result indicating whether to allow, skip, or abort
        """
        tool_name = context.tool_name or "unknown"
        args = context.tool_input
        try:
            # Check if we have a standing approval/rejection for this tool
            if tool_name in self._tool_approvals:
                standing_decision = self._tool_approvals[tool_name]
                match standing_decision:
                    case "always":
                        logger.debug("Auto-allowing tool", tool_name=tool_name, reason="always")
                        return "allow"
                    case "reject":
                        logger.debug("Auto-rejecting tool", tool_name=tool_name, reason="reject")
                        return "skip"

            # Create a pending permission request
            permission_id = self._generate_permission_id()
            future: asyncio.Future[PermissionReply] = asyncio.get_event_loop().create_future()
            pending = PendingPermission(
                permission_id=permission_id,
                tool_name=tool_name,
                args=args,
                future=future,
            )
            self._pending_permissions[permission_id] = pending
            max_preview_args = 3
            args_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:max_preview_args])
            if len(args) > max_preview_args:
                args_preview += ", ..."
            # Extract call_id from AgentContext if available
            # (set by non-native agents from streaming)
            # Fall back to a generated ID if not available
            call_id = context.tool_call_id
            if call_id is None:
                # Generate a synthetic call_id - won't match TUI tool parts but allows display
                call_id = f"toolu_{permission_id}"
            # TODO: Extract message_id from context when available

            event = PermissionRequestEvent.create(
                session_id=self.session_id,
                permission_id=permission_id,
                tool_name=tool_name,
                args_preview=args_preview,
                message=f"Tool '{tool_name}' wants to execute with args: {args_preview}",
                call_id=call_id,
            )

            await self.state.broadcast_event(event)
            logger.info("Permission requested", permission_id=permission_id, tool_name=tool_name)
            # Wait for the client to respond
            try:
                response = await future
            except asyncio.CancelledError:
                logger.warning("Permission request cancelled", permission_id=permission_id)
                return "skip"
            finally:
                # Clean up the pending permission
                self._pending_permissions.pop(permission_id, None)

            # Map OpenCode response to our confirmation result
            return self._handle_permission_response(response, tool_name)

        except Exception:
            logger.exception("Failed to get tool confirmation")
            return "abort_run"

    def _handle_permission_response(
        self, response: PermissionReply, tool_name: str
    ) -> ConfirmationResult:
        """Handle permission response and update tool approval state."""
        match response:
            case "once":
                return "allow"
            case "always":
                self._tool_approvals[tool_name] = "always"
                logger.info("Tool approval set", tool_name=tool_name, approval="always")
                return "allow"
            case "reject":
                return "skip"
            case _:
                logger.warning("Unknown permission response", response=response)
                return "abort_run"

    def resolve_permission(self, permission_id: str, response: PermissionReply) -> bool:
        """Resolve a pending permission request.

        Called by the REST endpoint when the client responds.

        Args:
            permission_id: The permission request ID
            response: The client's response ("once", "always", or "reject")

        Returns:
            True if the permission was found and resolved, False otherwise
        """
        pending = self._pending_permissions.get(permission_id)
        if pending is None:
            logger.warning("Permission not found", permission_id=permission_id)
            return False

        if pending.future.done():
            logger.warning("Permission already resolved", permission_id=permission_id)
            return False

        pending.future.set_result(response)
        logger.info(
            "Permission resolved",
            permission_id=permission_id,
            response=response,
        )
        return True

    def has_pending_permission(self, permission_id: str) -> bool:
        """Check whether a specific permission request is pending.

        Args:
            permission_id: The permission request ID to look up

        Returns:
            True if the permission is pending, False otherwise
        """
        return permission_id in self._pending_permissions

    def get_pending_permissions(self) -> list[PermissionAskedProperties]:
        """Get all pending permission requests.

        Returns:
            List of pending permission properties
        """
        result: list[PermissionAskedProperties] = []
        for p in self._pending_permissions.values():
            args_preview = ", ".join(f"{k}={v!r}" for k, v in list(p.args.items())[:3])
            pattern = f"{p.tool_name}: {args_preview}" if args_preview else p.tool_name
            props = PermissionAskedProperties(
                id=p.permission_id,
                session_id=self.session_id,
                permission=p.tool_name,
                patterns=[pattern],
                metadata=p.args,
                always=[pattern],
                tool=PermissionToolInfo(message_id="", call_id=None),
            )
            result.append(props)
        return result

    def get_pending_questions(self) -> list[QuestionRequest]:
        """Get all pending question requests for this session.

        Returns:
            List of pending question requests.
        """
        from wolfharness_server.opencode_server.models.question import QuestionRequest

        result: list[QuestionRequest] = []
        for question_id, pending in self._pending_questions_dict.items():
            if pending.session_id == self.session_id:
                result.append(
                    QuestionRequest(
                        id=question_id,
                        session_id=pending.session_id,
                        questions=pending.questions,
                        tool=pending.tool,
                    )
                )
        return result

    async def get_elicitation(
        self,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request via OpenCode questions.

        Translates MCP elicitation requests to OpenCode question system.
        Supports both single-select and multi-select questions.

        Args:
            params: MCP elicit request parameters

        Returns:
            Elicit result with user's response or error data
        """
        match params:
            case types.ElicitRequestURLParams(message=message, url=url):
                # For URL elicitation, we could open the URL.in browser?
                logger.info("URL elicitation request", message=message, url=url)
                return types.ElicitResult(action="decline")
            case types.ElicitRequestFormParams(
                requestedSchema=({"enum": _} | {"type": "array", "items": {"enum": _}}) as schema
            ):
                return await self._handle_question_elicitation(params, schema)
            case types.ElicitRequestFormParams(
                requestedSchema={"type": "object", "properties": dict() as props}
            ) if len(props) >= 1:
                return await self._handle_multi_question(params, props)
            case types.ElicitRequestFormParams(requestedSchema=schema, message=msg):
                # For other form elicitation, we don't have UI support yet
                logger.info("Form elicitation request (not supported)", message=msg, schema=schema)
                return types.ElicitResult(action="decline")

    async def _handle_question_elicitation(
        self,
        params: types.ElicitRequestFormParams,
        schema: dict[str, Any],
    ) -> types.ElicitResult | types.ErrorData:
        """Handle elicitation via OpenCode question system.

        Args:
            params: Form elicitation parameters
            schema: JSON schema with enum values

        Returns:
            Elicit result with user's answer
        """
        return await self._handle_single_enum(params, schema)

    async def _handle_single_enum(
        self,
        params: types.ElicitRequestFormParams,
        schema: dict[str, Any],
    ) -> types.ElicitResult | types.ErrorData:
        """Handle single enum/array question elicitation.

        Args:
            params: Form elicitation parameters
            schema: JSON schema with enum values (single or array type)

        Returns:
            Elicit result with user's answer
        """
        from wolfharness_server.opencode_server.models.events import QuestionAskedEvent
        from wolfharness_server.opencode_server.models.question import QuestionInfo, QuestionOption
        from wolfharness_server.opencode_server.state import PendingQuestion

        # Extract enum values
        match schema:
            case {"type": "array", "items": {"enum": [_, *_] as enum_values}}:
                is_multi = True
            case {"enum": [_, *_] as enum_values}:
                is_multi = False
            case _:
                return types.ElicitResult(action="decline")
        # Extract descriptions if available (custom x-option-descriptions field)
        descriptions = schema.get("x-option-descriptions", {})
        question_id = self._generate_question_id()
        opts = [
            QuestionOption(label=str(val), description=descriptions.get(str(val), ""))
            for val in enum_values
        ]
        question_info = QuestionInfo(
            question=params.message,
            header=params.message[:12],  # Truncate to 12 chars
            options=opts,
            multiple=is_multi or None,
        )
        # Create future to wait for answer
        future: asyncio.Future[list[list[str]]] = asyncio.get_event_loop().create_future()
        self._pending_questions_dict[question_id] = PendingQuestion(
            session_id=self.session_id,
            questions=[question_info],
            future=future,
        )
        # Broadcast event (serialize QuestionInfo to dict)
        event = QuestionAskedEvent.create(
            request_id=question_id,
            session_id=self.session_id,
            questions=[question_info],
        )
        await self.state.broadcast_event(event)
        logger.info(
            "Question asked",
            question_id=question_id,
            message=params.message,
            is_multi=is_multi,
        )
        # Wait for answer
        try:
            answers = await asyncio.wait_for(
                future, timeout=ELICITATION_TIMEOUT_SECONDS
            )  # list[list[str]]
            answer = answers[0] if answers else []  # Get first question's answer
            # ElicitResult content must be a dict, not a plain value
            # Wrap the answer in a dict with a "value" key
            # Multi-select: return list in dict
            # Single-select: return string in dict
            content: dict[str, str | list[str]] = (
                {"value": answer} if is_multi else {"value": answer[0] if answer else ""}
            )
            return types.ElicitResult(action="accept", content=content)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        except TimeoutError:
            logger.info("Question timed out", question_id=question_id)
            return types.ElicitResult(action="cancel")
        except asyncio.CancelledError:
            logger.info("Question cancelled", question_id=question_id)
            return types.ElicitResult(action="cancel")
        except Exception as e:
            logger.exception("Question failed", question_id=question_id)
            return types.ErrorData(code=-1, message=f"Elicitation failed: {e}")  # Generic err code
        finally:
            # Clean up pending question
            self._pending_questions_dict.pop(question_id, None)

    def _property_to_question(self, key: str, prop_schema: dict[str, Any]) -> QuestionInfo:
        """Convert JSON schema property definition to QuestionInfo.

        Supports enum, array+enum, string, and oneOf property types.
        Unsupported types fall back to text input behavior.

        Args:
            key: Property name (used as fallback for title/header)
            prop_schema: JSON schema property definition

        Returns:
            QuestionInfo configured for the property type
        """
        from wolfharness_server.opencode_server.models.question import (
            QuestionInfo,
            QuestionOption,
        )

        # Extract title with fallback to key
        title = prop_schema.get("title", key)
        # Use description if available, otherwise use title as secondary text
        description = prop_schema.get("description", title)
        # Header truncated to 12 chars per OpenCode spec
        header = title[:12]

        # Pattern match property schema types
        match prop_schema:
            case {"type": "array", "items": dict() as items}:
                # Multi-select array with enum items
                enum_values = items.get("enum", [])
                # Support x-option-descriptions for multi-select options
                descriptions = items.get("x-option-descriptions", {})
                opts = [
                    QuestionOption(label=str(val), description=descriptions.get(str(val), ""))
                    for val in enum_values
                ]
                return QuestionInfo(
                    question=description,
                    header=header,
                    options=opts,
                    multiple=True,
                )
            case {"enum": list() as enum_values}:
                # Single-select enum
                opts = [QuestionOption(label=str(val), description="") for val in enum_values]
                return QuestionInfo(
                    question=description,
                    header=header,
                    options=opts,
                    multiple=None,
                )
            case {"oneOf": list() as one_of}:
                # Single-select with const/title pairs (must check before type:string)
                one_of_opts: list[QuestionOption] = []
                for opt_def in one_of:
                    if isinstance(opt_def, dict):
                        const_val = opt_def.get("const") or ""
                        opt_title = opt_def.get("title") or ""
                        one_of_opts.append(
                            QuestionOption(label=str(const_val), description=opt_title)
                        )
                return QuestionInfo(
                    question=description,
                    header=header,
                    options=one_of_opts,
                    multiple=None,
                )
            case {"type": "string"}:
                # Text input - empty options (fallback if no oneOf)
                return QuestionInfo(
                    question=description,
                    header=header,
                    options=[],
                    multiple=None,
                )
            case _:
                # Unsupported types fallback to text input behavior
                return QuestionInfo(
                    question=description,
                    header=header,
                    options=[],
                    multiple=None,
                )

    async def _handle_multi_question(
        self,
        params: types.ElicitRequestFormParams,
        props: dict[str, Any],
    ) -> types.ElicitResult | types.ErrorData:
        """Handle multi-property object schema elicitation.

        Creates multiple questions from object properties, respecting order.
        Limits to 10 questions max with warning log.

        Args:
            params: Form elicitation parameters (contains message, description)
            props: Object properties dict (property name -> schema)

        Returns:
            Elicit result with dict of property names to answers
        """
        from wolfharness_server.opencode_server.models.events import QuestionAskedEvent
        from wolfharness_server.opencode_server.state import PendingQuestion

        max_questions = 10

        # Guard: empty properties should return decline (shouldn't happen due to match guard)
        if not props:
            logger.warning("Empty object schema properties, declining")
            return types.ElicitResult(action="decline")

        # Build property order and limit
        prop_items = list(props.items())
        original_keys = [k for k, _ in prop_items]

        if len(prop_items) > max_questions:
            logger.warning(
                "Object schema has too many properties, limiting to 10",
                property_count=len(prop_items),
            )
            prop_items = prop_items[:max_questions]
            original_keys = original_keys[:max_questions]

        # Convert each property to QuestionInfo
        questions: list[QuestionInfo] = []
        for prop_name, prop_schema in prop_items:
            q_info = self._property_to_question(prop_name, prop_schema)
            questions.append(q_info)

        if not questions:
            logger.warning("No valid questions could be created from object schema")
            return types.ElicitResult(action="decline")

        question_id = self._generate_question_id()

        # Create future to wait for answers
        future: asyncio.Future[list[list[str]]] = asyncio.get_event_loop().create_future()
        self._pending_questions_dict[question_id] = PendingQuestion(
            session_id=self.session_id,
            questions=questions,
            future=future,
        )

        # Broadcast QuestionAskedEvent with all questions
        event = QuestionAskedEvent.create(
            request_id=question_id,
            session_id=self.session_id,
            questions=questions,
        )
        await self.state.broadcast_event(event)
        logger.info(
            "Multi-question asked",
            question_id=question_id,
            question_count=len(questions),
            message=params.message,
        )

        # Wait for answers
        try:
            answers = await asyncio.wait_for(
                future, timeout=ELICITATION_TIMEOUT_SECONDS
            )  # list[list[str]] - indexed by question

            # Map answers back to property keys
            # answers[i] corresponds to questions[i] which corresponds to original_keys[i]
            result_content: dict[str, Any] = {}
            for i, key in enumerate(original_keys[: len(answers)]):
                answer_list = answers[i] if i < len(answers) else []
                # For multi-select, return list; for single-select, return string
                # Determined by checking if it was a multi question
                question_info: QuestionInfo | None = questions[i] if i < len(questions) else None
                if question_info is not None and question_info.multiple:
                    result_content[key] = answer_list
                else:
                    result_content[key] = answer_list[0] if answer_list else ""

            return types.ElicitResult(action="accept", content=result_content)  # pyright: ignore[reportArgumentType]

        except TimeoutError:
            logger.info("Multi-question timed out", question_id=question_id)
            return types.ElicitResult(action="cancel")
        except asyncio.CancelledError:
            logger.info("Multi-question cancelled", question_id=question_id)
            return types.ElicitResult(action="cancel")
        except Exception as e:
            logger.exception("Multi-question failed", question_id=question_id)
            return types.ErrorData(code=-1, message=f"Elicitation failed: {e}")
        finally:
            # Clean up pending question
            self._pending_questions_dict.pop(question_id, None)

    async def broadcast_elicitation_question(
        self,
        handle: str,
        params: types.ElicitRequestParams,
        shared_future: asyncio.Future[Any] | None = None,
    ) -> bool:
        """Broadcast a QuestionAskedEvent for a durable elicitation.

        Called by ``handle_elicitation`` Path 3 after the elicitation future
        is registered. This populates the pending questions dict and
        broadcasts the event so the OpenCode TUI can render the question UI.
        When the user answers via the REST endpoint, ``resolve_question``
        fires and the caller resolves the elicitation future.

        Args:
            handle: The elicitation handle (tool_call_id) used as question ID.
            params: The elicitation request parameters.
            shared_future: Optional future from the ElicitationFutureRegistry
                to share so ``resolve_question`` resolves both.

        Returns:
            True if the question was broadcast successfully, False if
            the schema type is not supported.
        """
        from wolfharness_server.opencode_server.models.events import QuestionAskedEvent
        from wolfharness_server.opencode_server.state import PendingQuestion

        match params:
            case types.ElicitRequestFormParams(
                requestedSchema=({"enum": _} | {"type": "array", "items": {"enum": _}}) as schema,
                message=msg,
            ):
                questions = [self._property_to_question("value", schema)]
            case types.ElicitRequestFormParams(
                requestedSchema={"type": "object", "properties": dict() as props},
                message=msg,
            ) if len(props) >= 1:
                questions = [self._property_to_question(k, s) for k, s in list(props.items())[:10]]  # ty: ignore[invalid-argument-type]
            case types.ElicitRequestFormParams(message=msg, requestedSchema=schema):
                logger.info(
                    "Durable elicitation schema not supported for question UI",
                    message=msg,
                    schema=schema,
                )
                return False
            case _:
                logger.info(
                    "Durable elicitation params type not supported",
                    params=str(params),
                )
                return False

        if shared_future is not None:
            future: asyncio.Future[list[list[str]]] = shared_future
        else:
            future = asyncio.get_event_loop().create_future()
        self._pending_questions_dict[handle] = PendingQuestion(
            session_id=self.session_id,
            questions=questions,
            future=future,
        )

        event = QuestionAskedEvent.create(
            request_id=handle,
            session_id=self.session_id,
            questions=questions,
        )
        await self.state.broadcast_event(event)
        logger.info(
            "Durable elicitation question broadcast",
            handle=handle,
            message=msg,
            question_count=len(questions),
        )
        return True

    def clear_tool_approvals(self) -> None:
        """Clear all stored tool approval decisions."""
        approval_count = len(self._tool_approvals)
        self._tool_approvals.clear()
        logger.info("Cleared tool approval decisions", count=approval_count)

    def resolve_question(self, question_id: str, answers: list[list[str]]) -> bool:
        """Resolve a pending question request.

        Called by the REST endpoint when the client responds.

        Args:
            question_id: The question request ID
            answers: User's answers (array of arrays per OpenCode format)

        Returns:
            True if the question was found and resolved, False otherwise
        """
        pending = self._pending_questions_dict.get(question_id)
        if pending is None:
            logger.warning("Question not found", question_id=question_id)
            return False

        future = pending.future
        if future.done():
            logger.warning("Question already resolved", question_id=question_id)
            return False

        future.set_result(answers)
        logger.info("Question resolved", question_id=question_id, answers=answers)
        return True

    def cancel_all_pending(self) -> int:
        """Cancel all pending permission requests.

        Returns:
            Number of permissions cancelled
        """
        count = 0
        for pending in list(self._pending_permissions.values()):
            if not pending.future.done():
                pending.future.cancel()
                count += 1
        self._pending_permissions.clear()
        logger.info("Cancelled all pending permissions", count=count)
        return count

    def cancel_pending_questions(self) -> int:
        """Cancel all pending question requests for this session.

        Returns:
            Number of questions cancelled.
        """
        count = 0
        for question_id, pending in list(self._pending_questions_dict.items()):
            if pending.session_id == self.session_id:
                future = getattr(pending, "future", None)
                if future is not None and not future.done():
                    future.cancel()
                    count += 1
                self._pending_questions_dict.pop(question_id, None)
        logger.info("Cancelled all pending questions", count=count, session_id=self.session_id)
        return count

    def cleanup_elicitation_question(self, handle: str) -> None:
        """Remove a pending elicitation question after timeout or cancellation."""
        self._pending_questions_dict.pop(handle, None)
