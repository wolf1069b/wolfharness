"""Permission routes for OpenCode TUI compatibility."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from wolfharness import log
from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.models import (
    PermissionAskedProperties,
    PermissionReplyRequest,
    PermissionResolvedEvent,
)


router = APIRouter(prefix="/permission", tags=["permission"])
logger = log.get_logger(__name__)


@router.get("")
async def list_permissions(state: StateDep) -> list[PermissionAskedProperties]:
    """List all pending permission requests across all sessions."""
    result: list[PermissionAskedProperties] = []
    if state.session_controller is not None:
        for session in state.session_controller._sessions.values():
            if session.input_provider is not None:
                result.extend(session.input_provider.get_pending_permissions())
    return result


@router.post("/{permission_id}/reply")
async def reply_to_permission(
    permission_id: str,
    body: PermissionReplyRequest,
    state: StateDep,
) -> bool:
    """Respond to a pending permission request (OpenCode TUI compatibility).

    This endpoint handles the OpenCode TUI's expected format:
    POST /permission/{permission_id}/reply

    The response can be:
    - "once": Allow this tool execution once
    - "always": Always allow this tool (remembered for session)
    - "reject": Reject this tool execution
    """
    logger.info("received reply", reply=body.reply, permission_id=permission_id)

    if state.session_controller is not None:
        for session_id, session in state.session_controller._sessions.items():
            input_provider = session.input_provider
            if input_provider is None:
                continue
            if not input_provider.has_pending_permission(permission_id):
                continue
            resolved = input_provider.resolve_permission(permission_id, body.reply)
            logger.info("Resolved permission", resolved=resolved)
            if not resolved:
                detail = "Permission not found or already resolved"
                raise HTTPException(status_code=404, detail=detail)
            event = PermissionResolvedEvent.create(
                session_id=session_id,
                request_id=permission_id,
                reply=body.reply,
            )
            await state.broadcast_event(event)
            return True

    raise HTTPException(status_code=404, detail="Permission not found")
