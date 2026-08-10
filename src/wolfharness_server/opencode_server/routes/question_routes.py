"""Question routes for OpenCode compatibility."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider
from wolfharness_server.opencode_server.models import (
    PermissionReply,
    PermissionResolvedEvent,
    QuestionRejectedEvent,
    QuestionRepliedEvent,
    QuestionReply,
    QuestionRequest,
)


router = APIRouter(prefix="/question", tags=["question"])


def _find_permission_provider(
    state: StateDep,
    permission_id: str,
) -> tuple[str, OpenCodeInputProvider] | None:
    if state.session_controller is None:
        return None
    for session_id, session in state.session_controller._sessions.items():
        provider = session.input_provider
        if (
            isinstance(provider, OpenCodeInputProvider)
            and permission_id in provider._pending_permissions
        ):
            return session_id, provider
    return None


def _extract_permission_reply(reply: QuestionReply) -> PermissionReply | None:
    if len(reply.answers) != 1:
        return None
    selected_answers = reply.answers[0]
    if len(selected_answers) != 1:
        return None

    selected_reply = selected_answers[0]
    match selected_reply:
        case "once" | "always" | "reject":
            return selected_reply
        case _:
            return None


def _get_all_pending_questions(state: StateDep) -> dict[str, Any]:
    """Get all pending questions from SessionController."""
    result: dict[str, Any] = {}
    if state.session_controller is not None:
        for session in state.session_controller._sessions.values():
            result.update(session.pending_questions)
    return result


def _get_pending_question(state: StateDep, question_id: str) -> Any | None:
    """Look up a pending question across SessionController."""
    if state.session_controller is not None:
        for session in state.session_controller._sessions.values():
            if question_id in session.pending_questions:
                return session.pending_questions[question_id]
    return None


def _remove_pending_question(state: StateDep, question_id: str) -> bool:
    """Remove a pending question from SessionController."""
    if state.session_controller is not None:
        for session in state.session_controller._sessions.values():
            if question_id in session.pending_questions:
                del session.pending_questions[question_id]
                return True
    return False


@router.get("/", response_model=list[QuestionRequest])
async def list_questions(state: StateDep) -> list[QuestionRequest]:
    """List all pending question requests.

    Returns a list of all pending questions awaiting user response.
    """
    pending = _get_all_pending_questions(state)
    return [
        QuestionRequest(id=question_id, session_id=i.session_id, questions=i.questions, tool=i.tool)
        for question_id, i in pending.items()
    ]


@router.post("/{requestID}/reply")
async def reply_to_question(requestID: str, reply: QuestionReply, state: StateDep) -> bool:  # noqa: N803
    """Reply to a question request.

    The user provides answers to the questions. Answers must be provided
    as an array of arrays, where each inner array contains the selected
    label(s) for that question.

    Args:
        requestID: The question request ID
        reply: The user's answers
        state: Server state

    Returns:
        True if the question was resolved successfully

    Raises:
        HTTPException: If question not found or invalid provider
    """
    pending = _get_pending_question(state, requestID)
    if not pending:
        permission_target = _find_permission_provider(state, requestID)
        if permission_target is None:
            raise HTTPException(status_code=404, detail="Question request not found")

        session_id, provider = permission_target
        permission_reply = _extract_permission_reply(reply)
        if permission_reply is None:
            raise HTTPException(status_code=400, detail="Invalid permission reply")

        if not provider.resolve_permission(requestID, permission_reply):
            raise HTTPException(status_code=404, detail="Permission not found or already resolved")

        resolved_event = PermissionResolvedEvent.create(
            session_id=session_id,
            request_id=requestID,
            reply=permission_reply,
        )
        await state.broadcast_event(resolved_event)
        return True

    session_id = pending.session_id
    session = (
        state.session_controller.get_session(session_id)
        if state.session_controller is not None
        else None
    )
    question_provider: OpenCodeInputProvider | None = (
        session.input_provider if session is not None else None
    )
    if not isinstance(question_provider, OpenCodeInputProvider):
        raise HTTPException(status_code=500, detail="Invalid provider for session")
    # Resolve via provider
    if not question_provider.resolve_question(requestID, reply.answers):
        raise HTTPException(status_code=404, detail="Question already resolved")
    # Broadcast replied event
    replied_event = QuestionRepliedEvent.create(
        session_id=session_id,
        request_id=requestID,
        answers=reply.answers,
    )
    await state.broadcast_event(replied_event)
    return True


@router.post("/{requestID}/reject")
async def reject_question(requestID: str, state: StateDep) -> bool:  # noqa: N803
    """Reject a question request.

    Called when the user dismisses the question without providing an answer.

    Args:
        requestID: The question request ID
        state: Server state

    Returns:
        True if the question was rejected successfully

    Raises:
        HTTPException: If question not found
    """
    pending = _get_pending_question(state, requestID)
    if not pending:
        permission_target = _find_permission_provider(state, requestID)
        if permission_target is None:
            raise HTTPException(status_code=404, detail="Question request not found")

        session_id, provider = permission_target
        if not provider.resolve_permission(requestID, "reject"):
            raise HTTPException(status_code=404, detail="Permission not found or already resolved")

        resolved_event = PermissionResolvedEvent.create(
            session_id=session_id,
            request_id=requestID,
            reply="reject",
        )
        await state.broadcast_event(resolved_event)
        return True
    # Cancel the future
    if not pending.future.done():
        pending.future.cancel()
    # Remove from pending
    _remove_pending_question(state, requestID)
    # Broadcast rejected event
    rejected_event = QuestionRejectedEvent.create(
        session_id=pending.session_id, request_id=requestID
    )
    await state.broadcast_event(rejected_event)
    return True
