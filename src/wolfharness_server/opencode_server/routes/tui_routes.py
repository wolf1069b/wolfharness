"""TUI routes for external control of the TUI.

These endpoints allow external integrations (e.g., VSCode extension) to control
the TUI by broadcasting events that the TUI listens for via SSE.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.models.events import (
    TuiCommandExecuteEvent,
    TuiCommandExecuteProperties,
    TuiPromptAppendEvent,
    TuiPromptAppendProperties,
    TuiSessionSelectEvent,
    TuiSessionSelectProperties,
    TuiToastShowEvent,
    TuiToastShowProperties,
)


router = APIRouter(prefix="/tui", tags=["tui"])


@router.post("/append-prompt")
async def append_prompt(request: TuiPromptAppendProperties, state: StateDep) -> bool:
    """Append text to the TUI prompt.

    Used by external integrations (e.g., VSCode) to insert text like file
    references into the prompt input.
    """
    await state.broadcast_event(TuiPromptAppendEvent.create(request.text))
    return True


@router.post("/submit-prompt")
async def submit_prompt(state: StateDep) -> bool:
    """Submit the current prompt.

    Triggers the TUI to submit whatever is currently in the prompt input.
    """
    await state.broadcast_event(TuiCommandExecuteEvent.create("prompt.submit"))
    return True


@router.post("/clear-prompt")
async def clear_prompt(state: StateDep) -> bool:
    """Clear the TUI prompt.

    Clears any text currently in the prompt input.
    """
    await state.broadcast_event(TuiCommandExecuteEvent.create("prompt.clear"))
    return True


@router.post("/execute-command")
async def execute_command(request: TuiCommandExecuteProperties, state: StateDep) -> bool:
    """Execute a TUI command.

    Available commands:
    - session.list, session.new, session.share, session.interrupt, session.compact
    - session.page.up, session.page.down, session.half.page.up, session.half.page.down
    - session.first, session.last
    - prompt.clear, prompt.submit
    - agent.cycle
    """
    await state.broadcast_event(TuiCommandExecuteEvent.create(request.command))
    return True


@router.post("/show-toast")
async def show_toast(request: TuiToastShowProperties, state: StateDep) -> bool:
    """Show a toast notification in the TUI."""
    event = TuiToastShowEvent.create(
        message=request.message,
        variant=request.variant,
        title=request.title,
        duration=request.duration,
    )
    await state.broadcast_event(event)
    return True


# Additional convenience endpoints matching OpenCode's API


@router.post("/open-help")
async def open_help(state: StateDep) -> bool:
    """Open the help dialog."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("help.open"))
    return True


@router.post("/open-sessions")
async def open_sessions(state: StateDep) -> bool:
    """Open the session selector."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("session.list"))
    return True


@router.post("/open-themes")
async def open_themes(state: StateDep) -> bool:
    """Open the theme selector."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("theme.list"))
    return True


@router.post("/open-models")
async def open_models(state: StateDep) -> bool:
    """Open the model selector."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("model.list"))
    return True


class PublishEventRequest(BaseModel):
    """Request body for publishing an arbitrary TUI event."""

    type: str = Field(..., description="Event type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Event properties")


@router.post("/publish")
async def publish_event(request: PublishEventRequest, state: StateDep) -> bool:
    """Publish an arbitrary TUI event.

    Supports all TUI event types like tui.prompt.append, tui.command.execute,
    tui.toast.show, tui.session.select.
    """
    match request.type:
        case "tui.prompt.append":
            text = request.properties.get("text", "")
            await state.broadcast_event(TuiPromptAppendEvent.create(text))
        case "tui.command.execute":
            command = request.properties.get("command", "")
            await state.broadcast_event(TuiCommandExecuteEvent.create(command))
        case "tui.toast.show":
            await state.broadcast_event(
                TuiToastShowEvent.create(
                    message=request.properties.get("message", ""),
                    variant=request.properties.get("variant", "info"),
                    title=request.properties.get("title"),
                    duration=request.properties.get("duration", 5000),
                )
            )
        case "tui.session.select":
            session_id = request.properties.get("sessionID", "")
            await state.broadcast_event(TuiSessionSelectEvent.create(session_id))
        case _:
            # Forward unknown event types as command execute
            await state.broadcast_event(TuiCommandExecuteEvent.create(request.type))
    return True


@router.post("/select-session")
async def select_session(request: TuiSessionSelectProperties, state: StateDep) -> bool:
    """Navigate the TUI to display the specified session."""
    await state.broadcast_event(TuiSessionSelectEvent.create(request.session_id))
    return True
