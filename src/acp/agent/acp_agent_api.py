"""ACP Agent API for simplified client-to-agent interactions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from acp.schema import (
    AuthenticateRequest,
    CancelNotification,
    ForkSessionRequest,
    InitializeRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    ResumeSessionRequest,
    SetSessionConfigOptionRequest,
    SetSessionModelRequest,
    SetSessionModeRequest,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from acp.agent.protocol import Agent
    from acp.schema import (
        AuthenticateResponse,
        ContentBlock,
        ForkSessionResponse,
        InitializeResponse,
        ListSessionsResponse,
        LoadSessionResponse,
        NewSessionResponse,
        PromptResponse,
        ResumeSessionResponse,
        SessionUpdate,
        SetSessionConfigOptionResponse,
        SetSessionModelResponse,
        SetSessionModeResponse,
    )
    from acp.schema.mcp import McpServer


@runtime_checkable
class _SessionStateProtocol(Protocol):
    """Protocol for session state objects that ACPAgentAPI can poll for updates."""

    def pop_update(self) -> SessionUpdate | None: ...
    def clear(self) -> None: ...


@runtime_checkable
class _UpdateEventProtocol(Protocol):
    """Protocol for update events that ACPAgentAPI can wait on."""

    async def wait_with_timeout(self, timeout: float | None = None) -> bool: ...
    def clear(self) -> None: ...


class ACPAgentAPI:
    """Thin wrapper for client-to-agent ACP interactions.

    Avoids manual instantiation of request/notification objects.

    When optional ``state`` and ``update_event`` are provided, the instance
    also satisfies the :class:`~wolfharness.agents.acp_agent.turn.ACPClientProtocol`
    protocol by implementing :meth:`stream_events` and :meth:`get_messages`.
    """

    def __init__(
        self,
        connection: Agent,
        *,
        state: _SessionStateProtocol | None = None,
        update_event: _UpdateEventProtocol | None = None,
    ) -> None:
        """Initialize agent API helper.

        Args:
            connection: The Agent protocol connection (e.g., ClientSideConnection)
            state: Optional session state for polling updates (enables stream_events)
            update_event: Optional event signaled when new updates arrive
        """
        self.connection = connection
        self._state: _SessionStateProtocol | None = state
        self._update_event: _UpdateEventProtocol | None = update_event
        self._consumed_updates: list[SessionUpdate] = []

    def _attach_state(
        self,
        state: _SessionStateProtocol,
        update_event: _UpdateEventProtocol,
    ) -> None:
        """Attach state and update event after construction.

        Allows deferred wiring when state/event are created after the API.
        """
        self._state = state
        self._update_event = update_event

    async def initialize(
        self,
        *,
        title: str,
        version: str,
        name: str,
        protocol_version: int = 1,
        terminal: bool = True,
        read_text_file: bool = True,
        write_text_file: bool = True,
        terminal_auth: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> InitializeResponse:
        """Initialize the ACP connection."""
        request = InitializeRequest.create(
            title=title,
            version=version,
            name=name,
            protocol_version=protocol_version,
            terminal=terminal,
            read_text_file=read_text_file,
            write_text_file=write_text_file,
            terminal_auth=terminal_auth,
            metadata=metadata,
        )
        return await self.connection.initialize(request)

    async def new_session(
        self,
        cwd: str | None = None,
        mcp_servers: Sequence[McpServer] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NewSessionResponse:
        """Create a new ACP session."""
        request = NewSessionRequest(
            cwd=cwd or str(Path.cwd()),
            mcp_servers=list(mcp_servers) if mcp_servers else None,
            field_meta=metadata,
        )
        return await self.connection.new_session(request)

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Sequence[McpServer] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LoadSessionResponse:
        """Load an existing session."""
        request = LoadSessionRequest(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=list(mcp_servers) if mcp_servers else None,
            field_meta=metadata,
        )
        return await self.connection.load_session(request)

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ListSessionsResponse:
        """List available sessions."""
        request = ListSessionsRequest(cwd=cwd, cursor=cursor, field_meta=metadata)
        return await self.connection.list_sessions(request)

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Sequence[McpServer] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ForkSessionResponse:
        """Fork an existing session."""
        request = ForkSessionRequest(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=list(mcp_servers) if mcp_servers else [],
            field_meta=metadata,
        )
        return await self.connection.fork_session(request)

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Sequence[McpServer] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResumeSessionResponse:
        """Resume a paused session."""
        servers = list(mcp_servers) if mcp_servers else []
        request = ResumeSessionRequest(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=servers,
            field_meta=metadata,
        )
        return await self.connection.resume_session(request)

    async def prompt(
        self,
        session_id: str,
        prompt: Sequence[ContentBlock],
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptResponse:
        """Send a prompt to the agent."""
        request = PromptRequest(
            session_id=session_id,
            prompt=list(prompt),
            message_id=message_id,
            field_meta=metadata,
        )
        return await self.connection.prompt(request)

    async def cancel(self, session_id: str) -> None:
        """Cancel the current operation in a session."""
        notification = CancelNotification(session_id=session_id)
        await self.connection.cancel(notification)

    async def set_session_mode(
        self,
        session_id: str,
        mode_id: str,
    ) -> SetSessionModeResponse | None:
        """Set the session mode."""
        request = SetSessionModeRequest(session_id=session_id, mode_id=mode_id)
        return await self.connection.set_session_mode(request)

    async def set_session_model(
        self,
        session_id: str,
        model_id: str,
    ) -> SetSessionModelResponse | None:
        """Set the session model."""
        request = SetSessionModelRequest(session_id=session_id, model_id=model_id)
        return await self.connection.set_session_model(request)

    async def set_session_config_option(
        self,
        session_id: str,
        config_id: str,
        value: str,
    ) -> SetSessionConfigOptionResponse | None:
        """Set a session configuration option."""
        request = SetSessionConfigOptionRequest(
            session_id=session_id,
            config_id=config_id,
            value=value,  # pyright: ignore[reportCallIssue]
        )
        return await self.connection.set_session_config_option(request)

    async def authenticate(
        self,
        method_id: str,
    ) -> AuthenticateResponse | None:
        """Authenticate with the agent."""
        request = AuthenticateRequest(method_id=method_id)
        return await self.connection.authenticate(request)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an extension method on the agent."""
        return await self.connection.ext_method(method, params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send an extension notification to the agent."""
        await self.connection.ext_notification(method, params)

    async def stream_events(
        self,
        response: PromptResponse,
    ) -> AsyncIterator[SessionUpdate]:
        """Yield raw ACP session updates from the state queue.

        Polls :meth:`_SessionStateProtocol.pop_update` in a loop, waiting
        up to 50 ms between drain cycles for new updates to arrive via
        ``_update_event``.  Once a full drain cycle produces no updates,
        the iterator ends.

        Updates are also collected in ``_consumed_updates`` so that
        :meth:`get_messages` can return them after streaming completes.

        Args:
            response: The prompt response (unused — updates come from state)
        """
        self._consumed_updates.clear()
        if self._state is None or self._update_event is None:
            return
        while True:
            try:
                await self._update_event.wait_with_timeout(0.05)
                self._update_event.clear()
            except TimeoutError:
                pass
            drained_any = False
            while (update := self._state.pop_update()) is not None:
                self._consumed_updates.append(update)
                yield update
                drained_any = True
            if not drained_any:
                break

    async def get_messages(self, session_id: str) -> list[SessionUpdate]:
        """Return all session updates consumed during :meth:`stream_events`.

        Args:
            session_id: The ACP session ID (unused — updates are already collected)
        """
        return list(self._consumed_updates)
