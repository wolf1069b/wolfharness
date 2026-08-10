"""WebSocket server that bridges to a stdio ACP agent.

Spawns a stdio-based ACP agent subprocess and exposes it via WebSocket,
allowing clients like mcp-ws to connect.

Usage:
    python -m acp.bridge.ws_server "uv run wolfharness serve-acp" --port 8765
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

import anyenv
import websockets

from acp.transports import spawn_stdio_transport


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from anyio.abc import ByteReceiveStream, ByteSendStream, Process
    from websockets.asyncio.server import ServerConnection

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ACPWebSocketServer:
    """WebSocket server that bridges to a stdio ACP agent."""

    command: str
    """command to spawn the WebSocket server."""

    args: list[str] = field(default_factory=list)
    """arguments for the command."""

    host: str = "localhost"
    """host to bind the WebSocket server to."""

    port: int = 8765
    """port for the WebSocket server."""

    env: Mapping[str, str] | None = None
    """environment variables for the subprocess."""

    cwd: str | Path | None = None
    """working directory for the subprocess."""

    def __post_init__(self) -> None:
        self._process: Process | None = None
        self._reader: ByteReceiveStream | None = None
        self._writer: ByteSendStream | None = None
        self._websocket: ServerConnection | None = None
        self._shutdown = asyncio.Event()
        self._read_buffer = b""

    async def _read_jsonrpc_message(self) -> dict[str, Any] | None:
        """Read a JSON-RPC message from the agent's stdout."""
        if self._reader is None:
            return None

        while True:
            # Check if we have a complete line in buffer
            if b"\n" in self._read_buffer:
                line, self._read_buffer = self._read_buffer.split(b"\n", 1)
                if line.strip():
                    try:
                        return anyenv.load_json(line.decode(), return_type=dict)
                    except anyenv.JsonLoadError:
                        logger.exception("Failed to parse JSON from agent")
                        continue

            # Read more data
            try:
                chunk = await self._reader.receive(65536)
                if not chunk:
                    return None
                self._read_buffer += chunk
            except Exception:  # noqa: BLE001
                return None

    async def _send_to_agent(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the agent's stdin."""
        if self._writer is None:
            return

        data = anyenv.dump_json(message) + "\n"
        await self._writer.send(data.encode())

    async def _agent_to_websocket(self) -> None:
        """Forward messages from agent stdout to WebSocket."""
        while not self._shutdown.is_set():
            try:
                message = await self._read_jsonrpc_message()
                if message is None:
                    logger.info("Agent stdout closed")
                    break

                if self._websocket is not None:
                    await self._websocket.send(anyenv.dump_json(message))
                    logger.debug("Agent → WebSocket: %s", message.get("method", message.get("id")))
            except Exception:
                logger.exception("Error forwarding agent → WebSocket")
                break

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a WebSocket client connection."""
        logger.info("WebSocket client connected")
        self._websocket = websocket

        try:
            async for raw_message in websocket:
                try:
                    message = anyenv.load_json(raw_message, return_type=dict)
                    logger.debug("WebSocket → Agent: %s", message.get("method", message.get("id")))
                    await self._send_to_agent(message)
                except anyenv.JsonLoadError:
                    logger.exception("Invalid JSON from WebSocket")
                    error = ({"code": -32700, "message": "Parse error"},)
                    error_response = {"jsonrpc": "2.0", "id": None, "error": error}
                    await websocket.send(anyenv.dump_json(error_response))
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        finally:
            self._websocket = None

    async def serve(self) -> None:
        """Start the WebSocket server and agent process."""
        cmd_str = f"{self.command} {' '.join(self.args)}".strip()
        logger.info("Starting ACP agent: %s", cmd_str)

        async with spawn_stdio_transport(self.command, *self.args, env=self.env, cwd=self.cwd) as (
            reader,
            writer,
            process,
        ):
            self._reader = reader
            self._writer = writer
            self._process = process
            # Start forwarding agent output to WebSocket
            forward_task = asyncio.create_task(self._agent_to_websocket())
            try:
                async with websockets.serve(self._handle_client, self.host, self.port):
                    logger.info("WebSocket server running on ws://%s:%d", self.host, self.port)
                    await self._shutdown.wait()
            finally:
                self._shutdown.set()
                forward_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forward_task
