"""CLI tests for serve-acp command with streamable-http transport.

Tests cover:
- --transport streamable-http creates ACPWebSocketTransport with correct host/port
- --transport websocket emits deprecation warning
- Default transport is StdioTransport
- --host and --port are passed correctly for streamable-http
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import typer as t
from typer.testing import CliRunner

from wolfharness_cli.serve_acp import acp_command


if TYPE_CHECKING:
    from pathlib import Path


# Minimal valid config content for testing
MINIMAL_CONFIG = """
agents:
  test_agent:
    type: native
    model: "openai:gpt-4o-mini"
    system_prompt: "Test agent"
"""


@pytest.fixture
def minimal_config_file(tmp_path: Path) -> str:
    """Create a minimal valid config file for CLI tests."""
    config_path = tmp_path / "agents.yml"
    config_path.write_text(MINIMAL_CONFIG)
    return str(config_path)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Typer test runner."""
    return CliRunner()


# Wrap the command into a Typer app for CliRunner compatibility
app = t.Typer()
app.command()(acp_command)


# =============================================================================
# Transport argument parsing tests
# =============================================================================


@pytest.mark.unit
def test_default_transport_is_stdio(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """Default --transport should create StdioTransport."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(app, [minimal_config_file])

        assert result.exit_code == 0, result.output
        mock_from_config.assert_called_once()
        transport_arg = mock_from_config.call_args.kwargs.get("transport")
        assert isinstance(transport_arg, object)
        assert type(transport_arg).__name__ == "StdioTransport"


@pytest.mark.unit
def test_transport_streamable_http_creates_acp_websocket_transport(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """--transport streamable-http should create ACPWebSocketTransport."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(app, [minimal_config_file, "--transport", "streamable-http"])

        assert result.exit_code == 0, result.output
        mock_from_config.assert_called_once()
        transport_arg = mock_from_config.call_args.kwargs.get("transport")
        assert type(transport_arg).__name__ == "ACPWebSocketTransport"


@pytest.mark.unit
def test_transport_streamable_http_with_custom_host_and_port(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """--host and --port should be passed to ACPWebSocketTransport."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(
            app,
            [
                minimal_config_file,
                "--transport",
                "streamable-http",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
            ],
        )

        assert result.exit_code == 0, result.output
        transport_arg = mock_from_config.call_args.kwargs["transport"]
        assert type(transport_arg).__name__ == "ACPWebSocketTransport"
        assert transport_arg.host == "0.0.0.0"
        assert transport_arg.port == 9000


@pytest.mark.unit
def test_transport_streamable_http_uses_default_host_port(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """ACPWebSocketTransport should use default host/port when not specified."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(app, [minimal_config_file, "--transport", "streamable-http"])

        assert result.exit_code == 0, result.output
        transport_arg = mock_from_config.call_args.kwargs["transport"]
        assert transport_arg.host == "localhost"
        assert transport_arg.port == 8080


@pytest.mark.unit
def test_transport_websocket_emits_deprecation_warning(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """--transport websocket should emit DeprecationWarning."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        with pytest.warns(DeprecationWarning, match="deprecated"):
            result = cli_runner.invoke(app, [minimal_config_file, "--transport", "websocket"])

        assert result.exit_code == 0, result.output
        transport_arg = mock_from_config.call_args.kwargs.get("transport")
        assert type(transport_arg).__name__ == "WebSocketTransport"


@pytest.mark.unit
def test_transport_websocket_uses_ws_host_and_ws_port(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """--ws-host and --ws-port should be passed to WebSocketTransport."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        with pytest.warns(DeprecationWarning, match="deprecated"):
            result = cli_runner.invoke(
                app,
                [
                    minimal_config_file,
                    "--transport",
                    "websocket",
                    "--ws-host",
                    "0.0.0.0",
                    "--ws-port",
                    "8766",
                ],
            )

        assert result.exit_code == 0, result.output
        transport_arg = mock_from_config.call_args.kwargs["transport"]
        assert type(transport_arg).__name__ == "WebSocketTransport"
        assert transport_arg.host == "0.0.0.0"
        assert transport_arg.port == 8766


# =============================================================================
# Short option tests
# =============================================================================


@pytest.mark.unit
def test_short_options_for_host_and_port(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """-h and -p should work as short options for --host and --port."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(
            app,
            [
                minimal_config_file,
                "--transport",
                "streamable-http",
                "-h",
                "127.0.0.1",
                "-p",
                "3000",
            ],
        )

        assert result.exit_code == 0, result.output
        transport_arg = mock_from_config.call_args.kwargs["transport"]
        assert transport_arg.host == "127.0.0.1"
        assert transport_arg.port == 3000


# =============================================================================
# Other CLI option passthrough tests
# =============================================================================


@pytest.mark.unit
def test_agent_option_is_passed_through(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """--agent option should be passed to ACPServer.from_config."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(
            app, [minimal_config_file, "--transport", "streamable-http", "--agent", "test_agent"]
        )

        assert result.exit_code == 0, result.output
        assert mock_from_config.call_args.kwargs.get("agent") == "test_agent"


@pytest.mark.unit
def test_debug_messages_option_is_passed_through(
    cli_runner: CliRunner,
    minimal_config_file: str,
) -> None:
    """--debug-messages option should be passed to ACPServer.from_config."""
    with (
        patch("wolfharness_server.acp_server.ACPServer.from_config") as mock_from_config,
        patch("wolfharness_cli.serve_acp.asyncio.run"),
    ):
        mock_server = MagicMock()
        mock_from_config.return_value = mock_server

        result = cli_runner.invoke(
            app, [minimal_config_file, "--transport", "streamable-http", "--debug-messages"]
        )

        assert result.exit_code == 0, result.output
        assert mock_from_config.call_args.kwargs.get("debug_messages") is True
