"""L4 OpenCode tests for MCP resource discovery and exact routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from pathlib import Path

    from tests.e2e.conftest import SubprocessServer


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows subprocess issues"),
]


@pytest.mark.parametrize(
    "subprocess_server_with_mcp_resources",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_resource_catalog_preserves_server_identity(
    subprocess_server_with_mcp_resources: SubprocessServer,
) -> None:
    """The Host catalog keeps both providers when they expose one URI."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        first_session = await client.post(
            f"{subprocess_server_with_mcp_resources.base_url}/session", json={}
        )
        second_session = await client.post(
            f"{subprocess_server_with_mcp_resources.base_url}/session", json={}
        )
        assert first_session.status_code in (200, 201), first_session.text
        assert second_session.status_code in (200, 201), second_session.text
        response = await client.get(
            f"{subprocess_server_with_mcp_resources.base_url}/experimental/resource"
        )
        second_response = await client.get(
            f"{subprocess_server_with_mcp_resources.base_url}/experimental/resource"
        )

    assert response.status_code == 200, response.text
    assert second_response.status_code == 200, second_response.text
    catalog = response.json()
    assert second_response.json() == catalog
    assert "alpha%3A%25:file:///shared.txt" in catalog
    assert "beta:file:///shared.txt" in catalog
    assert catalog["alpha%3A%25:file:///shared.txt"]["client"] == "alpha:%"
    assert catalog["beta:file:///shared.txt"]["client"] == "beta"
    assert catalog["beta:file:///shared.txt"]["mimeType"] == "text/plain"


@pytest.mark.parametrize(
    "subprocess_server_with_mcp_resources",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_resource_tools_are_the_only_formal_resource_tools(
    subprocess_server_with_mcp_resources: SubprocessServer,
) -> None:
    """The real HTTP tool surface exposes exactly the three formal resource tools."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_response = await client.post(
            f"{subprocess_server_with_mcp_resources.base_url}/session", json={}
        )
        assert session_response.status_code in (200, 201), session_response.text
        session_id = session_response.json().get("id") or session_response.json().get("sessionID")
        assert session_id
        message_response = await client.post(
            f"{subprocess_server_with_mcp_resources.base_url}/session/{session_id}/message",
            json={"parts": [{"type": "text", "text": "initialize tools"}]},
        )
        assert message_response.status_code in (200, 201, 202), message_response.text
        response = await client.get(
            f"{subprocess_server_with_mcp_resources.base_url}/experimental/tool/ids"
        )
        schema_response = await client.get(
            f"{subprocess_server_with_mcp_resources.base_url}/experimental/tool"
        )

    assert response.status_code == 200, response.text
    assert schema_response.status_code == 200, schema_response.text
    tool_ids = set(response.json())
    schema_tools = {item["id"]: item for item in schema_response.json()}
    assert {
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    } <= tool_ids
    assert {
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    } <= schema_tools.keys()
    assert schema_tools["read_mcp_resource"]["parameters"]["required"] == ["server", "uri"]
    assert (
        not {
            "list_resources",
            "list_resource_templates",
            "read_resource",
            "read_section",
            "read_section_text",
        }
        & tool_ids
    )


@pytest.mark.parametrize(
    "subprocess_server_with_mcp_resources_disabled",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_resource_gate_hides_tools_but_not_host_catalog(
    subprocess_server_with_mcp_resources_disabled: SubprocessServer,
) -> None:
    """resources.enabled=false only hides model tools, not Host resources."""
    base_url = subprocess_server_with_mcp_resources_disabled.base_url
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_response = await client.post(f"{base_url}/session", json={})
        assert session_response.status_code in (200, 201), session_response.text
        session_id = session_response.json().get("id") or session_response.json().get("sessionID")
        assert session_id
        message_response = await client.post(
            f"{base_url}/session/{session_id}/message",
            json={"parts": [{"type": "text", "text": "initialize tools"}]},
        )
        assert message_response.status_code in (200, 201, 202), message_response.text
        catalog_response = await client.get(f"{base_url}/experimental/resource")
        tools_response = await client.get(f"{base_url}/experimental/tool/ids")
        resource_message_response = await client.post(
            f"{base_url}/session/{session_id}/message",
            json={
                "noReply": True,
                "parts": [
                    {
                        "type": "file",
                        "mime": "text/plain",
                        "url": "",
                        "source": {
                            "type": "resource",
                            "clientName": "beta",
                            "uri": "file:///shared.txt",
                            "text": {"value": "shared resource", "start": 0, "end": 15},
                        },
                    }
                ],
            },
        )

    assert catalog_response.status_code == 200, catalog_response.text
    assert "beta:file:///shared.txt" in catalog_response.json()
    assert tools_response.status_code == 200, tools_response.text
    assert "read_mcp_resource" not in set(tools_response.json())
    assert resource_message_response.status_code in (200, 201, 202), resource_message_response.text


@pytest.mark.parametrize(
    "subprocess_server_with_mcp_resources",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_file_part_resource_source_is_accepted(
    subprocess_server_with_mcp_resources: SubprocessServer,
    e2e_config: Path,
) -> None:
    """OpenCode accepts a real FilePart carrying a ResourceSource."""
    _ = e2e_config
    base_url = subprocess_server_with_mcp_resources.base_url
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_response = await client.post(f"{base_url}/session", json={})
        assert session_response.status_code in (200, 201), session_response.text
        session_id = session_response.json().get("id") or session_response.json().get("sessionID")
        assert session_id

        payload: dict[str, Any] = {
            "noReply": True,
            "parts": [
                {
                    "type": "file",
                    "mime": "text/plain",
                    "url": "",
                    "source": {
                        "type": "resource",
                        "clientName": "beta",
                        "uri": "file:///shared.txt",
                        "text": {"value": "shared resource", "start": 0, "end": 15},
                    },
                }
            ],
        }
        message_response = await client.post(
            f"{base_url}/session/{session_id}/message",
            json=payload,
        )

    assert message_response.status_code in (200, 201, 202), message_response.text
