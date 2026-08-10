"""L4 subprocess E2E tests for the OpenAI-compatible API server (wolfharness serve-api).

L4a smoke tests (@pytest.mark.e2e, NOT slow):
    - test_server_startup: spawn serve-api, verify HTTP health
    - test_chat_completion: POST /v1/chat/completions, verify response
    - test_server_shutdown: verify clean process exit after test

L4b full tests (@pytest.mark.e2e + @pytest.mark.slow):
    - test_streaming_completion: streaming chat completion via SSE
    - test_multi_turn: multi-turn conversation

All tests use model: test (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


AUTH_HEADERS = {"Authorization": "Bearer test-key"}

# ---------------------------------------------------------------------------
# L4a Smoke Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-api", "is_stdio": False}],
    indirect=True,
)
async def test_server_startup(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Start serve-api, verify HTTP server is responding."""
    assert subprocess_server.process.returncode is None, "OpenAI API server process exited early"
    assert subprocess_server.port > 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{subprocess_server.base_url}/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "test_agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, (
            f"Server not responding properly: status={resp.status_code}, body={resp.text[:500]}"
        )


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-api", "is_stdio": False}],
    indirect=True,
)
async def test_chat_completion(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: POST /v1/chat/completions, verify chat completion response."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "test_agent",
                "messages": [{"role": "user", "content": "Hello, test agent!"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, (
            f"Chat completion unexpected status: {resp.status_code}: {resp.text[:500]}"
        )
        data = resp.json()
        assert data["model"] == "test_agent", (
            f"Expected model='test_agent', got: {data.get('model')}"
        )
        assert "choices" in data, f"Expected 'choices' in response: {data}"
        assert len(data["choices"]) > 0, f"Expected at least 1 choice: {data}"
        content = data["choices"][0].get("message", {}).get("content")
        assert content is not None, f"Expected content in message: {data['choices'][0]}"


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-api", "is_stdio": False}],
    indirect=True,
)
async def test_server_shutdown(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4a: Verify server is responsive then shuts down cleanly."""
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "test_agent", "messages": []},
        )
        # Any response (even 401) means the server is up.
        assert resp.status_code < 500

    assert subprocess_server.process.returncode is None


# ---------------------------------------------------------------------------
# L4b Full Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-api", "is_stdio": False}],
    indirect=True,
)
async def test_streaming_completion(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4b: Streaming chat completion via SSE."""
    base_url = subprocess_server.base_url

    async with (
        httpx.AsyncClient(timeout=30.0) as client,
        client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={
                "model": "test_agent",
                "messages": [{"role": "user", "content": "Stream a response"}],
                "stream": True,
            },
        ) as resp,
    ):
        assert resp.status_code == 200, f"Streaming completion failed: {resp.status_code}"
        # Read the SSE stream and verify we get chunks.
        chunks: list[str] = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line)
            # Stop at SSE done marker.
            stripped = line.strip()
            if stripped.endswith("DONE]"):
                break
        assert len(chunks) > 0, "Expected at least one SSE chunk"


@pytest.mark.slow
@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-api", "is_stdio": False}],
    indirect=True,
)
async def test_multi_turn(subprocess_server: SubprocessServer, e2e_config: Path) -> None:
    """L4b: Multi-turn conversation via chat completions."""
    base_url = subprocess_server.base_url

    messages: list[dict[str, str]] = [
        {"role": "user", "content": "First message in conversation"},
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        # First turn.
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={"model": "test_agent", "messages": messages, "stream": False},
        )
        assert resp.status_code == 200, f"First turn failed: {resp.status_code}"
        data = resp.json()
        assistant_content = data["choices"][0]["message"]["content"]
        assert assistant_content is not None

        # Second turn with conversation history.
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": "Second message"})

        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={"model": "test_agent", "messages": messages, "stream": False},
        )
        assert resp.status_code == 200, f"Second turn failed: {resp.status_code}"
        data2 = resp.json()
        assert data2["choices"][0]["message"]["content"] is not None
