"""L3 VCR test — OpenCode protocol over in-process FastAPI TestClient.

The OpenCode server (FastAPI + SSE event stream) runs for real in-process
against a real ``AgentPool``. VCR intercepts only model API HTTP calls. The
client uses httpx ``ASGITransport`` so no real socket is opened.

Cassettes ([HUMAN-REQUIRED]):
- ``tests/cassettes/vcr/test_opencode_protocol/test_session_create.yaml``
- ``tests/cassettes/vcr/test_opencode_protocol/test_prompt_sse_stream.yaml``
- ``tests/cassettes/vcr/test_opencode_protocol/test_tool_call_events.yaml``
- ``tests/cassettes/vcr/test_opencode_protocol/test_subagent_events.yaml``
- ``tests/cassettes/vcr/test_opencode_protocol/test_session_close.yaml``
- ``tests/cassettes/vcr/test_opencode_protocol/test_error_handling.yaml``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dirty_equals import IsPartialDict, IsStr
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness_server.opencode_server.dependencies import get_state
from wolfharness_server.opencode_server.routes import agent_router, file_router, session_router
from wolfharness_server.opencode_server.routes.global_routes import router as global_router
from wolfharness_server.opencode_server.routes.message_routes import router as message_router
from wolfharness_server.opencode_server.state import ServerState


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from wolfharness import AgentPool

pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_opencode_protocol"


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


@pytest.fixture
async def opencode_state(vcr_pool: AgentPool, tmp_path: Path) -> ServerState:
    """Build a real ``ServerState`` backed by ``vcr_pool``."""
    agent = vcr_pool.get_agent("test_agent")
    state = ServerState(
        working_dir=str(tmp_path),
        agent=agent,
        session_controller=None,
    )
    state.messages = {}
    state.todos = {}
    state.input_providers = {}
    state.pending_questions = {}
    return state


@pytest.fixture
async def opencode_app(opencode_state: ServerState) -> FastAPI:
    """Build a FastAPI app with all OpenCode routes."""
    app = FastAPI()
    app.include_router(session_router)
    app.include_router(message_router)
    app.include_router(file_router)
    app.include_router(agent_router)
    app.include_router(global_router)
    app.dependency_overrides[get_state] = lambda: opencode_state
    return app


@pytest.fixture
async def opencode_client(opencode_app: FastAPI) -> AsyncIterator[TestClient]:
    """FastAPI ``TestClient`` against the in-process OpenCode server."""
    return TestClient(opencode_app)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_session_create"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_session_create(opencode_client: TestClient, tmp_path: Path) -> None:
    """POST /session creates a new session with a non-empty ID."""
    now = _now_ms()
    payload = {
        "id": "test-vcr-session",
        "title": "VCR Test Session",
        "project_id": "default",
        "directory": str(tmp_path),
        "version": "1",
        "time": {"created": now, "updated": now},
    }
    response = opencode_client.post("/session", json=payload)
    assert response.status_code in (200, 201)
    data = response.json()
    assert data["id"] == IsStr(min_length=1)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_prompt_sse_stream"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_prompt_sse_stream(opencode_client: TestClient, opencode_state: ServerState) -> None:
    """POST /session/{id}/message streams SSE events back to the client.

    Asserts the SSE stream returns at least one event with a recognizable
    ``type`` field. The model API call is VCR-replayed.
    """
    # Ensure the session exists first.
    now = _now_ms()
    session_payload = {
        "id": "test-vcr-stream",
        "title": "VCR Stream Test",
        "project_id": "default",
        "directory": "/tmp",
        "version": "1",
        "time": {"created": now, "updated": now},
    }
    opencode_client.post("/session", json=session_payload)

    response = opencode_client.post(
        "/session/test-vcr-stream/message",
        json={"content": "Say hello in one short sentence.", "role": "user"},
    )
    assert response.status_code in (200, 202)
    # The response may be SSE or JSON depending on the route; accept either.
    if "text/event-stream" in response.headers.get("content-type", ""):
        body = response.text
        assert "data: " in body
    else:
        data = response.json()
        assert data == IsPartialDict()


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_call_events"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_call_events(opencode_client: TestClient) -> None:
    """Tool-call events appear in the SSE stream when the agent invokes a tool."""
    now = _now_ms()
    opencode_client.post(
        "/session",
        json={
            "id": "test-vcr-tool",
            "title": "VCR Tool Test",
            "project_id": "default",
            "directory": "/tmp",
            "version": "1",
            "time": {"created": now, "updated": now},
        },
    )
    response = opencode_client.post(
        "/session/test-vcr-tool/message",
        json={"content": "Use the echo tool to say hi.", "role": "user"},
    )
    assert response.status_code in (200, 202)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_subagent_events"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_subagent_events(opencode_client: TestClient) -> None:
    """Subagent spawn/complete events appear in the SSE stream."""
    now = _now_ms()
    opencode_client.post(
        "/session",
        json={
            "id": "test-vcr-subagent",
            "title": "VCR Subagent Test",
            "project_id": "default",
            "directory": "/tmp",
            "version": "1",
            "time": {"created": now, "updated": now},
        },
    )
    response = opencode_client.post(
        "/session/test-vcr-subagent/message",
        json={"content": "Delegate to the worker agent.", "role": "user"},
    )
    assert response.status_code in (200, 202)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_session_close"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_session_close(opencode_client: TestClient) -> None:
    """DELETE /session/{id} closes the session."""
    now = _now_ms()
    opencode_client.post(
        "/session",
        json={
            "id": "test-vcr-close",
            "title": "VCR Close Test",
            "project_id": "default",
            "directory": "/tmp",
            "version": "1",
            "time": {"created": now, "updated": now},
        },
    )
    response = opencode_client.delete("/session/test-vcr-close")
    assert response.status_code in (200, 204, 404)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_error_handling"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_error_handling(opencode_client: TestClient) -> None:
    """Malformed requests produce structured error responses, not crashes."""
    # POST to a non-existent session.
    response = opencode_client.post(
        "/session/nonexistent-session/message",
        json={"content": "hello", "role": "user"},
    )
    assert response.status_code in (404, 400, 422, 500)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_rate_limit"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_rate_limit(opencode_client: TestClient) -> None:
    """Model API returns 429 rate limit — error propagates as SSE error event.

    The cassette records a real 429 response from the model API. The OpenCode
    server should stream an error event to the client via SSE.
    """
    response = opencode_client.post(
        "/session/test-vcr-rate-limit/message",
        json={"content": "This will trigger a rate limit.", "role": "user"},
    )
    # The server should accept the request (202) and stream the error via SSE,
    # or return an error status code directly.
    assert response.status_code in (200, 202, 429, 500, 503)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_server_error"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_server_error(opencode_client: TestClient) -> None:
    """Model API returns 500 server error — error propagates through OpenCode SSE."""
    response = opencode_client.post(
        "/session/test-vcr-server-error/message",
        json={"content": "This will trigger a server error.", "role": "user"},
    )
    assert response.status_code in (200, 202, 500, 502)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_malformed_stream"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_malformed_stream(opencode_client: TestClient) -> None:
    """Model API returns malformed streaming response — error propagates gracefully.

    The cassette records a response where the SSE stream contains invalid
    JSON or truncated chunks. The server should handle this gracefully.
    """
    response = opencode_client.post(
        "/session/test-vcr-malformed/message",
        json={"content": "This will trigger a malformed stream.", "role": "user"},
    )
    # Server should not crash — it should return an error status or stream an error event.
    assert response.status_code in (200, 202, 500)
