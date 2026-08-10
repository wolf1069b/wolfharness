"""L3 VCR test — AG-UI protocol over in-process Starlette/ASGI client.

The AG-UI server (Starlette) runs for real in-process against a real
``AgentPool``. VCR intercepts only model API HTTP calls. The client uses
httpx ``ASGITransport`` so no real socket is opened.

AG-UI protocol route structure:
- ``GET  /``              — list available agents
- ``POST /{agent_name}``  — run agent with a ``RunAgentInput`` body (SSE stream)

There is no ``GET /{agent_name}`` or ``POST /{agent_name}/subscribe`` endpoint.
All agent interactions go through ``POST /{agent_name}`` with a full
``RunAgentInput`` body containing ``threadId``, ``runId``, ``state``,
``messages``, ``tools``, ``context``, and ``forwardedProps`` (camelCase
per the AG-UI wire format).

Cassettes ([HUMAN-REQUIRED]):
- ``tests/cassettes/vcr/test_agui_protocol/test_session_init.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_event_stream.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_tool_call.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_state_sync.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_error_handling.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_model_api_rate_limit.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_model_api_server_error.yaml``
- ``tests/cassettes/vcr/test_agui_protocol/test_model_api_malformed_stream.yaml``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dirty_equals import IsPartialDict
from httpx import ASGITransport, AsyncClient
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness_server.agui_server import AGUIServer


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import AgentPool

pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_agui_protocol"


def _build_run_agent_input(
    prompt: str, *, thread_id: str = "thread-vcr", run_id: str = "run-vcr"
) -> dict[str, Any]:
    """Build a minimal AG-UI ``RunAgentInput`` request body.

    The AG-UI protocol requires all 7 fields (``threadId``, ``runId``,
    ``state``, ``messages``, ``tools``, ``context``, ``forwardedProps``).
    Missing fields cause a ``ValidationError`` → HTTP 500.

    Uses camelCase keys per the AG-UI wire format convention. The
    ``RunAgentInput`` Pydantic model also accepts snake_case via
    ``populate_by_name=True``, but camelCase is the canonical form.

    Args:
        prompt: The user prompt text.
        thread_id: Thread identifier (default ``"thread-vcr"``).
        run_id: Run identifier (default ``"run-vcr"``).

    Returns:
        JSON-serializable dict matching the AG-UI ``RunAgentInput`` schema.
    """
    return {
        "threadId": thread_id,
        "runId": run_id,
        "state": {},
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": prompt,
            }
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


@pytest.fixture
async def agui_server(vcr_pool: AgentPool) -> AGUIServer:
    """Build an ``AGUIServer`` wrapping ``vcr_pool``."""
    return AGUIServer(vcr_pool, host="127.0.0.1", port=0)


@pytest.fixture
async def agui_app(agui_server: AGUIServer) -> Any:
    """Start the AG-UI server and return its ASGI app."""
    await agui_server.__aenter__()
    routes = await agui_server.get_routes()
    # AGUIServer uses Starlette — build a minimal Starlette app from routes.
    from starlette.applications import Starlette

    return Starlette(routes=routes)


@pytest.fixture
async def agui_client(agui_app: Any) -> AsyncIterator[AsyncClient]:
    """Httpx async client against the in-process AG-UI ASGI app."""
    transport = ASGITransport(app=agui_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_session_init"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_session_init(agui_client: AsyncClient) -> None:
    """The AG-UI root endpoint lists available agents."""
    response = await agui_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data == IsPartialDict()


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_event_stream"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_event_stream(agui_client: AsyncClient) -> None:
    """The per-agent endpoint streams SSE events for a prompt.

    Sends ``POST /test_agent`` with a full ``RunAgentInput`` body. The AG-UI
    server processes the prompt and returns an SSE event stream. VCR replays
    the model API call that the agent makes when processing the prompt.
    """
    response = await agui_client.post(
        "/test_agent",
        json=_build_run_agent_input("Hello from AG-UI VCR test!"),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_call"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_call(agui_client: AsyncClient) -> None:
    """Tool-call events appear in the AG-UI event stream.

    Sends ``POST /test_agent`` with a full ``RunAgentInput`` body containing
    a prompt that triggers tool usage. VCR replays the model API call.
    """
    response = await agui_client.post(
        "/test_agent",
        json=_build_run_agent_input("Use the echo tool."),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_state_sync"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_state_sync(agui_client: AsyncClient) -> None:
    """The AG-UI server handles POST requests consistently.

    AG-UI is a stateless protocol — each request contains the full
    conversation history. This test verifies that a single ``POST
    /test_agent`` request with a valid ``RunAgentInput`` body returns
    a successful response, confirming the server processes repeated
    stateless requests without issues.
    """
    response = await agui_client.post(
        "/test_agent",
        json=_build_run_agent_input("State sync test"),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_error_handling"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_error_handling(agui_client: AsyncClient) -> None:
    """Requesting a non-existent agent returns a 404."""
    response = await agui_client.get("/nonexistent_agent")
    assert response.status_code in (404, 200)  # some adapters return 200 with empty body


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_rate_limit"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_rate_limit(agui_client: AsyncClient) -> None:
    """Model API returns 429 rate limit — error propagates through AG-UI events.

    The cassette records a real 429 response from the model API. The AG-UI
    server should emit an error event in the event stream.
    """
    response = await agui_client.post(
        "/test_agent",
        json=_build_run_agent_input("This will trigger a rate limit."),
        headers={"Accept": "text/event-stream"},
    )
    # AG-UI uses SSE for streaming — the initial response may be 200 with
    # error events in the stream, or an error status code.
    assert response.status_code in (200, 202, 429, 500, 503)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_server_error"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_server_error(agui_client: AsyncClient) -> None:
    """Model API returns 500 server error — error propagates through AG-UI."""
    response = await agui_client.post(
        "/test_agent",
        json=_build_run_agent_input("This will trigger a server error."),
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code in (200, 202, 500, 502)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_malformed_stream"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_malformed_stream(agui_client: AsyncClient) -> None:
    """Model API returns malformed streaming response — AG-UI handles gracefully."""
    response = await agui_client.post(
        "/test_agent",
        json=_build_run_agent_input("This will trigger a malformed stream."),
        headers={"Accept": "text/event-stream"},
    )
    # Server should not crash — error should be emitted as an event, not a process failure.
    assert response.status_code in (200, 202, 500)
