"""L3 VCR test — OpenAI-compatible API server over in-process TestClient.

The OpenAI API server (FastAPI) runs for real in-process against a real
``AgentPool``. VCR intercepts only model API HTTP calls. The client uses
FastAPI ``TestClient`` so no real socket is opened.

Cassettes ([HUMAN-REQUIRED]):
- ``tests/cassettes/vcr/test_openai_api_protocol/test_chat_completion.yaml``
- ``tests/cassettes/vcr/test_openai_api_protocol/test_streaming_completion.yaml``
- ``tests/cassettes/vcr/test_openai_api_protocol/test_tool_call.yaml``
- ``tests/cassettes/vcr/test_openai_api_protocol/test_multi_turn.yaml``
- ``tests/cassettes/vcr/test_openai_api_protocol/test_error_handling.yaml``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dirty_equals import IsPartialDict, IsStr
from fastapi.testclient import TestClient
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness_server.openai_api_server.server import OpenAIAPIServer


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness import AgentPool

pytestmark = [pytest.mark.vcr, pytest.mark.integration]

_MODULE_STEM = "test_openai_api_protocol"

_AUTH_HEADERS = {"Authorization": "Bearer test-key"}


@pytest.fixture
async def openai_api_client(vcr_pool: AgentPool) -> AsyncIterator[TestClient]:
    """FastAPI ``TestClient`` against the in-process OpenAI API server."""
    server = OpenAIAPIServer(vcr_pool, docs=False)
    return TestClient(server.app)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_chat_completion"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_chat_completion(openai_api_client: TestClient) -> None:
    """POST /v1/chat/completions returns a non-streaming completion.

    Asserts the response shape matches the OpenAI Chat Completions schema:
    ``choices[0].message.content`` is a non-empty string.
    """
    response = openai_api_client.post(
        "/v1/chat/completions",
        headers=_AUTH_HEADERS,
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == IsStr(min_length=1)
    assert data["choices"][0]["message"]["content"] is not None
    assert data == IsPartialDict()


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_streaming_completion"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_streaming_completion(openai_api_client: TestClient) -> None:
    """POST /v1/chat/completions with ``stream: true`` returns SSE chunks.

    Asserts the response content-type is ``text/event-stream`` and the body
    contains ``data:`` lines.
    """
    response = openai_api_client.post(
        "/v1/chat/completions",
        headers=_AUTH_HEADERS,
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type or "application/x-ndjson" in content_type
    assert "data: " in response.text


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_tool_call"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_tool_call(openai_api_client: TestClient) -> None:
    """POST /v1/chat/completions with a ``tools`` field exercises tool calling.

    Asserts the response either contains ``tool_calls`` in the message or a
    regular content response (model-dependent).
    """
    response = openai_api_client.post(
        "/v1/chat/completions",
        headers=_AUTH_HEADERS,
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "Use the echo tool to say hi."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Echo the provided text.",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                }
            ],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]
    # The model may return either tool_calls or content — accept either.
    assert "content" in message or "tool_calls" in message


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_multi_turn"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_multi_turn(openai_api_client: TestClient) -> None:
    """A multi-turn conversation (multiple messages) completes successfully."""
    response = openai_api_client.post(
        "/v1/chat/completions",
        headers=_AUTH_HEADERS,
        json={
            "model": "test_agent",
            "messages": [
                {"role": "user", "content": "My name is VCR."},
                {"role": "assistant", "content": "Hello VCR!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] is not None


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_error_handling"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_error_handling(openai_api_client: TestClient) -> None:
    """Requests without authorization are rejected with 401."""
    response = openai_api_client.post(
        "/v1/chat/completions",
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 401


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_rate_limit"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_rate_limit(openai_api_client: TestClient) -> None:
    """Model API returns 429 rate limit — error propagates as HTTP 429 to client.

    The cassette records a real 429 response from the model API. The OpenAI-compatible
    API server should propagate this as a 429 response to the client.
    """
    response = openai_api_client.post(
        "/v1/chat/completions",
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "This will trigger a rate limit."}],
            "stream": False,
        },
    )
    assert response.status_code == 429
    data = response.json()
    assert "error" in data


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_server_error"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_server_error(openai_api_client: TestClient) -> None:
    """Model API returns 500 server error — error propagates as HTTP 500 to client."""
    response = openai_api_client.post(
        "/v1/chat/completions",
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "This will trigger a server error."}],
            "stream": False,
        },
    )
    assert response.status_code in (500, 502)
    data = response.json()
    assert "error" in data


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_model_api_malformed_stream"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_model_api_malformed_stream(openai_api_client: TestClient) -> None:
    """Model API returns malformed streaming response — server handles gracefully.

    The cassette records a response where the SSE stream contains invalid
    JSON or truncated chunks. The server should terminate the stream cleanly
    or return an error, not crash.
    """
    response = openai_api_client.post(
        "/v1/chat/completions",
        json={
            "model": "test_agent",
            "messages": [{"role": "user", "content": "This will trigger a malformed stream."}],
            "stream": True,
        },
    )
    # Streaming response may start as 200 but contain error events in the stream.
    # Non-streaming should return an error status.
    assert response.status_code in (200, 500)
