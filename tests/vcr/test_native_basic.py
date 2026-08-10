"""L3 VCR test — native agent basic completion (P1 pattern).

Pattern P1: single model call, assert response content/structure. Verifies
that model integration works at all — the agent receives a prompt, calls the
model once, and returns a non-empty response. VCR replays the recorded
``POST https://api.openai.com/v1/chat/completions`` exchange.

Cassette: ``tests/cassettes/vcr/test_native_basic/test_basic_completion.yaml``
([HUMAN-REQUIRED] — record with ``--record-mode=once`` and ``OPENAI_API_KEY``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dirty_equals import IsStr
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness.agents.events import StreamCompleteEvent


if TYPE_CHECKING:
    from wolfharness import AgentPool

pytestmark = pytest.mark.vcr

_MODULE_STEM = "test_native_basic"


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_basic_completion"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_basic_completion(vcr_pool: AgentPool) -> None:
    """A single model call returns a non-empty text response.

    Asserts the agent returns a ``StreamCompleteEvent`` whose message content
    is a non-empty string. This is the smoke test for the entire VCR stack:
    if this fails, either the cassette is missing/malformed or the model
    client wiring is broken.
    """
    agent = vcr_pool.get_agent("test_agent")
    result = await agent.run("Say hello in one short sentence.")

    # The result is a ChatMessage — assert it has non-empty content.
    assert result is not None
    assert result.content is not None
    # content may be a str or a list of content blocks; accept either.
    if isinstance(result.content, str):
        assert result.content == IsStr(min_length=1)
    else:
        assert len(result.content) > 0


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_basic_completion_streaming"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_basic_completion_streaming(vcr_pool: AgentPool) -> None:
    """Streaming the same prompt yields a ``StreamCompleteEvent`` at the end.

    Asserts the event stream terminates with ``StreamCompleteEvent`` carrying
    a non-empty message. Uses ``dirty_equals`` for fuzzy matching of the
    final event payload.
    """
    agent = vcr_pool.get_agent("test_agent")
    events: list[object] = [
        event async for event in agent.run_stream("Say hello in one short sentence.")
    ]

    assert events, "run_stream produced no events"
    last_event = events[-1]
    assert isinstance(last_event, StreamCompleteEvent)
    assert last_event.message is not None
    # content may be a str or a list of content blocks; verify non-empty.
    if isinstance(last_event.message.content, str):
        assert last_event.message.content == IsStr(min_length=1)
    else:
        assert len(last_event.message.content) > 0
