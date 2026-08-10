"""Test configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
import pytest
import yamling

# Import minimal_pool fixture so it's available to all tests without explicit import.
from tests.fixtures.minimal_pool import minimal_pool  # noqa: F401
from wolfharness import Agent, AgentPool, AgentsManifest, NativeAgentConfig


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from vcr import VCR
    from vcr.cassette import Cassette
    from vcr.record_mode import RecordMode


# ---------------------------------------------------------------------------
# VCR configuration — must be imported before any VCR-using fixtures.
# ---------------------------------------------------------------------------

# Configure VCR logger to WARNING — it logs every request/response including
# binary content, which causes GitHub Actions log downloads to fail.
import logging


logging.getLogger("vcr.cassette").setLevel(logging.WARNING)


TEST_RESPONSE = "I am a test response"

# Inject dummy OPENAI_API_KEY and OPENAI_BASE_URL when the real ones are absent
# (e.g. fork PRs where GitHub Actions secrets are not available). VCR cassette
# replay and TestModel-based tests never make real HTTP calls, but the OpenAI SDK
# client constructor validates the key at init time — without this shim, VCR and
# E2E smoke tests fail with "Missing credentials" on fork PRs.
#
# OPENAI_BASE_URL must point to the same endpoint the cassettes were recorded
# against. VCR's match_on is ["method"] but the SDK must still target the same
# URL so VCR can find and replay the recorded interaction.
# When the real secrets ARE available (same-repo PRs), this is a no-op.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-for-ci"
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "http://api.ai.rootcloud.info/v1"


@pytest.fixture
def default_model() -> str:
    """Default model for testing."""
    return os.getenv("TEST_DEFAULT_MODEL") or "openai-chat:svc/glm-4.7"


@pytest.fixture
def vision_model() -> str:
    """Vision-capable model for testing."""
    return os.getenv("TEST_VISION_MODEL") or "openai-chat:svc/kimi-k2"


@pytest.fixture(scope="session", autouse=True)
def unset_anthropic_api_key():
    os.environ["ANTHROPIC_API_KEY"] = ""


@pytest.fixture(autouse=True)
def _patch_toolset_lifecycle(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    """Patch MCPToolset.__aenter__/__aexit__ to avoid real MCP connections.

    ``get_capabilities()`` eagerly enters MCPToolset instances on cache miss
    (issue #175 fix). Without this patch, every test calling
    ``get_capabilities()`` would try to spawn subprocesses or open network
    connections.

    The fake implementations track ``_running_count`` exactly like the real
    pydantic-ai MCPToolset, so tests can verify reference-counting behaviour
    without real connections.

    Tests that need the real MCPToolset can opt out with
    ``@pytest.mark.real_mcp``.
    """
    if request.node.get_closest_marker("real_mcp"):
        yield
        return

    original_aenter = MCPToolset.__aenter__
    original_aexit = MCPToolset.__aexit__

    async def fake_aenter(self):
        self._running_count += 1
        return self

    async def fake_aexit(self, exc_type, exc_val, exc_tb):
        if self._running_count > 0:
            self._running_count -= 1

    MCPToolset.__aenter__ = fake_aenter  # type: ignore[assignment]
    MCPToolset.__aexit__ = fake_aexit  # type: ignore[assignment]
    try:
        yield
    finally:
        MCPToolset.__aenter__ = original_aenter  # type: ignore[assignment]
        MCPToolset.__aexit__ = original_aexit  # type: ignore[assignment]


@pytest.fixture(scope="session", autouse=True)
def disable_logfire(tmp_path_factory):
    """Disable logfire for all tests and set up test directories."""
    from pathlib import Path

    # Set environment variable to disable logfire
    os.environ["LOGFIRE_DISABLE"] = "true"
    # Also disable observability entirely
    os.environ["OBSERVABILITY_ENABLED"] = "false"
    # Skip config dir override in CI - not needed and credentials aren't available anyway
    is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
    if not is_ci:
        # Use temp directory for Claude storage during tests
        claude_test_dir = tmp_path_factory.mktemp("claude_config")
        # Copy credentials file if it exists so integration tests can authenticate
        # Use copy instead of symlink for cross-platform compatibility (Windows needs admin/dev
        # mode for symlinks)
        real_creds = Path.home() / ".claude" / ".credentials.json"
        if real_creds.exists():
            import shutil

            test_creds = claude_test_dir / ".credentials.json"
            shutil.copy2(real_creds, test_creds)
        os.environ["CLAUDE_CONFIG_DIR"] = str(claude_test_dir)
        # Use temp directory for Codex data during tests
        codex_test_dir = tmp_path_factory.mktemp("codex_home")
        # Copy Codex auth file if it exists so integration tests can authenticate
        real_codex_auth = Path.home() / ".codex" / "auth.json"
        if real_codex_auth.exists():
            import shutil

            test_codex_auth = codex_test_dir / "auth.json"
            shutil.copy2(real_codex_auth, test_codex_auth)
        os.environ["CODEX_HOME"] = str(codex_test_dir)

    # Mock logfire configure to be a no-op
    try:
        import logfire

        original_configure = logfire.configure
        logfire.configure = lambda *args, **kwargs: None  # type: ignore
        yield
        logfire.configure = original_configure
    except ImportError:
        # logfire not available, nothing to disable
        yield


VALID_CONFIG = """\
responses:
  SupportResult:
    response_schema:
        type: inline
        description: Support agent response
        fields:
            advice:
                type: str
                description: Support advice
            risk:
                type: int
                ge: 0
                le: 100
  ResearchResult:
    response_schema:
        type: inline
        description: Research agent response
        fields:
            findings:
                type: str
                description: Research findings

agents:
  support:
    type: native
    display_name: Support Agent
    model: {default_model}
    output_type: SupportResult
    system_prompt:
      - You are a support agent
      - "Context: {{data}}"
  researcher:
    type: native
    display_name: Research Agent
    model: {default_model}
    output_type: ResearchResult
    system_prompt: You are a researcher
"""


@pytest.fixture
def valid_config(default_model: str) -> dict[str, Any]:
    """Fixture providing valid agent configuration."""
    return yamling.load_yaml(VALID_CONFIG.format(default_model=default_model), verify_type=dict)


@pytest.fixture
def test_agent() -> Agent[None]:
    """Create an agent with TestModel for testing."""
    model = TestModel(custom_output_text=TEST_RESPONSE)
    return Agent(name="test-agent", model=model)


@pytest.fixture
def manifest():
    """Create test manifest with some agents."""
    agent_1 = NativeAgentConfig(name="agent1", model="test")
    agent_2 = NativeAgentConfig(name="agent2", model="test")
    return AgentsManifest(agents={"agent1": agent_1, "agent2": agent_2})


@pytest.fixture
async def pool(manifest):
    """Create test pool with agents."""
    async with AgentPool(manifest) as pool:
        yield pool


# Model override mapping for custom endpoints without gpt-4o access.
# Tests that hardcode "openai:gpt-4o" or "openai:gpt-4o-mini" are
# transparently remapped to a model available on the custom endpoint.
# Uses "openai-chat:" prefix so the remapped model uses OpenAIChatModel
# (Chat Completions API /v1/chat/completions), matching existing VCR
# cassettes which were all recorded against the chat completions endpoint.
_raw_override = os.getenv("TEST_MODEL_OVERRIDE", "openai-chat:gpt-5-nano")
# Ensure Chat Completions API for VCR cassette compatibility: convert
# "openai:" prefix to "openai-chat:" so OpenAIChatModel is created.
if _raw_override.startswith("openai:"):
    _raw_override = "openai-chat:" + _raw_override[len("openai:") :]
_DEFAULT_REMAP = _raw_override
_MODEL_REMAP = {
    "openai:gpt-4o": _DEFAULT_REMAP,
    "openai:gpt-4o-mini": _DEFAULT_REMAP,
}


@pytest.fixture(scope="session", autouse=True)
def remap_hardcoded_test_models():
    """Remap hardcoded gpt-4o/gpt-4o-mini to a custom-available model.

    Controlled via the ``TEST_MODEL_OVERRIDE`` environment variable.
    """
    from unittest.mock import patch

    from wolfharness.utils import model_helpers

    original = model_helpers.infer_model

    def _patched_infer(model):
        if isinstance(model, str) and model in _MODEL_REMAP:
            return original(_MODEL_REMAP[model])
        return original(model)

    with (
        patch.object(model_helpers, "infer_model", _patched_infer),
    ):
        yield


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip credential-dependent and thinking-incompatible tests.

    - ``real_model``: skipped when ``OPENAI_API_KEY`` is not set
    - ``incompatible_with_thinking``: skipped when ``TEST_DEFAULT_MODEL``
      points to a thinking-mode model (deepseek, kimi) — see issue #84
    """
    _thinking_model_prefixes = ("deepseek", "kimi", "moonshot")

    model = os.getenv("TEST_DEFAULT_MODEL", "")
    is_thinking_model = any(p in model for p in _thinking_model_prefixes)

    for item in items:
        if (
            "real_model" in item.keywords
            and not os.environ.get("OPENAI_API_KEY")
            and not os.environ.get("MODEL_GATEWAY_URL")
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="OPENAI_API_KEY not set — skipping credential-dependent test",
                )
            )
        if (
            "real_model" in item.keywords
            and os.environ.get("OPENAI_API_KEY") == "sk-test-dummy-for-ci"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="OPENAI_API_KEY is a dummy CI value — skipping real model test",
                )
            )
        if "incompatible_with_thinking" in item.keywords and is_thinking_model:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"TEST_DEFAULT_MODEL='{model}' uses thinking mode — "
                    "structured output (tool_choice: 'required') not supported (issue #84)",
                )
            )


# ---------------------------------------------------------------------------
# ALLOW_MODEL_REQUESTS gate (tasks 4.2-4.4)
# ---------------------------------------------------------------------------

ALLOW_MODEL_REQUESTS: bool = False
"""Global gate — blocks ALL real model API calls by default.

This is the single most important safety mechanism. It prevents tests from
accidentally making real (and potentially expensive) API calls.

- **Default**: ``False`` — all real model calls blocked at the httpx transport level.
- **Override**: Use the ``allow_model_requests`` fixture for tests that need
  real calls (e.g., recording cassettes).
- **VCR**: VCR-mocked tests do NOT need ``allow_model_requests`` — VCR
  intercepts at a higher level and the ``_block_model_requests`` fixture
  skips VCR-marked tests.

See ``tests/AGENTS.md`` § "Key Safety Mechanism: ALLOW_MODEL_REQUESTS" for details.
"""

# Sync pydantic-ai's own gate with ours so ``check_allow_model_requests()``
# inside model classes also blocks by default.
import pydantic_ai.models  # noqa: E402


pydantic_ai.models.ALLOW_MODEL_REQUESTS = ALLOW_MODEL_REQUESTS


def _model_requests_blocked() -> bool:
    """Return True if real model API calls are currently blocked."""
    return not pydantic_ai.models.ALLOW_MODEL_REQUESTS


@pytest.fixture(autouse=True)
def _block_model_requests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Install an httpx ``MockTransport`` blocking handler when gate is closed.

    This is the transport-level enforcement of the ``ALLOW_MODEL_REQUESTS`` gate.
    It prevents ANY httpx ``AsyncClient`` from making a real network request when
    the gate is closed, regardless of whether the model class calls
    ``check_allow_model_requests()``.

    Skipped for:
    - Tests marked ``@pytest.mark.vcr`` — VCR intercepts at a higher level.
    - Tests requesting the ``allow_model_requests`` fixture — the gate is opened.
    - Tests marked ``@pytest.mark.real_model`` — these are expected to make real
      calls (and are auto-skipped when ``OPENAI_API_KEY`` is unset).
    """
    # VCR tests: VCR cassette replay intercepts at a higher level.
    if request.node.get_closest_marker("vcr") is not None:
        yield
        return
    # Tests that explicitly request allow_model_requests open the gate.
    if "allow_model_requests" in request.fixturenames:
        yield
        return
    # real_model tests are expected to make real calls (auto-skipped without API key).
    if request.node.get_closest_marker("real_model") is not None:
        yield
        return
    # Gate already open (e.g. a parent fixture set it).
    if not _model_requests_blocked():
        yield
        return

    import httpx

    block_message = (
        "Real model API calls are blocked (ALLOW_MODEL_REQUESTS=False). "
        "Use @pytest.mark.vcr for VCR tests or the allow_model_requests "
        "fixture for real API tests."
    )

    def _blocking_handler(httpx_request: httpx.Request) -> httpx.Response:
        raise RuntimeError(
            f"{block_message} URL: {httpx_request.url}",
        )

    original_async_init = httpx.AsyncClient.__init__

    def _patched_async_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        # Only inject the blocking transport if no explicit transport was provided.
        transport = kwargs.get("transport")
        if transport is None:
            kwargs["transport"] = httpx.MockTransport(_blocking_handler)
        original_async_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _patched_async_init)

    original_sync_init = httpx.Client.__init__

    def _patched_sync_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        transport = kwargs.get("transport")
        if transport is None:
            kwargs["transport"] = httpx.MockTransport(_blocking_handler)
        original_sync_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", _patched_sync_init)

    yield


@pytest.fixture
def allow_model_requests() -> Iterator[None]:
    """Temporarily allow real model API calls for the duration of this test.

    Use this fixture when a test needs to make real API calls (e.g. recording
    a new VCR cassette with ``--record-mode=once``). VCR cassette replay tests
    do NOT need this fixture — VCR intercepts at a higher level.

    ```python
    async def test_record_cassette(allow_model_requests):
        agent = Agent(model="openai:gpt-4o-mini", ...)
        result = await agent.run("Hello")  # Real API call allowed
    ```
    """
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


# ---------------------------------------------------------------------------
# VCR configuration fixtures (tasks 4.5-4.7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vcr_config(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Module-scoped VCR configuration.

    Returns a dict consumed by ``pytest-recording`` to configure vcrpy.
    Cassettes are sanitized via the custom ``json_body_serializer`` (registered
    in ``pytest_recording_configure`` below) which decompresses bodies, filters
    headers, and scrubs credentials.
    """
    from pathlib import Path

    module_stem = request.module.__name__.rsplit(".", 1)[-1]
    cassettes_dir = str(Path(__file__).parent / "cassettes" / "vcr" / module_stem)
    return {
        "cassette_library_dir": cassettes_dir,
        "filter_headers": ["authorization", "x-api-key", "cookie", "set-cookie"],
        "decode_compressed_response": True,
        "match_on": ["method"],
    }


def pytest_recording_configure(config: Any, vcr: VCR) -> None:
    """Register the custom JSON body serializer and request/response hooks with VCR.

    This hook is called once per session by ``pytest-recording``. It registers:
    - ``json_body_serializer`` as the YAML serializer (replaces the default).
    - ``before_record_request`` for credential scrubbing on outgoing requests.
    - ``before_record_response`` for transient header stripping on responses.
    """
    from tests.vcr import json_body_serializer

    vcr.register_serializer("yaml", json_body_serializer)

    vcr.before_record_request = before_record_request
    vcr.before_record_response = before_record_response


def before_record_request(request: Any) -> Any:
    """VCR hook: scrub credentials and filter non-model requests before recording.

    Strips API key prefixes (``sk-...``) from the request URI and body so the
    recorded cassette contains no secrets. Also filters out litellm price
    lookup requests (``raw.githubusercontent.com``) which are model-dependent
    and non-deterministic across environments.
    """
    from tests.vcr.json_body_serializer import scrub_credentials

    # Filter out litellm price lookups — these are model-dependent and may not
    # be triggered in all environments (e.g., CI with a different model).
    if hasattr(request, "uri") and "raw.githubusercontent.com" in str(getattr(request, "uri", "")):
        return None

    # Scrub credentials from the URI
    if hasattr(request, "uri") and isinstance(request.uri, str):
        request.uri = scrub_credentials(request.uri)
    # Scrub credentials from the body
    if hasattr(request, "body") and isinstance(request.body, str | bytes):
        body: str
        if isinstance(request.body, bytes):
            body = request.body.decode("utf-8", errors="ignore")
        else:
            body = request.body
        scrubbed = scrub_credentials(body)
        request.body = scrubbed.encode("utf-8") if isinstance(request.body, bytes) else scrubbed
    return request


def before_record_response(response: Any) -> Any:
    """VCR hook: strip transient headers from a response before recording it.

    Removes headers that change between requests (``date``, ``cf-ray``,
    ``x-amz-request-id``, ``set-cookie``, ``request-id``, etc.) so cassettes
    are deterministic across replays.
    """
    transient_headers: set[str] = {
        "date",
        "cf-ray",
        "cf-cache-status",
        "x-amz-request-id",
        "x-amz-id-2",
        "x-request-id",
        "request-id",
        "server",
        "via",
        "set-cookie",
        "x-served-by",
        "x-cache",
        "x-cache-hits",
        "x-timer",
        "age",
        "etag",
        "last-modified",
        "strict-transport-security",
        "x-content-type-options",
        "x-xss-protection",
        "x-frame-options",
    }
    headers: Any = getattr(response, "headers", None)
    if headers is None:
        return response
    # vcrpy response headers are a dict-like; lowercase keys for matching.
    if isinstance(headers, dict):
        new_headers = {k: v for k, v in headers.items() if k.lower() not in transient_headers}
        response.headers = new_headers  # type: ignore[attr-defined]
    return response


# ---------------------------------------------------------------------------
# httpx resource management (tasks 4.8-4.9)
# ---------------------------------------------------------------------------

_HttpClientCache: type = dict[tuple[Any, Any], Any]


@pytest.fixture(autouse=True)
def track_httpx_clients(monkeypatch: pytest.MonkeyPatch) -> Iterator[_HttpClientCache]:
    """Track all httpx ``AsyncClient`` instances created during a test.

    Monkeypatches ``pydantic_ai.models.create_async_http_client`` (and any
    module-level aliases) so that calls with the same ``(timeout, connect)``
    args reuse the same client. On teardown, all tracked clients are closed —
    no process-global state leaks across tests.

    This is a sync fixture so it applies to both sync and async tests. For
    async tests, the companion ``close_httpx_clients`` fixture handles async
    cleanup first.
    """
    cache: dict[tuple[Any, Any], Any] = {}
    original = pydantic_ai.models.create_async_http_client

    def _cached_per_test(**kwargs: Any) -> Any:
        import httpx

        timeout = kwargs.get("timeout", httpx.USE_CLIENT_DEFAULT)
        connect = kwargs.get("connect", 5)
        key = (timeout, connect)
        client = cache.get(key)
        if client is None or client.is_closed:
            client = original(**kwargs)
            cache[key] = client
        return client

    # Patch the function in every loaded module that imported it directly.
    for mod in list(sys.modules.values()):
        mod_dict = getattr(mod, "__dict__", None)
        if mod_dict is not None and mod_dict.get("create_async_http_client", None) is original:
            monkeypatch.setattr(mod, "create_async_http_client", _cached_per_test)

    yield cache

    # Fallback sync close for any clients not closed by close_httpx_clients.
    import asyncio

    unclosed = [c for c in cache.values() if not c.is_closed]
    if unclosed:  # pragma: no cover
        for client in unclosed:
            try:
                asyncio.get_event_loop().run_until_complete(client.aclose())
            except RuntimeError:
                # No event loop — create one for cleanup.
                asyncio.run(client.aclose())


@pytest.fixture(autouse=True)
async def close_httpx_clients(
    track_httpx_clients: _HttpClientCache,
) -> AsyncIterator[None]:
    """Close tracked httpx clients after async tests."""
    yield
    for client in track_httpx_clients.values():
        if not client.is_closed:
            await client.aclose()


@pytest.fixture
def disable_ssrf_protection_for_vcr() -> Iterator[None]:
    """Disable SSRF protection for VCR compatibility.

    VCR cassettes record requests with the original hostname. Since pydantic-ai's
    SSRF protection resolves hostnames to IPs before making requests, we need to
    disable the validation for VCR tests to match the pre-recorded cassettes.

    This fixture patches ``validate_and_resolve_url`` to return the hostname in
    place of the resolved IP, allowing the request URL to use the original
    hostname so VCR matching succeeds.
    """
    from unittest.mock import patch

    try:
        from pydantic_ai._ssrf import ResolvedUrl, extract_host_and_port
    except ImportError:
        # SSRF protection not available — nothing to patch.
        yield
        return

    async def _mock_validate_and_resolve(url: str, allow_local: bool) -> ResolvedUrl:
        hostname, path, port, is_https = extract_host_and_port(url)
        return ResolvedUrl(
            resolved_ip=hostname,
            hostname=hostname,
            port=port,
            is_https=is_https,
            path=path,
        )

    with patch("pydantic_ai._ssrf.validate_and_resolve_url", _mock_validate_and_resolve):
        yield


# ---------------------------------------------------------------------------
# Strict cassette usage (task 4.11)
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    """Register pytest command-line options."""
    parser.addoption(
        "--strict-vcr-cassette-usage",
        action="store_true",
        default=False,
        help="Fail when a loaded VCR cassette has no interactions played, "
        "not only when playback leaves a stale tail.",
    )


def _check_vcr_cassette_usage(vcr: Cassette, strict_usage: bool) -> None:
    """Fail if a VCR cassette has unplayed interactions."""
    if vcr.play_count == 0 and not strict_usage:
        return
    unused = [i for i in range(len(vcr)) if vcr.play_counts.get(i, 0) == 0]
    if unused:
        pytest.fail(
            f"Cassette {getattr(vcr, '_path', '<unknown>')} did not play all "
            f"interactions: played {vcr.play_count}/{len(vcr)}; unused indexes: {unused}",
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[Any]:
    """Attach setup/call/teardown reports to the item for fail_partially_used_vcr_cassettes."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture(autouse=True)
def fail_partially_used_vcr_cassettes(
    request: pytest.FixtureRequest,
    vcr: Cassette | None,
) -> Iterator[None]:
    """Fail VCR-marked tests that leave cassette interactions unplayed.

    This catches stale cassettes where the test logic changed to make fewer
    requests than the cassette contains, or where the test errored before
    playing all interactions.

    Skipped when:
    - The test is not VCR-marked (``vcr`` fixture is ``None``).
    - The cassette is in record mode (recording, not replaying).
    - The test was skipped or failed in setup/call (don't pile on).
    - All interactions were played.
    """
    yield
    setup_report = getattr(request.node, "rep_setup", None)
    call_report = getattr(request.node, "rep_call", None)
    if any(
        getattr(report, "skipped", False) or getattr(report, "failed", False)
        for report in (setup_report, call_report)
        if report is not None
    ):
        return
    if vcr is None:
        return
    record_mode: RecordMode = vcr.record_mode  # type: ignore[assignment]
    if record_mode != "none" or vcr.all_played:
        return
    strict_usage = bool(request.config.getoption("--strict-vcr-cassette-usage"))
    _check_vcr_cassette_usage(vcr, strict_usage)
