# AgentPool Testing Guide

This document defines the testing architecture, layer requirements, and best practices for AgentPool.
All contributors MUST follow these guidelines when writing or modifying tests.

## L1-L4 Layer Taxonomy

AgentPool uses a 4-layer conceptual taxonomy (L1-L4) for discussing test types. These are **conceptual labels**, not actual pytest markers. The actual pytest markers are `unit`, `integration`, `vcr`, and `e2e`.

```
     ┌──────────────────┐
     │  L4 — E2E        │  @pytest.mark.e2e       — real server process + protocol client  (nightly)
     ├──────────────────┤
     │  L3 — VCR        │  @pytest.mark.vcr       — real API responses replayed from cassette (CI)
     ├──────────────────┤
     │  L2 — Integration │  @pytest.mark.integration — component wiring, mocked deps (CI)
     ├──────────────────┤
     │  L1 — Unit       │  @pytest.mark.unit      — logic-level, TestModel/FunctionModel (CI)
     └──────────────────┘
```

| Layer | Concept | Marker | Tools | Speed | Network | CI | Purpose |
|-------|---------|--------|-------|-------|---------|-----|---------|
| **L1** | Unit | `@pytest.mark.unit` | `TestModel`, `FunctionModel`, mocks | Fast (<100ms) | None | ✅ Always | Logic branches, boundary conditions, data transforms |
| **L2** | Integration | `@pytest.mark.integration` | Real `AgentPool` + `TestModel`, FastAPI `TestClient` | Fast (<500ms) | None | ✅ Always | Component wiring, capability registration, event conversion, protocol handlers (mocked deps) |
| **L3** | VCR | `@pytest.mark.vcr` | `pytest-recording` + `vcrpy` + cassettes | Fast-Medium (<5s) | None (replay) | ✅ Always | Real model API response format, streaming event sequences, tool call structures, protocol transport |
| **L4** | Subprocess E2E | `@pytest.mark.e2e` | Real `wolfharness serve-*` process + protocol client | Slow (5-30s) | Optional | L4a: ✅ PR smoke, L4b: ❌ Nightly | Server startup, stdio transport, process lifecycle, real I/O timing |

Tests carry exactly ONE primary layer marker. L3 VCR tests that also exercise protocol integration MAY additionally carry `@pytest.mark.integration` as a secondary marker for filtering. The primary marker determines CI stage assignment.

### L4 Sub-layers

L4 is split into two sub-layers for CI efficiency:

| Sub-layer | Scope | Speed | CI | Marker |
|-----------|-------|-------|-----|--------|
| **L4a** — Smoke | Server startup + 1 basic prompt + shutdown | ~30s | ✅ PR (optional stage) | `@pytest.mark.e2e` (NOT `@pytest.mark.slow`) |
| **L4b** — Full | Multi-turn, tool calls, subagent, cancellation, error paths | 5-30min | ❌ Nightly | `@pytest.mark.e2e` + `@pytest.mark.slow` |

L4a catches "server won't start" and "basic prompt fails" regressions within minutes of a PR. L4b catches deeper integration issues overnight.

## Which Layers Do I Need?

| Feature Type | L1 | L2 | L3 | L4a (smoke) | L4b (full) |
|--------------|----|----|----|-------------|------------|
| New capability (`AbstractCapability` subclass) | ✅ Required | ✅ Required | — | — | — |
| New protocol handler / event type | ✅ Required | ✅ Required | ✅ Required | ✅ Recommended | Recommended |
| New agent type | ✅ Required | ✅ Required | ✅ Required | ✅ Recommended | Recommended |
| Bug fix | ✅ Required (reproducer) | — | If bug is in model/protocol layer | If bug is in process lifecycle | If bug is in deep integration |
| Refactor | Existing tests pass | ✅ For new logic | — | — | — |
| CI / infra change | Existing tests pass | — | — | — |

**Rule of thumb**: If your change touches model API calls or protocol event emission, you need L3 (VCR) tests. If your change touches server startup or process lifecycle, you need L4 (E2E) tests.

## Test Markers Reference

### Layer Markers (one primary per test; vcr tests may add integration as secondary)

| Marker | Purpose | CI Default | How to Use |
|--------|---------|------------|------------|
| `@pytest.mark.unit` | L1: Fast logic tests | ✅ Run | `@pytest.mark.unit` |
| `@pytest.mark.integration` | L2: Component integration | ✅ Run | `@pytest.mark.integration` |
| `@pytest.mark.vcr` | L3: VCR cassette replay | ✅ Run | `pytestmark = pytest.mark.vcr` at module level |
| `@pytest.mark.e2e` | L4: Subprocess e2e | ❌ Skip (nightly) | `@pytest.mark.e2e` |

### Orthogonal Markers (can combine with any layer)

| Marker | Purpose | CI Default | How to Use |
|--------|---------|------------|------------|
| `@pytest.mark.slow` | Test takes >1s | ❌ Skip | `@pytest.mark.slow` |
| `@pytest.mark.flaky` | Known intermittent failure | ✅ Run | `@pytest.mark.flaky` (should be deflaked and removed) |
| `@pytest.mark.incompatible_with_thinking` | Fails with thinking models | ✅ Run | `@pytest.mark.incompatible_with_thinking` |
| `@pytest.mark.snapshot` | Syrupy snapshot tests | ❌ Skip | `@pytest.mark.snapshot` (renamed from `acp_snapshot`) |
| `@pytest.mark.security` | Security-focused tests | ✅ Run | `@pytest.mark.security` |
| `@pytest.mark.known_bug` | Test that reproduces a known bug (xfail until fixed) | ✅ Run | `@pytest.mark.known_bug` (always paired with `@pytest.mark.xfail`) |

### xfail Strategy

Use `@pytest.mark.xfail` for tests that are expected to fail due to a known bug.
Always pair it with `@pytest.mark.known_bug` to make known-bug tests easily discoverable.

```python
@pytest.mark.xfail(reason="serve-api doesn't init SessionPool (#185)", strict=False, raises=AssertionError)
@pytest.mark.known_bug
async def test_chat_completion(...):
    ...
```

Guidelines:
- Use `strict=False` — the test should not fail the suite if it unexpectedly passes (bug fixed).
- Use `raises=ExpectedException` to narrow the xfail to a specific failure type. Unexpected exceptions still fail the test.
- Include a GitHub issue reference in `reason`.
- Always add `@pytest.mark.known_bug` alongside `@pytest.mark.xfail`.

### Skip Strategy

Use `@pytest.mark.skip(reason="...")` for features not yet implemented. Unlike `xfail`,
a skipped test is never executed. The test intent MUST be documented in the docstring
so the test clearly communicates what behavior is expected once the feature is implemented.

```python
@pytest.mark.skip(reason="mcp/connect client-side not implemented in wolfharness ACP server yet")
async def test_mcp_connect_client_side(...):
    """Send mcp/connect request with a valid MCP server config.
    Expect result.connected = true and result.server_info containing
    server name and version. Verify mcp_server_added SessionUpdate emitted.
    Error case: invalid URL or missing required fields → error code -32602."""
    ...
```

### L4 Sub-qualifiers (only with `@pytest.mark.e2e`)

| Sub-qualifier | Purpose | CI Default | How to Use |
|---------------|---------|------------|------------|
| `@pytest.mark.real_model` | Makes real model API calls | ❌ Skip (auto-skip if no `OPENAI_API_KEY`) | `@pytest.mark.real_model` (replaces `requires_openai_key`) |
| `@pytest.mark.real_mcp` | Connects to real MCP server | ❌ Skip | `@pytest.mark.real_mcp` |

### Running Tests by Layer

```bash
# Default: L1 + L2 + L3 (excludes slow, snapshot, e2e, real_model, real_mcp)
uv run pytest

# Only L1 unit tests (fastest feedback)
uv run pytest -m unit

# Only L3 VCR tests (cassette replay)
uv run pytest tests/vcr/ --strict-vcr-cassette-usage

# L4 E2E tests (requires local server binary)
uv run pytest -m e2e

# Record new cassettes (requires OPENAI_API_KEY)
uv run pytest tests/vcr/test_my_feature.py --record-mode=once

# Include slow tests
uv run pytest -m slow

# Everything (all markers)
uv run pytest -m ""
```

## VCR Recording Workflow

VCR tests replay recorded HTTP interactions (cassettes) so tests are deterministic and network-free in CI.

> **[HUMAN-REQUIRED]**: Cassette recording requires `OPENAI_API_KEY` and must be done by a human. CI never records — it only replays existing cassettes. See steps below.

### Writing a VCR Test

1. **Create the test file** in `tests/vcr/`:

```python
# tests/vcr/test_my_feature.py
import pytest

pytestmark = pytest.mark.vcr  # Module-level marker — all tests in this file use VCR


async def test_basic_completion(real_pool):
    """Test basic text completion via native agent."""
    agent = real_pool.get_agent("test_agent")
    result = await agent.run("Say hello")
    assert result.content is not None
```

2. **Record the cassette**:

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Record (makes real API call, saves cassette)
uv run pytest tests/vcr/test_my_feature.py --record-mode=once
```

3. **Verify the cassette**:
   - Located at `tests/cassettes/vcr/test_my_feature/test_basic_completion.yaml`
   - Check: `authorization` header is `REDACTED`
   - Check: response bodies are decoded (not gzip bytes)
   - Check: no API keys in URLs or body content

4. **Commit the cassette** alongside the test file.

5. **Verify replay** (CI mode — no `--record-mode`):

```bash
uv run pytest tests/vcr/test_my_feature.py --strict-vcr-cassette-usage
```

### Cassette Storage Convention

```
tests/cassettes/
  vcr/
    test_native_basic/
      test_basic_completion.yaml
    test_acp_protocol/
      test_session_lifecycle.yaml
      test_tool_call.yaml
```

Cassettes are auto-named based on the test module path and test function name. Do NOT manually rename cassette files.

### Cassette Sanitization

The `vcr_config` fixture in `tests/conftest.py` automatically:
- **Filters headers**: `authorization`, `x-api-key`, `cookie`, `set-cookie` → `REDACTED`
- **Decodes compressed bodies**: gzip/deflate responses stored as decoded text
- **Normalizes unicode**: Smart quotes and variant characters normalized for deterministic matching
- **Scrubs URLs**: API keys in query parameters removed

A pre-commit hook (or CI step) scans all cassettes for `authorization` header values. If found, the check fails.

### Strict Cassette Usage

`--strict-vcr-cassette-usage` enforces that:
1. Every HTTP interaction in the cassette is played during the test
2. No extra HTTP requests are made beyond what's in the cassette
3. A cassette with zero plays causes the test to fail

This prevents stale cassettes from silently passing tests.

### Cassette Hygiene

The `tests/check_cassettes.py` script verifies every `.yaml` cassette has a corresponding test function. Run as:

```bash
python tests/check_cassettes.py
```

This is a CI step. To fix orphaned cassettes, either add the missing test or delete the cassette.

## Test Directory Structure

```
tests/
├── AGENTS.md                  # This document
├── conftest.py                # Root conftest: ALLOW_MODEL_REQUESTS gate, VCR config, fixtures
├── check_cassettes.py         # Cassette hygiene script
├── cassettes/                 # VCR cassettes (committed to git)
│   └── vcr/
│       └── <test_module>/
│           └── <test_function>.yaml
├── vcr/                       # VCR-backed tests (cassette replay)
│   ├── conftest.py            # Shared fixtures: real_pool, minimal YAML config
│   ├── test_native_basic.py
│   ├── test_native_tool_call.py
│   ├── test_native_streaming.py
│   ├── test_acp_protocol.py
│   ├── test_opencode_protocol.py
│   ├── test_agui_protocol.py
│   └── test_openai_api_protocol.py
├── e2e/                       # Subprocess e2e tests (nightly CI)
│   ├── conftest.py            # Subprocess spawn/teardown, health check, port assignment
│   ├── test_acp_subprocess.py
│   ├── test_opencode_subprocess.py
│   └── test_acp_tool_call_e2e.py
├── fixtures/                  # Shared test fixtures (e.g., minimal_pool.py, sample configs)
│   └── minimal_pool.py        # Minimal AgentPool setup for integration tests
├── servers/                   # Server-level integration tests (mocked pool/agent)
│   ├── acp_server/
│   └── opencode_server/
├── orchestrator/              # RunLoop, Turn, EventBus unit tests
├── lifecycle/                 # Lifecycle dimension unit tests
├── acp/                       # ACP library-level tests
├── integration/               # Cross-component integration tests
└── ...                        # Other test directories
```

## Key Safety Mechanism: ALLOW_MODEL_REQUESTS

```python
# tests/conftest.py
ALLOW_MODEL_REQUESTS = False  # Global gate — blocks ALL real model API calls
```

This is the single most important safety mechanism. It prevents tests from accidentally making real (and potentially expensive) API calls.

- **Default**: `False` — all real model calls blocked
- **Override**: Use the `allow_model_requests` fixture for tests that need real calls (e.g., recording cassettes)
- **VCR**: VCR-mocked tests do NOT need `allow_model_requests` — VCR intercepts at the HTTP level

```python
# Test that needs real API calls (e.g., for recording)
async def test_real_model(allow_model_requests):
    agent = Agent(model="openai:gpt-4o-mini", ...)
    result = await agent.run("Hello")
    # This works because allow_model_requests fixture set the gate to True
```

## Writing Good Protocol Tests

### Event Sequence Assertion

Use `dirty-equals` for partial matching of event sequences:

```python
from dirty_equals import IsStr, IsPartialDict, IsNow


async def test_acp_streaming_events(acp_client, real_pool):
    """Verify ACP streaming event order and structure."""
    events = await acp_client.send_prompt("test_agent", "Say hello")

    # Assert event types in order
    event_types = [e.type for e in events]
    assert event_types == [
        "run_started",
        "part_start",
        "part_delta",  # may repeat
        "part_delta",
        "part_end",
        "stream_complete",
    ]

    # Assert event content with dirty-equals for fuzzy matching
    assert events[0] == IsPartialDict(
        type="run_started",
        session_id=IsStr(),
    )
    assert events[-1] == IsPartialDict(
        type="stream_complete",
        message=IsPartialDict(
            content=IsStr(),
        ),
    )
```

### Tool Call Verification

```python
async def test_tool_call_through_acp(acp_client, real_pool):
    """Verify tool call events propagate through ACP protocol."""
    events = await acp_client.send_prompt("test_agent", "Read the file")

    tool_starts = [e for e in events if e.type == "tool_call_start"]
    tool_completes = [e for e in events if e.type == "tool_call_complete"]

    assert len(tool_starts) == 1
    assert tool_starts[0].tool_name == "read"
    assert len(tool_completes) == 1
    assert tool_completes[0].tool_result is not None
```

### Subagent Delegation Verification

```python
async def test_subagent_through_acp(acp_client, real_pool):
    """Verify subagent delegation events propagate through ACP."""
    events = await acp_client.send_prompt("coordinator", "Delegate to worker")

    spawn_starts = [e for e in events if e.type == "spawn_session_start"]
    spawn_completes = [e for e in events if e.type == "spawn_session_complete"]

    assert len(spawn_starts) == 1
    assert spawn_starts[0].child_agent_name == "worker"
    assert len(spawn_completes) == 1
```

## Anti-Patterns

### ❌ Mocking the entire pool for a protocol test

```python
# BAD: Mocks everything, tests nothing real
async def test_acp():
    mock_pool = MagicMock()
    mock_agent = MagicMock()
    mock_agent.run.return_value = "hello"
    mock_pool.get_agent.return_value = mock_agent
    # ... this only tests that your mock returns what you told it to return
```

```python
# GOOD: Real pool + VCR cassette, tests actual event flow
pytestmark = pytest.mark.vcr


async def test_acp(real_pool):
    agent = real_pool.get_agent("test_agent")
    result = await agent.run("Say hello")
    assert result.content  # Real event flow, real capability wiring
```

### ❌ No VCR test for model-touching code

If your code makes model API calls (directly or through an agent), you MUST have a VCR test. Unit tests with `TestModel` verify logic but not API response format compatibility.

### ❌ Skipping L4a for protocol changes

If you change how the ACP server handles sessions, streaming, or tool calls, a VCR test is the minimum. An L4a smoke E2E test is strongly recommended because subprocess-level issues (startup, stdio, signals) are invisible to in-process tests. L4a runs in ~30s and catches "server won't start" regressions before merge.

### ❌ Running VCR tests concurrently

VCR cassettes are stateful — they record a specific sequence of HTTP interactions. Running VCR tests in parallel (e.g., `pytest -n auto` with VCR tests) can cause cassette mismatches, race conditions on shared cassette files, or spurious "unplayed interaction" errors. VCR tests MUST run sequentially. The default `uv run pytest` already excludes parallel execution for `tests/vcr/`, but do not override this with `-n auto` for VCR tests.

### ❌ Using `as any` or `@ts-ignore` equivalents in tests

Python equivalent: `# type: ignore` or `cast(Any, ...)`. Tests should be fully typed. If you can't type a test properly, the interface under test is probably wrong.

### ❌ Deleting failing tests to make CI pass

Never. Fix the test or fix the code. If a test is legitimately obsolete, mark it `@pytest.mark.skip` with a reason and create an issue to remove it.

## CI Pipeline Stages

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Lint   │ →  │  Unit   │ →  │  VCR    │ →  │  Integ  │
│ Format  │    │  -m unit│    │ tests/  │    │  -m     │
│ Typechk │    │         │    │  vcr/   │    │  integ  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  E2E (nightly)  │
                                           │  -m e2e          │
                                           └─────────────────┘
```

- **Lint/Format/Typecheck**: Fast, blocks everything
- **Unit**: `pytest -m unit` — fastest test feedback
- **VCR**: `pytest tests/vcr/ --strict-vcr-cassette-usage` — cassette replay + hygiene check
- **Integration**: `pytest -m integration` — includes existing mocked tests + new VCR protocol tests
- **E2E Nightly**: `pytest -m e2e` in `e2e-nightly.yml` workflow at 02:00 UTC

## Fixtures Reference

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `real_pool` | function | Real `AgentPool` from minimal YAML config (for VCR/E2E tests) |
| `allow_model_requests` | function | Temporarily enables real model API calls |
| `vcr_config` | module | VCR configuration (header filtering, body decoding) |
| `acp_client` | function | ACP client connected to in-process ACP server |
| `opencode_client` | function | FastAPI `TestClient` for OpenCode server |
| `subprocess_server` | function | Spawned `wolfharness serve-*` process (cached by serve_command; use `--no-server-cache` to bypass) |
| `minimal_config` | session | Minimal YAML config with single test agent |
| `session_e2e_config` | session | Session-scoped e2e config with in-memory storage (for cached servers) |
| `session_e2e_config_with_tool` | session | Session-scoped e2e config with bash tool + in-memory storage |

## Subprocess Server Cache

The `subprocess_server` fixture uses a **session-scoped cache** to reuse `wolfharness serve-*` processes across tests with the same `serve_command`. This reduces ~100+ subprocess spawns to ~4-5, cutting L4a e2e time from ~14min to ~4-5min.

### How It Works

- Cached servers use `subprocess.Popen` (event-loop-agnostic) instead of `asyncio.create_subprocess_exec`
- Cache key: `(serve_command, is_stdio, health_path, extra_args, config_type)`
- Before yielding a cached server, OpenCode servers get `_clear_sessions()` (GET /session + DELETE each)
- ACP stdio servers bypass the cache (stdio connections are stateful)
- Crash detection via `popen.poll()` + socket health check on each cache hit

### Cache Bypass

Use `--no-server-cache` to force fresh subprocess spawning per test (for debugging):

```bash
uv run pytest -m "e2e and not slow" --no-server-cache
```

Tests that mutate server-global state should be marked `@pytest.mark.isolated` to bypass the cache.

### xdist Constraint

The cache auto-disables when `pytest-xdist` is detected (`-n` flag). Each worker spawns its own servers.

## Further Reading

- [pydantic-ai testing docs](https://ai.pydantic.dev/testing/) — TestModel, FunctionModel, VCR patterns
- [vcrpy documentation](https://vcrpy.readthedocs.io/) — Cassette configuration, record modes
- [pytest-recording](https://github.com/kiwicom/pytest-recording) — pytest plugin for VCR
- [dirty-equals](https://dirty-equals.helpmanual.io/) — Fuzzy equality assertions
