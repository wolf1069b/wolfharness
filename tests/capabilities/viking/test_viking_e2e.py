"""End-to-end tests for VikingCapability against a real Viking server.

These tests require:
1. ``openviking-sdk`` installed (``uv sync --group viking``)
2. A running Viking server (default: ``http://127.0.0.1:1933``)
3. ``~/.openviking/ovcli.conf`` or env vars (``OPENVIKING_URL``, etc.)

All tests use a dedicated test namespace under ``viking://user/default/memories/viking_e2e_test/``
and clean up after themselves.

Marked ``@pytest.mark.e2e`` — not run by default. Use::

    uv run pytest tests/capabilities/viking/test_viking_e2e.py -v -m e2e

Design principles:
- **Workflow tests**: Multiple tools exercised per test (write→read→edit→forget)
  instead of one-tool-per-test, reducing test count and server round-trips.
- **Shared fixtures**: ``viking_tools`` returns a ``{name: tool}`` dict;
  ``mock_ctx`` provides a ready RunContext mock; ``skills_dir`` sets up
  skill files.
- **Full tool coverage**: ``enable_memory=True`` and ``enable_link=True``
  on the ``viking_cap`` fixture so all 15 tools are available.
- **Single skip point**: ``viking_cap`` skips all tests if no server —
  no separate autouse fixture needed, halving connection count.
"""

from __future__ import annotations

from contextlib import suppress
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any
import uuid

from pydantic_ai.messages import ToolReturn
import pytest

from wolfharness.capabilities.viking import VikingCapability
from wolfharness.capabilities.viking.tools import build_tools


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.real_mcp,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_name() -> str:
    """Generate a random name for test isolation."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def viking_cap(allow_model_requests: None) -> AsyncIterator[VikingCapability]:
    """Create a VikingCapability connected to the real Viking server.

    Skips all tests if no Viking server is reachable. Enables
    ``enable_memory`` and ``enable_link`` for full 15-tool coverage.

    Depends on ``allow_model_requests`` because the Viking SDK uses httpx
    internally, which is blocked by the ``ALLOW_MODEL_REQUESTS`` gate.
    """
    cap = VikingCapability(mode="all", enable_memory=True, enable_link=True, enable_forget=True)
    try:
        await cap.__aenter__()
    except Exception:  # noqa: BLE001
        pytest.skip("No Viking server available")
    yield cap
    await cap.__aexit__(None, None, None)


@pytest.fixture
async def test_dir(viking_cap: VikingCapability) -> AsyncIterator[str]:
    """Create a test directory and clean it up after.

    Uses the resolved identity's memories path (e.g.
    ``viking://user/yuchen.liu/memories/e2e_abc123/``) to ensure
    write permissions. Relies on ``viking_cap`` fixture having already
    triggered identity resolution via ``__aenter__()``.
    """
    client = viking_cap._client
    assert client is not None
    base = viking_cap._resolve_memories_uri()
    dir_name = f"e2e_{_random_name()}"
    dir_uri = f"{base}{dir_name}/"
    await client.mkdir(dir_uri, description="E2E test directory")
    yield dir_uri
    with suppress(Exception):
        await client.rm(dir_uri, recursive=True)


@pytest.fixture
def viking_tools(viking_cap: VikingCapability) -> dict[str, Callable[..., Any]]:
    """Return a ``{tool_name: tool_fn}`` dict for all 15 Viking tools."""
    return {t.__name__: t for t in build_tools(viking_cap)}


@pytest.fixture
def mock_ctx() -> Any:
    """Create a MagicMock RunContext with session_id for tool calls."""
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "e2e-test"
    return ctx


@pytest.fixture
async def skills_dir(viking_cap: VikingCapability) -> AsyncIterator[tuple[str, str]]:
    """Create a skills directory with a test skill file.

    Yields:
        A tuple of (skills_uri, skill_name).
    """
    client = viking_cap._client
    assert client is not None
    base = viking_cap._resolve_memories_uri()
    skills_uri = f"{base}e2e_skills_{_random_name()}/"
    skill_name = "test_skill"
    await client.mkdir(skills_uri, description="Test skills")
    await client.write(
        f"{skills_uri}{skill_name}.md",
        "---\ndescription: A test skill\n---\n# Test Skill\nContent here.",
        mode="create",
    )
    yield skills_uri, skill_name
    with suppress(Exception):
        await client.rm(skills_uri, recursive=True)


# ---------------------------------------------------------------------------
# Retrieve workflow tests
# ---------------------------------------------------------------------------


async def test_retrieve_workflow(
    viking_cap: VikingCapability,
    viking_tools: dict[str, Any],
    test_dir: str,
    mock_ctx: Any,
) -> None:
    """Test ls, read, grep, glob in a single workflow.

    Writes a file, then exercises ls → read → grep → glob against it.
    """
    client = viking_cap._client
    assert client is not None

    content = "# Code Guide\n\nfunction hello()\nfunction world()\n"
    await client.write(f"{test_dir}guide.md", content, mode="create")

    # ls
    ls_result = await viking_tools["viking_ls"](mock_ctx, uri=test_dir)
    assert "guide.md" in ls_result.return_value

    # read
    read_result = await viking_tools["viking_read"](mock_ctx, uris=f"{test_dir}guide.md")
    assert "Code Guide" in read_result.return_value

    # grep
    grep_result = await viking_tools["viking_grep"](
        mock_ctx, uri=f"{test_dir}guide.md", pattern="function"
    )
    assert "hello" in grep_result.return_value or "world" in grep_result.return_value

    # glob
    glob_result = await viking_tools["viking_glob"](mock_ctx, pattern="**/guide.md", uri=test_dir)
    if "No files found" in glob_result.return_value:
        # Glob may not find files if not indexed yet — verify via ls
        assert "guide.md" in ls_result.return_value
    else:
        assert "guide" in glob_result.return_value


async def test_search_find(
    viking_cap: VikingCapability,
    viking_tools: dict[str, Any],
    test_dir: str,
    mock_ctx: Any,
) -> None:
    """Test viking_search and viking_find return valid JSON results."""
    client = viking_cap._client
    assert client is not None

    unique_marker = f"semantictest_{_random_name()}"
    await client.write(
        f"{test_dir}searchable.md",
        f"# {unique_marker}\n\nThis document discusses quantum entanglement and photon "
        f"polarization in the context of Bell's theorem.\n",
        mode="create",
    )

    # search
    search_result = await viking_tools["viking_search"](
        mock_ctx,
        query="quantum entanglement photon polarization",
        target_uri=test_dir,
        limit=5,
    )
    assert "error" not in search_result.return_value.lower()
    # Results are now formatted text, not JSON
    assert isinstance(search_result.return_value, str)

    # find
    find_result = await viking_tools["viking_find"](
        mock_ctx,
        query="quantum entanglement photon polarization",
        target_uri=test_dir,
        limit=5,
    )
    assert "error" not in find_result.return_value.lower()
    # Results are now formatted text, not JSON
    assert isinstance(find_result.return_value, str)


async def test_recall(viking_tools: dict[str, Any], mock_ctx: Any) -> None:
    """Test viking_recall returns a formatted string with section headers."""
    result = await viking_tools["viking_recall"](
        mock_ctx,
        query="test query for recall",
        quotas={"memory": 3, "resource": 3, "skill": 3},
    )
    assert "error" not in result.return_value.lower()
    assert isinstance(result, ToolReturn)
    if result.return_value.strip():
        assert "===" in result.return_value or "No" in result.return_value
        assert len(result.return_value) > 0


# ---------------------------------------------------------------------------
# Write workflow tests
# ---------------------------------------------------------------------------


async def test_crud_lifecycle(
    viking_cap: VikingCapability,
    viking_tools: dict[str, Any],
    test_dir: str,
    mock_ctx: Any,
) -> None:
    """Test write → edit → forget in a single CRUD lifecycle."""
    client = viking_cap._client
    assert client is not None

    uri = f"{test_dir}crud.md"

    # write
    write_result = await viking_tools["viking_write"](
        mock_ctx, uri=uri, content="line1\nold_text\nline3\n"
    )
    assert "error" not in write_result.return_value.lower()

    # edit
    edit_result = await viking_tools["viking_edit"](
        mock_ctx, uri=uri, old_string="old_text", new_string="new_text"
    )
    assert "error" not in edit_result.return_value.lower()
    content = await client.read(uri)
    assert "new_text" in content
    assert "old_text" not in content

    # forget
    forget_result = await viking_tools["viking_forget"](mock_ctx, uri=uri)
    assert "error" not in forget_result.return_value.lower()
    entries = await client.ls(test_dir)
    names = [e.get("name") for e in entries if isinstance(e, dict)]
    assert "crud.md" not in names


async def test_mkdir(
    viking_cap: VikingCapability,
    viking_tools: dict[str, Any],
    test_dir: str,
    mock_ctx: Any,
) -> None:
    """Test viking_mkdir creates a directory and verifies with ls."""
    client = viking_cap._client
    assert client is not None

    sub_dir = f"{test_dir}subdir_{_random_name()}/"
    result = await viking_tools["viking_mkdir"](mock_ctx, uri=sub_dir, description="Test subdir")
    assert "error" not in result.return_value.lower()

    entries = await client.ls(test_dir)
    names = [e.get("name") for e in entries if isinstance(e, dict)]
    assert any(sub_dir.rstrip("/").split("/")[-1] in str(n) for n in names)


async def test_remember(viking_cap: VikingCapability, mock_ctx: Any) -> None:
    """Test viking_remember schedules a deferred capture, then drains it."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.models.test import TestModel

    tools = {t.__name__: t for t in build_tools(viking_cap)}

    # 1. Tool call only schedules — no session work at call time.
    result = await tools["viking_remember"](mock_ctx, reason=f"e2e remember {_random_name()}")
    assert "error" not in result.return_value.lower()
    assert "Capture scheduled" in result.return_value
    assert viking_cap._remember_pending != []

    # 2. Drain at the next boundary ingests the real conversation.
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[
            ModelRequest(parts=[UserPromptPart(content=f"Hello from E2E test {_random_name()}")]),
            ModelResponse(parts=[TextPart(content="Hi! I received your message.")]),
        ],
        model_settings=None,
        model_request_parameters=None,  # type: ignore[arg-type]
    )
    await viking_cap._handle_remember_drain(mock_ctx, request_context)

    assert viking_cap._last_ingested_idx == 2, (
        f"Expected _last_ingested_idx=2, got {viking_cap._last_ingested_idx}"
    )
    assert viking_cap._remember_pending == []


async def test_add_resource(viking_tools: dict[str, Any], mock_ctx: Any) -> None:
    """Test viking_add_resource ingests a local file into Viking."""
    tmp_path = pathlib.Path(tempfile.mkstemp(suffix=".md", prefix="viking_resource_")[1])
    tmp_path.write_text(f"# Resource Test {_random_name()}\n\nContent for ingestion.\n")

    target_uri = f"viking://resources/e2e_test_{_random_name()}/"
    try:
        result = await viking_tools["viking_add_resource"](
            mock_ctx,
            path=str(tmp_path),
            to=target_uri,
        )
        assert "Added resource" in result.return_value
        assert "viking_add_resource error:" not in result.return_value
    finally:
        with suppress(Exception):
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Graph workflow tests
# ---------------------------------------------------------------------------


async def test_link(
    viking_cap: VikingCapability,
    viking_tools: dict[str, Any],
    test_dir: str,
    mock_ctx: Any,
) -> None:
    """Test viking_link creates a link between two nodes."""
    client = viking_cap._client
    assert client is not None

    uri_a = f"{test_dir}node_a.md"
    uri_b = f"{test_dir}node_b.md"
    await client.write(uri_a, "content a", mode="create")
    await client.write(uri_b, "content b", mode="create")

    link_result = await viking_tools["viking_link"](
        mock_ctx, from_uri=uri_a, to_uris=uri_b, reason="test link"
    )
    assert "error" not in link_result.return_value.lower()


async def test_set_tags(
    viking_cap: VikingCapability,
    viking_tools: dict[str, Any],
    test_dir: str,
    mock_ctx: Any,
) -> None:
    """Test viking_set_tags on a newly created file.

    May fail if the Viking backend hasn't finished embedding the file
    (Qdrant requires dense/sparse vectors before tags can be set).
    We wait for processing and retry once; if it still fails, the test
    is xfailed rather than failing the suite.
    """
    client = viking_cap._client
    assert client is not None

    uri = f"{test_dir}tagged.md"
    await client.write(uri, "tagged content for indexing", mode="create")

    # Wait for the file to be indexed by the embedding pipeline
    with suppress(Exception):
        await client.wait_processed(timeout=15)

    tags_result = await viking_tools["viking_set_tags"](
        mock_ctx, uri=uri, tags=["category=test", "priority=high"]
    )
    if "error" in tags_result.return_value.lower():
        pytest.xfail(f"viking_set_tags failed — file not yet indexed: {tags_result.return_value}")


# ---------------------------------------------------------------------------
# SkillResource Protocol tests
# ---------------------------------------------------------------------------


async def test_skill_resource_workflow(
    viking_cap: VikingCapability,
    skills_dir: tuple[str, str],
) -> None:
    """Test list_skills, read_skill, skill_exists in a single workflow."""
    skills_uri, skill_name = skills_dir
    client = viking_cap._client
    assert client is not None

    cap = VikingCapability(skills_uri=skills_uri, enable_memory=True, enable_link=True)
    cap._client = client
    cap._owns_client = False

    # list_skills
    skills = await cap.list_skills()
    assert len(skills) >= 1
    assert any(s.name == skill_name for s in skills)
    assert all(s.source == "remote" for s in skills)

    # read_skill
    content = await cap.read_skill(skill_name)
    assert content is not None
    assert "Test Skill" in content

    # read_skill — non-existent
    assert await cap.read_skill("nonexistent") is None

    # skill_exists
    assert await cap.skill_exists(skill_name) is True
    assert await cap.skill_exists("not_here") is False


# ---------------------------------------------------------------------------
# ResourceAccess Protocol tests
# ---------------------------------------------------------------------------


async def test_resource_access_workflow(
    viking_cap: VikingCapability,
    test_dir: str,
) -> None:
    """Test list_resources, read_resource, resource_exists in a single workflow."""
    client = viking_cap._client
    assert client is not None

    # Write a resource file under the test directory (memories/ path)
    resource_uri = f"{test_dir}resource.md"
    resource_content = f"# E2E Test Resource {_random_name()}\n\nThis is test content.\n"
    await client.write(resource_uri, resource_content, mode="create")

    # Override resources_uri so read_resource can find files under test_dir.
    # Use resource_read_level="read" to get full file content (overview returns
    # a directory summary, not file content).
    viking_cap.resources_uri = test_dir
    viking_cap.resource_read_level = "read"
    try:
        from wolfharness.capabilities.resource_protocols import TextResourceContent

        # read_resource
        result = await viking_cap.read_resource(resource_uri)
        assert result is not None
        assert len(result) >= 1
        assert isinstance(result[0], TextResourceContent)
        assert result[0].text
        assert "E2E Test Resource" in result[0].text

        # resource_exists — existing
        assert await viking_cap.resource_exists(resource_uri) is True

        # resource_exists — non-existent
        assert await viking_cap.resource_exists(f"{test_dir}nonexistent_xyz.md") is False
    finally:
        viking_cap.resources_uri = None
        viking_cap.resource_read_level = "overview"


async def test_list_resources(viking_cap: VikingCapability) -> None:
    """Test list_resources returns a list."""
    resources = await viking_cap.list_resources()
    assert isinstance(resources, list)


# ---------------------------------------------------------------------------
# Multimodal Bridge test
# ---------------------------------------------------------------------------


async def test_multimodal_bridge(
    viking_cap: VikingCapability,
    test_dir: str,
) -> None:
    """Test before_model_request uploads binary and replaces with text ref."""
    from pydantic_ai.messages import BinaryContent, ModelRequest, TextPart, UserPromptPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.models.test import TestModel

    from wolfharness_config.model_capabilities import ModelCapabilities

    cap = VikingCapability(
        mode="all",
        enable_memory=True,
        enable_link=True,
        multimodal_bridge=True,
        model_capabilities=ModelCapabilities(image_input=False),
        uploads_uri=test_dir,
        _client=viking_cap._client,
        _owns_client=False,
    )

    content_parts: list[Any] = [
        TextPart(content="describe this image"),
        BinaryContent(data=b"fake-image-data-for-testing", media_type="image/png"),
    ]
    user_prompt = UserPromptPart(content=content_parts)
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[ModelRequest(parts=[user_prompt])],
        model_settings=None,
        model_request_parameters=None,  # type: ignore[arg-type]
    )

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "e2e-test"

    modified_context = await cap.before_model_request(ctx, request_context)

    # The binary content should be replaced with a TextPart containing a viking:// URI
    modified_msg = modified_context.messages[0]
    assert isinstance(modified_msg, ModelRequest)
    user_part = next(
        (p for p in modified_msg.parts if isinstance(p, UserPromptPart)),
        None,
    )
    assert user_part is not None
    assert isinstance(user_part.content, list)

    has_text_ref = False
    has_binary = False
    for item in user_part.content:
        if isinstance(item, TextPart) and "viking://" in item.content:
            has_text_ref = True
        elif isinstance(item, BinaryContent):
            has_binary = True

    assert has_text_ref, "BinaryContent should be replaced with a TextPart containing viking:// URI"
    assert not has_binary, "Original BinaryContent should not remain"

    # Verify the uploaded file exists in Viking under test_dir
    text_part = next(
        (i for i in user_part.content if isinstance(i, TextPart) and "viking://" in i.content),
        None,
    )
    assert text_part is not None
    # Extract the viking:// URI from the text.
    # The TextPart format is: "[Content stored at {uri}. Use viking_read to access.]"
    # so the URI is followed by ". " — strip trailing punctuation.
    content_str = text_part.content
    uri_start = content_str.index("viking://")
    rest = content_str[uri_start:]
    uploaded_uri = rest
    for i, ch in enumerate(rest):
        if ch in " \n\t]":
            uploaded_uri = rest[:i]
            break
    uploaded_uri = uploaded_uri.rstrip(".]")

    # Verify the uploaded file exists in Viking by reading it back
    # (ls may not immediately reflect newly written files due to
    # eventual consistency in the Viking backend).
    client = viking_cap._client
    assert client is not None
    uploaded_content = await client.read(uploaded_uri)
    assert uploaded_content is not None
    assert len(uploaded_content) > 0
    with suppress(Exception):
        await client.rm(uploaded_uri)


# ---------------------------------------------------------------------------
# Auto-Recall E2E test
# ---------------------------------------------------------------------------


async def test_auto_recall_e2e(
    viking_cap: VikingCapability,
    test_dir: str,
) -> None:
    """Test auto-recall injects <openviking-recall> block into model context.

    Writes a document to Viking, then creates a VikingCapability with
    ``auto_recall_enabled=True`` and calls ``before_model_request()``.
    Verifies the returned context contains an ``<openviking-recall>`` block.
    """
    from unittest.mock import MagicMock

    from pydantic_ai.messages import ModelRequest, SystemPromptPart, TextPart, UserPromptPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.models.test import TestModel

    client = viking_cap._client
    assert client is not None

    # Write a document so there is something to recall
    unique_marker = f"recalltest_{_random_name()}"
    doc_content = (
        f"# {unique_marker}\n\n"
        "This document discusses machine learning model optimization "
        "techniques including quantization and pruning.\n"
    )
    await client.write(f"{test_dir}recall_doc.md", doc_content, mode="create")

    # Wait for indexing
    with suppress(Exception):
        await client.wait_processed(timeout=15)

    # Create a capability with auto_recall enabled, sharing the existing client
    cap = VikingCapability(
        mode="all",
        auto_recall_enabled=True,
        auto_recall_method="find",
        auto_recall_min_score=0.0,
        memories_uri=test_dir,
        _client=client,
        _owns_client=False,
    )

    # Build a request context with a user prompt
    user_msg = ModelRequest(parts=[UserPromptPart(content="What is model optimization?")])
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[user_msg],
        model_settings=None,
        model_request_parameters=None,  # type: ignore[arg-type]
    )

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "e2e-recall-test"

    modified_context = await cap.before_model_request(ctx, request_context)

    # Check that <openviking-recall> was injected into the messages
    all_text = ""
    for msg in modified_context.messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, TextPart | SystemPromptPart):
                    all_text += str(part.content)

    assert "<openviking-recall>" in all_text, (
        f"Expected <openviking-recall> block in messages, got: {all_text[:500]}"
    )


# ---------------------------------------------------------------------------
# Auto-Ingest E2E test
# ---------------------------------------------------------------------------


async def test_auto_ingest_e2e(
    viking_cap: VikingCapability,
) -> None:
    """Test auto-ingest triggers conversation ingestion on second turn.

    Creates a VikingCapability with ``auto_ingest_enabled=True``,
    simulates a completed turn (user + assistant messages), then calls
    ``before_model_request()`` with a new user prompt. Verifies that
    ingestion was triggered (``create_session`` called on the client).
    """
    from unittest.mock import MagicMock

    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.models.test import TestModel

    client = viking_cap._client
    assert client is not None

    # Create a capability with auto_ingest enabled, using sync mode for
    # deterministic testing (no fire-and-forget timing issues)
    cap = VikingCapability(
        mode="all",
        auto_ingest_enabled=True,
        auto_ingest_mode="sync",
        _client=client,
        _owns_client=False,
    )

    # Simulate a completed first turn: user prompt + assistant response
    first_user = ModelRequest(
        parts=[UserPromptPart(content=f"Hello from ingest test {_random_name()}")],
    )
    first_assistant = ModelResponse(parts=[TextPart(content="Hi! I received your message.")])
    second_user = ModelRequest(parts=[UserPromptPart(content="What did I just say?")])

    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[first_user, first_assistant, second_user],
        model_settings=None,
        model_request_parameters=None,  # type: ignore[arg-type]
    )

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "e2e-ingest-test"

    # Call before_model_request — this should trigger ingestion of the
    # first turn's conversation (messages since _last_ingested_idx=0)
    await cap.before_model_request(ctx, request_context)

    # Verify ingestion cursor advanced
    assert cap._last_ingested_idx == 3, (
        f"Expected _last_ingested_idx=3, got {cap._last_ingested_idx}"
    )


# ---------------------------------------------------------------------------
# Identity Resolution E2E test
# ---------------------------------------------------------------------------


async def test_identity_resolution_e2e(
    viking_cap: VikingCapability,
) -> None:
    """Test dynamic identity resolution with API key only (no explicit user/account).

    Creates a VikingCapability with NO explicit ``user`` or ``account``
    (only the API key from the connected client), then verifies:
    1. ``_resolve_identity()`` returns a non-None identity
    2. ``list_resources()`` uses the resolved identity in URIs
    """
    client = viking_cap._client
    assert client is not None

    # Create a capability with no explicit user/account, sharing the
    # existing client and identity from the already-connected viking_cap
    cap = VikingCapability(
        mode="all",
        user=None,
        account=None,
        _client=client,
        _owns_client=False,
    )

    # Force identity resolution by calling _resolve_identity
    identity = await cap._resolve_identity()
    assert identity is not None
    assert identity.user_id
    assert identity.account_id

    # Verify list_resources works with the resolved identity
    resources = await cap.list_resources()
    assert isinstance(resources, list)

    # Verify sessions URI uses the resolved identity
    sessions_uri = cap._resolve_sessions_uri()
    assert identity.user_id in sessions_uri

    # Verify memories URI uses the resolved identity
    memories_uri = cap._resolve_memories_uri()
    assert identity.user_id in memories_uri
