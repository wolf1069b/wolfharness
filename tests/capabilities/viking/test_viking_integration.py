"""Integration tests for VikingCapability.

Covers tasks 8.1-8.8 and 9.1-9.5 from openspec/changes/viking-capability-lifecycle/tasks.md.
Tests YAML config loading, config fallback, mode-based tool exposure,
error handling, and L2 lifecycle integration (auto-recall, auto-ingest,
URI guard, for_run isolation, compaction, profile injection, after_run flush)
— all with mocked AsyncHTTPClient.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturn,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
import pytest
import yamling

from wolfharness.capabilities.viking import VikingCapability
from wolfharness.capabilities.viking.identity import VikingIdentity
from wolfharness.capabilities.viking.tools import build_tools
from wolfharness_config.capabilities import VikingCapabilityConfig, build_capability


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# L2 Integration Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> AsyncMock:
    """Create a fully populated mock AsyncHTTPClient for L2 integration tests.

    Delegates to the shared factory so the canned SDK surface is defined
    exactly once (see ``tests/capabilities/viking/conftest.py``).
    """
    from tests.capabilities.viking.conftest import build_mock_client

    return build_mock_client()


def _make_request_context(messages: list[Any]) -> ModelRequestContext:
    """Build a minimal ModelRequestContext with a real TestModel."""
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
        ),
    )


def _make_run_context(session_id: str = "test-session") -> MagicMock:
    """Create a mock RunContext with session_id on deps."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = session_id
    return ctx


def _build_cap_from_config(
    config: VikingCapabilityConfig,
    client: AsyncMock,
    identity: VikingIdentity | None = None,
) -> VikingCapability:
    """Build a VikingCapability from config, inject mock client and identity.

    This exercises the real ``build_capability()`` factory wiring,
    then injects the mock client (bypassing SDK import) and optionally
    sets the resolved identity.
    """
    cap = build_capability(config)
    assert isinstance(cap, VikingCapability)
    cap._client = client
    if identity is not None:
        cap._identity = identity
    return cap


# ---------------------------------------------------------------------------
# 9.1 — Test YAML config loading
# ---------------------------------------------------------------------------


def test_yaml_config_loading_viking_all() -> None:
    """YAML config with type=viking, mode=all produces VikingCapabilityConfig."""
    from wolfharness import AgentsManifest

    yaml_str = """
agents:
  test_agent:
    type: native
    model: test
    capabilities:
      - type: viking
        mode: all
"""
    d = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest.model_validate(d)
    cap_configs = manifest.agents["test_agent"].capabilities
    assert len(cap_configs) == 1
    cfg = cap_configs[0]
    assert isinstance(cfg, VikingCapabilityConfig)
    assert cfg.type == "viking"
    assert cfg.mode == "all"


def test_yaml_config_loading_viking_retrieve() -> None:
    """YAML config with mode=retrieve produces VikingCapabilityConfig with that mode."""
    from wolfharness import AgentsManifest

    yaml_str = """
agents:
  test_agent:
    type: native
    model: test
    capabilities:
      - type: viking
        mode: retrieve
"""
    d = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest.model_validate(d)
    cfg = manifest.agents["test_agent"].capabilities[0]
    assert isinstance(cfg, VikingCapabilityConfig)
    assert cfg.mode == "retrieve"


def test_yaml_config_loading_viking_with_fields() -> None:
    """YAML config with all fields populated parses correctly."""
    from wolfharness import AgentsManifest

    yaml_str = """
agents:
  test_agent:
    type: native
    model: test
    capabilities:
      - type: viking
        mode: write
        url: https://viking.example.com
        api_key: secret
        account: acct123
        user: alice
        timeout: 30.0
        skills_uri: viking://user/alice/skills/
        multimodal_bridge: true
"""
    d = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest.model_validate(d)
    cfg = manifest.agents["test_agent"].capabilities[0]
    assert isinstance(cfg, VikingCapabilityConfig)
    assert cfg.url == "https://viking.example.com"
    assert cfg.api_key == "secret"
    assert cfg.account == "acct123"
    assert cfg.user == "alice"
    assert cfg.timeout == 30.0
    assert cfg.skills_uri == "viking://user/alice/skills/"
    assert cfg.multimodal_bridge is True


def test_yaml_config_loading_default_mode() -> None:
    """YAML config without mode defaults to 'all'."""
    from wolfharness import AgentsManifest

    yaml_str = """
agents:
  test_agent:
    type: native
    model: test
    capabilities:
      - type: viking
"""
    d = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest.model_validate(d)
    cfg = manifest.agents["test_agent"].capabilities[0]
    assert isinstance(cfg, VikingCapabilityConfig)
    assert cfg.mode == "all"


# ---------------------------------------------------------------------------
# 9.2 — Test config fallback
# ---------------------------------------------------------------------------


def test_config_fallback_no_url_no_api_key() -> None:
    """Config with no url/api_key has url=None, api_key=None (SDK resolves from env)."""
    cfg = VikingCapabilityConfig()
    assert cfg.url is None
    assert cfg.api_key is None


def test_config_fallback_no_account_no_user() -> None:
    """Config with no account/user has account=None, user=None."""
    cfg = VikingCapabilityConfig()
    assert cfg.account is None
    assert cfg.user is None


def test_config_fallback_no_timeout() -> None:
    """Config with no timeout has timeout=None (SDK uses default 60s)."""
    cfg = VikingCapabilityConfig()
    assert cfg.timeout is None


def test_config_fallback_no_skills_uri() -> None:
    """Config with no skills_uri has skills_uri=None (capability uses default convention)."""
    cfg = VikingCapabilityConfig()
    assert cfg.skills_uri is None


def test_config_fallback_no_resources_uri() -> None:
    """Config with no resources_uri has resources_uri=None."""
    cfg = VikingCapabilityConfig()
    assert cfg.resources_uri is None


def test_config_fallback_no_uploads_uri() -> None:
    """Config with no uploads_uri has uploads_uri=None."""
    cfg = VikingCapabilityConfig()
    assert cfg.uploads_uri is None


def test_config_fallback_no_public_download_base_url() -> None:
    """Config with no public_download_base_url has public_download_base_url=None."""
    cfg = VikingCapabilityConfig()
    assert cfg.public_download_base_url is None


def test_config_fallback_multimodal_bridge_default_false() -> None:
    """Config defaults multimodal_bridge to False."""
    cfg = VikingCapabilityConfig()
    assert cfg.multimodal_bridge is False


# ---------------------------------------------------------------------------
# 9.3 — Test mode-based tool exposure end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_tool_exposure_retrieve() -> None:
    """Load config, build capability, enter context, get toolset — retrieve mode."""
    cfg = VikingCapabilityConfig(mode="retrieve")
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)

    mock_client = AsyncMock()
    cap._client = mock_client

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert len(tool_names) == 7  # 7 retrieve tools (recall gated by enable_memory, expand always)


@pytest.mark.asyncio
async def test_mode_tool_exposure_write() -> None:
    """Load config, build capability, enter context, get toolset — write mode."""
    cfg = VikingCapabilityConfig(mode="write")
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)

    mock_client = AsyncMock()
    cap._client = mock_client

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    # 4 write tools (remember gated by enable_memory, forget by enable_forget)
    assert len(tool_names) == 4


@pytest.mark.asyncio
async def test_mode_tool_exposure_graph() -> None:
    """Load config, build capability, enter context, get toolset — graph mode."""
    cfg = VikingCapabilityConfig(mode="graph")
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)

    mock_client = AsyncMock()
    cap._client = mock_client

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert len(tool_names) == 1  # 1 graph tool (link gated by enable_link)


@pytest.mark.asyncio
async def test_mode_tool_exposure_all() -> None:
    """Load config, build capability, enter context, get toolset — all mode."""
    cfg = VikingCapabilityConfig(mode="all")
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)

    mock_client = AsyncMock()
    cap._client = mock_client

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert len(tool_names) == 12  # 7 retrieve + 4 write + 1 graph (expand included in retrieve)


@pytest.mark.asyncio
async def test_mode_tool_exposure_with_config_fields() -> None:
    """Build capability from config with all fields, verify toolset works."""
    cfg = VikingCapabilityConfig(
        mode="all",
        url="https://viking.example.com",
        api_key="key",
        user="alice",
        timeout=30.0,
    )
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)
    assert cap.url == "https://viking.example.com"
    assert cap.user == "alice"

    mock_client = AsyncMock()
    cap._client = mock_client

    toolset = cap.get_toolset()
    assert toolset is not None
    tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
    assert len(tool_names) == 12  # default flags: enable_link=False, enable_memory=False


# ---------------------------------------------------------------------------
# 9.4 — Test error handling (integration-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_network_error_graceful() -> None:
    """Network errors are caught and returned as error strings."""
    cap = VikingCapability(mode="all")
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=ConnectionError("network down"))
    cap._client = mock_client

    tools = build_tools(cap)
    search_tool = next(t for t in tools if t.__name__ == "viking_search")

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "test"

    result = await search_tool(ctx, query="test")
    assert "viking_search error: network down" in result.return_value
    assert isinstance(result, ToolReturn)


@pytest.mark.asyncio
async def test_invalid_uri_graceful() -> None:
    """Invalid URI errors are caught and returned as error strings."""
    cap = VikingCapability(mode="all")
    mock_client = AsyncMock()
    mock_client.read = AsyncMock(side_effect=ValueError("invalid URI format"))
    cap._client = mock_client

    tools = build_tools(cap)
    read_tool = next(t for t in tools if t.__name__ == "viking_read")

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "test"

    result = await read_tool(ctx, uris="not-a-valid-uri")
    assert "viking_read error: invalid URI format" in result.return_value


@pytest.mark.asyncio
async def test_permission_error_graceful() -> None:
    """Permission errors are caught and returned as error strings."""
    cap = VikingCapability(mode="all")
    mock_client = AsyncMock()
    mock_client.write = AsyncMock(side_effect=PermissionError("access denied"))
    cap._client = mock_client

    tools = build_tools(cap)
    write_tool = next(t for t in tools if t.__name__ == "viking_write")

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "test"

    result = await write_tool(ctx, uri="viking://protected/doc.md", content="data")
    assert "viking_write error: access denied" in result.return_value


@pytest.mark.asyncio
async def test_timeout_error_graceful() -> None:
    """Timeout errors are caught and returned as error strings."""
    cap = VikingCapability(mode="all")
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=TimeoutError("request timed out"))
    cap._client = mock_client

    tools = build_tools(cap)
    search_tool = next(t for t in tools if t.__name__ == "viking_search")

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "test"

    result = await search_tool(ctx, query="slow query")
    assert "viking_search error: request timed out" in result.return_value


@pytest.mark.asyncio
async def test_generic_exception_graceful() -> None:
    """Generic exceptions are caught and returned as error strings."""
    cap = VikingCapability(mode="all")
    mock_client = AsyncMock()
    mock_client.ls = AsyncMock(side_effect=Exception("unexpected error"))
    cap._client = mock_client

    tools = build_tools(cap)
    ls_tool = next(t for t in tools if t.__name__ == "viking_ls")

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "test"

    result = await ls_tool(ctx, uri="viking://broken/")
    assert "viking_ls error: unexpected error" in result.return_value


@pytest.mark.asyncio
async def test_build_capability_from_config_and_use() -> None:
    """Full end-to-end: build capability from config, inject client, use a tool."""
    cfg = VikingCapabilityConfig(mode="retrieve", user="alice")
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(
        return_value={"results": [{"uri": "viking://found.md", "score": 0.95}]}
    )
    cap._client = mock_client

    tools = build_tools(cap)
    search_tool = next(t for t in tools if t.__name__ == "viking_search")

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = "session-1"

    result = await search_tool(ctx, query="find me", limit=5)
    assert "viking://found.md" in result.return_value
    mock_client.search.assert_called_once()


# ---------------------------------------------------------------------------
# 9.5 — Test resource_read_level config parsing
# ---------------------------------------------------------------------------


def test_resource_read_level_default_overview() -> None:
    """Default resource_read_level is 'overview'."""
    cfg = VikingCapabilityConfig()
    assert cfg.resource_read_level == "overview"


def test_resource_read_level_abstract() -> None:
    """resource_read_level='abstract' parses correctly."""
    cfg = VikingCapabilityConfig(resource_read_level="abstract")
    assert cfg.resource_read_level == "abstract"


def test_resource_read_level_read() -> None:
    """resource_read_level='read' parses correctly."""
    cfg = VikingCapabilityConfig(resource_read_level="read")
    assert cfg.resource_read_level == "read"


def test_resource_read_level_yaml_parsing() -> None:
    """YAML config with resource_read_level parses correctly."""
    from wolfharness import AgentsManifest

    yaml_str = """
agents:
  test_agent:
    type: native
    model: test
    capabilities:
      - type: viking
        mode: all
        resource_read_level: abstract
"""
    d = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest.model_validate(d)
    cfg = manifest.agents["test_agent"].capabilities[0]
    assert isinstance(cfg, VikingCapabilityConfig)
    assert cfg.resource_read_level == "abstract"


@pytest.mark.asyncio
async def test_resource_read_level_propagates_to_capability() -> None:
    """Build capability from config with resource_read_level='abstract'."""
    cfg = VikingCapabilityConfig(resource_read_level="abstract")
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)
    assert cap.resource_read_level == "abstract"

    # Verify it propagates through for_run
    mock_client = AsyncMock()
    cap._client = mock_client
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.deps = MagicMock()
    copy_cap = await cap.for_run(ctx)
    assert copy_cap.resource_read_level == "abstract"


# ---------------------------------------------------------------------------
# 8.1-8.2 — L2: Auto-recall through real before_model_request chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_auto_recall_before_model_request_fires() -> None:
    """L2: auto_recall fires, client.search() called, recall block injected.

    Given: a VikingCapability built from VikingCapabilityConfig with
        auto_recall_enabled=True, injected with a mock client returning
        a search hit.
    When: before_model_request is called with a real ModelRequestContext
        containing a user prompt.
    Then: client.search() is called, and the result contains a
        <openviking-recall> system message injected before the user prompt.
    """
    client = _make_mock_client()
    client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "uri": "viking://user/alice/memories/doc.md",
                    "score": 0.9,
                    "content": "hydraulic diagnosis info",
                    "context_type": "memory",
                }
            ]
        }
    )
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        auto_recall_enabled=True,
        auto_recall_method="search",
    )
    cap = _build_cap_from_config(cfg, client, identity)

    msg = ModelRequest(parts=[UserPromptPart(content="hydraulic pressure issue")])
    rc = _make_request_context([msg])
    ctx = _make_run_context(session_id="sess-123")

    result = await cap.before_model_request(ctx, rc)

    # Verify search was called with the user prompt and memories URI
    client.search.assert_called_once()
    call_args = client.search.call_args
    assert call_args.args[0] == "hydraulic pressure issue"
    assert "viking://user/alice/memories/" in call_args.kwargs["target_uri"]

    # Verify recall block was injected as a system message
    assert result is not rc
    assert len(result.messages) == 2
    sys_msg = result.messages[0]
    assert isinstance(sys_msg, ModelRequest)
    sys_part = sys_msg.parts[0]
    assert isinstance(sys_part, SystemPromptPart)
    assert "<openviking-recall>" in sys_part.content
    assert "viking://user/alice/memories/doc.md" in sys_part.content


# ---------------------------------------------------------------------------
# 8.3 — L2: Auto-ingest fires on second turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_auto_ingest_second_turn_triggers_ingestion() -> None:
    """L2: auto_ingest_enabled=True → second turn triggers create_session + commit_session.

    Given: a VikingCapability built from config with auto_ingest_enabled=True
        and auto_ingest_mode="sync".
    When: before_model_request is called with messages from a previous
        turn (user + assistant) plus a new user prompt.
    Then: client.create_session() and client.commit_session() are called,
        and the ingestion cursor is advanced.
    """
    client = _make_mock_client()
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        auto_ingest_enabled=True,
        auto_ingest_mode="sync",
    )
    cap = _build_cap_from_config(cfg, client, identity)

    # Previous turn (user + assistant) + new user prompt
    messages = [
        ModelRequest(parts=[UserPromptPart(content="What is X?")]),
        ModelResponse(parts=[TextPart(content="X is a thing.")]),
        ModelRequest(parts=[UserPromptPart(content="Tell me more about X")]),
    ]
    rc = _make_request_context(messages)
    ctx = _make_run_context()

    await cap.before_model_request(ctx, rc)

    # Ingest should have created a session and committed it
    client.create_session.assert_called_once()
    client.commit_session.assert_called_once()
    # At least 2 add_message calls (user + assistant from previous turn)
    assert client.add_message.call_count >= 2

    # Cursor should be advanced to the current message count
    assert cap._last_ingested_idx == 3


# ---------------------------------------------------------------------------
# 8.4 — L2: URI guard blocks viking:// URIs in protected tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_uri_guard_blocks_viking_uri_in_read_tool() -> None:
    """L2: uri_guard_enabled=True → read tool with viking:// URI returns error string.

    Given: a VikingCapability built from config with uri_guard_enabled=True.
    When: wrap_tool_execute is called for the "read" tool with args
        containing a viking:// URI.
    Then: the handler is NOT called, and an error string is returned
        directing the user to viking_read or viking_search.
    """
    client = _make_mock_client()

    cfg = VikingCapabilityConfig(
        mode="all",
        uri_guard_enabled=True,
        uri_guard_protected_tools=["read", "bash", "grep", "glob"],
    )
    cap = _build_cap_from_config(cfg, client)

    call = MagicMock()
    call.tool_name = "read"
    handler = AsyncMock(return_value="should not reach")
    args = {"file_path": "viking://user/alice/doc.md"}

    result = await cap.wrap_tool_execute(
        MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
    )

    assert isinstance(result, str)
    assert "viking://" in result
    assert "read" in result
    assert "viking_read" in result or "viking_search" in result
    handler.assert_not_called()


# ---------------------------------------------------------------------------
# 8.5 — L2: allowed_uri_prefixes config-based restriction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_allowed_uri_prefixes_block_read_outside_scope() -> None:
    """L2: viking_read rejects URIs outside the configured prefix allowlist.

    Given: a VikingCapability built from config with
        allowed_uri_prefixes=["viking://resources/wiki/"].
    When: viking_read is called with a URI under viking://resources/raw/.
    Then: the tool returns an error string and the SDK read is NOT called.
    """
    client = _make_mock_client()
    cfg = VikingCapabilityConfig(
        mode="retrieve",
        allowed_uri_prefixes=["viking://resources/wiki/"],
    )
    cap = _build_cap_from_config(cfg, client)
    tools = build_tools(cap)
    read_tool = next(t for t in tools if t.__name__ == "viking_read")

    ctx = _make_run_context()
    result = await read_tool(ctx, uris="viking://resources/raw/engine.md")

    assert "outside the allowed prefixes" in result.return_value
    client.read.assert_not_called()


@pytest.mark.asyncio
async def test_l2_allowed_uri_prefixes_allow_read_in_scope() -> None:
    """L2: viking_read succeeds for URIs within the configured prefix allowlist.

    Given: a VikingCapability built from config with
        allowed_uri_prefixes=["viking://resources/wiki/"].
    When: viking_read is called with a URI under that prefix.
    Then: the SDK read is called with the URI and content is returned.
    """
    client = _make_mock_client()
    client.read = AsyncMock(return_value="wiki content")
    cfg = VikingCapabilityConfig(
        mode="retrieve",
        allowed_uri_prefixes=["viking://resources/wiki/"],
    )
    cap = _build_cap_from_config(cfg, client)
    tools = build_tools(cap)
    read_tool = next(t for t in tools if t.__name__ == "viking_read")

    ctx = _make_run_context()
    result = await read_tool(ctx, uris="viking://resources/wiki/Device/SY215.md")

    assert "1\u2502 wiki content" in result.return_value
    assert client.read.call_args.args[0] == "viking://resources/wiki/Device/SY215.md"


@pytest.mark.asyncio
async def test_l2_allowed_uri_prefixes_search_scoped_when_no_target() -> None:
    """L2: viking_search defaults target_uri to the first allowed prefix.

    Given: allowed_uri_prefixes configured, no target_uri passed.
    When: viking_search is called.
    Then: the SDK search receives target_uri equal to the first allowed prefix.
    """
    client = _make_mock_client()
    cfg = VikingCapabilityConfig(
        mode="retrieve",
        allowed_uri_prefixes=["viking://resources/wiki/"],
    )
    cap = _build_cap_from_config(cfg, client)
    tools = build_tools(cap)
    search_tool = next(t for t in tools if t.__name__ == "viking_search")

    ctx = _make_run_context()
    await search_tool(ctx, query="hydraulic")

    assert client.search.call_args.kwargs["target_uri"] == "viking://resources/wiki/"


@pytest.mark.asyncio
async def test_l2_allowed_uri_prefixes_scopes_list_resources() -> None:
    """L2: list_resources narrows the resources tree, own sessions pass through.

    Given: allowed_uri_prefixes=["viking://resources/wiki/"].
    When: list_resources() is called.
    Then: resources entries come from the allowed prefix; the SDK ls is
        never invoked on the whole viking://resources/ tree, and the
        non-resources sessions tree is listed as-is.
    """
    client = _make_mock_client()
    cfg = VikingCapabilityConfig(
        mode="all",
        allowed_uri_prefixes=["viking://resources/wiki/"],
    )
    cap = _build_cap_from_config(cfg, client)
    cap._client = client

    client.ls = AsyncMock(
        return_value=[
            {
                "uri": "viking://resources/wiki/Device/SY215.md",
                "name": "SY215.md",
                "isDir": False,
            },
            {
                "uri": "viking://resources/wiki/engine/torque.md",
                "name": "torque.md",
                "isDir": False,
            },
        ]
    )
    resources = await cap.list_resources()

    assert [r.uri for r in resources] == [
        "viking://resources/wiki/Device/SY215.md",
        "viking://resources/wiki/engine/torque.md",
    ]
    ls_uris = [call.args[0] for call in client.ls.await_args_list]
    assert set(ls_uris) == {
        "viking://resources/wiki/",
        "viking://user/default/sessions/",
    }
    assert "viking://resources/" not in ls_uris


@pytest.mark.asyncio
async def test_l2_allowed_uri_prefixes_config_roundtrip() -> None:
    """L2: allowed_uri_prefixes propagates from YAML config to capability.

    Given: an agent YAML fragment with allowed_uri_prefixes.
    When: the manifest is loaded.
    Then: the resulting VikingCapabilityConfig carries the prefixes.
    """
    from wolfharness import AgentsManifest

    yaml_str = """
agents:
  test_agent:
    type: native
    model: test
    capabilities:
      - type: viking
        mode: retrieve
        allowed_uri_prefixes:
          - viking://resources/wiki/
          - viking://resources/raw/
"""
    d = yamling.load_yaml(yaml_str, verify_type=dict)
    manifest = AgentsManifest.model_validate(d)
    cap_configs = manifest.agents["test_agent"].capabilities
    assert len(cap_configs) == 1
    cfg = cap_configs[0]
    assert isinstance(cfg, VikingCapabilityConfig)
    assert cfg.allowed_uri_prefixes == [
        "viking://resources/wiki/",
        "viking://resources/raw/",
    ]


@pytest.mark.asyncio
async def test_l2_memory_features_implicitly_allowed_outside_prefixes() -> None:
    """L2: memory features run even though memories URI is outside the allowlist.

    Given: a VikingCapabilityConfig with auto_recall_enabled=True and an
        allowlist covering only viking://resources/ sub-prefixes.
    When: before_model_request is called with a user prompt.
    Then: auto_recall fires — client.search() is called against the agent's
        memories URI, since the allowlist only restricts the
        viking://resources/ namespace.
    """
    client = _make_mock_client()
    client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "uri": "viking://user/alice/memories/doc.md",
                    "score": 0.9,
                    "content": "hydraulic diagnosis info",
                    "context_type": "memory",
                }
            ]
        }
    )
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        auto_recall_enabled=True,
        auto_recall_method="search",
        allowed_uri_prefixes=["viking://resources/wiki/"],
    )
    cap = _build_cap_from_config(cfg, client, identity)

    msg = ModelRequest(parts=[UserPromptPart(content="hydraulic pressure issue")])
    rc = _make_request_context([msg])
    ctx = _make_run_context(session_id="sess-123")

    result = await cap.before_model_request(ctx, rc)

    client.search.assert_called_once()
    assert "viking://user/alice/memories/" in client.search.call_args.kwargs["target_uri"]
    assert result is not rc
    assert any(
        "<openviking-recall>" in part.content
        for msg_ in result.messages
        if isinstance(msg_, ModelRequest)
        for part in msg_.parts
        if isinstance(part, SystemPromptPart)
    )


# ---------------------------------------------------------------------------
# 8.5 — L2: for_run() state isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_for_run_state_isolation() -> None:
    """L2: for_run() produces independent per-run state, shared identity, passed sessions_uri.

    Given: a parent VikingCapability with identity set, sessions_uri configured,
        _last_ingested_idx and _profile_injected modified.
    When: two per-run copies are created via for_run().
    Then: each copy has _last_ingested_idx=0 and _profile_injected=False
        (reset), _identity is shared (same object), sessions_uri is passed
        through, and modifying one copy's state does not affect the other.
    """
    client = _make_mock_client()
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        sessions_uri="viking://user/alice/sessions/",
        auto_ingest_enabled=True,
        profile_enabled=True,
    )
    cap = build_capability(cfg)
    assert isinstance(cap, VikingCapability)
    cap._client = client
    cap._identity = identity
    # Simulate state from a previous run
    cap._last_ingested_idx = 5
    cap._profile_injected = True

    ctx = _make_run_context()
    copy1 = await cap.for_run(ctx)
    copy2 = await cap.for_run(ctx)

    # Per-run state is reset
    assert copy1._last_ingested_idx == 0
    assert copy1._profile_injected is False
    assert copy2._last_ingested_idx == 0
    assert copy2._profile_injected is False

    # Identity is shared (same object reference)
    assert copy1._identity is identity
    assert copy2._identity is identity

    # sessions_uri is passed through (bug fix verification)
    assert copy1.sessions_uri == "viking://user/alice/sessions/"
    assert copy2.sessions_uri == "viking://user/alice/sessions/"

    # Modifying copy1 does not affect copy2
    copy1._last_ingested_idx = 10
    copy1._profile_injected = True
    assert copy2._last_ingested_idx == 0
    assert copy2._profile_injected is False

    # Parent state is unchanged
    assert cap._last_ingested_idx == 5
    assert cap._profile_injected is True


# ---------------------------------------------------------------------------
# 8.6 — L2: Compaction archives large context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_compaction_archives_large_context() -> None:
    """L2: compaction fires on large context, client.write() archives old messages.

    Given: a VikingCapability built from config with compaction_enabled=True
        and a very low threshold (to trigger compaction with test messages).
    When: before_model_request is called with messages exceeding the threshold.
    Then: client.write() is called to archive old messages, and the result
        context has fewer messages (replaced with summary + URI reference).
    """
    client = _make_mock_client()
    client.write = AsyncMock(return_value={"status": "ok"})
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        compaction_enabled=True,
        compaction_threshold=100,  # Very low threshold to trigger with test data
        compaction_keep_recent_turns=2,
    )
    cap = _build_cap_from_config(cfg, client, identity)

    # Build enough messages to exceed the threshold (100 tokens ≈ 400 chars)
    long_text = "x" * 500  # ~125 tokens per message
    messages: list[Any] = []
    for i in range(10):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"Q{i}: {long_text}")]))
        messages.append(ModelResponse(parts=[TextPart(content=f"A{i}: {long_text}")]))
    # Final user prompt
    messages.append(ModelRequest(parts=[UserPromptPart(content="final question")]))

    rc = _make_request_context(messages)
    ctx = _make_run_context()

    result = await cap.before_model_request(ctx, rc)

    # Compaction should have called client.write() to archive
    client.write.assert_called_once()
    write_args = client.write.call_args.args
    write_kwargs = client.write.call_args.kwargs
    assert "viking://user/alice/memories/compacted/" in write_args[0]
    assert write_kwargs["mode"] == "create"

    # Result should have fewer messages (old ones archived, replaced with summary)
    assert len(result.messages) < len(messages)
    assert result is not rc


# ---------------------------------------------------------------------------
# 8.7 — L2: Profile injection fires on first turn, skips on second
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_profile_injection_first_turn_fires_second_skips() -> None:
    """L2: profile_enabled=True → first turn calls client.find(), second turn does not.

    Given: a VikingCapability built from config with profile_enabled=True.
    When: before_model_request is called twice (simulating two turns).
    Then: the first call triggers client.find() for profile lookup,
        and the second call does NOT call find() (profile already injected).
    """
    client = _make_mock_client()
    client.find = AsyncMock(
        return_value={
            "hits": [
                {
                    "uri": "viking://user/alice/memories/profile.md",
                    "content": "project context data",
                    "context_type": "memory",
                    "score": 0.9,
                }
            ]
        }
    )
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        profile_enabled=True,
        profile_first_turn_only=True,
    )
    cap = _build_cap_from_config(cfg, client, identity)

    # First turn: single user message
    msg1 = ModelRequest(parts=[UserPromptPart(content="Hello, help me with diagnosis")])
    rc1 = _make_request_context([msg1])
    ctx = _make_run_context()

    result1 = await cap.before_model_request(ctx, rc1)

    # First turn should call find() for profile
    client.find.assert_called_once()

    # Result should contain the profile block
    assert result1 is not rc1
    sys_msgs = [
        m
        for m in result1.messages
        if isinstance(m, ModelRequest) and any(isinstance(p, SystemPromptPart) for p in m.parts)
    ]
    assert len(sys_msgs) >= 1
    profile_content = next(
        p.content for m in sys_msgs for p in m.parts if isinstance(p, SystemPromptPart)
    )
    assert "<openviking-profile>" in profile_content

    # Second turn: previous turn + new user message
    client.find.reset_mock()
    messages2 = [
        msg1,
        ModelResponse(parts=[TextPart(content="Sure, I can help.")]),
        ModelRequest(parts=[UserPromptPart(content="Tell me more")]),
    ]
    rc2 = _make_request_context(messages2)

    await cap.before_model_request(ctx, rc2)

    # Second turn should NOT call find() (profile already injected)
    client.find.assert_not_called()


# ---------------------------------------------------------------------------
# 8.8 — L2: after_run() flushes pending fire-and-forget tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_after_run_flushes_pending_tasks() -> None:
    """L2: after_run() flushes pending fire-and-forget ingestion tasks before client close.

    Given: a VikingCapability with auto_ingest_enabled=True and
        auto_ingest_mode="async", and a pending ingestion task in
        _pending_tasks.
    When: after_run() is called.
    Then: the pending task is awaited (asyncio.gather is effectively called),
        and _pending_tasks is emptied after the flush.
    """
    client = _make_mock_client()
    identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

    cfg = VikingCapabilityConfig(
        mode="all",
        auto_ingest_enabled=True,
        auto_ingest_mode="async",
    )
    cap = _build_cap_from_config(cfg, client, identity)

    # Simulate a fire-and-forget ingestion by creating a pending task
    ingestion_called = asyncio.Event()

    async def _mock_ingest() -> None:
        ingestion_called.set()

    task = asyncio.create_task(_mock_ingest())
    cap._pending_tasks.add(task)
    task.add_done_callback(cap._pending_tasks.discard)

    ctx = _make_run_context()
    await cap.after_run(ctx, result="agent result")

    # The task should have been awaited and completed
    assert ingestion_called.is_set()
    # _pending_tasks should be empty (task completed and callback removed it)
    assert len(cap._pending_tasks) == 0


@pytest.mark.asyncio
async def test_l2_after_run_no_pending_tasks_is_noop() -> None:
    """L2: after_run() with no pending tasks is a no-op, returns result unchanged.

    Given: a VikingCapability with no pending tasks.
    When: after_run() is called with a result.
    Then: the result is returned unchanged.
    """
    client = _make_mock_client()
    cfg = VikingCapabilityConfig(mode="all")
    cap = _build_cap_from_config(cfg, client)

    ctx = _make_run_context()
    result = await cap.after_run(ctx, result="test result")

    assert result == "test result"
    assert len(cap._pending_tasks) == 0
