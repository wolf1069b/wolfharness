"""Unit tests for VikingCapability.

Covers tasks 8.1-8.14 from openspec/changes/viking-capability/tasks.md.
All tests mock ``AsyncHTTPClient`` — no real Viking server required.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.capabilities.viking import VikingCapability, _normalize_search_results
from wolfharness.capabilities.viking.identity import VikingIdentity, _try_decode_api_key
from wolfharness.capabilities.viking.profile import (
    _derive_context_hint,
    _format_profile_block,
)
from wolfharness.capabilities.viking.recall import (
    _extract_latest_user_prompt,
    _format_recall_block,
    _inject_system_message,
    _rank_and_dedup,
)
from wolfharness.capabilities.viking.tools import build_tools
from wolfharness.capabilities.viking.utils import (
    add_line_numbers,
    format_ls_entries,
    format_search_results,
    is_viking_uri,
    truncate_text,
)
from wolfharness_config.capabilities import VikingCapabilityConfig, build_capability


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx(session_id: str | None = "test-session") -> MagicMock:
    """Create a mock RunContext with session_id on deps."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.session_id = session_id
    return ctx


def _get_tool(tools: list[Any], name: str) -> Any:
    """Find a tool by name from the list returned by build_tools."""
    return next(t for t in tools if t.__name__ == name)


def _make_request_context(messages: list[Any]) -> ModelRequestContext:
    """Build a minimal ModelRequestContext for before_model_request tests."""
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
        ),
    )


# ---------------------------------------------------------------------------
# 8.1 — Test VikingCapabilityConfig parsing
# ---------------------------------------------------------------------------


class TestVikingCapabilityConfig:
    """Tests for VikingCapabilityConfig parsing and validation."""

    def test_default_config(self) -> None:
        """Default config has mode='all' and all optional fields as None."""
        cfg = VikingCapabilityConfig()
        assert cfg.type == "viking"
        assert cfg.mode == "all"
        assert cfg.url is None
        assert cfg.api_key is None
        assert cfg.account is None
        assert cfg.user is None
        assert cfg.timeout is None
        assert cfg.skills_uri is None
        assert cfg.resources_uri is None
        assert cfg.multimodal_bridge is False
        assert cfg.uploads_uri is None
        assert cfg.public_download_base_url is None
        assert cfg.resource_read_level == "overview"

    def test_default_support_vision_none(self) -> None:
        """support_vision defaults to None (auto-detect from model capabilities)."""
        cfg = VikingCapabilityConfig()
        assert cfg.support_vision is None

    def test_support_vision_true(self) -> None:
        """support_vision=True forces image bytes for image URIs."""
        cfg = VikingCapabilityConfig(support_vision=True)
        assert cfg.support_vision is True

    def test_support_vision_false(self) -> None:
        """support_vision=False forces text URI descriptions for image URIs."""
        cfg = VikingCapabilityConfig(support_vision=False)
        assert cfg.support_vision is False

    def test_support_vision_build_passthrough(self) -> None:
        """build_capability passes support_vision=... to VikingCapability.

        Explicit False must reach the capability (only None is filtered by
        _import_and_instantiate), so forced-text mode survives the build.
        """
        cap = build_capability(VikingCapabilityConfig(support_vision=False))
        assert cap.support_vision is False
        cap_none = build_capability(VikingCapabilityConfig())
        assert cap_none.support_vision is None

    def test_mode_retrieve(self) -> None:
        """Mode 'retrieve' is accepted."""
        cfg = VikingCapabilityConfig(mode="retrieve")
        assert cfg.mode == "retrieve"

    def test_mode_write(self) -> None:
        """Mode 'write' is accepted."""
        cfg = VikingCapabilityConfig(mode="write")
        assert cfg.mode == "write"

    def test_mode_graph(self) -> None:
        """Mode 'graph' is accepted."""
        cfg = VikingCapabilityConfig(mode="graph")
        assert cfg.mode == "graph"

    def test_mode_all(self) -> None:
        """Mode 'all' is accepted."""
        cfg = VikingCapabilityConfig(mode="all")
        assert cfg.mode == "all"

    def test_mode_invalid_rejected(self) -> None:
        """Invalid mode value is rejected by validation."""
        with pytest.raises(ValidationError):
            VikingCapabilityConfig(mode="invalid")  # type: ignore[arg-type]

    def test_all_fields_populated(self) -> None:
        """All fields can be populated at once."""
        cfg = VikingCapabilityConfig(
            mode="retrieve",
            url="https://viking.example.com",
            api_key="secret-key",
            account="acct123",
            user="alice",
            timeout=30.0,
            skills_uri="viking://user/alice/skills/",
            resources_uri="viking://resources/",
            multimodal_bridge=True,
            uploads_uri="viking://uploads/",
            public_download_base_url="https://download.example.com",
        )
        assert cfg.url == "https://viking.example.com"
        assert cfg.api_key == "secret-key"
        assert cfg.account == "acct123"
        assert cfg.user == "alice"
        assert cfg.timeout == 30.0
        assert cfg.skills_uri == "viking://user/alice/skills/"
        assert cfg.resources_uri == "viking://resources/"
        assert cfg.multimodal_bridge is True
        assert cfg.uploads_uri == "viking://uploads/"
        assert cfg.public_download_base_url == "https://download.example.com"

    def test_discriminator_works(self) -> None:
        """The 'type' field discriminator correctly identifies VikingCapabilityConfig."""
        import typing

        from wolfharness_config.capabilities import BuiltinCapabilityConfig

        cfg = VikingCapabilityConfig()
        assert cfg.type == "viking"
        # BuiltinCapabilityConfig is Annotated[Union[...], Field(discriminator="type")]
        # The union type is the first arg; extract its member types.
        union_type = typing.get_args(BuiltinCapabilityConfig)[0]
        member_types = typing.get_args(union_type)
        assert VikingCapabilityConfig in member_types


# ---------------------------------------------------------------------------
# support_vision — _should_return_image_bytes tri-state matrix
# ---------------------------------------------------------------------------


class TestShouldReturnImageBytes:
    """Tri-state matrix for ``_should_return_image_bytes``."""

    @staticmethod
    def _cap(
        support_vision: bool | None = None, image_input: bool | None = None
    ) -> VikingCapability:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(mode="all", support_vision=support_vision)
        cap.model_capabilities = ModelCapabilities(image_input=image_input)
        return cap

    @pytest.mark.parametrize(
        "image_input",
        [None, False, True],
        ids=["unknown", "false", "true"],
    )
    def test_explicit_true_overrides_all(self, image_input: bool | None) -> None:
        """support_vision=True forces bytes regardless of model capabilities."""
        cap = self._cap(support_vision=True, image_input=image_input)
        assert cap._should_return_image_bytes() is True

    @pytest.mark.parametrize(
        "image_input",
        [None, False, True],
        ids=["unknown", "false", "true"],
    )
    def test_explicit_false_overrides_all(self, image_input: bool | None) -> None:
        """support_vision=False forces text regardless of model capabilities."""
        cap = self._cap(support_vision=False, image_input=image_input)
        assert cap._should_return_image_bytes() is False

    def test_auto_vision_model(self) -> None:
        """support_vision=None + image_input=True auto-detects vision."""
        cap = self._cap(support_vision=None, image_input=True)
        assert cap._should_return_image_bytes() is True

    def test_auto_text_only_model(self) -> None:
        """support_vision=None + image_input=False auto-detects text-only."""
        cap = self._cap(support_vision=None, image_input=False)
        assert cap._should_return_image_bytes() is False

    def test_auto_unknown_capability_field(self) -> None:
        """support_vision=None + image_input=None (cache miss) degrades to text."""
        cap = self._cap(support_vision=None, image_input=None)
        assert cap._should_return_image_bytes() is False

    def test_auto_uninjected_capabilities(self) -> None:
        """support_vision=None + model_capabilities=None (not injected) → text.

        Safe degradation: the capability *produces* image content and must
        not emit BinaryImage it cannot guarantee the model accepts (unlike
        ModalityFilterCapability which passes through on None).
        """
        cap = VikingCapability(mode="all", support_vision=None)
        cap.model_capabilities = None
        assert cap._should_return_image_bytes() is False


# ---------------------------------------------------------------------------
# 8.2 — Test __aenter__/__aexit__ lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Tests for __aenter__/__aexit__ lifecycle management."""

    @pytest.mark.asyncio
    async def test_aenter_noop_when_client_already_set(self, mock_client: AsyncMock) -> None:
        """__aenter__ is a no-op when client is already set (for_run copy)."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client

        result = await cap.__aenter__()

        assert result is cap
        assert cap._client is mock_client
        mock_client.initialize.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_aenter_import_error_when_sdk_not_installed(self) -> None:
        """__aenter__ raises ImportError when openviking_sdk is not installed."""
        try:
            import openviking_sdk  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("openviking_sdk is installed — ImportError test not applicable")
        cap = VikingCapability(mode="all")
        assert cap._client is None
        with pytest.raises(ImportError):
            await cap.__aenter__()

    @pytest.mark.asyncio
    async def test_aexit_closes_client_when_owned(self, mock_client: AsyncMock) -> None:
        """__aexit__ closes the client when _owns_client is True."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client
        cap._owns_client = True

        await cap.__aexit__(None, None, None)

        mock_client.close.assert_called_once()
        assert cap._client is None

    @pytest.mark.asyncio
    async def test_aexit_does_not_close_client_when_not_owned(self, mock_client: AsyncMock) -> None:
        """__aexit__ does not close the client when _owns_client is False."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client
        cap._owns_client = False

        await cap.__aexit__(None, None, None)

        mock_client.close.assert_not_called()
        assert cap._client is None

    @pytest.mark.asyncio
    async def test_aexit_sets_client_none_regardless_of_ownership(
        self, mock_client: AsyncMock
    ) -> None:
        """__aexit__ sets _client to None even when not owning the client."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client
        cap._owns_client = False

        await cap.__aexit__(None, None, None)

        assert cap._client is None

    @pytest.mark.asyncio
    async def test_aexit_with_exception_still_closes(self, mock_client: AsyncMock) -> None:
        """__aexit__ closes the client even when an exception was raised."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client
        cap._owns_client = True

        await cap.__aexit__(ValueError, ValueError("test"), None)

        mock_client.close.assert_called_once()
        assert cap._client is None


# ---------------------------------------------------------------------------
# 8.3 — Test for_run()
# ---------------------------------------------------------------------------


class TestForRun:
    """Tests for for_run() method."""

    @pytest.mark.asyncio
    async def test_for_run_shares_client(self, mock_client: AsyncMock) -> None:
        """for_run() returns a copy that shares the same client reference."""
        cap = VikingCapability(mode="all", user="alice")
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap is not cap
        assert copy_cap._client is mock_client
        assert copy_cap._owns_client is False
        assert copy_cap.mode == cap.mode
        assert copy_cap.user == cap.user

    @pytest.mark.asyncio
    async def test_for_run_preserves_all_fields(self, mock_client: AsyncMock) -> None:
        """for_run() preserves all configuration fields."""
        cap = VikingCapability(
            mode="retrieve",
            url="https://viking.example.com",
            api_key="key",
            account="acct",
            user="alice",
            timeout=30.0,
            skills_uri="viking://user/alice/skills/",
            resources_uri="viking://resources/",
            multimodal_bridge=True,
            uploads_uri="viking://uploads/",
            public_download_base_url="https://dl.example.com",
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.url == cap.url
        assert copy_cap.api_key == cap.api_key
        assert copy_cap.account == cap.account
        assert copy_cap.user == cap.user
        assert copy_cap.timeout == cap.timeout
        assert copy_cap.skills_uri == cap.skills_uri
        assert copy_cap.resources_uri == cap.resources_uri
        assert copy_cap.multimodal_bridge == cap.multimodal_bridge
        assert copy_cap.uploads_uri == cap.uploads_uri
        assert copy_cap.public_download_base_url == cap.public_download_base_url
        assert copy_cap.resource_read_level == cap.resource_read_level

    @pytest.mark.asyncio
    async def test_for_run_copy_does_not_close_parent_client(self, mock_client: AsyncMock) -> None:
        """Closing the for_run copy does not close the parent's client."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        await copy_cap.__aexit__(None, None, None)
        mock_client.close.assert_not_called()
        assert cap._client is mock_client


# ---------------------------------------------------------------------------
# 8.4 — Test each retrieve tool with mocked client
# ---------------------------------------------------------------------------


class TestRetrieveTools:
    """Tests for the 7 retrieve tools."""

    @pytest.mark.asyncio
    async def test_viking_search(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_search maps params correctly and injects session_id."""
        mock_client.search = AsyncMock(
            return_value={"results": [{"uri": "viking://doc.md", "score": 0.9}]}
        )
        tools = build_tools(viking_cap)
        search_tool = _get_tool(tools, "viking_search")

        ctx = _make_ctx(session_id="sess-123")
        result = await search_tool(ctx, query="test query", limit=5, min_score=0.5, level=[1])

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        call_args = mock_client.search.call_args.args
        assert call_args[0] == "test query"
        assert call_kwargs["limit"] == 5
        assert call_kwargs["score_threshold"] == 0.5
        assert call_kwargs["filter"] == {"level": [1]}
        assert call_kwargs["session_id"] == "sess-123"
        assert "viking://doc.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_search_no_level(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_search passes filter=None when level is not specified."""
        mock_client.search = AsyncMock(return_value={"results": []})
        tools = build_tools(viking_cap)
        search_tool = _get_tool(tools, "viking_search")

        ctx = _make_ctx()
        await search_tool(ctx, query="test")

        assert mock_client.search.call_args.kwargs["filter"] is None

    @pytest.mark.asyncio
    async def test_viking_search_no_session_id(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_search passes session_id=None when deps has no session_id."""
        mock_client.search = AsyncMock(return_value={"results": []})
        tools = build_tools(viking_cap)
        search_tool = _get_tool(tools, "viking_search")

        # Use a context where deps does not have session_id attribute
        ctx = MagicMock()
        ctx.deps = MagicMock(spec=[])  # spec=[] means no attributes
        await search_tool(ctx, query="test")

        assert mock_client.search.call_args.kwargs["session_id"] is None

    @pytest.mark.asyncio
    async def test_viking_find(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """viking_find maps params correctly but does NOT pass session_id."""
        mock_client.find = AsyncMock(return_value={"results": [{"uri": "viking://doc.md"}]})
        tools = build_tools(viking_cap)
        find_tool = _get_tool(tools, "viking_find")

        ctx = _make_ctx(session_id="sess-123")
        result = await find_tool(ctx, query="find query", limit=3, min_score=0.2, level=[0])

        mock_client.find.assert_called_once()
        call_kwargs = mock_client.find.call_args.kwargs
        call_args = mock_client.find.call_args.args
        assert call_args[0] == "find query"
        assert call_kwargs["limit"] == 3
        assert call_kwargs["score_threshold"] == 0.2
        assert call_kwargs["filter"] == {"level": [0]}
        assert "session_id" not in call_kwargs
        assert "viking://doc.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_recall(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_recall makes multiple find() calls with different context_types."""
        mock_client.find = AsyncMock(
            return_value={"hits": [{"uri": "viking://mem.md", "content": "memory"}]}
        )
        tools = build_tools(viking_cap)
        recall_tool = _get_tool(tools, "viking_recall")

        ctx = _make_ctx()
        result = await recall_tool(ctx, query="remember when")

        assert mock_client.find.call_count == 3
        context_types = [c.kwargs["context_type"] for c in mock_client.find.call_args_list]
        assert "memory" in context_types
        assert "resource" in context_types
        assert "skill" in context_types
        for call in mock_client.find.call_args_list:
            assert call.kwargs["query"] == "remember when"
        assert "=== memory ===" in result.return_value
        assert "=== resource ===" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_recall_custom_quotas(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_recall respects custom quotas."""
        mock_client.find = AsyncMock(return_value={"results": []})
        tools = build_tools(viking_cap)
        recall_tool = _get_tool(tools, "viking_recall")

        ctx = _make_ctx()
        custom_quotas = {"memory": 2, "resource": 3}
        result = await recall_tool(ctx, query="test", quotas=custom_quotas)

        assert mock_client.find.call_count == 2
        quotas_used = [c.kwargs["limit"] for c in mock_client.find.call_args_list]
        assert 2 in quotas_used
        assert 3 in quotas_used
        assert "=== memory ===" in result.return_value
        assert "=== resource ===" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_grep(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """viking_grep passes uri, pattern, case_insensitive correctly."""
        mock_client.grep = AsyncMock(
            return_value={"matches": [{"line": 10, "content": "matched line"}]}
        )
        tools = build_tools(viking_cap)
        grep_tool = _get_tool(tools, "viking_grep")

        ctx = _make_ctx()
        result = await grep_tool(ctx, uri="viking://doc.md", pattern="hello", case_insensitive=True)

        mock_client.grep.assert_called_once()
        call_args = mock_client.grep.call_args.args
        call_kwargs = mock_client.grep.call_args.kwargs
        assert call_args[0] == "viking://doc.md"
        assert call_args[1] == "hello"
        assert call_kwargs["case_insensitive"] is True
        assert "L10" in result.return_value
        assert "matched line" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_grep_no_matches(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_grep returns 'No matches found.' when empty."""
        mock_client.grep = AsyncMock(return_value={"matches": []})
        tools = build_tools(viking_cap)
        grep_tool = _get_tool(tools, "viking_grep")

        ctx = _make_ctx()
        result = await grep_tool(ctx, uri="viking://doc.md", pattern="nothing")
        assert "No matches found" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_glob(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """viking_glob passes pattern and uri correctly."""
        mock_client.glob = AsyncMock(
            return_value={"matches": ["viking://doc1.md", "viking://doc2.md"]}
        )
        tools = build_tools(viking_cap)
        glob_tool = _get_tool(tools, "viking_glob")

        ctx = _make_ctx()
        result = await glob_tool(ctx, pattern="**/*.md", uri="viking://user/")

        mock_client.glob.assert_called_once()
        call_args = mock_client.glob.call_args.args
        call_kwargs = mock_client.glob.call_args.kwargs
        assert call_args[0] == "**/*.md"
        assert call_kwargs["uri"] == "viking://user/"
        assert "viking://doc1.md" in result.return_value
        assert "viking://doc2.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_glob_no_results(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_glob returns 'No URIs found.' when empty."""
        mock_client.glob = AsyncMock(return_value={"matches": []})
        tools = build_tools(viking_cap)
        glob_tool = _get_tool(tools, "viking_glob")

        ctx = _make_ctx()
        result = await glob_tool(ctx, pattern="**/*.txt")
        assert "No files found" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_ls(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """viking_ls passes uri and recursive, outputs [dir]/[file] markers."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "folder1", "type": "directory"},
                {"name": "file1.md", "type": "file"},
            ]
        )
        tools = build_tools(viking_cap)
        ls_tool = _get_tool(tools, "viking_ls")

        ctx = _make_ctx()
        result = await ls_tool(ctx, uri="viking://user/alice/", recursive=True)

        mock_client.ls.assert_called_once()
        call_args = mock_client.ls.call_args.args
        call_kwargs = mock_client.ls.call_args.kwargs
        assert call_args[0] == "viking://user/alice/"
        assert call_kwargs["recursive"] is True
        assert "[dir] folder1" in result.return_value
        assert "[file] file1.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_ls_empty(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_ls returns '(empty)' for empty listing."""
        mock_client.ls = AsyncMock(return_value=[])
        tools = build_tools(viking_cap)
        ls_tool = _get_tool(tools, "viking_ls")

        ctx = _make_ctx()
        result = await ls_tool(ctx, uri="viking://empty/")
        assert result.return_value == "(empty)"

    @pytest.mark.asyncio
    async def test_viking_read_single_uri(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read reads a single URI with line number prefixes."""
        mock_client.read = AsyncMock(return_value="line1\nline2\nline3")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md", line=1, limit=-1)

        mock_client.read.assert_called_once()
        call_args = mock_client.read.call_args.args
        call_kwargs = mock_client.read.call_args.kwargs
        assert call_args[0] == "viking://doc.md"
        assert call_kwargs["offset"] == 0
        assert call_kwargs["limit"] == -1
        assert "1\u2502 line1" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_line_to_offset_conversion(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read converts line=51 to offset=50."""
        mock_client.read = AsyncMock(return_value="content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        await read_tool(ctx, uris="viking://doc.md", line=51)

        assert mock_client.read.call_args.kwargs["offset"] == 50

    @pytest.mark.asyncio
    async def test_viking_read_multi_uri(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read handles multiple URIs with === {uri} === headers."""
        mock_client.read = AsyncMock(return_value="content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris=["viking://a.md", "viking://b.md"])

        assert mock_client.read.call_count == 2
        assert "=== viking://a.md ===" in result.return_value
        assert "=== viking://b.md ===" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_multi_uri_no_header_for_single(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read does not add === header for single URI."""
        mock_client.read = AsyncMock(return_value="content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://single.md")

        assert "===" not in result.return_value

    # ------------------------------------------------------------------
    # viking_read image branch (support_vision)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_viking_read_image_support_vision_true(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """support_vision=True returns BinaryImage with correct data & mime."""
        viking_cap.support_vision = True
        mock_client.download_bytes = AsyncMock(return_value=b"\x89PNG-fake-image-bytes")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://photo.png")

        mock_client.download_bytes.assert_called_once_with("viking://photo.png")
        mock_client.read.assert_not_called()
        assert result.content is not None
        parts = list(result.content)
        assert any(isinstance(p, BinaryImage) for p in parts)
        img = next(p for p in parts if isinstance(p, BinaryImage))
        assert img.data == b"\x89PNG-fake-image-bytes"
        assert img.media_type == "image/png"

    @pytest.mark.asyncio
    async def test_viking_read_image_support_vision_false(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """support_vision=False returns text URI hint, no download."""
        viking_cap.support_vision = False
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://photo.png")

        mock_client.download_bytes.assert_not_called()
        mock_client.read.assert_not_called()
        assert result.content is None
        assert "viking://photo.png" in result.return_value
        assert "Image resource" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_image_auto_vision_model(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """support_vision=None + image_input=True returns image bytes."""
        from wolfharness_config.model_capabilities import ModelCapabilities

        viking_cap.support_vision = None
        viking_cap.model_capabilities = ModelCapabilities(image_input=True)
        mock_client.download_bytes = AsyncMock(return_value=b"webp-data")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://pic.webp")

        parts = list(result.content) if result.content is not None else []
        assert any(isinstance(p, BinaryImage) for p in parts)
        img = next(p for p in parts if isinstance(p, BinaryImage))
        assert img.media_type == "image/webp"

    @pytest.mark.asyncio
    async def test_viking_read_image_auto_text_only_model(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """support_vision=None + image_input=False returns text URI hint."""
        from wolfharness_config.model_capabilities import ModelCapabilities

        viking_cap.support_vision = None
        viking_cap.model_capabilities = ModelCapabilities(image_input=False)
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://photo.png")

        assert result.content is None
        assert "Image resource" in result.return_value
        mock_client.download_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_viking_read_non_image_ignores_support_vision(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Non-image URIs keep the text path regardless of the switch."""
        viking_cap.support_vision = True
        mock_client.read = AsyncMock(return_value="text content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md")

        mock_client.download_bytes.assert_not_called()
        mock_client.read.assert_called_once()
        assert result.content is None
        assert "text content" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_image_download_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """download_bytes failure returns viking_read error text, no raise."""
        viking_cap.support_vision = True
        mock_client.download_bytes = AsyncMock(side_effect=RuntimeError("boom"))
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://photo.png")

        assert "viking_read error" in result.return_value
        assert "boom" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_image_mixed_uris(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Mixed image + text URIs: image bytes + text sections, order kept."""
        viking_cap.support_vision = True
        mock_client.read = AsyncMock(return_value="doc body")
        mock_client.download_bytes = AsyncMock(return_value=b"img-bytes")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris=["viking://a.md", "viking://photo.jpg", "viking://b.md"])

        assert mock_client.read.call_count == 2
        mock_client.download_bytes.assert_called_once_with("viking://photo.jpg")
        assert "=== viking://photo.jpg ===" in result.return_value
        parts = list(result.content) if result.content is not None else []
        assert any(isinstance(p, BinaryImage) for p in parts)

    @pytest.mark.asyncio
    async def test_viking_read_multi_image_index_maps_to_content_order(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Multi-image reads: #N markers in return_value map to content order.

        The text return_value must carry indexed markers ([Image #1], [#2], ...)
        in URIs order, and the BinaryImage parts in ToolReturn.content must
        follow the same order — so the model can disambiguate which image
        belongs to which URI.
        """
        viking_cap.support_vision = True
        mock_client.download_bytes = AsyncMock(side_effect=[b"a-bytes", b"b-bytes", b"c-bytes"])
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(
            ctx,
            uris=[
                "viking://one.png",
                "viking://two.png",
                "viking://three.png",
            ],
        )

        # Markers appear in URI order, 1-based.
        rv = result.return_value
        i1, i2, i3 = rv.index("[Image #1"), rv.index("[Image #2"), rv.index("[Image #3")
        assert i1 < i2 < i3
        # Content mirrors the same order.
        imgs = [p for p in (result.content or []) if isinstance(p, BinaryImage)]
        assert [p.data for p in imgs] == [b"a-bytes", b"b-bytes", b"c-bytes"]
        assert [p.media_type for p in imgs] == ["image/png"] * 3

    @pytest.mark.asyncio
    async def test_viking_read_image_svg_never_returns_bytes(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """SVG images degrade to text URI hint even with support_vision=True."""
        viking_cap.support_vision = True
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://diagram.svg")

        mock_client.download_bytes.assert_not_called()
        assert result.content is None
        assert "Image resource" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_image_jpeg_mime_extension_map(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Unknown image extension falls back to application/octet-stream."""
        viking_cap.support_vision = True
        mock_client.download_bytes = AsyncMock(return_value=b"raw-bytes")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        await read_tool(ctx, uris="viking://data.xyz")

        # .xyz is not a known image extension → text path untouched.
        mock_client.read = AsyncMock(return_value="content")
        result2 = await read_tool(ctx, uris="viking://data.xyz")
        assert result2.content is None

        # .jpeg maps to image/jpeg.
        result3 = await read_tool(ctx, uris="viking://pic.jpeg")
        parts = list(result3.content) if result3.content is not None else []
        img = next(p for p in parts if isinstance(p, BinaryImage))
        assert img.media_type == "image/jpeg"


# ---------------------------------------------------------------------------
# 8.5 — Test each write tool with mocked client
# ---------------------------------------------------------------------------


class TestWriteTools:
    """Tests for the 6 write tools."""

    @pytest.mark.asyncio
    async def test_viking_remember_schedules_deferred_capture(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_remember schedules a capture without touching the client."""
        tools = build_tools(viking_cap)
        remember_tool = _get_tool(tools, "viking_remember")

        result = await remember_tool(_make_ctx())

        # No session work happens at call time — the capture is deferred.
        mock_client.create_session.assert_not_called()
        mock_client.add_message.assert_not_called()
        mock_client.commit_session.assert_not_called()
        assert viking_cap._remember_pending == [""]
        assert "Capture scheduled" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_remember_records_reason(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_remember appends the optional reason to the pending queue."""
        tools = build_tools(viking_cap)
        remember_tool = _get_tool(tools, "viking_remember")

        result = await remember_tool(_make_ctx(), reason="SY215 oil pressure is 34.3 MPa")

        assert viking_cap._remember_pending == ["SY215 oil pressure is 34.3 MPa"]
        assert "Capture scheduled" in result.return_value
        mock_client.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_viking_write_default_mode(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_write uses mode='create' by default."""
        tools = build_tools(viking_cap)
        write_tool = _get_tool(tools, "viking_write")

        ctx = _make_ctx()
        result = await write_tool(ctx, uri="viking://new.md", content="hello world")

        mock_client.write.assert_called_once()
        call_args = mock_client.write.call_args.args
        call_kwargs = mock_client.write.call_args.kwargs
        assert call_args[0] == "viking://new.md"
        assert call_args[1] == "hello world"
        assert call_kwargs["mode"] == "create"
        assert "Wrote" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_write_replace_mode(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_write passes mode='replace' when specified."""
        tools = build_tools(viking_cap)
        write_tool = _get_tool(tools, "viking_write")

        ctx = _make_ctx()
        await write_tool(ctx, uri="viking://doc.md", content="new", mode="replace")

        assert mock_client.write.call_args.kwargs["mode"] == "replace"

    @pytest.mark.asyncio
    async def test_viking_edit_success(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_edit successfully replaces a string in a document."""
        mock_client.read = AsyncMock(return_value="hello world")
        tools = build_tools(viking_cap)
        edit_tool = _get_tool(tools, "viking_edit")

        ctx = _make_ctx()
        result = await edit_tool(ctx, uri="viking://doc.md", old_string="hello", new_string="hi")

        mock_client.read.assert_called_once()
        mock_client.write.assert_called_once()
        written_content = mock_client.write.call_args.args[1]
        assert written_content == "hi world"
        assert "Replaced 1 occurrence" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_edit_multiple_matches_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_edit returns error when multiple matches found without replace_all."""
        mock_client.read = AsyncMock(return_value="hello world hello")
        tools = build_tools(viking_cap)
        edit_tool = _get_tool(tools, "viking_edit")

        ctx = _make_ctx()
        result = await edit_tool(
            ctx, uri="viking://doc.md", old_string="hello", new_string="hi", replace_all=False
        )

        mock_client.write.assert_not_called()
        assert "error" in result.return_value.lower()
        assert "2 times" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_edit_replace_all(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_edit replaces all occurrences when replace_all=True."""
        mock_client.read = AsyncMock(return_value="hello world hello")
        tools = build_tools(viking_cap)
        edit_tool = _get_tool(tools, "viking_edit")

        ctx = _make_ctx()
        result = await edit_tool(
            ctx, uri="viking://doc.md", old_string="hello", new_string="hi", replace_all=True
        )

        mock_client.write.assert_called_once()
        written_content = mock_client.write.call_args.args[1]
        assert written_content == "hi world hi"
        assert "Replaced 2 occurrence" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_edit_no_matches_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_edit returns error when old_string not found."""
        mock_client.read = AsyncMock(return_value="hello world")
        tools = build_tools(viking_cap)
        edit_tool = _get_tool(tools, "viking_edit")

        ctx = _make_ctx()
        result = await edit_tool(
            ctx, uri="viking://doc.md", old_string="nonexistent", new_string="x"
        )

        mock_client.write.assert_not_called()
        assert "error" in result.return_value.lower()
        assert "not found" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_mkdir(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """viking_mkdir passes uri and description correctly."""
        tools = build_tools(viking_cap)
        mkdir_tool = _get_tool(tools, "viking_mkdir")

        ctx = _make_ctx()
        result = await mkdir_tool(ctx, uri="viking://new/dir/", description="My directory")

        mock_client.mkdir.assert_called_once()
        call_args = mock_client.mkdir.call_args.args
        call_kwargs = mock_client.mkdir.call_args.kwargs
        assert call_args[0] == "viking://new/dir/"
        assert call_kwargs["description"] == "My directory"
        assert "Created directory" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_mkdir_no_description(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_mkdir works without a description."""
        tools = build_tools(viking_cap)
        mkdir_tool = _get_tool(tools, "viking_mkdir")

        ctx = _make_ctx()
        result = await mkdir_tool(ctx, uri="viking://new/dir/")

        assert mock_client.mkdir.call_args.kwargs["description"] is None
        assert "Created directory" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_add_resource(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_add_resource passes all params correctly."""
        mock_client.add_resource = AsyncMock(return_value={"status": "ok", "id": "res-123"})
        tools = build_tools(viking_cap)
        add_tool = _get_tool(tools, "viking_add_resource")

        ctx = _make_ctx()
        result = await add_tool(
            ctx,
            path="/local/file.txt",
            to="viking://user/alice/files/",
            parent="viking://user/alice/",
            processing_mode="auto",
            watch_interval=5.0,
        )

        mock_client.add_resource.assert_called_once()
        call_args = mock_client.add_resource.call_args.args
        call_kwargs = mock_client.add_resource.call_args.kwargs
        assert call_args[0] == "/local/file.txt"
        assert call_kwargs["to"] == "viking://user/alice/files/"
        assert call_kwargs["parent"] == "viking://user/alice/"
        # processing_mode is NOT passed to SDK (not a supported kwarg)
        assert "processing_mode" not in call_kwargs
        assert call_kwargs["watch_interval"] == 5.0
        assert "Added resource" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_forget(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_forget calls rm() with the recursive flag."""
        tools = build_tools(viking_cap)
        forget_tool = _get_tool(tools, "viking_forget")

        ctx = _make_ctx()
        result = await forget_tool(ctx, uri="viking://doc.md", recursive=True)

        mock_client.rm.assert_called_once()
        call_args = mock_client.rm.call_args.args
        call_kwargs = mock_client.rm.call_args.kwargs
        assert call_args[0] == "viking://doc.md"
        assert call_kwargs["recursive"] is True
        assert "Removed" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_forget_non_recursive(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_forget passes recursive=False by default."""
        tools = build_tools(viking_cap)
        forget_tool = _get_tool(tools, "viking_forget")

        ctx = _make_ctx()
        await forget_tool(ctx, uri="viking://doc.md")

        assert mock_client.rm.call_args.kwargs["recursive"] is False


# ---------------------------------------------------------------------------
# 8.6 — Test each graph tool with mocked client
# ---------------------------------------------------------------------------


class TestGraphTools:
    """Tests for the 2 graph tools."""

    @pytest.mark.asyncio
    async def test_viking_link_single_target(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_link passes from_uri, to_uris (string), reason correctly."""
        tools = build_tools(viking_cap)
        link_tool = _get_tool(tools, "viking_link")

        ctx = _make_ctx()
        result = await link_tool(
            ctx, from_uri="viking://a.md", to_uris="viking://b.md", reason="depends-on"
        )

        mock_client.link.assert_called_once()
        call_args = mock_client.link.call_args.args
        call_kwargs = mock_client.link.call_args.kwargs
        assert call_args[0] == "viking://a.md"
        assert call_args[1] == "viking://b.md"
        assert call_kwargs["reason"] == "depends-on"
        assert "Linked" in result.return_value
        assert "viking://a.md" in result.return_value
        assert "viking://b.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_link_multiple_targets(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_link handles a list of target URIs."""
        tools = build_tools(viking_cap)
        link_tool = _get_tool(tools, "viking_link")

        ctx = _make_ctx()
        result = await link_tool(
            ctx,
            from_uri="viking://a.md",
            to_uris=["viking://b.md", "viking://c.md"],
            reason="references",
        )

        mock_client.link.assert_called_once()
        call_args = mock_client.link.call_args.args
        assert call_args[0] == "viking://a.md"
        assert call_args[1] == ["viking://b.md", "viking://c.md"]
        assert "viking://b.md" in result.return_value
        assert "viking://c.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_set_tags(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_set_tags passes uri, tags, recursive correctly."""
        tools = build_tools(viking_cap)
        tags_tool = _get_tool(tools, "viking_set_tags")

        ctx = _make_ctx()
        result = await tags_tool(
            ctx, uri="viking://doc.md", tags=["status=active", "priority=high"], recursive=True
        )

        mock_client.set_tags.assert_called_once()
        call_args = mock_client.set_tags.call_args.args
        call_kwargs = mock_client.set_tags.call_args.kwargs
        assert call_args[0] == "viking://doc.md"
        assert call_args[1] == ["status=active", "priority=high"]
        assert call_kwargs["recursive"] is True
        assert "Set 2 tag" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_set_tags_non_recursive(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_set_tags passes recursive=False by default."""
        tools = build_tools(viking_cap)
        tags_tool = _get_tool(tools, "viking_set_tags")

        ctx = _make_ctx()
        await tags_tool(ctx, uri="viking://doc.md", tags=["key=val"])

        assert mock_client.set_tags.call_args.kwargs["recursive"] is False


# ---------------------------------------------------------------------------
# 8.7 — Test viking_read pagination (detailed)
# ---------------------------------------------------------------------------


class TestVikingReadPagination:
    """Detailed tests for viking_read pagination and formatting."""

    @pytest.mark.asyncio
    async def test_line_to_offset_conversion(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """line=1 -> offset=0, line=10 -> offset=9, line=51 -> offset=50."""
        mock_client.read = AsyncMock(return_value="content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        for line, expected_offset in [(1, 0), (10, 9), (51, 50), (100, 99)]:
            mock_client.read.reset_mock()
            await read_tool(ctx, uris="viking://doc.md", line=line)
            assert mock_client.read.call_args.kwargs["offset"] == expected_offset

    @pytest.mark.asyncio
    async def test_limit_passed_correctly(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Limit is passed to SDK read() correctly."""
        mock_client.read = AsyncMock(return_value="content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        await read_tool(ctx, uris="viking://doc.md", line=1, limit=50)
        assert mock_client.read.call_args.kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_line_number_prefixes(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read adds line number prefixes to output."""
        mock_client.read = AsyncMock(return_value="first\nsecond\nthird")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md", line=1)

        lines = result.return_value.split("\n")
        assert len(lines) == 3
        assert "1" in lines[0]
        assert "first" in lines[0]
        assert "2" in lines[1]
        assert "second" in lines[1]
        assert "3" in lines[2]
        assert "third" in lines[2]

    @pytest.mark.asyncio
    async def test_multi_uri_batch_headers(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read includes === {uri} === headers for multi-URI reads."""
        mock_client.read = AsyncMock(return_value="content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        uris = ["viking://a.md", "viking://b.md", "viking://c.md"]
        result = await read_tool(ctx, uris=uris)

        assert mock_client.read.call_count == 3
        for uri in uris:
            assert f"=== {uri} ===" in result.return_value


# ---------------------------------------------------------------------------
# 8.8 — Test viking_edit (additional edge cases)
# ---------------------------------------------------------------------------


class TestVikingEditEdgeCases:
    """Additional edge case tests for viking_edit."""

    @pytest.mark.asyncio
    async def test_edit_read_modify_write_cycle(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_edit performs a full read-modify-write cycle."""
        original = "The quick brown fox"
        mock_client.read = AsyncMock(return_value=original)
        tools = build_tools(viking_cap)
        edit_tool = _get_tool(tools, "viking_edit")

        ctx = _make_ctx()
        await edit_tool(ctx, uri="viking://doc.md", old_string="quick", new_string="slow")

        mock_client.read.assert_called_once()
        assert mock_client.read.call_args.args[0] == "viking://doc.md"
        mock_client.write.assert_called_once()
        assert mock_client.write.call_args.args[0] == "viking://doc.md"
        assert mock_client.write.call_args.args[1] == "The slow brown fox"
        assert mock_client.write.call_args.kwargs["mode"] == "replace"

    @pytest.mark.asyncio
    async def test_edit_file_not_found(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_edit returns error string when read raises an exception."""
        mock_client.read = AsyncMock(side_effect=FileNotFoundError("not found"))
        tools = build_tools(viking_cap)
        edit_tool = _get_tool(tools, "viking_edit")

        ctx = _make_ctx()
        result = await edit_tool(ctx, uri="viking://missing.md", old_string="old", new_string="new")

        mock_client.write.assert_not_called()
        assert "viking_edit error" in result.return_value


# ---------------------------------------------------------------------------
# 8.9 — Test viking_recall (detailed)
# ---------------------------------------------------------------------------


class TestVikingRecallDetailed:
    """Detailed tests for viking_recall quota enforcement and result merging."""

    @pytest.mark.asyncio
    async def test_default_quotas(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Default quotas are {memory: 5, resource: 3, skill: 2}."""
        mock_client.find = AsyncMock(return_value={"results": []})
        tools = build_tools(viking_cap)
        recall_tool = _get_tool(tools, "viking_recall")

        ctx = _make_ctx()
        await recall_tool(ctx, query="test")

        calls = mock_client.find.call_args_list
        quota_map = {c.kwargs["context_type"]: c.kwargs["limit"] for c in calls}
        assert quota_map == {"memory": 5, "resource": 3, "skill": 2}

    @pytest.mark.asyncio
    async def test_result_merge(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """Results from multiple find() calls are merged with section headers."""
        mock_client.find = AsyncMock(
            return_value={"hits": [{"uri": "viking://mem.md", "content": "data"}]}
        )
        tools = build_tools(viking_cap)
        recall_tool = _get_tool(tools, "viking_recall")

        ctx = _make_ctx()
        result = await recall_tool(ctx, query="test", max_chars=10000)

        assert "=== memory ===" in result.return_value
        assert "=== resource ===" in result.return_value
        assert "=== skill ===" in result.return_value
        assert result.return_value.count("viking://mem.md") == 3

    @pytest.mark.asyncio
    async def test_truncation(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """Output is truncated when it exceeds max_chars."""
        long_content = "x" * 5000
        mock_client.find = AsyncMock(
            return_value={"hits": [{"uri": "viking://mem.md", "content": long_content}]}
        )
        tools = build_tools(viking_cap)
        recall_tool = _get_tool(tools, "viking_recall")

        ctx = _make_ctx()
        result = await recall_tool(ctx, query="test", max_chars=100)

        assert len(result.return_value) <= 200
        assert "truncated" in result.return_value


# ---------------------------------------------------------------------------
# 8.10 — Test viking_remember (detailed)
# ---------------------------------------------------------------------------


class TestVikingRememberDeferred:
    """Deferred ``viking_remember`` capture semantics.

    The tool only queues reasons; the drain runs at the next
    ``before_model_request`` (or ``after_run``) and ingests the real
    conversation into a ``remember-`` session.
    """

    @pytest.mark.asyncio
    async def test_drain_ingests_real_conversation_with_marker(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Drain ingests real pairs to a remember session, appends the marker."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        tools = build_tools(viking_cap)
        await _get_tool(tools, "viking_remember")(_make_ctx(), reason="remember this")

        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is X?")]),
            ModelResponse(parts=[TextPart(content="X is a thing.")]),
        ]
        rc = _make_request_context(messages)
        result = await viking_cap._handle_remember_drain(_make_ctx(), rc)

        assert result is rc
        assert mock_client.create_session.call_args.kwargs["session_id"].startswith("remember-")
        add_calls = mock_client.add_message.call_args_list
        assert (add_calls[0].args[1], add_calls[0].args[2]) == ("user", "What is X?")
        assert (add_calls[1].args[1], add_calls[1].args[2]) == ("assistant", "X is a thing.")
        # Intent marker appended as a trailing message
        assert "<memory-intent>remember this</memory-intent>" in add_calls[2].args[2]
        mock_client.commit_session.assert_called_once()
        # Success semantics: cursor advanced, reasons cleared
        assert viking_cap._last_ingested_idx == 2
        assert viking_cap._remember_pending == []

    @pytest.mark.asyncio
    async def test_drain_sanitizes_unconditionally(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Drain strips injected XML blocks regardless of auto_ingest_sanitize."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        viking_cap.auto_ingest_sanitize = False  # remember must sanitize anyway
        tools = build_tools(viking_cap)
        await _get_tool(tools, "viking_remember")(_make_ctx())

        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="Q <openviking-recall>secret</openviking-recall>")]
            )
        ]
        rc = _make_request_context(messages)
        await viking_cap._handle_remember_drain(_make_ctx(), rc)

        add_calls = mock_client.add_message.call_args_list
        assert "[recalled context omitted]" in add_calls[0].args[2]
        assert "secret" not in add_calls[0].args[2]

    @pytest.mark.asyncio
    async def test_drain_multiple_reasons_merge_into_one_commit(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Two remember calls within a boundary merge into one capture."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        tools = build_tools(viking_cap)
        remember_tool = _get_tool(tools, "viking_remember")
        await remember_tool(_make_ctx(), reason="reason-a")
        await remember_tool(_make_ctx(), reason="reason-b")

        messages = [ModelRequest(parts=[UserPromptPart(content="prompt")])]
        rc = _make_request_context(messages)
        await viking_cap._handle_remember_drain(_make_ctx(), rc)

        # One session, one commit; marker per reason.
        mock_client.create_session.assert_called_once()
        mock_client.commit_session.assert_called_once()
        marker_texts = " ".join(c.args[2] for c in mock_client.add_message.call_args_list)
        assert "<memory-intent>reason-a</memory-intent>" in marker_texts
        assert "<memory-intent>reason-b</memory-intent>" in marker_texts

    @pytest.mark.asyncio
    async def test_drain_no_op_without_pending(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Without pending reasons the drain leaves the context untouched."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        messages = [ModelRequest(parts=[UserPromptPart(content="prompt")])]
        rc = _make_request_context(messages)
        result = await viking_cap._handle_remember_drain(_make_ctx(), rc)

        assert result is rc
        mock_client.create_session.assert_not_called()
        assert viking_cap._last_ingested_idx == 0

    @pytest.mark.asyncio
    async def test_drain_failed_commit_keeps_cursor_and_reasons(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """A failed drain retries: cursor unchanged and reasons retained."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.commit_session = AsyncMock(side_effect=RuntimeError("server down"))
        tools = build_tools(viking_cap)
        remember_tool = _get_tool(tools, "viking_remember")
        await remember_tool(_make_ctx(), reason="keep me")

        messages = [ModelRequest(parts=[UserPromptPart(content="prompt")])]
        rc = _make_request_context(messages)
        await viking_cap._handle_remember_drain(_make_ctx(), rc)

        assert viking_cap._last_ingested_idx == 0
        assert viking_cap._remember_pending == ["keep me"]

    @pytest.mark.asyncio
    async def test_drain_drops_reasons_after_retry_cap(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Consecutive failures drop pending reasons after the retry cap."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.commit_session = AsyncMock(side_effect=RuntimeError("server down"))
        tools = build_tools(viking_cap)
        remember_tool = _get_tool(tools, "viking_remember")
        rc = _make_request_context([ModelRequest(parts=[UserPromptPart(content="prompt")])])

        for _ in range(3):
            await remember_tool(_make_ctx(), reason="r")
            await viking_cap._handle_remember_drain(_make_ctx(), rc)

        assert viking_cap._remember_drain_failures == 3
        assert viking_cap._remember_pending == []

    @pytest.mark.asyncio
    async def test_drain_success_advances_cursor(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """A successful drain with a full commit result advances cursor/clears reasons."""
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            UserPromptPart,
        )

        mock_client.commit_session = AsyncMock(
            return_value={"archive_uri": "viking://user/u/sessions/s1", "task_id": "task-1"}
        )
        tools = build_tools(viking_cap)
        await _get_tool(tools, "viking_remember")(_make_ctx(), reason="n")

        messages = [
            ModelRequest(parts=[UserPromptPart(content="P")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        await viking_cap._handle_remember_drain(_make_ctx(), rc)

        assert viking_cap._last_ingested_idx == 2
        assert viking_cap._remember_pending == []
        assert viking_cap._remember_drain_failures == 0

    @pytest.mark.asyncio
    async def test_notify_task_steers_formatted_summary(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """The notification task steers the formatted memory diff into the session."""
        commit_result = {
            "archive_uri": "viking://user/u/sessions/s1",
            "task_id": "task-1",
        }
        mock_client._request = AsyncMock(return_value={"status": "completed"})
        mock_client.read = AsyncMock(
            return_value={"added": ["viking://user/u/memories/x.md"], "updated": [], "deleted": []}
        )
        session_pool = AsyncMock()
        session_pool.steer_from_background_task = AsyncMock(return_value="steer-1")

        await viking_cap._notify_memory_diff(mock_client, commit_result, session_pool, "run-1")

        session_pool.steer_from_background_task.assert_awaited_once()
        steer_msg = session_pool.steer_from_background_task.await_args.args[1]
        assert "added: viking://user/u/memories/x.md" in steer_msg

    @pytest.mark.asyncio
    async def test_notify_task_failure_is_swallowed(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """A broken poll or steer never raises into the run."""
        commit_result = {"archive_uri": "viking://a", "task_id": "t1"}

        # Poll raises -> extraction wait fails -> no steer, no exception.
        mock_client._request = AsyncMock(side_effect=RuntimeError("poll exploded"))
        session_pool = AsyncMock()
        await viking_cap._notify_memory_diff(mock_client, commit_result, session_pool, "run-1")
        session_pool.steer_from_background_task.assert_not_awaited()

        # Steer raises after a successful poll -> swallowed.
        mock_client._request = AsyncMock(return_value={"status": "completed"})
        mock_client.read = AsyncMock(return_value={"added": ["viking://m"]})
        session_pool.steer_from_background_task = AsyncMock(
            side_effect=RuntimeError("steer dropped")
        )
        await viking_cap._notify_memory_diff(mock_client, commit_result, session_pool, "run-1")

    @pytest.mark.asyncio
    async def test_wait_for_extraction_returns_false_on_failed_status(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """A task that ended in a failed state stops the poll with False."""
        mock_client._request = AsyncMock(return_value={"status": "failed"})
        assert await viking_cap._wait_for_extraction(mock_client, "task-1", timeout=2.0) is False

    @pytest.mark.asyncio
    async def test_notify_task_never_spawned_when_disabled(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """remember_notify=False skips the steer notification."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        viking_cap.remember_notify = False
        mock_client.commit_session = AsyncMock(
            return_value={"archive_uri": "viking://a", "task_id": "t1"}
        )
        tools = build_tools(viking_cap)
        await _get_tool(tools, "viking_remember")(_make_ctx())

        rc = _make_request_context([ModelRequest(parts=[UserPromptPart(content="P")])])
        await viking_cap._handle_remember_drain(_make_ctx(), rc)

        # No stray notification task was spawned — the drain returns cleanly.
        assert viking_cap._last_ingested_idx == 1
        assert viking_cap._remember_pending == []


# ---------------------------------------------------------------------------
# 8.10b — Test remember boundary wiring (before_model_request / after_run)
# ---------------------------------------------------------------------------


class TestRememberBoundaryIntegration:
    """Remember capture wired into the model-request boundary and run end."""

    @pytest.mark.asyncio
    async def test_remember_captures_at_boundary_with_auto_ingest_disabled(
        self, mock_client: AsyncMock
    ) -> None:
        """The drain runs at before_model_request even with auto_ingest off."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(mode="all", enable_memory=True, auto_ingest_enabled=False)
        cap._client = mock_client
        tools = build_tools(cap)
        await _get_tool(tools, "viking_remember")(_make_ctx(), reason="P")

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        await cap.before_model_request(_make_ctx(), rc)

        mock_client.create_session.assert_called_once()
        assert mock_client.create_session.call_args.kwargs["session_id"].startswith("remember-")
        mock_client.commit_session.assert_called_once()
        assert cap._last_ingested_idx == 2

    @pytest.mark.asyncio
    async def test_remember_and_auto_ingest_drain_disjoint_ranges(
        self, mock_client: AsyncMock
    ) -> None:
        """With auto_ingest on, remember drains first — auto_ingest skips its range."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            enable_memory=True,
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
        )
        cap._client = mock_client
        tools = build_tools(cap)
        await _get_tool(tools, "viking_remember")(_make_ctx())

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        await cap.before_model_request(_make_ctx(), rc)

        # Exactly one session (the remember one) — no double commit.
        mock_client.create_session.assert_called_once()
        assert mock_client.create_session.call_args.kwargs["session_id"].startswith("remember-")
        mock_client.commit_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_after_run_flushes_final_assistant_message(self, mock_client: AsyncMock) -> None:
        """after_run captures trailing messages the cursor never saw (auto_ingest path)."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(mode="all", auto_ingest_enabled=True, auto_ingest_mode="sync")
        cap._client = mock_client
        cap._last_ingested_idx = 1  # user turn already ingested at its boundary

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="final answer")]),
        ]
        ctx = _make_ctx()
        ctx.messages = messages
        result = await cap.after_run(ctx, result="done")

        assert result == "done"
        mock_client.create_session.assert_called_once()
        add_calls = mock_client.add_message.call_args_list
        assert (add_calls[0].args[1], add_calls[0].args[2]) == ("assistant", "final answer")
        assert cap._last_ingested_idx == 2

    @pytest.mark.asyncio
    async def test_after_run_flushes_last_moment_remember(self, mock_client: AsyncMock) -> None:
        """after_run flushes a remember intent from the run's final turn."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(mode="all", enable_memory=True)
        cap._client = mock_client
        tools = build_tools(cap)
        await _get_tool(tools, "viking_remember")(_make_ctx(), reason="final")

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        ctx = _make_ctx()
        ctx.messages = messages
        await cap.after_run(ctx, result="done")

        mock_client.create_session.assert_called_once()
        assert mock_client.create_session.call_args.kwargs["session_id"].startswith("remember-")
        marker_texts = " ".join(c.args[2] for c in mock_client.add_message.call_args_list)
        assert "<memory-intent>final</memory-intent>" in marker_texts
        assert cap._remember_pending == []

    @pytest.mark.asyncio
    async def test_after_run_no_ingest_when_all_capture_disabled(
        self, mock_client: AsyncMock
    ) -> None:
        """after_run does NOT capture when both auto_ingest and remember are off."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(mode="all")  # auto_ingest_enabled=False, no remember
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        ctx = _make_ctx()
        ctx.messages = messages
        result = await cap.after_run(ctx, result="done")

        assert result == "done"
        mock_client.create_session.assert_not_called()
        mock_client.commit_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_add_message_failure_keeps_cursor_and_reasons(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """add_message failing mid-pipeline behaves like a commit failure."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.add_message = AsyncMock(side_effect=RuntimeError("write rejected"))
        tools = build_tools(viking_cap)
        remember_tool = _get_tool(tools, "viking_remember")
        await remember_tool(_make_ctx(), reason="keep")

        rc = _make_request_context([ModelRequest(parts=[UserPromptPart(content="P")])])
        await viking_cap._handle_remember_drain(_make_ctx(), rc)

        assert viking_cap._last_ingested_idx == 0
        assert viking_cap._remember_pending == ["keep"]


# ---------------------------------------------------------------------------
# 8.11 — Test error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests that all tools return error strings, never raise exceptions."""

    @pytest.mark.asyncio
    async def test_search_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.search = AsyncMock(side_effect=RuntimeError("connection failed"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_search")(ctx, query="test")
        assert "viking_search error (RuntimeError): connection failed" in result.return_value

    @pytest.mark.asyncio
    async def test_find_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.find = AsyncMock(side_effect=RuntimeError("timeout"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_find")(ctx, query="test")
        assert "viking_find error (RuntimeError): timeout" in result.return_value

    @pytest.mark.asyncio
    async def test_recall_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.find = AsyncMock(side_effect=RuntimeError("server error"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_recall")(ctx, query="test")
        assert "viking_recall error (RuntimeError): server error" in result.return_value

    @pytest.mark.asyncio
    async def test_grep_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        """viking_grep catches per-pattern errors and returns no matches."""
        mock_client.grep = AsyncMock(side_effect=RuntimeError("bad pattern"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_grep")(ctx, uri="viking://doc.md", pattern="test")
        # With multi-pattern support, individual grep errors are caught silently
        assert "No matches found" in result.return_value

    @pytest.mark.asyncio
    async def test_glob_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.glob = AsyncMock(side_effect=RuntimeError("error"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_glob")(ctx, pattern="**/*.md")
        assert "viking_glob error (RuntimeError): error" in result.return_value

    @pytest.mark.asyncio
    async def test_ls_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.ls = AsyncMock(side_effect=RuntimeError("not found"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_ls")(ctx, uri="viking://missing/")
        assert "viking_ls error (RuntimeError): not found" in result.return_value

    @pytest.mark.asyncio
    async def test_read_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.read = AsyncMock(side_effect=RuntimeError("permission denied"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_read")(ctx, uris="viking://secret.md")
        assert "viking_read error (RuntimeError): permission denied" in result.return_value

    @pytest.mark.asyncio
    async def test_write_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.write = AsyncMock(side_effect=RuntimeError("disk full"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_write")(ctx, uri="viking://doc.md", content="data")
        assert "viking_write error (RuntimeError): disk full" in result.return_value

    @pytest.mark.asyncio
    async def test_edit_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.read = AsyncMock(side_effect=RuntimeError("network error"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_edit")(
            ctx, uri="viking://doc.md", old_string="a", new_string="b"
        )
        assert "viking_edit error (RuntimeError): network error" in result.return_value

    @pytest.mark.asyncio
    async def test_mkdir_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.mkdir = AsyncMock(side_effect=RuntimeError("exists"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_mkdir")(ctx, uri="viking://exists/")
        assert "viking_mkdir error (RuntimeError): exists" in result.return_value

    @pytest.mark.asyncio
    async def test_add_resource_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.add_resource = AsyncMock(side_effect=RuntimeError("invalid path"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_add_resource")(ctx, path="/bad/path")
        assert "viking_add_resource error (RuntimeError): invalid path" in result.return_value

    @pytest.mark.asyncio
    async def test_forget_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.rm = AsyncMock(side_effect=RuntimeError("protected"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_forget")(ctx, uri="viking://protected.md")
        assert "viking_forget error (RuntimeError): protected" in result.return_value

    @pytest.mark.asyncio
    async def test_link_error(self, viking_cap: VikingCapability, mock_client: AsyncMock) -> None:
        mock_client.link = AsyncMock(side_effect=RuntimeError("cycle detected"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_link")(
            ctx, from_uri="viking://a.md", to_uris="viking://b.md"
        )
        assert "viking_link error (RuntimeError): cycle detected" in result.return_value

    @pytest.mark.asyncio
    async def test_set_tags_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.set_tags = AsyncMock(side_effect=RuntimeError("invalid tag"))
        tools = build_tools(viking_cap)
        ctx = _make_ctx()
        result = await _get_tool(tools, "viking_set_tags")(ctx, uri="viking://doc.md", tags=["bad"])
        assert "viking_set_tags error (RuntimeError): invalid tag" in result.return_value

    @pytest.mark.asyncio
    async def test_ensure_client_lazy_init(self) -> None:
        """_ensure_client lazily initializes the SDK client when not set."""
        cap = VikingCapability(mode="all", url="https://dummy.example.com")
        # _ensure_client should create a client (lazy init), not raise.
        client = await cap._ensure_client()
        assert client is not None
        # Second call returns the same client (no re-init).
        client2 = await cap._ensure_client()
        assert client is client2
        # Clean up
        await cap.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# 8.12 — Test SkillResource methods
# ---------------------------------------------------------------------------


class TestSkillResource:
    """Tests for SkillResource protocol methods."""

    @pytest.mark.asyncio
    async def test_list_skills_success(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_skills returns SkillEntry list with source='remote'."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "ponytail.md", "type": "file"},
                {"name": "brainstorming.md", "type": "file"},
                {"name": "notes", "type": "directory"},
            ]
        )
        skills = await viking_cap.list_skills()

        assert len(skills) == 2
        assert all(s.source == "remote" for s in skills)
        assert all(s.skill_path is None for s in skills)
        names = [s.name for s in skills]
        assert "ponytail" in names
        assert "brainstorming" in names

    @pytest.mark.asyncio
    async def test_list_skills_empty(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_skills returns empty list when no skills found."""
        mock_client.ls = AsyncMock(return_value=[])
        skills = await viking_cap.list_skills()
        assert skills == []

    @pytest.mark.asyncio
    async def test_list_skills_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_skills returns empty list on error."""
        mock_client.ls = AsyncMock(side_effect=RuntimeError("connection failed"))
        skills = await viking_cap.list_skills()
        assert skills == []

    @pytest.mark.asyncio
    async def test_list_skills_not_initialized(self) -> None:
        """list_skills returns empty list when client is not initialized."""
        cap = VikingCapability(mode="all")
        skills = await cap.list_skills()
        assert skills == []

    @pytest.mark.asyncio
    async def test_list_skills_non_list_response(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_skills returns empty list when SDK returns non-list."""
        mock_client.ls = AsyncMock(return_value={"error": "unexpected"})
        skills = await viking_cap.list_skills()
        assert skills == []

    @pytest.mark.asyncio
    async def test_list_skills_string_entries(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_skills handles string entries from ls."""
        mock_client.ls = AsyncMock(return_value=["skill1.md", "skill2.md", "not_a_skill"])
        skills = await viking_cap.list_skills()

        assert len(skills) == 2
        assert all(s.source == "remote" for s in skills)
        names = [s.name for s in skills]
        assert "skill1" in names
        assert "skill2" in names

    @pytest.mark.asyncio
    async def test_read_skill_success(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """read_skill returns skill content as string."""
        mock_client.read = AsyncMock(return_value="# Ponytail Skill\n\nInstructions...")
        content = await viking_cap.read_skill("ponytail")

        assert content is not None
        assert "Ponytail Skill" in content
        expected_uri = "viking://user/default/skills/ponytail.md"
        assert mock_client.read.call_args.args[0] == expected_uri

    @pytest.mark.asyncio
    async def test_read_skill_not_found(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """read_skill returns None when skill doesn't exist."""
        mock_client.read = AsyncMock(side_effect=FileNotFoundError("not found"))
        content = await viking_cap.read_skill("nonexistent")
        assert content is None

    @pytest.mark.asyncio
    async def test_read_skill_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """read_skill returns None on error."""
        mock_client.read = AsyncMock(side_effect=RuntimeError("server error"))
        content = await viking_cap.read_skill("test")
        assert content is None

    @pytest.mark.asyncio
    async def test_read_skill_empty_content(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """read_skill returns None when content is empty."""
        mock_client.read = AsyncMock(return_value="")
        content = await viking_cap.read_skill("empty")
        assert content is None

    @pytest.mark.asyncio
    async def test_read_skill_not_initialized(self) -> None:
        """read_skill returns None when client is not initialized."""
        cap = VikingCapability(mode="all")
        content = await cap.read_skill("test")
        assert content is None

    @pytest.mark.asyncio
    async def test_skill_exists_true(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """skill_exists returns True when skill is found."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "ponytail.md", "type": "file"},
                {"name": "other.md", "type": "file"},
            ]
        )
        exists = await viking_cap.skill_exists("ponytail")
        assert exists is True

    @pytest.mark.asyncio
    async def test_skill_exists_false(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """skill_exists returns False when skill is not found."""
        mock_client.ls = AsyncMock(return_value=[{"name": "other.md"}])
        exists = await viking_cap.skill_exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_skill_exists_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """skill_exists returns False on error."""
        mock_client.ls = AsyncMock(side_effect=RuntimeError("error"))
        exists = await viking_cap.skill_exists("test")
        assert exists is False

    @pytest.mark.asyncio
    async def test_skill_exists_not_initialized(self) -> None:
        """skill_exists returns False when client is not initialized."""
        cap = VikingCapability(mode="all")
        exists = await cap.skill_exists("test")
        assert exists is False

    @pytest.mark.asyncio
    async def test_skill_exists_non_list_response(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """skill_exists returns False when SDK returns non-list."""
        mock_client.ls = AsyncMock(return_value="unexpected")
        exists = await viking_cap.skill_exists("test")
        assert exists is False

    def test_resolve_skills_uri_with_override(self) -> None:
        """_resolve_skills_uri returns override when set."""
        cap = VikingCapability(mode="all", skills_uri="viking://custom/skills/")
        assert cap._resolve_skills_uri() == "viking://custom/skills/"

    def test_resolve_skills_uri_default(self) -> None:
        """_resolve_skills_uri uses default convention when no override."""
        cap = VikingCapability(mode="all", user="alice")
        assert cap._resolve_skills_uri() == "viking://user/alice/skills/"

    def test_resolve_skills_uri_default_user(self) -> None:
        """_resolve_skills_uri uses 'default' when user is None."""
        cap = VikingCapability(mode="all")
        assert cap._resolve_skills_uri() == "viking://user/default/skills/"


# ---------------------------------------------------------------------------
# 8.13 — Test mode filtering
# ---------------------------------------------------------------------------


class TestModeFiltering:
    """Tests that mode filtering exposes the correct number of tools."""

    def test_retrieve_mode_7_tools_default(self) -> None:
        """Retrieve mode exposes 7 tools (viking_expand added by compaction feature)."""
        cap = VikingCapability(mode="retrieve")
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 7
        names = {t.__name__ for t in tools}
        assert names == {
            "viking_search",
            "viking_find",
            "viking_grep",
            "viking_glob",
            "viking_ls",
            "viking_read",
            "viking_expand",
        }

    def test_retrieve_mode_8_tools_with_memory(self) -> None:
        """Retrieve mode exposes 8 tools when enable_memory=True (7 + recall)."""
        cap = VikingCapability(mode="retrieve", enable_memory=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 8
        names = {t.__name__ for t in tools}
        assert "viking_recall" in names

    def test_write_mode_4_tools_default(self) -> None:
        """Write mode exposes 4 tools (forget gated by enable_forget, remember by enable_memory)."""
        cap = VikingCapability(mode="write")
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 4
        names = {t.__name__ for t in tools}
        assert names == {
            "viking_write",
            "viking_edit",
            "viking_mkdir",
            "viking_add_resource",
        }

    def test_write_mode_6_tools_with_memory_and_forget(self) -> None:
        """Write mode exposes 6 tools when enable_memory=True and enable_forget=True."""
        cap = VikingCapability(mode="write", enable_memory=True, enable_forget=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 6
        names = {t.__name__ for t in tools}
        assert "viking_remember" in names
        assert "viking_forget" in names

    def test_graph_mode_1_tool_default(self) -> None:
        """Graph mode exposes 1 tool (link gated by enable_link)."""
        cap = VikingCapability(mode="graph")
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 1
        names = {t.__name__ for t in tools}
        assert names == {"viking_set_tags"}

    def test_graph_mode_2_tools_with_link(self) -> None:
        """Graph mode exposes 2 tools when enable_link=True."""
        cap = VikingCapability(mode="graph", enable_link=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 2
        names = {t.__name__ for t in tools}
        assert names == {"viking_link", "viking_set_tags"}

    def test_all_mode_12_tools_default(self) -> None:
        """All mode exposes 12 tools by default (link, memory, and forget gated off)."""
        cap = VikingCapability(mode="all")
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 12

    def test_all_mode_16_tools_with_flags(self) -> None:
        """All mode exposes 16 tools when enable_link + enable_memory + enable_forget."""
        cap = VikingCapability(mode="all", enable_link=True, enable_memory=True, enable_forget=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        assert len(tools) == 16

    def test_disabled_tools_excludes_search_find(self) -> None:
        """disabled_tools blacklist removes viking_search and viking_find."""
        cap = VikingCapability(
            mode="retrieve",
            disabled_tools=["viking_search", "viking_find"],
        )
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_search" not in names
        assert "viking_find" not in names
        assert "viking_read" in names
        assert "viking_grep" in names

    def test_enabled_tools_whitelist(self) -> None:
        """enabled_tools whitelist keeps only the listed tools."""
        cap = VikingCapability(mode="retrieve", enabled_tools=["viking_ls", "viking_read"])
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert names == {"viking_ls", "viking_read"}

    def test_enabled_tools_unknown_names_ignored(self) -> None:
        """enabled_tools entries that don't match any tool are ignored."""
        cap = VikingCapability(mode="retrieve", enabled_tools=["viking_ls", "nope_tool"])
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert names == {"viking_ls"}

    def test_get_toolset_retrieve(self) -> None:
        """get_toolset() returns a FunctionToolset with 7 tools for retrieve mode (default)."""
        from pydantic_ai.toolsets import FunctionToolset

        cap = VikingCapability(mode="retrieve")
        cap._client = AsyncMock()
        toolset = cap.get_toolset()
        assert toolset is not None
        assert isinstance(toolset, FunctionToolset)
        tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
        assert len(tool_names) == 7

    def test_get_toolset_write(self) -> None:
        """get_toolset() returns a FunctionToolset with 4 tools for write mode (default)."""
        from pydantic_ai.toolsets import FunctionToolset

        cap = VikingCapability(mode="write")
        cap._client = AsyncMock()
        toolset = cap.get_toolset()
        assert toolset is not None
        assert isinstance(toolset, FunctionToolset)
        tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
        assert len(tool_names) == 4

    def test_get_toolset_graph(self) -> None:
        """get_toolset() returns a FunctionToolset with 1 tool for graph mode (default)."""
        from pydantic_ai.toolsets import FunctionToolset

        cap = VikingCapability(mode="graph")
        cap._client = AsyncMock()
        toolset = cap.get_toolset()
        assert toolset is not None
        assert isinstance(toolset, FunctionToolset)
        tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
        assert len(tool_names) == 1

    def test_get_toolset_all(self) -> None:
        """get_toolset() returns a FunctionToolset with 12 tools for all mode (default)."""
        from pydantic_ai.toolsets import FunctionToolset

        cap = VikingCapability(mode="all")
        cap._client = AsyncMock()
        toolset = cap.get_toolset()
        assert toolset is not None
        assert isinstance(toolset, FunctionToolset)
        tool_names = list(toolset.tools.keys())  # type: ignore[attr-defined]
        assert len(tool_names) == 12

    def test_get_toolset_id_is_viking(self) -> None:
        """get_toolset() returns a FunctionToolset with id='viking'."""
        cap = VikingCapability(mode="all")
        cap._client = AsyncMock()
        toolset = cap.get_toolset()
        assert toolset is not None
        assert toolset.id == "viking"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 8.14 — Test get_instructions()
# ---------------------------------------------------------------------------


class TestGetInstructions:
    """Tests for get_instructions() method."""

    def test_returns_non_empty_string(self) -> None:
        """get_instructions() returns a non-empty string."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_contains_two_step_retrieval(self) -> None:
        """Instructions contain the two-step retrieval pattern section."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Two-Step Retrieval" in instructions
        assert "Search" in instructions
        assert "Read" in instructions

    def test_contains_tool_selection_priority(self) -> None:
        """Instructions contain the tool selection section."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Tool Selection" in instructions

    def test_contains_three_tier_model(self) -> None:
        """Instructions contain the three-tier content model section."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Three-Tier Content Model" in instructions

    def test_contains_writing_strategy(self) -> None:
        """Instructions contain the writing strategy section."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Writing Strategy" in instructions

    def test_contains_uri_conventions(self) -> None:
        """Instructions contain URI conventions section."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "URI Path Rules" in instructions

    def test_contains_memory_tools(self) -> None:
        """Instructions contain memory tools section."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Memory Tools" in instructions

    def test_instructions_consistent_across_modes(self) -> None:
        """Instructions are the same regardless of mode."""
        cap_all = VikingCapability(mode="all")
        cap_retrieve = VikingCapability(mode="retrieve")
        assert cap_all.get_instructions() == cap_retrieve.get_instructions()

    def test_instructions_no_prefix_block_when_unrestricted(self) -> None:
        """get_instructions() omits the allowed-prefix block when unrestricted."""
        cap = VikingCapability(mode="all")
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Allowed URI Prefixes" not in instructions

    def test_instructions_include_prefix_block_when_restricted(self) -> None:
        """get_instructions() lists the allowed prefixes.

        So the model can pass a target_uri and skip discovery probing.
        """
        cap = VikingCapability(
            mode="all",
            allowed_uri_prefixes=[
                "viking://resources/wiki/",
                "viking://resources/raw/",
            ],
        )
        instructions = cap.get_instructions()
        assert instructions is not None
        assert "Allowed URI Prefixes" in instructions
        assert "viking://resources/wiki/" in instructions
        assert "viking://resources/raw/" in instructions

    def test_on_change_returns_none(self) -> None:
        """on_change() returns None."""
        cap = VikingCapability(mode="all")
        assert cap.on_change() is None

    def test_has_wrap_node_run_false(self) -> None:
        """has_wrap_node_run returns False."""
        cap = VikingCapability(mode="all")
        assert cap.has_wrap_node_run is False


# ---------------------------------------------------------------------------
# Utils tests (supplementary)
# ---------------------------------------------------------------------------


class TestUtils:
    """Tests for utility functions in utils.py."""

    def test_format_search_results_dict_with_hits(self) -> None:
        results = {"hits": [{"uri": "viking://doc.md", "score": 0.9, "content": "hello"}]}
        formatted = format_search_results(results)
        assert "viking://doc.md" in formatted
        assert "90%" in formatted
        assert "Found 1 item(s):" in formatted

    def test_format_search_results_dict_with_results(self) -> None:
        results = {"results": [{"uri": "viking://doc.md"}]}
        formatted = format_search_results(results)
        assert "viking://doc.md" in formatted

    def test_format_search_results_list(self) -> None:
        results = [{"uri": "viking://doc.md", "content": "data"}]
        formatted = format_search_results(results)
        assert "viking://doc.md" in formatted

    def test_format_search_results_empty(self) -> None:
        assert format_search_results([]) == "No matching context found."
        assert format_search_results({}) == "No matching context found."

    def test_format_ls_entries_with_markers(self) -> None:
        entries = [
            {"name": "folder1", "type": "directory"},
            {"name": "file1.md", "type": "file"},
        ]
        formatted = format_ls_entries(entries)
        assert "[dir] folder1" in formatted
        assert "[file] file1.md" in formatted

    def test_format_ls_entries_empty(self) -> None:
        assert format_ls_entries([]) == "(empty)"

    def test_format_ls_entries_string(self) -> None:
        formatted = format_ls_entries(["doc.md"])
        assert "[file] doc.md" in formatted

    def test_add_line_numbers(self) -> None:
        result = add_line_numbers("a\nb\nc", start_line=1)
        lines = result.split("\n")
        assert "1" in lines[0]
        assert "a" in lines[0]
        assert "2" in lines[1]
        assert "b" in lines[1]
        assert "3" in lines[2]
        assert "c" in lines[2]

    def test_add_line_numbers_start_line(self) -> None:
        result = add_line_numbers("a\nb", start_line=10)
        lines = result.split("\n")
        assert "10" in lines[0]
        assert "11" in lines[1]

    def test_add_line_numbers_empty(self) -> None:
        assert add_line_numbers("") == ""

    def test_is_viking_uri_true(self) -> None:
        assert is_viking_uri("viking://user/alice/doc.md") is True

    def test_is_viking_uri_false(self) -> None:
        assert is_viking_uri("https://example.com") is False
        assert is_viking_uri("file:///local/path") is False

    def test_truncate_text_no_truncation(self) -> None:
        assert truncate_text("short", 100) == "short"

    def test_truncate_text_with_truncation(self) -> None:
        text = "x" * 200
        result = truncate_text(text, 100)
        assert len(result) < 200
        assert "truncated" in result

    def test_truncate_text_exact_length(self) -> None:
        text = "x" * 50
        assert truncate_text(text, 50) == text


# ---------------------------------------------------------------------------
# Phase 5: ResourceAccess Protocol Tests
# ---------------------------------------------------------------------------


class TestResourceAccessProtocol:
    """Tests for ResourceAccess Protocol implementation (Phase 5)."""

    def test_isinstance_resource_access(self, viking_cap: VikingCapability) -> None:
        """VikingCapability should be recognized as ResourceAccess."""
        from wolfharness.capabilities.resource_protocols import ResourceAccess

        assert isinstance(viking_cap, ResourceAccess)

    def test_resolve_resources_uri_default(self) -> None:
        cap = VikingCapability()
        assert cap._resolve_resources_uri() == "viking://resources/"

    def test_resolve_resources_uri_override(self) -> None:
        cap = VikingCapability(resources_uri="viking://resources/plm/templates/")
        assert cap._resolve_resources_uri() == "viking://resources/plm/templates/"

    async def test_list_resources_success(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        # Mock ls returns different results based on URI.
        # list_resources() now lists from both viking://resources/ and
        # viking://user/default/sessions/ in parallel.
        async def mock_ls(uri, **kwargs):
            if "sessions" in uri:
                return []  # No sessions in test
            if uri == "viking://resources/":
                return [
                    {"name": "doc1.md", "uri": "viking://resources/doc1.md", "isDir": False},
                    {"name": "doc2.txt", "uri": "viking://resources/doc2.txt", "isDir": False},
                    {"name": "subdir", "uri": "viking://resources/subdir", "isDir": True},
                ]
            if "subdir" in uri:
                return [
                    {"name": "doc3.md", "uri": "viking://resources/subdir/doc3.md", "isDir": False},
                ]
            return []

        mock_client.ls = AsyncMock(side_effect=mock_ls)
        result = await viking_cap.list_resources()
        assert len(result) == 3  # only text files, no directories
        assert result[0].name == "doc1.md"
        assert result[0].uri == "viking://resources/doc1.md"
        assert result[0].mime_type == "text/markdown"
        # Descriptions use relative path (Viking abstract() returns parent dir
        # abstract, not file-level, so we don't enrich files with abstracts)
        assert result[0].description == "doc1.md"
        assert result[1].name == "doc2.txt"
        assert result[1].mime_type == "text/plain"
        assert result[1].description == "doc2.txt"
        assert result[2].name == "doc3.md"
        assert result[2].description == "subdir/doc3.md"

    async def test_list_resources_empty(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(return_value=[])
        result = await viking_cap.list_resources()
        assert result == []

    async def test_list_resources_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(side_effect=RuntimeError("network error"))
        result = await viking_cap.list_resources()
        assert result == []

    async def test_list_resources_not_list(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(return_value={"error": "bad"})
        result = await viking_cap.list_resources()
        assert result == []

    async def test_read_resource_success(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        # Default resource_read_level is "overview", so overview() is called
        mock_client.overview = AsyncMock(return_value="resource content here")
        result = await viking_cap.read_resource("viking://resources/doc.md")
        assert result is not None
        assert len(result) == 1
        assert result[0].text == "resource content here"
        assert result[0].uri == "viking://resources/doc.md"
        assert result[0].mime_type == "text/markdown"
        mock_client.overview.assert_called_once_with("viking://resources/doc.md")

    async def test_read_resource_non_markdown(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.overview = AsyncMock(return_value="plain text")
        result = await viking_cap.read_resource("viking://resources/doc.txt")
        assert result is not None
        assert result[0].mime_type is None

    async def test_read_resource_empty(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.overview = AsyncMock(return_value="")
        result = await viking_cap.read_resource("viking://resources/missing.md")
        assert result is None

    async def test_read_resource_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        # Both overview and read fail
        mock_client.overview = AsyncMock(side_effect=RuntimeError("not found"))
        mock_client.read = AsyncMock(side_effect=RuntimeError("not found"))
        result = await viking_cap.read_resource("viking://resources/missing.md")
        assert result is None

    async def test_resource_exists_true(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "doc.md", "uri": "viking://resources/doc.md"},
                {"name": "other.md", "uri": "viking://resources/other.md"},
            ]
        )
        result = await viking_cap.resource_exists("viking://resources/doc.md")
        assert result is True

    async def test_resource_exists_false(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(
            return_value=[{"name": "other.md", "uri": "viking://resources/other.md"}]
        )
        result = await viking_cap.resource_exists("viking://resources/doc.md")
        assert result is False

    async def test_resource_exists_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(side_effect=RuntimeError("network error"))
        result = await viking_cap.resource_exists("viking://resources/doc.md")
        assert result is False

    async def test_resource_exists_not_list(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        mock_client.ls = AsyncMock(return_value="not a list")
        result = await viking_cap.resource_exists("viking://resources/doc.md")
        assert result is False


# ---------------------------------------------------------------------------
# Phase 6: Multimodal Bridge Tests
# ---------------------------------------------------------------------------


class TestMultimodalBridge:
    """Tests for multimodal bridge implementation (Phase 6)."""

    def test_supports_modality_no_caps(self) -> None:
        cap = VikingCapability()
        assert cap._supports_modality("image/png") is False
        assert cap._supports_modality("audio/mpeg") is False

    def test_supports_modality_image(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(model_capabilities=ModelCapabilities(image_input=True))
        assert cap._supports_modality("image/png") is True
        assert cap._supports_modality("image/jpeg") is True

    def test_supports_modality_image_false(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(model_capabilities=ModelCapabilities(image_input=False))
        assert cap._supports_modality("image/png") is False

    def test_supports_modality_audio(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(model_capabilities=ModelCapabilities(audio_input=True))
        assert cap._supports_modality("audio/mpeg") is True

    def test_supports_modality_video(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(model_capabilities=ModelCapabilities(video_input=True))
        assert cap._supports_modality("video/mp4") is True

    def test_supports_modality_document(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(model_capabilities=ModelCapabilities(document_input=True))
        assert cap._supports_modality("application/pdf") is True

    def test_supports_modality_unknown(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(model_capabilities=ModelCapabilities(image_input=True))
        assert cap._supports_modality("application/zip") is False

    def test_guess_extension_known(self) -> None:
        from wolfharness.capabilities.viking import _guess_extension

        assert _guess_extension("image/png") == "png"
        assert _guess_extension("image/jpeg") == "jpg"
        assert _guess_extension("audio/mpeg") == "mp3"
        assert _guess_extension("video/mp4") == "mp4"
        assert _guess_extension("application/pdf") == "pdf"

    def test_guess_extension_unknown(self) -> None:
        from wolfharness.capabilities.viking import _guess_extension

        assert _guess_extension("application/zip") == "bin"

    async def test_before_model_request_disabled(self, viking_cap: VikingCapability) -> None:
        """Should return request_context unchanged when bridge is disabled."""
        rc = _make_request_context([])
        result = await viking_cap.before_model_request(MagicMock(), rc)
        assert result is rc

    async def test_before_model_request_no_client(self) -> None:
        """Should return request_context unchanged when client is None."""
        cap = VikingCapability(multimodal_bridge=True)
        cap._client = None
        rc = _make_request_context([])
        result = await cap.before_model_request(MagicMock(), rc)
        assert result is rc

    async def test_before_model_request_no_binary(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Should return request_context unchanged when no binary content."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(multimodal_bridge=True)
        cap._client = mock_client

        msg = ModelRequest(parts=[UserPromptPart(content="hello world")])
        rc = _make_request_context([msg])
        result = await cap.before_model_request(MagicMock(), rc)
        assert result is rc  # No modification

    async def test_before_model_request_text_only_model(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Text-only model: binary should be replaced with text reference."""
        from pydantic_ai.messages import (
            BinaryContent,
            ModelRequest,
            TextPart,
            UserPromptPart,
        )

        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(
            multimodal_bridge=True,
            model_capabilities=ModelCapabilities(image_input=False),
        )
        cap._client = mock_client
        mock_client.write = AsyncMock(return_value={"status": "ok"})

        binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
        msg = ModelRequest(parts=[UserPromptPart(content=["look at this", binary])])
        rc = _make_request_context([msg])
        result = await cap.before_model_request(MagicMock(), rc)

        assert result is not rc
        mock_client.write.assert_called_once()
        new_msg = result.messages[0]
        content = new_msg.parts[0].content
        text_parts = [c for c in content if isinstance(c, TextPart)]
        binary_parts = [c for c in content if isinstance(c, BinaryContent)]
        assert len(text_parts) == 1
        assert "viking://" in text_parts[0].content
        assert len(binary_parts) == 0

    async def test_before_model_request_multimodal_with_url(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Multimodal model + public_download_base_url: replace with HTTP URL."""
        from pydantic_ai.messages import (
            BinaryContent,
            ModelRequest,
            TextPart,
            UserPromptPart,
        )

        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(
            multimodal_bridge=True,
            public_download_base_url="https://download.example.com",
            model_capabilities=ModelCapabilities(image_input=True),
        )
        cap._client = mock_client
        mock_client.write = AsyncMock(return_value={"status": "ok"})

        binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
        msg = ModelRequest(parts=[UserPromptPart(content=["look", binary])])
        rc = _make_request_context([msg])
        result = await cap.before_model_request(MagicMock(), rc)

        assert result is not rc
        new_msg = result.messages[0]
        content = new_msg.parts[0].content
        text_parts = [c for c in content if isinstance(c, TextPart)]
        assert len(text_parts) == 1
        assert text_parts[0].content.startswith("https://download.example.com?uri=")

    async def test_before_model_request_multimodal_no_url(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Multimodal model + no URL: keep original binary (persisted)."""
        from pydantic_ai.messages import (
            BinaryContent,
            ModelRequest,
            UserPromptPart,
        )

        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(
            multimodal_bridge=True,
            model_capabilities=ModelCapabilities(image_input=True),
        )
        cap._client = mock_client
        mock_client.write = AsyncMock(return_value={"status": "ok"})

        binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
        msg = ModelRequest(parts=[UserPromptPart(content=["look", binary])])
        rc = _make_request_context([msg])
        result = await cap.before_model_request(MagicMock(), rc)
        assert result is rc  # No modification — binary kept as-is

    async def test_before_model_request_upload_failure(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """Upload failure: keep original binary content."""
        from pydantic_ai.messages import (
            BinaryContent,
            ModelRequest,
            UserPromptPart,
        )

        from wolfharness_config.model_capabilities import ModelCapabilities

        cap = VikingCapability(
            multimodal_bridge=True,
            model_capabilities=ModelCapabilities(image_input=False),
        )
        cap._client = mock_client
        mock_client.write = AsyncMock(side_effect=RuntimeError("upload failed"))

        binary = BinaryContent(data=b"\x89PNG", media_type="image/png")
        msg = ModelRequest(parts=[UserPromptPart(content=["look", binary])])
        rc = _make_request_context([msg])
        result = await cap.before_model_request(MagicMock(), rc)
        assert result is rc  # Upload failed — keep original

    async def test_upload_binary_success(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        from pydantic_ai.messages import BinaryContent

        mock_client.write = AsyncMock(return_value={"status": "ok"})
        binary = BinaryContent(data=b"\x89PNG test", media_type="image/png")
        uri = await viking_cap._upload_binary(binary)
        assert uri is not None
        assert uri.startswith("viking://user/default/memories/uploads/")
        assert uri.endswith(".md")  # Viking only allows .md files
        mock_client.write.assert_called_once()
        call_kwargs = mock_client.write.call_args
        assert call_kwargs.kwargs["mode"] == "create"

    async def test_upload_binary_custom_uploads_uri(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        from pydantic_ai.messages import BinaryContent

        cap = VikingCapability(uploads_uri="viking://custom/uploads/")
        cap._client = mock_client
        mock_client.write = AsyncMock(return_value={"status": "ok"})
        binary = BinaryContent(data=b"data", media_type="image/jpeg")
        uri = await cap._upload_binary(binary)
        assert uri is not None
        assert uri.startswith("viking://custom/uploads/")
        assert uri.endswith(".md")  # Viking only allows .md files

    async def test_upload_binary_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        from pydantic_ai.messages import BinaryContent

        mock_client.write = AsyncMock(side_effect=RuntimeError("write failed"))
        binary = BinaryContent(data=b"data", media_type="image/png")
        uri = await viking_cap._upload_binary(binary)
        assert uri is None

    async def test_upload_binary_no_client(self) -> None:
        from pydantic_ai.messages import BinaryContent

        cap = VikingCapability()
        binary = BinaryContent(data=b"data", media_type="image/png")
        uri = await cap._upload_binary(binary)
        assert uri is None

    async def test_for_run_preserves_model_capabilities(self) -> None:
        from wolfharness_config.model_capabilities import ModelCapabilities

        caps = ModelCapabilities(image_input=True)
        cap = VikingCapability(model_capabilities=caps, multimodal_bridge=True)
        copy = await cap.for_run(MagicMock())
        assert copy.model_capabilities is caps
        assert copy.multimodal_bridge is True


# ---------------------------------------------------------------------------
# Tiered Loading Tests (L0/L1/L2)
# ---------------------------------------------------------------------------


class TestTieredLoadingVikingRead:
    """Tests for viking_read level parameter (L0/L1/L2)."""

    @pytest.mark.asyncio
    async def test_viking_read_level_abstract(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read with level='abstract' calls client.abstract()."""
        mock_client.abstract = AsyncMock(return_value="Short summary")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md", level="abstract")

        mock_client.abstract.assert_called_once_with("viking://doc.md")
        mock_client.read.assert_not_called()
        assert "Short summary" in result.return_value
        # Abstracts don't get line numbers
        assert "\u2502" not in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_level_overview(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read with level='overview' calls client.overview()."""
        mock_client.overview = AsyncMock(return_value="Overview content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md", level="overview")

        mock_client.overview.assert_called_once_with("viking://doc.md")
        mock_client.read.assert_not_called()
        assert "Overview content" in result.return_value
        # Overviews don't get line numbers
        assert "\u2502" not in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_level_read_default(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read with default level='read' calls client.read()."""
        mock_client.read = AsyncMock(return_value="full content")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md")

        mock_client.read.assert_called_once()
        mock_client.abstract.assert_not_called()
        mock_client.overview.assert_not_called()
        # Read level gets line numbers
        assert "\u2502" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_abstract_multi_uri(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read with level='abstract' and multiple URIs."""
        mock_client.abstract = AsyncMock(return_value="summary")
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris=["viking://a.md", "viking://b.md"], level="abstract")

        assert mock_client.abstract.call_count == 2
        assert "=== viking://a.md ===" in result.return_value
        assert "=== viking://b.md ===" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_read_abstract_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_read with level='abstract' handles errors."""
        mock_client.abstract = AsyncMock(side_effect=RuntimeError("not available"))
        tools = build_tools(viking_cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://doc.md", level="abstract")

        assert "viking_read error (RuntimeError): not available" in result.return_value


class TestTieredLoadingReadResource:
    """Tests for read_resource with resource_read_level config."""

    @pytest.mark.asyncio
    async def test_read_resource_level_overview(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """read_resource with default resource_read_level='overview' calls overview()."""
        mock_client.overview = AsyncMock(return_value="overview text")
        result = await viking_cap.read_resource("viking://resources/doc.md")

        assert result is not None
        assert result[0].text == "overview text"
        mock_client.overview.assert_called_once_with("viking://resources/doc.md")
        mock_client.read.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_resource_level_abstract(self, mock_client: AsyncMock) -> None:
        """read_resource with resource_read_level='abstract' calls abstract()."""
        cap = VikingCapability(mode="all", resource_read_level="abstract")
        cap._client = mock_client
        mock_client.abstract = AsyncMock(return_value="abstract text")

        result = await cap.read_resource("viking://resources/doc.md")

        assert result is not None
        assert result[0].text == "abstract text"
        mock_client.abstract.assert_called_once_with("viking://resources/doc.md")
        mock_client.read.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_resource_level_read(self, mock_client: AsyncMock) -> None:
        """read_resource with resource_read_level='read' calls read()."""
        cap = VikingCapability(mode="all", resource_read_level="read")
        cap._client = mock_client
        mock_client.read = AsyncMock(return_value="full content")

        result = await cap.read_resource("viking://resources/doc.md")

        assert result is not None
        assert result[0].text == "full content"
        mock_client.read.assert_called_once_with("viking://resources/doc.md")

    @pytest.mark.asyncio
    async def test_read_resource_overview_fallback_to_read(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """read_resource falls back to read() when overview() fails."""
        mock_client.overview = AsyncMock(side_effect=RuntimeError("not available"))
        mock_client.read = AsyncMock(return_value="fallback content")

        result = await viking_cap.read_resource("viking://resources/doc.md")

        assert result is not None
        assert result[0].text == "fallback content"
        mock_client.overview.assert_called_once()
        mock_client.read.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_resource_abstract_fallback_to_read(self, mock_client: AsyncMock) -> None:
        """read_resource with abstract level falls back to read() when abstract fails."""
        cap = VikingCapability(mode="all", resource_read_level="abstract")
        cap._client = mock_client
        mock_client.abstract = AsyncMock(side_effect=RuntimeError("not available"))
        mock_client.read = AsyncMock(return_value="fallback content")

        result = await cap.read_resource("viking://resources/doc.md")

        assert result is not None
        assert result[0].text == "fallback content"
        mock_client.abstract.assert_called_once()
        mock_client.read.assert_called_once()


class TestTieredLoadingListResources:
    """Tests for list_resources with L0 abstract enrichment."""

    @pytest.mark.asyncio
    async def test_list_resources_with_abstracts(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_resources uses relative path as description (no abstract enrichment).

        Viking's abstract() returns the parent directory's abstract, not the
        file's own, so we don't enrich file descriptions with abstracts.
        """
        mock_client.ls = AsyncMock(
            side_effect=[
                [
                    {"name": "doc1.md", "uri": "viking://resources/doc1.md", "isDir": False},
                ],
            ]
        )
        result = await viking_cap.list_resources()

        assert len(result) == 1
        assert result[0].description == "doc1.md"
        # abstract should not be called for files
        mock_client.abstract.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_resources_abstract_failure_keeps_path(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_resources uses path-based description (no abstract calls)."""
        mock_client.ls = AsyncMock(
            side_effect=[
                [
                    {"name": "doc1.md", "uri": "viking://resources/doc1.md", "isDir": False},
                ],
            ]
        )
        result = await viking_cap.list_resources()

        assert len(result) == 1
        assert result[0].description == "doc1.md"

    @pytest.mark.asyncio
    async def test_list_resources_empty_abstract_keeps_path(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """list_resources uses path-based description (no abstract calls)."""
        mock_client.ls = AsyncMock(
            side_effect=[
                [
                    {"name": "doc1.md", "uri": "viking://resources/doc1.md", "isDir": False},
                ],
            ]
        )
        result = await viking_cap.list_resources()

        assert len(result) == 1
        assert result[0].description == "doc1.md"


class TestTieredLoadingVikingLs:
    """Tests for viking_ls with show_abstract parameter."""

    @pytest.mark.asyncio
    async def test_viking_ls_show_abstract(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_ls with show_abstract=True fetches abstracts for directories."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "chapters", "type": "directory", "uri": "viking://resources/chapters/"},
                {"name": "file1.md", "type": "file", "uri": "viking://resources/file1.md"},
            ]
        )
        mock_client.abstract = AsyncMock(return_value="Knowledge base about machines")
        tools = build_tools(viking_cap)
        ls_tool = _get_tool(tools, "viking_ls")

        ctx = _make_ctx()
        result = await ls_tool(ctx, uri="viking://resources/", show_abstract=True)

        mock_client.abstract.assert_called_once_with("viking://resources/chapters/")
        assert "[dir] chapters" in result.return_value
        assert "Knowledge base about machines" in result.return_value
        assert "[file] file1.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_ls_show_abstract_no_dirs(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_ls with show_abstract=True and no directories doesn't call abstract."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "file1.md", "type": "file", "uri": "viking://resources/file1.md"},
            ]
        )
        mock_client.abstract = AsyncMock(return_value="should not be called")
        tools = build_tools(viking_cap)
        ls_tool = _get_tool(tools, "viking_ls")

        ctx = _make_ctx()
        result = await ls_tool(ctx, uri="viking://resources/", show_abstract=True)

        mock_client.abstract.assert_not_called()
        assert "[file] file1.md" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_ls_show_abstract_error(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_ls with show_abstract=True handles abstract errors gracefully."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "chapters", "type": "directory", "uri": "viking://resources/chapters/"},
            ]
        )
        mock_client.abstract = AsyncMock(side_effect=RuntimeError("not available"))
        tools = build_tools(viking_cap)
        ls_tool = _get_tool(tools, "viking_ls")

        ctx = _make_ctx()
        result = await ls_tool(ctx, uri="viking://resources/", show_abstract=True)

        # Still shows the directory without abstract
        assert "[dir] chapters" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_ls_default_no_show_abstract(
        self, viking_cap: VikingCapability, mock_client: AsyncMock
    ) -> None:
        """viking_ls without show_abstract doesn't call abstract."""
        mock_client.ls = AsyncMock(
            return_value=[
                {"name": "dir1", "type": "directory", "uri": "viking://resources/dir1/"},
                {"name": "file1.md", "type": "file", "uri": "viking://resources/file1.md"},
            ]
        )
        mock_client.abstract = AsyncMock(return_value="should not be called")
        tools = build_tools(viking_cap)
        ls_tool = _get_tool(tools, "viking_ls")

        ctx = _make_ctx()
        result = await ls_tool(ctx, uri="viking://resources/")

        mock_client.abstract.assert_not_called()
        assert "[dir] dir1" in result.return_value
        assert "[file] file1.md" in result.return_value


class TestTieredLoadingForRun:
    """Tests that for_run() preserves resource_read_level."""

    @pytest.mark.asyncio
    async def test_for_run_preserves_resource_read_level(self, mock_client: AsyncMock) -> None:
        """for_run() preserves resource_read_level='abstract'."""
        cap = VikingCapability(mode="all", resource_read_level="abstract")
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.resource_read_level == "abstract"

    @pytest.mark.asyncio
    async def test_for_run_preserves_resource_read_level_overview(
        self, mock_client: AsyncMock
    ) -> None:
        """for_run() preserves resource_read_level='overview' (default)."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.resource_read_level == "overview"

    @pytest.mark.asyncio
    async def test_for_run_preserves_resource_read_level_read(self, mock_client: AsyncMock) -> None:
        """for_run() preserves resource_read_level='read'."""
        cap = VikingCapability(mode="all", resource_read_level="read")
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.resource_read_level == "read"


class TestTieredLoadingFormatSearchResults:
    """Tests for format_search_results with Viking grouped format and abstracts."""

    def test_format_search_results_viking_memories(self) -> None:
        """format_search_results handles Viking's memories/resources/skills format."""
        results = {
            "memories": [{"uri": "viking://mem.md", "score": 0.9, "abstract": "test abstract"}]
        }
        formatted = format_search_results(results)
        assert "viking://mem.md" in formatted
        assert "test abstract" in formatted
        assert "90%" in formatted
        assert "[memory" in formatted

    def test_format_search_results_viking_resources(self) -> None:
        """format_search_results handles resources key."""
        results = {"resources": [{"uri": "viking://res.md", "abstract": "resource abstract"}]}
        formatted = format_search_results(results)
        assert "viking://res.md" in formatted
        assert "resource abstract" in formatted

    def test_format_search_results_viking_skills(self) -> None:
        """format_search_results handles skills key."""
        results = {"skills": [{"uri": "viking://skill.md", "abstract": "skill abstract"}]}
        formatted = format_search_results(results)
        assert "viking://skill.md" in formatted
        assert "skill abstract" in formatted

    def test_format_search_results_viking_combined(self) -> None:
        """format_search_results combines memories + resources + skills."""
        results = {
            "memories": [{"uri": "viking://mem.md", "abstract": "mem abstract"}],
            "resources": [{"uri": "viking://res.md", "abstract": "res abstract"}],
            "skills": [{"uri": "viking://skill.md", "abstract": "skill abstract"}],
        }
        formatted = format_search_results(results)
        assert "viking://mem.md" in formatted
        assert "viking://res.md" in formatted
        assert "viking://skill.md" in formatted

    def test_format_search_results_with_abstract_and_content(self) -> None:
        """format_search_results shows abstract when present."""
        results = {
            "hits": [
                {
                    "uri": "viking://doc.md",
                    "score": 0.95,
                    "abstract": "L0 summary",
                    "content": "Full content snippet",
                }
            ]
        }
        formatted = format_search_results(results)
        assert "viking://doc.md" in formatted
        assert "L0 summary" in formatted
        assert "95%" in formatted

    def test_format_search_results_viking_empty_groups(self) -> None:
        """format_search_results returns 'No matching context' when all groups are empty."""
        results = {"memories": [], "resources": [], "skills": []}
        formatted = format_search_results(results)
        assert formatted == "No matching context found."


# ---------------------------------------------------------------------------
# 1.11 — Test identity resolution (Tasks 1.1-1.10)
# ---------------------------------------------------------------------------


class TestIdentityResolution:
    """Tests for VikingIdentity, _try_decode_api_key, and _resolve_identity."""

    # ---- Pure function tests (no mock) ----

    def test_try_decode_new_format_key(self) -> None:
        """_try_decode_api_key decodes a new-format key successfully."""
        import base64

        account = base64.b64encode(b"myaccount").decode("ascii")
        user = base64.b64encode(b"alice").decode("ascii")
        secret = base64.b64encode(b"secretkey").decode("ascii")
        api_key = f"{account}.{user}.{secret}"

        result = _try_decode_api_key(api_key)
        assert result is not None
        assert result == ("myaccount", "alice")

    def test_try_decode_legacy_key_no_dots(self) -> None:
        """_try_decode_api_key returns None for legacy keys without dots."""
        result = _try_decode_api_key("legacy-key-no-dots")
        assert result is None

    def test_try_decode_malformed_key(self) -> None:
        """_try_decode_api_key returns None for malformed base64 parts."""
        result = _try_decode_api_key("!!!not-base64!!!.!!!also-bad!!.sig")
        assert result is None

    def test_try_decode_key_wrong_part_count(self) -> None:
        """_try_decode_api_key returns None for keys with fewer than 3 parts."""
        result = _try_decode_api_key("part1.part2")
        assert result is None

    def test_try_decode_empty_key(self) -> None:
        """_try_decode_api_key returns None for empty string."""
        assert _try_decode_api_key("") is None

    def test_try_decode_empty_decoded_values(self) -> None:
        """_try_decode_api_key returns None when decoded parts are empty."""
        import base64

        empty_b64 = base64.b64encode(b"").decode("ascii")
        user_b64 = base64.b64encode(b"alice").decode("ascii")
        secret_b64 = base64.b64encode(b"sig").decode("ascii")
        result = _try_decode_api_key(f"{empty_b64}.{user_b64}.{secret_b64}")
        assert result is None

    def test_viking_identity_is_frozen(self) -> None:
        """VikingIdentity is a frozen dataclass — assignment raises FrozenInstanceError."""
        identity = VikingIdentity(account_id="acct", user_id="alice", role="user")
        with pytest.raises(dataclasses.FrozenInstanceError):
            identity.user_id = "bob"  # type: ignore[misc]

    # ---- Three-tier resolution tests (using mock_client) ----

    @pytest.mark.asyncio
    async def test_resolve_identity_explicit_config(self, mock_client: AsyncMock) -> None:
        """Tier 1: explicit config fields take precedence."""
        cap = VikingCapability(account="myacct", user="bob")
        cap._client = mock_client
        cap._identity = None

        identity = await cap._resolve_identity()

        assert identity.account_id == "myacct"
        assert identity.user_id == "bob"
        assert identity.role == "user"
        # /health should NOT be called when explicit config is present
        mock_client._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_identity_api_key_decode(self, mock_client: AsyncMock) -> None:
        """Tier 2: API key decode succeeds when config fields are None."""
        import base64

        account_b64 = base64.b64encode(b"keyaccount").decode("ascii")
        user_b64 = base64.b64encode(b"keyuser").decode("ascii")
        secret_b64 = base64.b64encode(b"secret").decode("ascii")
        api_key = f"{account_b64}.{user_b64}.{secret_b64}"

        cap = VikingCapability(api_key=api_key)
        cap._client = mock_client
        cap._identity = None

        identity = await cap._resolve_identity()

        assert identity.account_id == "keyaccount"
        assert identity.user_id == "keyuser"
        assert identity.role == "user"
        # /health should NOT be called when API key decode succeeds
        mock_client._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_identity_health_fallback(self, mock_client: AsyncMock) -> None:
        """Tier 3: /health endpoint is queried when API key decode fails."""
        mock_client._request = AsyncMock(
            return_value={
                "account_id": "health_acct",
                "user_id": "health_user",
                "role": "admin",
            }
        )
        # Legacy API key that can't be decoded (no dots)
        cap = VikingCapability(api_key="legacy-key-no-dots")
        cap._client = mock_client
        cap._identity = None

        identity = await cap._resolve_identity()

        assert identity.account_id == "health_acct"
        assert identity.user_id == "health_user"
        assert identity.role == "admin"
        mock_client._request.assert_called_once_with("GET", "/health")

    @pytest.mark.asyncio
    async def test_resolve_identity_all_fail_fallback(self, mock_client: AsyncMock) -> None:
        """Tier 4: fallback to default when all tiers fail."""
        mock_client._request = AsyncMock(side_effect=RuntimeError("connection refused"))
        cap = VikingCapability()
        cap._client = mock_client
        cap._identity = None

        identity = await cap._resolve_identity()

        assert identity.account_id == "default"
        assert identity.user_id == "default"
        assert identity.role == "user"

    @pytest.mark.asyncio
    async def test_resolve_identity_cached(self, mock_client: AsyncMock) -> None:
        """_resolve_identity returns cached identity on second call."""
        cap = VikingCapability(account="acct", user="alice")
        cap._client = mock_client
        cap._identity = None

        first = await cap._resolve_identity()
        second = await cap._resolve_identity()

        assert first is second
        assert first.user_id == "alice"

    @pytest.mark.asyncio
    async def test_resolve_identity_health_missing_fields(self, mock_client: AsyncMock) -> None:
        """/health response missing account_id falls through to default."""
        mock_client._request = AsyncMock(return_value={"user_id": "partial_user"})
        cap = VikingCapability()
        cap._client = mock_client
        cap._identity = None

        identity = await cap._resolve_identity()

        assert identity.account_id == "default"
        assert identity.user_id == "default"

    # ---- URI methods use resolved identity ----

    @pytest.mark.asyncio
    async def test_resolve_skills_uri_uses_identity(self, mock_client: AsyncMock) -> None:
        """_resolve_skills_uri uses _identity.user_id when set."""
        cap = VikingCapability()
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        uri = cap._resolve_skills_uri()
        assert uri == "viking://user/alice/skills/"

    def test_resolve_skills_uri_explicit_override(self) -> None:
        """_resolve_skills_uri uses explicit skills_uri when set."""
        cap = VikingCapability(skills_uri="viking://resources/shared-skills/")
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        uri = cap._resolve_skills_uri()
        assert uri == "viking://resources/shared-skills/"

    @pytest.mark.asyncio
    async def test_resolve_sessions_uri_uses_identity(self, mock_client: AsyncMock) -> None:
        """_resolve_sessions_uri uses _identity.user_id when set."""
        cap = VikingCapability()
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        uri = cap._resolve_sessions_uri()
        assert uri == "viking://user/alice/sessions/"

    def test_resolve_sessions_uri_explicit_override(self) -> None:
        """_resolve_sessions_uri uses explicit sessions_uri when set."""
        cap = VikingCapability(sessions_uri="viking://resources/shared-sessions/")
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        uri = cap._resolve_sessions_uri()
        assert uri == "viking://resources/shared-sessions/"

    @pytest.mark.asyncio
    async def test_resolve_memories_uri_uses_identity(self, mock_client: AsyncMock) -> None:
        """_resolve_memories_uri uses _identity.user_id when set."""
        cap = VikingCapability()
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        uri = cap._resolve_memories_uri()
        assert uri == "viking://user/alice/memories/"

    def test_resolve_memories_uri_explicit_override(self) -> None:
        """_resolve_memories_uri uses explicit memories_uri when set."""
        cap = VikingCapability(memories_uri="viking://resources/shared-memories/")
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        uri = cap._resolve_memories_uri()
        assert uri == "viking://resources/shared-memories/"

    def test_resolve_uri_fallback_to_user_config(self) -> None:
        """URI methods fall back to self.user when _identity is None."""
        cap = VikingCapability(user="configuser")
        # _identity is None

        assert cap._resolve_skills_uri() == "viking://user/configuser/skills/"
        assert cap._resolve_sessions_uri() == "viking://user/configuser/sessions/"
        assert cap._resolve_memories_uri() == "viking://user/configuser/memories/"

    def test_resolve_uri_fallback_to_default(self) -> None:
        """URI methods fall back to 'default' when both _identity and self.user are None."""
        cap = VikingCapability()
        # _identity is None, user is None

        assert cap._resolve_skills_uri() == "viking://user/default/skills/"
        assert cap._resolve_sessions_uri() == "viking://user/default/sessions/"
        assert cap._resolve_memories_uri() == "viking://user/default/memories/"

    # ---- for_run() bug fix ----

    @pytest.mark.asyncio
    async def test_for_run_passes_sessions_uri(self, mock_client: AsyncMock) -> None:
        """for_run() preserves sessions_uri (existing bug fix)."""
        cap = VikingCapability(
            mode="all",
            sessions_uri="viking://user/alice/sessions/",
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.sessions_uri == "viking://user/alice/sessions/"

    @pytest.mark.asyncio
    async def test_for_run_shares_identity(self, mock_client: AsyncMock) -> None:
        """for_run() shares _identity with the parent."""
        cap = VikingCapability(mode="all")
        cap._client = mock_client
        identity = VikingIdentity(account_id="acct", user_id="alice", role="user")
        cap._identity = identity

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap._identity is identity

    @pytest.mark.asyncio
    async def test_for_run_preserves_all_new_fields(self, mock_client: AsyncMock) -> None:
        """for_run() preserves auto_resolve_identity, memories_uri, actor_peer_id."""
        cap = VikingCapability(
            mode="all",
            auto_resolve_identity=False,
            memories_uri="viking://user/alice/memories/",
            actor_peer_id="diagnosis",
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.auto_resolve_identity is False
        assert copy_cap.memories_uri == "viking://user/alice/memories/"
        assert copy_cap.actor_peer_id == "diagnosis"

    # ---- Config field tests ----

    def test_config_auto_resolve_identity_default(self) -> None:
        """VikingCapabilityConfig has auto_resolve_identity=True by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_resolve_identity is True

    def test_config_memories_uri_default(self) -> None:
        """VikingCapabilityConfig has memories_uri=None by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.memories_uri is None

    def test_config_actor_peer_id_default(self) -> None:
        """VikingCapabilityConfig has actor_peer_id=None by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.actor_peer_id is None

    def test_config_actor_peer_id_set(self) -> None:
        """VikingCapabilityConfig accepts actor_peer_id."""
        cfg = VikingCapabilityConfig(actor_peer_id="diagnosis")
        assert cfg.actor_peer_id == "diagnosis"


# ---------------------------------------------------------------------------
# 2.8 — Auto Semantic Recall Tests (Tasks 2.1-2.7)
# ---------------------------------------------------------------------------


class TestAutoRecall:
    """Tests for auto semantic recall helpers and _handle_auto_recall()."""

    # ---- Pure function tests: _extract_latest_user_prompt ----

    def test_extract_latest_user_prompt_simple(self) -> None:
        """Extracts the text content of the latest UserPromptPart."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="SY55C 液压压力不足")])
        result = _extract_latest_user_prompt([msg])
        assert result == "SY55C 液压压力不足"

    def test_extract_latest_user_prompt_multiple_messages(self) -> None:
        """Returns the latest user prompt when multiple messages exist."""
        from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart

        msg1 = ModelRequest(parts=[UserPromptPart(content="first prompt")])
        msg2 = ModelRequest(parts=[TextPart(content="assistant reply")])
        msg3 = ModelRequest(parts=[UserPromptPart(content="second prompt")])
        result = _extract_latest_user_prompt([msg1, msg2, msg3])
        assert result == "second prompt"

    def test_extract_latest_user_prompt_no_user_prompt(self) -> None:
        """Returns None when no UserPromptPart is found."""
        from pydantic_ai.messages import ModelRequest, TextPart

        msg = ModelRequest(parts=[TextPart(content="no user here")])
        result = _extract_latest_user_prompt([msg])
        assert result is None

    def test_extract_latest_user_prompt_empty_messages(self) -> None:
        """Returns None for empty message list."""
        assert _extract_latest_user_prompt([]) is None

    def test_extract_latest_user_prompt_skips_multimodal(self) -> None:
        """Skips UserPromptPart with list (multimodal) content."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg1 = ModelRequest(parts=[UserPromptPart(content=["image", "text"])])
        msg2 = ModelRequest(parts=[UserPromptPart(content="plain text")])
        result = _extract_latest_user_prompt([msg1, msg2])
        assert result == "plain text"

    def test_extract_latest_user_prompt_skips_whitespace_only(self) -> None:
        """Skips UserPromptPart with whitespace-only content."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="   ")])
        result = _extract_latest_user_prompt([msg])
        assert result is None

    # ---- Pure function tests: _rank_and_dedup ----

    def test_rank_and_dedup_basic_sorting(self) -> None:
        """Hits are sorted by composite score descending."""
        hits = [
            {"uri": "viking://low.md", "score": 0.3, "content": "low score content"},
            {"uri": "viking://high.md", "score": 0.9, "content": "high score content"},
        ]
        result = _rank_and_dedup(hits, query="test", min_score=0.0)
        assert len(result) == 2
        assert result[0]["uri"] == "viking://high.md"
        assert result[1]["uri"] == "viking://low.md"

    def test_rank_and_dedup_category_boost(self) -> None:
        """Memory-type hits get category boost, ranking them higher."""
        hits = [
            {
                "uri": "viking://res.md",
                "score": 0.80,
                "context_type": "resource",
                "content": "resource content",
            },
            {
                "uri": "viking://mem.md",
                "score": 0.80,
                "context_type": "memory",
                "content": "memory content",
            },
        ]
        result = _rank_and_dedup(
            hits, query="q", lexical_boost=0.0, category_boost=0.05, min_score=0.0
        )
        assert result[0]["uri"] == "viking://mem.md"
        assert result[0]["_composite_score"] == pytest.approx(0.85)
        assert result[1]["_composite_score"] == pytest.approx(0.80)

    def test_rank_and_dedup_lexical_boost(self) -> None:
        """Lexical overlap increases the composite score."""
        hits = [
            {"uri": "viking://no_overlap.md", "score": 0.5, "content": "unrelated text"},
            {
                "uri": "viking://overlap.md",
                "score": 0.5,
                "content": "hydraulic pressure diagnosis",
            },
        ]
        result = _rank_and_dedup(hits, query="hydraulic pressure", lexical_boost=0.1, min_score=0.0)
        # The overlap hit should rank higher due to 2 overlapping words
        assert result[0]["uri"] == "viking://overlap.md"
        assert result[0]["_composite_score"] == pytest.approx(0.7)

    def test_rank_and_dedup_dedup_by_content(self) -> None:
        """Duplicate content (first 200 chars) is deduplicated."""
        long_content = "x" * 200 + " different suffix"
        hits = [
            {"uri": "viking://a.md", "score": 0.9, "content": long_content},
            {"uri": "viking://b.md", "score": 0.5, "content": long_content},
        ]
        result = _rank_and_dedup(hits, query="q", min_score=0.0)
        assert len(result) == 1
        assert result[0]["uri"] == "viking://a.md"

    def test_rank_and_dedup_min_score_filter(self) -> None:
        """Hits below min_score are filtered out."""
        hits = [
            {"uri": "viking://low.md", "score": 0.1, "content": "low score content"},
            {"uri": "viking://high.md", "score": 0.9, "content": "high score content"},
        ]
        result = _rank_and_dedup(hits, query="q", min_score=0.5)
        assert len(result) == 1
        assert result[0]["uri"] == "viking://high.md"

    def test_rank_and_dedup_context_type_filter(self) -> None:
        """Hits are filtered by context_types when specified."""
        hits = [
            {"uri": "viking://mem.md", "score": 0.9, "context_type": "memory", "content": "x"},
            {"uri": "viking://res.md", "score": 0.9, "context_type": "resource", "content": "x"},
            {"uri": "viking://skl.md", "score": 0.9, "context_type": "skill", "content": "x"},
        ]
        result = _rank_and_dedup(hits, query="q", context_types=["memory"], min_score=0.0)
        assert len(result) == 1
        assert result[0]["uri"] == "viking://mem.md"

    def test_rank_and_dedup_empty_hits(self) -> None:
        """Empty hits list returns empty result."""
        assert _rank_and_dedup([], query="q") == []

    def test_rank_and_dedup_no_context_type_filter(self) -> None:
        """When context_types is None, all hits are included."""
        hits = [
            {"uri": "viking://a.md", "score": 0.9, "context_type": "memory", "content": "x"},
            {"uri": "viking://b.md", "score": 0.9, "context_type": "skill", "content": "y"},
        ]
        result = _rank_and_dedup(hits, query="q", context_types=None, min_score=0.0)
        assert len(result) == 2

    # ---- Pure function tests: _format_recall_block ----

    def test_format_recall_block_basic(self) -> None:
        """Formats hits as <openviking-recall> XML block."""
        hits = [
            {"uri": "viking://doc.md", "_composite_score": 0.85, "content": "important info"},
        ]
        block = _format_recall_block(hits, max_tokens=2000)
        assert "<openviking-recall>" in block
        assert "</openviking-recall>" in block
        assert 'uri="viking://doc.md"' in block
        assert "important info" in block

    def test_format_recall_block_with_session_context(self) -> None:
        """Includes session context when provided."""
        hits = [{"uri": "viking://doc.md", "_composite_score": 0.9, "content": "data"}]
        session_ctx = {"recent_topic": "hydraulic diagnosis"}
        block = _format_recall_block(hits, session_context=session_ctx, max_tokens=2000)
        assert "<session-context>" in block
        assert "hydraulic diagnosis" in block

    def test_format_recall_block_truncation(self) -> None:
        """Content exceeding max_tokens is truncated."""
        long_content = "x" * 10000
        hits = [{"uri": "viking://big.md", "_composite_score": 0.9, "content": long_content}]
        block = _format_recall_block(hits, max_tokens=100)
        assert "truncated" in block
        assert len(block) < 10000

    def test_format_recall_block_empty(self) -> None:
        """Empty hits and no session context returns empty string."""
        assert _format_recall_block([]) == ""
        assert _format_recall_block([], session_context=None) == ""

    def test_format_recall_block_multiple_hits(self) -> None:
        """Multiple hits each get their own <hit> element."""
        hits = [
            {"uri": "viking://a.md", "_composite_score": 0.9, "content": "content a"},
            {"uri": "viking://b.md", "_composite_score": 0.7, "content": "content b"},
        ]
        block = _format_recall_block(hits, max_tokens=2000)
        assert block.count("<hit ") == 2
        assert "viking://a.md" in block
        assert "viking://b.md" in block

    # ---- Pure function tests: _inject_system_message ----

    def test_inject_system_message_inserts_before_user(self) -> None:
        """System message is inserted before the latest user message."""
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        msg1 = ModelRequest(parts=[UserPromptPart(content="first")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="second")])
        rc = _make_request_context([msg1, msg2])

        result = _inject_system_message(rc, "recall block text")
        assert len(result.messages) == 3
        # The system message should be at index 1 (before msg2)
        sys_msg = result.messages[1]
        assert isinstance(sys_msg, ModelRequest)
        sys_part = sys_msg.parts[0]
        assert isinstance(sys_part, SystemPromptPart)
        assert "recall block text" in sys_part.content

    def test_inject_system_message_no_user_prompt(self) -> None:
        """Returns original context when no user message is found."""
        from pydantic_ai.messages import ModelRequest, TextPart

        msg = ModelRequest(parts=[TextPart(content="no user")])
        rc = _make_request_context([msg])
        result = _inject_system_message(rc, "recall text")
        assert result is rc

    def test_inject_system_message_empty_block(self) -> None:
        """Returns original context when recall block is empty."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="test")])
        rc = _make_request_context([msg])
        result = _inject_system_message(rc, "")
        assert result is rc

    # ---- _normalize_search_results tests ----

    def test_normalize_search_results_dict_with_hits(self) -> None:
        """Normalizes dict with 'hits' key."""
        results = {"hits": [{"uri": "viking://a.md"}]}
        assert _normalize_search_results(results) == [{"uri": "viking://a.md"}]

    def test_normalize_search_results_dict_with_results(self) -> None:
        """Normalizes dict with 'results' key."""
        results = {"results": [{"uri": "viking://b.md"}]}
        assert _normalize_search_results(results) == [{"uri": "viking://b.md"}]

    def test_normalize_search_results_dict_grouped(self) -> None:
        """Normalizes dict with Viking grouped keys (memories/resources/skills)."""
        results = {
            "memories": [{"uri": "viking://mem.md"}],
            "resources": [{"uri": "viking://res.md"}],
            "skills": [{"uri": "viking://skl.md"}],
        }
        normalized = _normalize_search_results(results)
        assert len(normalized) == 3

    def test_normalize_search_results_list(self) -> None:
        """Normalizes list input directly."""
        results = [{"uri": "viking://a.md"}]
        assert _normalize_search_results(results) == results

    def test_normalize_search_results_empty(self) -> None:
        """Returns empty list for empty/None input."""
        assert _normalize_search_results({}) == []
        assert _normalize_search_results([]) == []
        assert _normalize_search_results(None) == []

    # ---- _handle_auto_recall integration tests ----

    @pytest.mark.asyncio
    async def test_handle_auto_recall_disabled(self, mock_client: AsyncMock) -> None:
        """Disabled recall (auto_recall_enabled=False) is a no-op."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(mode="all", auto_recall_enabled=False)
        cap._client = mock_client

        msg = ModelRequest(parts=[UserPromptPart(content="test prompt")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap._handle_auto_recall(ctx, rc)
        assert result is rc
        mock_client.search.assert_not_called()
        mock_client.find.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_auto_recall_search_method(self, mock_client: AsyncMock) -> None:
        """Search method calls client.search() with session_id and memories_uri."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.search = AsyncMock(
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
        mock_client.get_session_context = AsyncMock(return_value={"recent_topic": "hydraulics"})

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_recall_method="search",
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="hydraulic pressure issue")])
        rc = _make_request_context([msg])
        ctx = _make_ctx(session_id="sess-123")

        result = await cap._handle_auto_recall(ctx, rc)

        # Verify search was called with correct params
        mock_client.get_session_context.assert_called_once()
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.args[0] == "hydraulic pressure issue"
        assert call_args.kwargs["session_id"] == "sess-123"
        assert "viking://user/alice/memories/" in call_args.kwargs["target_uri"]

        # Verify recall block was injected
        assert result is not rc
        assert len(result.messages) == 2
        from pydantic_ai.messages import SystemPromptPart

        sys_msg = result.messages[0]
        sys_part = sys_msg.parts[0]
        assert isinstance(sys_part, SystemPromptPart)
        assert "<openviking-recall>" in sys_part.content
        assert "viking://user/alice/memories/doc.md" in sys_part.content

    @pytest.mark.asyncio
    async def test_handle_auto_recall_find_method(self, mock_client: AsyncMock) -> None:
        """Find method calls client.find() without session_id."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.find = AsyncMock(
            return_value={
                "results": [
                    {
                        "uri": "viking://user/alice/memories/doc.md",
                        "score": 0.8,
                        "content": "translation memory",
                        "context_type": "memory",
                    }
                ]
            }
        )

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_recall_method="find",
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="翻译这段文字")])
        rc = _make_request_context([msg])
        ctx = _make_ctx(session_id="sess-456")

        result = await cap._handle_auto_recall(ctx, rc)

        # Verify find was called (not search)
        mock_client.find.assert_called_once()
        mock_client.search.assert_not_called()
        mock_client.get_session_context.assert_not_called()

        call_args = mock_client.find.call_args
        assert call_args.args[0] == "翻译这段文字"
        assert "viking://user/alice/memories/" in call_args.kwargs["target_uri"]

        # Verify recall block injected
        assert result is not rc
        from pydantic_ai.messages import SystemPromptPart

        sys_part = result.messages[0].parts[0]
        assert isinstance(sys_part, SystemPromptPart)
        assert "<openviking-recall>" in sys_part.content

    @pytest.mark.asyncio
    async def test_handle_auto_recall_no_user_prompt(self, mock_client: AsyncMock) -> None:
        """Recall skips when no user prompt is found."""
        from pydantic_ai.messages import ModelRequest, TextPart

        cap = VikingCapability(mode="all", auto_recall_enabled=True)
        cap._client = mock_client

        msg = ModelRequest(parts=[TextPart(content="no user prompt here")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap._handle_auto_recall(ctx, rc)
        assert result is rc
        mock_client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_auto_recall_graceful_failure(self, mock_client: AsyncMock) -> None:
        """Recall fails gracefully — returns original context on error."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.search = AsyncMock(side_effect=RuntimeError("server unreachable"))

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_recall_method="search",
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="test query")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap._handle_auto_recall(ctx, rc)
        # Should return original context unchanged
        assert result is rc

    @pytest.mark.asyncio
    async def test_handle_auto_recall_session_context_failure_fallback(
        self, mock_client: AsyncMock
    ) -> None:
        """When get_session_context fails, search still proceeds."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.get_session_context = AsyncMock(
            side_effect=RuntimeError("session context unavailable")
        )
        mock_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "uri": "viking://user/alice/memories/doc.md",
                        "score": 0.7,
                        "content": "content",
                        "context_type": "memory",
                    }
                ]
            }
        )

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_recall_method="search",
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="query")])
        rc = _make_request_context([msg])
        ctx = _make_ctx(session_id="sess-789")

        result = await cap._handle_auto_recall(ctx, rc)

        # Search should still be called
        mock_client.search.assert_called_once()
        # Recall block should still be injected (without session context)
        assert result is not rc
        from pydantic_ai.messages import SystemPromptPart

        sys_part = result.messages[0].parts[0]
        assert isinstance(sys_part, SystemPromptPart)
        assert "<openviking-recall>" in sys_part.content
        # No <session-context> section
        assert "<session-context>" not in sys_part.content

    @pytest.mark.asyncio
    async def test_handle_auto_recall_empty_results(self, mock_client: AsyncMock) -> None:
        """Empty search results returns original context unchanged."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.search = AsyncMock(return_value={"results": []})

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_recall_method="search",
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="obscure query")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap._handle_auto_recall(ctx, rc)
        # Empty results → no recall block → original context returned
        assert result is rc

    @pytest.mark.asyncio
    async def test_handle_auto_recall_for_run_preserves_fields(
        self, mock_client: AsyncMock
    ) -> None:
        """for_run() preserves auto-recall config fields."""
        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_recall_method="find",
            auto_recall_max_tokens=500,
            auto_recall_limit=5,
            auto_recall_min_score=0.5,
            auto_recall_lexical_boost=0.2,
            auto_recall_category_boost=0.1,
            auto_recall_context_types=["memory"],
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.auto_recall_enabled is True
        assert copy_cap.auto_recall_method == "find"
        assert copy_cap.auto_recall_max_tokens == 500
        assert copy_cap.auto_recall_limit == 5
        assert copy_cap.auto_recall_min_score == 0.5
        assert copy_cap.auto_recall_lexical_boost == 0.2
        assert copy_cap.auto_recall_category_boost == 0.1
        assert copy_cap.auto_recall_context_types == ["memory"]

    # ---- Config field tests ----

    def test_config_auto_recall_enabled_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_enabled=False by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_enabled is False

    def test_config_auto_recall_method_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_method='search' by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_method == "search"

    def test_config_auto_recall_max_tokens_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_max_tokens=2000 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_max_tokens == 2000

    def test_config_auto_recall_limit_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_limit=10 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_limit == 10

    def test_config_auto_recall_min_score_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_min_score=0.3 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_min_score == 0.3

    def test_config_auto_recall_lexical_boost_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_lexical_boost=0.1 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_lexical_boost == 0.1

    def test_config_auto_recall_category_boost_default(self) -> None:
        """VikingCapabilityConfig has auto_recall_category_boost=0.05 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_category_boost == 0.05

    def test_config_auto_recall_context_types_default(self) -> None:
        """VikingCapabilityConfig has default context_types=['memory', 'resource']."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_recall_context_types == ["memory", "resource"]

    def test_config_auto_recall_all_fields_set(self) -> None:
        """All auto-recall config fields can be set at once."""
        cfg = VikingCapabilityConfig(
            auto_recall_enabled=True,
            auto_recall_method="find",
            auto_recall_max_tokens=1000,
            auto_recall_limit=5,
            auto_recall_min_score=0.5,
            auto_recall_lexical_boost=0.2,
            auto_recall_category_boost=0.1,
            auto_recall_context_types=["memory", "resource", "skill"],
        )
        assert cfg.auto_recall_enabled is True
        assert cfg.auto_recall_method == "find"
        assert cfg.auto_recall_max_tokens == 1000
        assert cfg.auto_recall_limit == 5
        assert cfg.auto_recall_min_score == 0.5
        assert cfg.auto_recall_lexical_boost == 0.2
        assert cfg.auto_recall_category_boost == 0.1
        assert cfg.auto_recall_context_types == ["memory", "resource", "skill"]


# ---------------------------------------------------------------------------
# 4.7 — URI Guard tests (Tasks 4.1-4.2)
# ---------------------------------------------------------------------------


class TestURIGuard:
    """Tests for wrap_tool_execute URI guard interception."""

    @pytest.mark.asyncio
    async def test_blocked_read_with_viking_uri(self) -> None:
        """Read tool with viking:// URI is blocked when guard is enabled."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "bash", "grep", "glob"],
        )
        call = MagicMock()
        call.tool_name = "read"
        handler = AsyncMock(return_value="tool result")
        args = {"file_path": "viking://user/alice/doc.md"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert isinstance(result, str)
        assert "viking://" in result
        assert "read" in result
        assert "viking_read" in result or "viking_search" in result
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_non_protected_tool(self) -> None:
        """Non-protected tool (not in list) with viking:// URI is allowed."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "bash"],
        )
        call = MagicMock()
        call.tool_name = "grep"
        handler = AsyncMock(return_value="grep result")
        args = {"path": "viking://user/alice/doc.md", "pattern": "hello"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert result == "grep result"
        handler.assert_called_once_with(args)

    @pytest.mark.asyncio
    async def test_allowed_protected_tool_no_viking_uri(self) -> None:
        """Protected tool (read) without viking:// URI is allowed."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "bash", "grep", "glob"],
        )
        call = MagicMock()
        call.tool_name = "read"
        handler = AsyncMock(return_value="file content")
        args = {"file_path": "/local/path/to/file.txt"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert result == "file content"
        handler.assert_called_once_with(args)

    @pytest.mark.asyncio
    async def test_disabled_guard_always_passes(self) -> None:
        """When uri_guard_enabled=False, handler is always called."""
        cap = VikingCapability(
            uri_guard_enabled=False,
            uri_guard_protected_tools=["read", "bash", "grep", "glob"],
        )
        call = MagicMock()
        call.tool_name = "read"
        handler = AsyncMock(return_value="result")
        args = {"file_path": "viking://user/alice/doc.md"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert result == "result"
        handler.assert_called_once_with(args)

    @pytest.mark.asyncio
    async def test_custom_protected_tools_list(self) -> None:
        """Custom protected tools list excludes bash, allowing it through."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "write_file"],
        )
        call = MagicMock()
        call.tool_name = "bash"
        handler = AsyncMock(return_value="bash result")
        args = {"command": "cat viking://user/alice/doc.md"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert result == "bash result"
        handler.assert_called_once_with(args)

    @pytest.mark.asyncio
    async def test_blocked_bash_with_viking_uri(self) -> None:
        """Bash tool with viking:// URI in args is blocked."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "bash", "grep", "glob"],
        )
        call = MagicMock()
        call.tool_name = "bash"
        handler = AsyncMock(return_value="should not reach")
        args = {"command": "cat viking://user/alice/secret.md"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert isinstance(result, str)
        assert "bash" in result
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_grep_with_viking_uri(self) -> None:
        """Grep tool with viking:// URI in args is blocked."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "bash", "grep", "glob"],
        )
        call = MagicMock()
        call.tool_name = "grep"
        handler = AsyncMock(return_value="should not reach")
        args = {"path": "viking://resources/doc.md", "pattern": "hello"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert isinstance(result, str)
        assert "grep" in result
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_glob_with_viking_uri(self) -> None:
        """Glob tool with viking:// URI in args is blocked."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "bash", "grep", "glob"],
        )
        call = MagicMock()
        call.tool_name = "glob"
        handler = AsyncMock(return_value="should not reach")
        args = {"path": "viking://resources/", "pattern": "**/*.md"}

        result = await cap.wrap_tool_execute(
            MagicMock(), call=call, tool_def=MagicMock(), args=args, handler=handler
        )

        assert isinstance(result, str)
        assert "glob" in result
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_for_run_preserves_uri_guard_config(self, mock_client: AsyncMock) -> None:
        """for_run() preserves uri_guard_enabled and uri_guard_protected_tools."""
        cap = VikingCapability(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "custom_tool"],
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.uri_guard_enabled is True
        assert copy_cap.uri_guard_protected_tools == ["read", "custom_tool"]


# ---------------------------------------------------------------------------
# 9.0 — allowed_uri_prefixes tests
# ---------------------------------------------------------------------------


class TestAllowedUriPrefixes:
    """Tests for the allowed_uri_prefixes access restriction."""

    def test_default_unrestricted(self) -> None:
        """Empty allowed_uri_prefixes means unrestricted."""
        cap = VikingCapability()
        assert cap.allowed_uri_prefixes == []
        assert cap._check_uri_allowed("viking://resources/wiki/Device.md") is None
        assert cap._check_uri_allowed("viking://user/alice/memories/x.md") is None

    def test_matching_prefix_allowed(self) -> None:
        """URI under an allowed prefix passes validation."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        assert cap._check_uri_allowed("viking://resources/wiki/Device/SY215.md") is None

    def test_non_matching_prefix_blocked(self) -> None:
        """URI outside allowed prefixes returns an error message."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        err = cap._check_uri_allowed("viking://resources/raw/engine.md")
        assert err is not None
        assert "outside the allowed prefixes" in err
        assert "viking://resources/raw/engine.md" in err

    def test_multiple_prefixes(self) -> None:
        """Multiple allowed prefixes all pass; others are blocked."""
        cap = VikingCapability(
            allowed_uri_prefixes=["viking://resources/wiki/", "viking://resources/raw/"]
        )
        assert cap._check_uri_allowed("viking://resources/wiki/a.md") is None
        assert cap._check_uri_allowed("viking://resources/raw/b.md") is None
        assert cap._check_uri_allowed("viking://resources/docs/c.md") is not None

    def test_empty_uri_allowed_when_restricted(self) -> None:
        """Empty URI is allowed (nothing to restrict)."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        assert cap._check_uri_allowed("") is None

    def test_tool_name_in_error(self) -> None:
        """Tool name appears in the error message."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        err = cap._check_uri_allowed("viking://resources/raw/x.md", tool_name="viking_read")
        assert err is not None
        assert "viking_read" in err

    def test_non_resources_namespace_always_allowed(self) -> None:
        """Allowlist only applies to the viking://resources/ namespace."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        # Own user namespace (memories/sessions/skills) is not resources.
        assert cap._check_uri_allowed("viking://user/alice/memories/x.md") is None
        assert cap._check_uri_allowed("viking://user/alice/sessions/y.md") is None
        assert cap._check_uri_allowed("viking://user/alice/skills/z.md") is None
        # Other users' namespaces are also not restricted by this allowlist.
        assert cap._check_uri_allowed("viking://user/bob/memories/x.md") is None
        assert cap._check_uri_allowed("viking://skills/foo.md") is None
        assert cap._check_uri_allowed("viking://resources/raw/engine.md") is not None

    def test_prefixed_subtree_matches(self) -> None:
        """Subtree under an allowed prefix passes validation."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        assert cap._check_uri_allowed("viking://resources/wiki/Device/SY215.md") is None
        # A shorter namespace under resources but not in the allowlist is blocked.
        assert cap._check_uri_allowed("viking://resources/fta-eval/x.md") is not None

    def test_allowed_prefix_for(self) -> None:
        """_allowed_prefix_for returns matched prefix or None."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        assert cap._allowed_prefix_for("viking://resources/wiki/a.md") == "viking://resources/wiki/"
        assert cap._allowed_prefix_for("viking://resources/other/") is None

    def test_allowed_prefix_for_unrestricted(self) -> None:
        """_allowed_prefix_for returns the URI itself when unrestricted."""
        cap = VikingCapability()
        assert (
            cap._allowed_prefix_for("viking://resources/wiki/a.md")
            == "viking://resources/wiki/a.md"
        )

    @pytest.mark.asyncio
    async def test_viking_read_blocks_outside_prefix(self, mock_client: AsyncMock) -> None:
        """viking_read rejects URIs outside the allowed prefixes."""
        cap = VikingCapability(mode="retrieve", allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        tools = build_tools(cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://resources/raw/engine.md")

        assert "outside the allowed prefixes" in result.return_value
        mock_client.read.assert_not_called()

    @pytest.mark.asyncio
    async def test_viking_read_allows_inside_prefix(self, mock_client: AsyncMock) -> None:
        """viking_read reads URIs within the allowed prefixes."""
        mock_client.read = AsyncMock(return_value="content")
        cap = VikingCapability(mode="retrieve", allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        tools = build_tools(cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(ctx, uris="viking://resources/wiki/Device/SY215.md")

        assert "1\u2502 content" in result.return_value
        assert mock_client.read.call_args.args[0] == "viking://resources/wiki/Device/SY215.md"

    @pytest.mark.asyncio
    async def test_viking_read_multi_uri_blocks_first_outside(self, mock_client: AsyncMock) -> None:
        """viking_read rejects the batch if any URI is outside the prefixes."""
        cap = VikingCapability(mode="retrieve", allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        tools = build_tools(cap)
        read_tool = _get_tool(tools, "viking_read")

        ctx = _make_ctx()
        result = await read_tool(
            ctx, uris=["viking://resources/wiki/a.md", "viking://resources/raw/b.md"]
        )

        assert "outside the allowed prefixes" in result.return_value
        mock_client.read.assert_not_called()

    @pytest.mark.asyncio
    async def test_viking_write_blocks_outside_prefix(self, mock_client: AsyncMock) -> None:
        """viking_write rejects URIs outside the allowed prefixes."""
        cap = VikingCapability(mode="all", allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        tools = build_tools(cap)
        write_tool = _get_tool(tools, "viking_write")

        ctx = _make_ctx()
        result = await write_tool(ctx, uri="viking://resources/raw/note.md", content="hi")

        assert "outside the allowed prefixes" in result.return_value
        mock_client.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_viking_search_defaults_to_first_prefix(self, mock_client: AsyncMock) -> None:
        """viking_search passes the single allowed prefix as a one-element list.

        A list is the SDK's multi-prefix scoping contract, so a single
        prefix is passed as a one-element list rather than a bare string.
        """
        cap = VikingCapability(mode="retrieve", allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        tools = build_tools(cap)
        search_tool = _get_tool(tools, "viking_search")

        ctx = _make_ctx()
        await search_tool(ctx, query="hydraulic")

        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["target_uri"] == ["viking://resources/wiki/"]

    @pytest.mark.asyncio
    async def test_viking_search_multi_prefix_defaults_to_all(self, mock_client: AsyncMock) -> None:
        """viking_search without target_uri passes ALL allowed prefixes to the SDK.

        The SDK's target_uri accepts a list and the server searches each
        prefix — the old behavior of silently using only the first prefix
        dropped results from the other allowed trees.
        """
        cap = VikingCapability(
            mode="retrieve",
            allowed_uri_prefixes=["viking://resources/wiki/", "viking://resources/raw/"],
        )
        cap._client = mock_client
        tools = build_tools(cap)
        search_tool = _get_tool(tools, "viking_search")

        ctx = _make_ctx()
        await search_tool(ctx, query="hydraulic")

        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["target_uri"] == [
            "viking://resources/wiki/",
            "viking://resources/raw/",
        ]

    @pytest.mark.asyncio
    async def test_viking_find_multi_prefix_defaults_to_all(self, mock_client: AsyncMock) -> None:
        """viking_find without target_uri passes ALL allowed prefixes to the SDK."""
        cap = VikingCapability(
            mode="retrieve",
            allowed_uri_prefixes=["viking://resources/wiki/", "viking://resources/raw/"],
        )
        cap._client = mock_client
        tools = build_tools(cap)
        find_tool = _get_tool(tools, "viking_find")

        ctx = _make_ctx()
        await find_tool(ctx, query="hydraulic")

        kwargs = mock_client.find.call_args.kwargs
        assert kwargs["target_uri"] == [
            "viking://resources/wiki/",
            "viking://resources/raw/",
        ]

    @pytest.mark.asyncio
    async def test_viking_search_blocks_outside_target(self, mock_client: AsyncMock) -> None:
        """viking_search rejects a target_uri outside the allowed prefixes."""
        cap = VikingCapability(mode="retrieve", allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        tools = build_tools(cap)
        search_tool = _get_tool(tools, "viking_search")

        ctx = _make_ctx()
        result = await search_tool(ctx, query="hydraulic", target_uri="viking://resources/raw/")

        assert "outside the allowed prefixes" in result.return_value
        mock_client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_resources_filters_by_prefix(self, mock_client: AsyncMock) -> None:
        """list_resources narrows resources tree, non-resources trees pass through."""
        cap = VikingCapability(
            allowed_uri_prefixes=["viking://resources/wiki/"],
            user="alice",
        )
        cap._client = mock_client
        mock_client.ls = AsyncMock(return_value=[])
        resources = await cap.list_resources()
        assert resources == []
        ls_uris = [call.args[0] for call in mock_client.ls.await_args_list]
        assert set(ls_uris) == {
            "viking://resources/wiki/",
            "viking://user/alice/sessions/",
        }

    @pytest.mark.asyncio
    async def test_list_resources_includes_own_sessions(self, mock_client: AsyncMock) -> None:
        """list_resources includes the non-resources sessions tree as-is."""
        cap = VikingCapability(
            allowed_uri_prefixes=["viking://resources/wiki/"],
            user="alice",
        )
        cap._client = mock_client
        mock_client.ls = AsyncMock(
            return_value=[
                {
                    "uri": "viking://user/alice/sessions/s1.md",
                    "name": "s1.md",
                    "isDir": False,
                }
            ]
        )
        resources = await cap.list_resources()
        assert len(resources) == 1
        assert resources[0].uri == "viking://user/alice/sessions/s1.md"

    @pytest.mark.asyncio
    async def test_list_resources_includes_allowed_tree(self, mock_client: AsyncMock) -> None:
        """list_resources lists the allowed prefix tree when it matches."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        mock_client.ls = AsyncMock(
            return_value=[
                {
                    "uri": "viking://resources/wiki/Device/SY215.md",
                    "name": "SY215.md",
                    "isDir": False,
                }
            ]
        )
        resources = await cap.list_resources()
        assert len(resources) == 1
        assert resources[0].uri == "viking://resources/wiki/Device/SY215.md"

    @pytest.mark.asyncio
    async def test_read_resource_blocks_outside_prefix(self, mock_client: AsyncMock) -> None:
        """read_resource returns None for URIs outside the allowed prefixes."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        result = await cap.read_resource("viking://resources/raw/engine.md")
        assert result is None
        mock_client.read.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_resource_allows_inside_prefix(self, mock_client: AsyncMock) -> None:
        """read_resource returns content for URIs within the allowed prefixes."""
        mock_client.read = AsyncMock(return_value="content")
        cap = VikingCapability(
            allowed_uri_prefixes=["viking://resources/wiki/"],
            resource_read_level="read",
        )
        cap._client = mock_client
        result = await cap.read_resource("viking://resources/wiki/a.md")
        assert result is not None
        assert result[0].text == "content"

    @pytest.mark.asyncio
    async def test_resource_exists_blocks_outside_prefix(self, mock_client: AsyncMock) -> None:
        """resource_exists returns False for URIs outside the allowed prefixes."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        assert await cap.resource_exists("viking://resources/raw/a.md") is False
        mock_client.ls.assert_not_called()

    @pytest.mark.asyncio
    async def test_for_run_preserves_allowed_prefixes(self, mock_client: AsyncMock) -> None:
        """for_run() preserves the allowed_uri_prefixes field."""
        cap = VikingCapability(allowed_uri_prefixes=["viking://resources/wiki/"])
        cap._client = mock_client
        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)
        assert copy_cap.allowed_uri_prefixes == ["viking://resources/wiki/"]

    @pytest.mark.asyncio
    async def test_auto_recall_fires_when_memories_outside_prefixes(
        self, mock_client: AsyncMock
    ) -> None:
        """auto_recall still fires when memories_uri is not in the allowlist.

        The agent's own memory namespace is implicitly allowed — the
        knowledge-base allowlist does not gate memory features.
        """
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.search = AsyncMock(
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
        cap = VikingCapability(
            mode="retrieve",
            auto_recall_enabled=True,
            allowed_uri_prefixes=["viking://resources/wiki/"],
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        rc = _make_request_context([ModelRequest(parts=[UserPromptPart(content="query")])])
        result = await cap._handle_auto_recall(ctx, rc)

        assert result is not rc
        mock_client.search.assert_called_once()
        assert mock_client.search.call_args.kwargs["target_uri"] == "viking://user/alice/memories/"


# ---------------------------------------------------------------------------
# 4.7 — viking_forget gating tests (Tasks 4.3-4.4)
# ---------------------------------------------------------------------------


class TestForgetGating:
    """Tests for viking_forget tool gating behind enable_forget."""

    def test_forget_excluded_by_default(self) -> None:
        """viking_forget is excluded when enable_forget=False (default)."""
        cap = VikingCapability(mode="write")
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_forget" not in names

    def test_forget_included_when_enabled(self) -> None:
        """viking_forget is included when enable_forget=True."""
        cap = VikingCapability(mode="write", enable_forget=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_forget" in names

    def test_forget_excluded_with_memory_enabled(self) -> None:
        """viking_forget is excluded when enable_memory=True but enable_forget=False."""
        cap = VikingCapability(mode="write", enable_memory=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_forget" not in names
        assert "viking_remember" in names

    def test_forget_included_with_all_flags(self) -> None:
        """viking_forget is included when both enable_memory and enable_forget are True."""
        cap = VikingCapability(mode="all", enable_memory=True, enable_forget=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_forget" in names
        assert "viking_remember" in names

    def test_forget_excluded_in_all_mode_default(self) -> None:
        """viking_forget is excluded in all mode by default."""
        cap = VikingCapability(mode="all")
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_forget" not in names

    def test_forget_excluded_in_retrieve_mode(self) -> None:
        """viking_forget is excluded in retrieve mode even with enable_forget=True."""
        cap = VikingCapability(mode="retrieve", enable_forget=True)
        cap._client = AsyncMock()
        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_forget" not in names

    def test_config_enable_forget_default(self) -> None:
        """VikingCapabilityConfig has enable_forget=False by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.enable_forget is False

    def test_config_enable_forget_set(self) -> None:
        """VikingCapabilityConfig accepts enable_forget=True."""
        cfg = VikingCapabilityConfig(enable_forget=True)
        assert cfg.enable_forget is True

    def test_config_uri_guard_enabled_default(self) -> None:
        """VikingCapabilityConfig has uri_guard_enabled=False by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.uri_guard_enabled is False

    def test_config_uri_guard_protected_tools_default(self) -> None:
        """VikingCapabilityConfig has default protected tools list."""
        cfg = VikingCapabilityConfig()
        assert cfg.uri_guard_protected_tools == ["read", "bash", "grep", "glob"]

    def test_config_uri_guard_protected_tools_custom(self) -> None:
        """VikingCapabilityConfig accepts custom protected tools list."""
        cfg = VikingCapabilityConfig(
            uri_guard_enabled=True,
            uri_guard_protected_tools=["read", "write_file"],
        )
        assert cfg.uri_guard_protected_tools == ["read", "write_file"]

    @pytest.mark.asyncio
    async def test_for_run_preserves_enable_forget(self, mock_client: AsyncMock) -> None:
        """for_run() preserves enable_forget setting."""
        cap = VikingCapability(mode="all", enable_forget=True)
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.enable_forget is True


# ---------------------------------------------------------------------------
# 5.10 — Test Compaction Archive (Tasks 5.1-5.9)
# ---------------------------------------------------------------------------


class TestCompaction:
    """Tests for compaction archive feature (Tasks 5.1-5.9)."""

    # ---- Task 5.1: _estimate_tokens() pure function tests ----

    def test_estimate_tokens_ascii(self) -> None:
        """ASCII text: chars / 4."""
        from wolfharness.capabilities.viking.compaction import _estimate_tokens

        assert _estimate_tokens("") == 0
        assert _estimate_tokens("hello") == 1  # 5 // 4 = 1
        assert _estimate_tokens("hello world!") == 3  # 12 // 4 = 3
        assert _estimate_tokens("a" * 400) == 100

    def test_estimate_tokens_cjk(self) -> None:
        """CJK characters count as 1 token each."""
        from wolfharness.capabilities.viking.compaction import _estimate_tokens

        # Each CJK char is 1 token
        assert _estimate_tokens("你好") == 2
        assert _estimate_tokens("你好世界") == 4
        # CJK Extension A
        assert _estimate_tokens("\u3400\u3401") == 2

    def test_estimate_tokens_mixed(self) -> None:
        """Mixed ASCII + CJK: ASCII at 4:1, CJK at 1:1."""
        from wolfharness.capabilities.viking.compaction import _estimate_tokens

        # 2 CJK chars (2 tokens) + 8 ASCII chars (2 tokens) = 4
        assert _estimate_tokens("你好abcdefgh") == 4

    def test_estimate_tokens_empty(self) -> None:
        """Empty string returns 0."""
        from wolfharness.capabilities.viking.compaction import _estimate_tokens

        assert _estimate_tokens("") == 0

    # ---- Task 5.2: _split_archivable() pure function tests ----

    def test_split_archivable_basic(self) -> None:
        """Split messages keeping last N turns."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.compaction import _split_archivable

        # 4 user turns, keep last 2
        messages: list[Any] = []
        for i in range(4):
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"User turn {i}")]))
            messages.append(ModelResponse(parts=[TextPart(content=f"Assistant turn {i}")]))

        archivable, keep = _split_archivable(messages, keep_recent_turns=2)
        assert len(archivable) == 4  # 2 old turns (2 msgs each)
        assert len(keep) == 4  # 2 recent turns (2 msgs each)
        # Keep should start with the 3rd user message
        first_keep = keep[0]
        assert isinstance(first_keep, ModelRequest)
        user_part = first_keep.parts[0]
        assert isinstance(user_part, UserPromptPart)
        assert "User turn 2" in str(user_part.content)

    def test_split_archivable_keep_all_when_fewer_turns(self) -> None:
        """Keep all messages when fewer turns than keep_recent_turns."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        from wolfharness.capabilities.viking.compaction import _split_archivable

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Turn 1")]),
            ModelRequest(parts=[UserPromptPart(content="Turn 2")]),
        ]
        archivable, keep = _split_archivable(messages, keep_recent_turns=5)
        assert archivable == []
        assert len(keep) == 2

    def test_split_archivable_keep_zero(self) -> None:
        """keep_recent_turns=0 means all messages are archivable."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        from wolfharness.capabilities.viking.compaction import _split_archivable

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Turn 1")]),
            ModelRequest(parts=[UserPromptPart(content="Turn 2")]),
        ]
        archivable, keep = _split_archivable(messages, keep_recent_turns=0)
        assert len(archivable) == 2
        assert keep == []

    def test_split_archivable_empty(self) -> None:
        """Empty messages list returns empty tuple."""
        from wolfharness.capabilities.viking.compaction import _split_archivable

        archivable, keep = _split_archivable([], keep_recent_turns=3)
        assert archivable == []
        assert keep == []

    # ---- Task 5.3: _serialize_messages() pure function tests ----

    def test_serialize_messages_basic(self) -> None:
        """Serialize messages as markdown with role headers."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.compaction import _serialize_messages

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello there")]),
            ModelResponse(parts=[TextPart(content="Hi! How can I help?")]),
        ]
        result = _serialize_messages(messages)
        assert "## User" in result
        assert "Hello there" in result
        assert "## Assistant" in result
        assert "Hi! How can I help?" in result

    def test_serialize_messages_empty(self) -> None:
        """Empty messages produce empty string."""
        from wolfharness.capabilities.viking.compaction import _serialize_messages

        assert _serialize_messages([]) == ""

    # ---- Task 5.4: _summarize_messages() pure function tests ----

    def test_summarize_messages_basic(self) -> None:
        """Summary contains first 200 chars of each message."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.compaction import _summarize_messages

        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is Python?")]),
            ModelResponse(parts=[TextPart(content="A programming language.")]),
        ]
        result = _summarize_messages(messages)
        assert "**User:** What is Python?" in result
        assert "**Assistant:** A programming language." in result

    def test_summarize_messages_truncation(self) -> None:
        """Summary truncates long content to 200 chars."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        from wolfharness.capabilities.viking.compaction import _summarize_messages

        long_text = "x" * 300
        messages = [
            ModelRequest(parts=[UserPromptPart(content=long_text)]),
        ]
        result = _summarize_messages(messages)
        # Should contain first 200 chars
        assert "x" * 200 in result
        assert "x" * 201 not in result

    def test_summarize_messages_empty(self) -> None:
        """Empty messages produce empty string."""
        from wolfharness.capabilities.viking.compaction import _summarize_messages

        assert _summarize_messages([]) == ""

    # ---- Task 5.6: _handle_compaction() tests ----

    @pytest.mark.asyncio
    async def test_handle_compaction_above_threshold(self, mock_client: AsyncMock) -> None:
        """Above threshold: archive and replace messages."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            compaction_enabled=True,
            compaction_threshold=10,  # very low threshold
            compaction_keep_recent_turns=1,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        messages: list[Any] = []
        for i in range(3):
            messages.append(
                ModelRequest(parts=[UserPromptPart(content=f"User turn {i} with some text")])
            )
            messages.append(
                ModelResponse(parts=[TextPart(content=f"Assistant reply {i} with some text")])
            )

        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await cap._handle_compaction(ctx, rc)

        # Should have written to Viking
        mock_client.write.assert_called_once()
        write_args = mock_client.write.call_args
        archive_uri = write_args.args[0]
        assert "viking://user/alice/memories/compacted/" in archive_uri
        assert archive_uri.endswith(".md")

        # Result should have fewer messages than original (archived replaced with summary)
        assert len(result.messages) < len(messages)
        # First message should be the archive summary (SystemPromptPart)
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        from pydantic_ai.messages import SystemPromptPart

        sys_part = first_msg.parts[0]
        assert isinstance(sys_part, SystemPromptPart)
        assert archive_uri in sys_part.content

    @pytest.mark.asyncio
    async def test_handle_compaction_below_threshold(self, mock_client: AsyncMock) -> None:
        """Below threshold: no-op, return original context."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(
            mode="all",
            compaction_enabled=True,
            compaction_threshold=1_000_000,  # very high threshold
            compaction_keep_recent_turns=2,
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="short message")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await cap._handle_compaction(ctx, rc)

        # Should return the same context unchanged
        assert result is rc
        mock_client.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_compaction_disabled(self, mock_client: AsyncMock) -> None:
        """Disabled compaction is a no-op."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(
            mode="all",
            compaction_enabled=False,
            compaction_threshold=1,  # would trigger if enabled
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="text")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await cap._handle_compaction(ctx, rc)

        assert result is rc
        mock_client.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_compaction_server_unreachable(self, mock_client: AsyncMock) -> None:
        """Server unreachable: graceful failure, return original context."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            compaction_enabled=True,
            compaction_threshold=10,
            compaction_keep_recent_turns=1,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")
        mock_client.write = AsyncMock(side_effect=RuntimeError("connection refused"))

        messages: list[Any] = []
        for i in range(3):
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"User turn {i}")]))
            messages.append(ModelResponse(parts=[TextPart(content=f"Assistant {i}")]))

        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await cap._handle_compaction(ctx, rc)

        # Should return original context unchanged
        assert result is rc

    @pytest.mark.asyncio
    async def test_handle_compaction_cursor_adjustment(self, mock_client: AsyncMock) -> None:
        """Cursor (_last_ingested_idx) is decremented by N after compaction."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            compaction_enabled=True,
            compaction_threshold=10,
            compaction_keep_recent_turns=1,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")
        cap._last_ingested_idx = 5  # Simulate prior ingestion

        messages: list[Any] = []
        for i in range(3):
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"User turn {i}")]))
            messages.append(ModelResponse(parts=[TextPart(content=f"Assistant {i}")]))

        rc = _make_request_context(messages)
        ctx = _make_ctx()
        await cap._handle_compaction(ctx, rc)

        # 4 messages were archivable (2 turns), so cursor should be 5-4=1
        assert cap._last_ingested_idx == 1

    @pytest.mark.asyncio
    async def test_handle_compaction_cursor_clamped_to_zero(self, mock_client: AsyncMock) -> None:
        """Cursor is clamped to 0 when N exceeds current value."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            compaction_enabled=True,
            compaction_threshold=10,
            compaction_keep_recent_turns=1,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")
        cap._last_ingested_idx = 2  # Less than N=4 archivable

        messages: list[Any] = []
        for i in range(3):
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"User turn {i}")]))
            messages.append(ModelResponse(parts=[TextPart(content=f"Assistant {i}")]))

        rc = _make_request_context(messages)
        ctx = _make_ctx()
        await cap._handle_compaction(ctx, rc)

        assert cap._last_ingested_idx == 0

    # ---- Task 5.7: viking_expand tool tests ----

    @pytest.mark.asyncio
    async def test_viking_expand_tool_calls_client_read(self, mock_client: AsyncMock) -> None:
        """viking_expand calls client.read(uri) and returns content."""
        mock_client.read = AsyncMock(return_value="Archived conversation content")
        cap = VikingCapability(
            mode="retrieve",
            compaction_expand_tool=True,
        )
        cap._client = mock_client

        tools = build_tools(cap)
        expand_tool = _get_tool(tools, "viking_expand")

        ctx = _make_ctx()
        result = await expand_tool(ctx, uri="viking://user/alice/memories/compacted/abc.md")

        mock_client.read.assert_called_once_with("viking://user/alice/memories/compacted/abc.md")
        assert result.return_value == "Archived conversation content"

    @pytest.mark.asyncio
    async def test_viking_expand_tool_error(self, mock_client: AsyncMock) -> None:
        """viking_expand returns error string on failure."""
        mock_client.read = AsyncMock(side_effect=RuntimeError("not found"))
        cap = VikingCapability(
            mode="retrieve",
            compaction_expand_tool=True,
        )
        cap._client = mock_client

        tools = build_tools(cap)
        expand_tool = _get_tool(tools, "viking_expand")

        ctx = _make_ctx()
        result = await expand_tool(ctx, uri="viking://missing.md")

        assert "viking_expand error (RuntimeError): not found" in result.return_value

    @pytest.mark.asyncio
    async def test_viking_expand_tool_empty_content(self, mock_client: AsyncMock) -> None:
        """viking_expand returns 'No content found' for empty response."""
        mock_client.read = AsyncMock(return_value="")
        cap = VikingCapability(
            mode="retrieve",
            compaction_expand_tool=True,
        )
        cap._client = mock_client

        tools = build_tools(cap)
        expand_tool = _get_tool(tools, "viking_expand")

        ctx = _make_ctx()
        result = await expand_tool(ctx, uri="viking://empty.md")

        assert result.return_value == "No content found at URI."

    def test_viking_expand_not_exposed_when_disabled(self) -> None:
        """viking_expand not in tools when compaction_expand_tool=False."""
        cap = VikingCapability(
            mode="retrieve",
            compaction_expand_tool=False,
        )
        cap._client = AsyncMock()

        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_expand" not in names

    def test_viking_expand_exposed_when_enabled(self) -> None:
        """viking_expand in tools when compaction_expand_tool=True."""
        cap = VikingCapability(
            mode="retrieve",
            compaction_expand_tool=True,
        )
        cap._client = AsyncMock()

        tools = build_tools(cap)
        names = {t.__name__ for t in tools}
        assert "viking_expand" in names

    # ---- Task 5.8: Config field tests ----

    def test_config_compaction_disabled_default(self) -> None:
        """VikingCapabilityConfig has compaction_enabled=False by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.compaction_enabled is False

    def test_config_compaction_threshold_default(self) -> None:
        """VikingCapabilityConfig has compaction_threshold=100000 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.compaction_threshold == 100_000

    def test_config_compaction_keep_recent_turns_default(self) -> None:
        """VikingCapabilityConfig has compaction_keep_recent_turns=5 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.compaction_keep_recent_turns == 5

    def test_config_compaction_expand_tool_default(self) -> None:
        """VikingCapabilityConfig has compaction_expand_tool=True by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.compaction_expand_tool is True

    def test_config_compaction_fields_set(self) -> None:
        """All compaction fields can be set at once."""
        cfg = VikingCapabilityConfig(
            compaction_enabled=True,
            compaction_threshold=50_000,
            compaction_keep_recent_turns=3,
            compaction_expand_tool=False,
        )
        assert cfg.compaction_enabled is True
        assert cfg.compaction_threshold == 50_000
        assert cfg.compaction_keep_recent_turns == 3
        assert cfg.compaction_expand_tool is False

    # ---- Task 5.9: for_run() preserves compaction fields ----

    @pytest.mark.asyncio
    async def test_for_run_preserves_compaction_fields(self, mock_client: AsyncMock) -> None:
        """for_run() preserves compaction config fields."""
        cap = VikingCapability(
            mode="all",
            compaction_enabled=True,
            compaction_threshold=50_000,
            compaction_keep_recent_turns=3,
            compaction_expand_tool=False,
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.compaction_enabled is True
        assert copy_cap.compaction_threshold == 50_000
        assert copy_cap.compaction_keep_recent_turns == 3
        assert copy_cap.compaction_expand_tool is False


# ---------------------------------------------------------------------------
# 3.10 — Auto Conversation Ingestion tests (Tasks 3.1-3.9)
# ---------------------------------------------------------------------------


class TestAutoIngest:
    """Tests for auto conversation ingestion helpers and _handle_auto_ingest()."""

    # ---- Pure function tests: _sanitize_message ----

    def test_sanitize_strips_recall_block(self) -> None:
        """_sanitize_message replaces <openviking-recall> with placeholder."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        content = "Hello <openviking-recall>secret data</openviking-recall> world"
        result = _sanitize_message(content)
        assert "<openviking-recall>" not in result
        assert "secret data" not in result
        assert "[recalled context omitted]" in result
        assert "Hello" in result
        assert "world" in result

    def test_sanitize_strips_profile_block(self) -> None:
        """_sanitize_message replaces <openviking-profile> with placeholder."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        content = "Context: <openviking-profile>user profile data</openviking-profile> end"
        result = _sanitize_message(content)
        assert "<openviking-profile>" not in result
        assert "user profile data" not in result
        assert "[recalled context omitted]" in result

    def test_sanitize_strips_both_blocks(self) -> None:
        """_sanitize_message strips both recall and profile blocks."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        content = (
            "<openviking-recall>recall data</openviking-recall>"
            " middle "
            "<openviking-profile>profile data</openviking-profile>"
        )
        result = _sanitize_message(content)
        assert "recall data" not in result
        assert "profile data" not in result
        assert result.count("[recalled context omitted]") == 2

    def test_sanitize_multiline_blocks(self) -> None:
        """_sanitize_message handles multi-line XML blocks."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        content = (
            "<openviking-recall>\n  <hit uri='viking://doc.md'/>\n  data\n</openviking-recall> rest"
        )
        result = _sanitize_message(content)
        assert "<openviking-recall>" not in result
        assert "[recalled context omitted]" in result
        assert "rest" in result

    def test_sanitize_disabled_returns_original(self) -> None:
        """When enabled=False, content is returned unchanged."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        content = "<openviking-recall>keep this</openviking-recall>"
        result = _sanitize_message(content, enabled=False)
        assert result == content

    def test_sanitize_no_xml_blocks(self) -> None:
        """Content without XML blocks is returned unchanged."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        content = "just plain text without any xml blocks"
        result = _sanitize_message(content)
        assert result == content

    def test_sanitize_empty_string(self) -> None:
        """Empty string returns empty string."""
        from wolfharness.capabilities.viking.ingest import _sanitize_message

        assert _sanitize_message("") == ""

    # ---- Pure function tests: _extract_conversation_pairs ----

    def test_extract_conversation_pairs_basic(self) -> None:
        """Extracts user+assistant pairs from messages."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.ingest import _extract_conversation_pairs

        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is X?")]),
            ModelResponse(parts=[TextPart(content="X is a thing.")]),
        ]
        result = _extract_conversation_pairs(messages, start_idx=0)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "What is X?"}
        assert result[1] == {"role": "assistant", "content": "X is a thing."}

    def test_extract_conversation_pairs_cursor_tracking(self) -> None:
        """Only extracts messages after start_idx."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.ingest import _extract_conversation_pairs

        messages = [
            ModelRequest(parts=[UserPromptPart(content="old question")]),
            ModelResponse(parts=[TextPart(content="old answer")]),
            ModelRequest(parts=[UserPromptPart(content="new question")]),
            ModelResponse(parts=[TextPart(content="new answer")]),
        ]
        result = _extract_conversation_pairs(messages, start_idx=2)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "new question"}
        assert result[1] == {"role": "assistant", "content": "new answer"}

    def test_extract_conversation_pairs_no_new_messages(self) -> None:
        """Returns empty list when start_idx >= len(messages)."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        from wolfharness.capabilities.viking.ingest import _extract_conversation_pairs

        messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
        result = _extract_conversation_pairs(messages, start_idx=1)
        assert result == []

    def test_extract_conversation_pairs_multiple_turns(self) -> None:
        """Extracts multiple user+assistant turns."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.ingest import _extract_conversation_pairs

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q1")]),
            ModelResponse(parts=[TextPart(content="A1")]),
            ModelRequest(parts=[UserPromptPart(content="Q2")]),
            ModelResponse(parts=[TextPart(content="A2")]),
            ModelRequest(parts=[UserPromptPart(content="Q3")]),
            ModelResponse(parts=[TextPart(content="A3")]),
        ]
        result = _extract_conversation_pairs(messages, start_idx=0)
        assert len(result) == 6
        assert result[0]["content"] == "Q1"
        assert result[1]["content"] == "A1"
        assert result[4]["content"] == "Q3"
        assert result[5]["content"] == "A3"

    def test_extract_conversation_pairs_skips_non_text_content(self) -> None:
        """Skips UserPromptPart with list (multimodal) content."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.ingest import _extract_conversation_pairs

        messages = [
            ModelRequest(parts=[UserPromptPart(content=["image_data", "text"])]),
            ModelResponse(parts=[TextPart(content="response")]),
            ModelRequest(parts=[UserPromptPart(content="plain text")]),
        ]
        result = _extract_conversation_pairs(messages, start_idx=0)
        # First user prompt is skipped (list content), assistant text is included
        assert len(result) == 2
        assert result[0] == {"role": "assistant", "content": "response"}
        assert result[1] == {"role": "user", "content": "plain text"}

    def test_extract_conversation_pairs_empty_messages(self) -> None:
        """Empty messages list returns empty list."""
        from wolfharness.capabilities.viking.ingest import _extract_conversation_pairs

        assert _extract_conversation_pairs([], start_idx=0) == []

    # ---- _handle_auto_ingest integration tests ----

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_disabled(self, mock_client: AsyncMock) -> None:
        """Disabled ingest (auto_ingest_enabled=False) is a no-op."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(mode="all", auto_ingest_enabled=False)
        cap._client = mock_client

        msg = ModelRequest(parts=[UserPromptPart(content="test")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        # Even when called directly, disabled ingest does nothing
        # (Group 7 gates this; here we test the handler logic itself)
        result = await cap._handle_auto_ingest(ctx, rc)
        assert result is rc
        mock_client.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_no_new_messages(self, mock_client: AsyncMock) -> None:
        """When cursor is at current message count, ingestion is skipped."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(mode="all", auto_ingest_enabled=True)
        cap._client = mock_client
        cap._last_ingested_idx = 1  # Cursor at current count

        msg = ModelRequest(parts=[UserPromptPart(content="test")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap._handle_auto_ingest(ctx, rc)
        assert result is rc
        mock_client.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_extracts_and_ingests(self, mock_client: AsyncMock) -> None:
        """Ingestion extracts conversation pairs and creates a Viking session."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",  # sync for deterministic test
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is X?")]),
            ModelResponse(parts=[TextPart(content="X is a thing.")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        result = await cap._handle_auto_ingest(ctx, rc)

        # Cursor should be updated
        assert cap._last_ingested_idx == 2
        # Session creation should have been called
        mock_client.create_session.assert_called_once()
        # Two messages should have been added
        assert mock_client.add_message.call_count == 2
        # Session should have been committed
        mock_client.commit_session.assert_called_once()

        # Verify message content
        add_calls = mock_client.add_message.call_args_list
        assert add_calls[0].args[1] == "user"
        assert add_calls[0].args[2] == "What is X?"
        assert add_calls[1].args[1] == "assistant"
        assert add_calls[1].args[2] == "X is a thing."

        # Result should be unchanged
        assert result is rc

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_sanitizes_messages(self, mock_client: AsyncMock) -> None:
        """Ingestion sanitizes XML blocks before writing to Viking."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_sanitize=True,
        )
        cap._client = mock_client

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content="Question <openviking-recall>secret</openviking-recall>")
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content="Answer <openviking-profile>profile data</openviking-profile>")
                ]
            ),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        # Verify sanitized content was ingested
        add_calls = mock_client.add_message.call_args_list
        assert "[recalled context omitted]" in add_calls[0].args[2]
        assert "secret" not in add_calls[0].args[2]
        assert "[recalled context omitted]" in add_calls[1].args[2]
        assert "profile data" not in add_calls[1].args[2]

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_no_sanitize(self, mock_client: AsyncMock) -> None:
        """When auto_ingest_sanitize=False, messages are ingested verbatim."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_sanitize=False,
        )
        cap._client = mock_client

        xml_content = "<openviking-recall>keep this</openviking-recall>"
        messages = [
            ModelRequest(parts=[UserPromptPart(content=xml_content)]),
            ModelResponse(parts=[TextPart(content="response")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        add_calls = mock_client.add_message.call_args_list
        assert add_calls[0].args[2] == xml_content  # verbatim

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_commit_with_retention(self, mock_client: AsyncMock) -> None:
        """Commit passes keep_recent_count when configured."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_keep_recent_turns=3,
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        commit_kwargs = mock_client.commit_session.call_args.kwargs
        assert commit_kwargs["keep_recent_count"] == 3

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_commit_without_retention(
        self, mock_client: AsyncMock
    ) -> None:
        """Commit does not pass keep_recent_count when it's 0."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_keep_recent_turns=0,
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        commit_kwargs = mock_client.commit_session.call_args.kwargs
        assert "keep_recent_count" not in commit_kwargs

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_async_mode_creates_task(self, mock_client: AsyncMock) -> None:
        """Async mode creates a fire-and-forget task."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="async",
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        # Cursor should be updated immediately
        assert cap._last_ingested_idx == 2
        # A task must have been spawned and tracked
        assert len(cap._pending_tasks) == 1

        await asyncio.gather(*cap._pending_tasks, return_exceptions=True)

        # After task completes, SDK calls should have been made
        mock_client.create_session.assert_called_once()
        mock_client.commit_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_graceful_failure(self, mock_client: AsyncMock) -> None:
        """Ingestion failure does not block and still updates cursor."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        mock_client.create_session = AsyncMock(side_effect=RuntimeError("server unreachable"))

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        # Should not raise
        result = await cap._handle_auto_ingest(ctx, rc)

        # Cursor should still be updated (to avoid retrying)
        assert cap._last_ingested_idx == 2
        # Result should be unchanged
        assert result is rc

    @pytest.mark.asyncio
    async def test_handle_auto_ingest_cursor_update_on_no_pairs(
        self, mock_client: AsyncMock
    ) -> None:
        """Cursor is updated even when no conversation pairs are extracted."""
        from pydantic_ai.messages import ModelRequest, TextPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
        )
        cap._client = mock_client

        # Message without UserPromptPart (e.g., system-only)
        messages = [ModelRequest(parts=[TextPart(content="no user prompt")])]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        # Cursor should be updated even though no pairs were extracted
        assert cap._last_ingested_idx == 1
        mock_client.create_session.assert_not_called()

    # ---- after_run() tests ----

    @pytest.mark.asyncio
    async def test_after_run_no_pending_tasks(self, mock_client: AsyncMock) -> None:
        """after_run() is a no-op when there are no pending tasks."""
        cap = VikingCapability(mode="all", auto_ingest_enabled=True)
        cap._client = mock_client

        ctx = _make_ctx()
        result = await cap.after_run(ctx, result="test_result")
        assert result == "test_result"

    @pytest.mark.asyncio
    async def test_after_run_flushes_pending_tasks(self, mock_client: AsyncMock) -> None:
        """after_run() awaits pending fire-and-forget tasks."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="async",
        )
        cap._client = mock_client

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Q")]),
            ModelResponse(parts=[TextPart(content="A")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap._handle_auto_ingest(ctx, rc)

        # after_run should flush
        await cap.after_run(ctx, result="done")

        # All tasks should be completed
        assert len(cap._pending_tasks) == 0
        mock_client.create_session.assert_called_once()
        mock_client.commit_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_after_run_timeout(self, mock_client: AsyncMock) -> None:
        """after_run() logs warning on timeout but does not raise."""
        import asyncio

        cap = VikingCapability(mode="all", auto_ingest_enabled=True)
        cap._client = mock_client

        # Create a task that never completes
        async def _slow_task() -> None:
            await asyncio.sleep(100)

        task = asyncio.create_task(_slow_task())
        cap._pending_tasks.add(task)
        task.add_done_callback(cap._pending_tasks.discard)

        ctx = _make_ctx()
        # Should return within 5 seconds (timeout)
        result = await cap.after_run(ctx, result="done")
        assert result == "done"

        # Clean up the task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # ---- for_run() state isolation ----

    @pytest.mark.asyncio
    async def test_for_run_resets_ingestion_state(self, mock_client: AsyncMock) -> None:
        """for_run() resets the ingestion cursor and deferred-remember state."""
        cap = VikingCapability(mode="all", auto_ingest_enabled=True)
        cap._client = mock_client
        cap._last_ingested_idx = 42
        cap._remember_pending = ["reason"]
        cap._remember_drain_failures = 2

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap._last_ingested_idx == 0
        assert copy_cap._remember_pending == []
        assert copy_cap._remember_drain_failures == 0
        assert copy_cap._pending_tasks == set()
        # Identity should be shared
        assert copy_cap._identity is cap._identity

    @pytest.mark.asyncio
    async def test_for_run_preserves_auto_ingest_config(self, mock_client: AsyncMock) -> None:
        """for_run() preserves auto_ingest config fields."""
        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_sanitize=False,
            auto_ingest_source_type="custom",
            auto_ingest_keep_recent_turns=5,
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.auto_ingest_enabled is True
        assert copy_cap.auto_ingest_mode == "sync"
        assert copy_cap.auto_ingest_sanitize is False
        assert copy_cap.auto_ingest_source_type == "custom"
        assert copy_cap.auto_ingest_keep_recent_turns == 5

    # ---- Config field tests ----

    def test_config_auto_ingest_enabled_default(self) -> None:
        """VikingCapabilityConfig has auto_ingest_enabled=False by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_ingest_enabled is False

    def test_config_auto_ingest_mode_default(self) -> None:
        """VikingCapabilityConfig has auto_ingest_mode='async' by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_ingest_mode == "async"

    def test_config_auto_ingest_sanitize_default(self) -> None:
        """VikingCapabilityConfig has auto_ingest_sanitize=True by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_ingest_sanitize is True

    def test_config_auto_ingest_source_type_default(self) -> None:
        """VikingCapabilityConfig has auto_ingest_source_type='wolfharness' by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_ingest_source_type == "wolfharness"

    def test_config_auto_ingest_keep_recent_turns_default(self) -> None:
        """VikingCapabilityConfig has auto_ingest_keep_recent_turns=0 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.auto_ingest_keep_recent_turns == 0

    def test_config_auto_ingest_all_fields_set(self) -> None:
        """All auto-ingest config fields can be set at once."""
        cfg = VikingCapabilityConfig(
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_sanitize=False,
            auto_ingest_source_type="myapp",
            auto_ingest_keep_recent_turns=10,
        )
        assert cfg.auto_ingest_enabled is True
        assert cfg.auto_ingest_mode == "sync"
        assert cfg.auto_ingest_sanitize is False
        assert cfg.auto_ingest_source_type == "myapp"
        assert cfg.auto_ingest_keep_recent_turns == 10


# ---------------------------------------------------------------------------
# 6.7 — Test Profile Injection (Tasks 6.1-6.6)
# ---------------------------------------------------------------------------


class TestProfileInjection:
    """Tests for profile injection — formatting, context hint, injection, config.

    Covers _format_profile_block, _derive_context_hint, _handle_profile_inject,
    config fields, and for_run() reset.
    """

    # ---- _format_profile_block (pure function) ----

    def test_format_profile_block_basic(self) -> None:
        """_format_profile_block renders XML with memory hits."""
        results = {
            "hits": [
                {
                    "uri": "viking://user/alice/memories/project.md",
                    "score": 0.9,
                    "content": "Project uses Python 3.13",
                    "context_type": "memory",
                },
            ]
        }
        block = _format_profile_block(results, max_tokens=1000)
        assert "<openviking-profile>" in block
        assert "</openviking-profile>" in block
        assert "<project-context>" in block
        assert "viking://user/alice/memories/project.md" in block
        assert "Project uses Python 3.13" in block

    def test_format_profile_block_with_resources(self) -> None:
        """_format_profile_block groups resource hits into <relevant-resources>."""
        results = {
            "hits": [
                {
                    "uri": "viking://resources/doc.md",
                    "score": 0.8,
                    "content": "Resource content",
                    "context_type": "resource",
                },
            ]
        }
        block = _format_profile_block(results, max_tokens=1000)
        assert "<relevant-resources>" in block
        assert "viking://resources/doc.md" in block

    def test_format_profile_block_empty_results(self) -> None:
        """_format_profile_block returns empty string for empty results."""
        assert _format_profile_block({"hits": []}) == ""
        assert _format_profile_block({"results": []}) == ""
        assert _format_profile_block([]) == ""
        assert _format_profile_block({}) == ""

    def test_format_profile_block_token_budget_truncation(self) -> None:
        """_format_profile_block truncates content exceeding token budget."""
        long_content = "x" * 5000
        results = {
            "hits": [
                {
                    "uri": "viking://mem.md",
                    "score": 0.9,
                    "content": long_content,
                    "context_type": "memory",
                },
            ]
        }
        block = _format_profile_block(results, max_tokens=100)
        # max_tokens=100 → max_chars=400
        assert len(block) < 600
        assert "truncated" in block

    def test_format_profile_block_within_budget(self) -> None:
        """_format_profile_block does not truncate when within budget."""
        results = {
            "hits": [
                {
                    "uri": "viking://mem.md",
                    "score": 0.9,
                    "content": "Short content",
                    "context_type": "memory",
                },
            ]
        }
        block = _format_profile_block(results, max_tokens=1000)
        assert "truncated" not in block

    def test_format_profile_block_viking_grouped_format(self) -> None:
        """_format_profile_block handles Viking's grouped format (memories/resources/skills)."""
        results = {
            "memories": [
                {"uri": "viking://mem.md", "content": "memory", "context_type": "memory"},
            ],
            "resources": [
                {"uri": "viking://res.md", "content": "resource", "context_type": "resource"},
            ],
        }
        block = _format_profile_block(results, max_tokens=1000)
        assert "<project-context>" in block
        assert "<relevant-resources>" in block
        assert "viking://mem.md" in block
        assert "viking://res.md" in block

    # ---- _derive_context_hint (pure function with mock ctx) ----

    def test_derive_context_hint_agent_name(self) -> None:
        """_derive_context_hint returns agent_name when available."""
        ctx = MagicMock()
        ctx.deps = MagicMock()
        ctx.deps.agent_name = "diagnostic_agent"
        ctx.deps.session_metadata = None
        ctx.messages = []
        hint = _derive_context_hint(ctx)
        assert hint == "diagnostic_agent"

    def test_derive_context_hint_session_metadata_topic(self) -> None:
        """_derive_context_hint falls back to session_metadata topic."""
        ctx = MagicMock()
        ctx.deps = MagicMock()
        ctx.deps.agent_name = ""
        ctx.deps.session_metadata = {"topic": "hydraulic diagnosis"}
        ctx.messages = []
        hint = _derive_context_hint(ctx)
        assert hint == "hydraulic diagnosis"

    def test_derive_context_hint_session_metadata_description(self) -> None:
        """_derive_context_hint falls back to session_metadata description."""
        ctx = MagicMock()
        ctx.deps = MagicMock()
        ctx.deps.agent_name = ""
        ctx.deps.session_metadata = {"description": "excavator repair session"}
        ctx.messages = []
        hint = _derive_context_hint(ctx)
        assert hint == "excavator repair session"

    def test_derive_context_hint_fallback_to_prompt(self) -> None:
        """_derive_context_hint falls back to first 100 chars of latest user prompt."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        prompt = "A" * 150  # longer than 100 chars
        msg = ModelRequest(parts=[UserPromptPart(content=prompt)])
        ctx = MagicMock()
        ctx.deps = MagicMock()
        ctx.deps.agent_name = ""
        ctx.deps.session_metadata = None
        ctx.messages = [msg]
        hint = _derive_context_hint(ctx)
        assert len(hint) == 100
        assert hint == "A" * 100

    def test_derive_context_hint_empty_when_nothing_available(self) -> None:
        """_derive_context_hint returns empty string when nothing is available."""
        ctx = MagicMock()
        ctx.deps = MagicMock(spec=[])  # no attributes
        ctx.messages = []
        hint = _derive_context_hint(ctx)
        assert hint == ""

    # ---- _handle_profile_inject (using mock_client + _make_request_context) ----

    @pytest.mark.asyncio
    async def test_handle_profile_inject_first_turn(self, mock_client: AsyncMock) -> None:
        """_handle_profile_inject injects profile on first turn."""
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        mock_client.find = AsyncMock(
            return_value={
                "hits": [
                    {
                        "uri": "viking://mem.md",
                        "content": "Project context",
                        "context_type": "memory",
                        "score": 0.9,
                    },
                ]
            }
        )
        cap = VikingCapability(mode="all", profile_enabled=True)
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        rc = _make_request_context([msg])

        result = await cap._handle_profile_inject(ctx, rc)

        assert cap._profile_injected is True
        mock_client.find.assert_called_once()
        # Check find() call args
        call_kwargs = mock_client.find.call_args.kwargs
        assert call_kwargs["limit"] == 5  # default profile_limit
        assert call_kwargs["context_type"] == "memory"
        assert "viking://user/alice/memories/" in call_kwargs["target_uri"]
        # Check that a SystemPromptPart was injected before the user message
        assert len(result.messages) == 2
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        assert any(isinstance(p, SystemPromptPart) for p in first_msg.parts)

    @pytest.mark.asyncio
    async def test_handle_profile_inject_skips_when_already_injected(
        self, mock_client: AsyncMock
    ) -> None:
        """_handle_profile_inject skips when _profile_injected is already True."""
        cap = VikingCapability(mode="all", profile_enabled=True)
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")
        cap._profile_injected = True

        ctx = _make_ctx()
        rc = _make_request_context([])

        result = await cap._handle_profile_inject(ctx, rc)

        mock_client.find.assert_not_called()
        assert result is rc

    @pytest.mark.asyncio
    async def test_handle_profile_inject_skips_subsequent_turn(
        self, mock_client: AsyncMock
    ) -> None:
        """_handle_profile_inject skips when message count > 2 (subsequent turn)."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(mode="all", profile_enabled=True)
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        # 3 messages = subsequent turn
        messages = [
            ModelRequest(parts=[UserPromptPart(content="msg1")]),
            ModelRequest(parts=[UserPromptPart(content="msg2")]),
            ModelRequest(parts=[UserPromptPart(content="msg3")]),
        ]
        rc = _make_request_context(messages)

        result = await cap._handle_profile_inject(ctx, rc)

        mock_client.find.assert_not_called()
        assert result is rc
        # _profile_injected is set to True to prevent future attempts
        assert cap._profile_injected is True

    @pytest.mark.asyncio
    async def test_handle_profile_inject_graceful_failure(self, mock_client: AsyncMock) -> None:
        """_handle_profile_inject handles errors gracefully — returns original context."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.find = AsyncMock(side_effect=RuntimeError("server unreachable"))
        cap = VikingCapability(mode="all", profile_enabled=True)
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        rc = _make_request_context([msg])

        result = await cap._handle_profile_inject(ctx, rc)

        # Should return original context, not raise
        assert result is rc
        # _profile_injected should be True (set before the try block to avoid retrying)
        assert cap._profile_injected is True

    @pytest.mark.asyncio
    async def test_handle_profile_inject_disabled_is_noop(self, mock_client: AsyncMock) -> None:
        """_handle_profile_inject is a no-op when profile_enabled=False."""
        cap = VikingCapability(mode="all", profile_enabled=False)
        cap._client = mock_client

        ctx = _make_ctx()
        rc = _make_request_context([])

        result = await cap._handle_profile_inject(ctx, rc)

        mock_client.find.assert_not_called()
        assert result is rc
        assert cap._profile_injected is False

    @pytest.mark.asyncio
    async def test_handle_profile_inject_token_budget(self, mock_client: AsyncMock) -> None:
        """_handle_profile_inject truncates profile to profile_max_tokens."""
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        long_content = "x" * 5000
        mock_client.find = AsyncMock(
            return_value={
                "hits": [
                    {
                        "uri": "viking://mem.md",
                        "content": long_content,
                        "context_type": "memory",
                        "score": 0.9,
                    },
                ]
            }
        )
        cap = VikingCapability(
            mode="all",
            profile_enabled=True,
            profile_max_tokens=100,  # 100 tokens → 400 chars max
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        rc = _make_request_context([msg])

        result = await cap._handle_profile_inject(ctx, rc)

        # The injected SystemPromptPart content should be truncated
        first_msg = result.messages[0]
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert len(sys_parts) == 1
        assert len(sys_parts[0].content) < 600
        assert "truncated" in sys_parts[0].content

    @pytest.mark.asyncio
    async def test_handle_profile_inject_empty_results_no_injection(
        self, mock_client: AsyncMock
    ) -> None:
        """_handle_profile_inject does not inject when find() returns empty results."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.find = AsyncMock(return_value={"hits": []})
        cap = VikingCapability(mode="all", profile_enabled=True)
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        rc = _make_request_context([msg])

        result = await cap._handle_profile_inject(ctx, rc)

        # No injection — result is the same context
        assert result is rc
        # But _profile_injected is True (we did attempt)
        assert cap._profile_injected is True

    @pytest.mark.asyncio
    async def test_handle_profile_inject_first_turn_only_false(
        self, mock_client: AsyncMock
    ) -> None:
        """_handle_profile_inject runs on subsequent turns when first_turn_only=False."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.find = AsyncMock(
            return_value={
                "hits": [
                    {
                        "uri": "viking://mem.md",
                        "content": "context",
                        "context_type": "memory",
                        "score": 0.9,
                    },
                ]
            }
        )
        cap = VikingCapability(
            mode="all",
            profile_enabled=True,
            profile_first_turn_only=False,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        ctx = _make_ctx()
        # 4 messages — subsequent turn, but first_turn_only=False
        messages = [
            ModelRequest(parts=[UserPromptPart(content="msg1")]),
            ModelRequest(parts=[UserPromptPart(content="msg2")]),
            ModelRequest(parts=[UserPromptPart(content="msg3")]),
            ModelRequest(parts=[UserPromptPart(content="msg4")]),
        ]
        rc = _make_request_context(messages)

        await cap._handle_profile_inject(ctx, rc)

        mock_client.find.assert_called_once()
        assert cap._profile_injected is True

    # ---- Config fields ----

    def test_config_profile_enabled_default_false(self) -> None:
        """VikingCapabilityConfig has profile_enabled=False by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.profile_enabled is False

    def test_config_profile_max_tokens_default(self) -> None:
        """VikingCapabilityConfig has profile_max_tokens=1000 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.profile_max_tokens == 1000

    def test_config_profile_limit_default(self) -> None:
        """VikingCapabilityConfig has profile_limit=5 by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.profile_limit == 5

    def test_config_profile_first_turn_only_default(self) -> None:
        """VikingCapabilityConfig has profile_first_turn_only=True by default."""
        cfg = VikingCapabilityConfig()
        assert cfg.profile_first_turn_only is True

    def test_config_profile_fields_set(self) -> None:
        """VikingCapabilityConfig accepts all profile fields."""
        cfg = VikingCapabilityConfig(
            profile_enabled=True,
            profile_max_tokens=500,
            profile_limit=10,
            profile_first_turn_only=False,
        )
        assert cfg.profile_enabled is True
        assert cfg.profile_max_tokens == 500
        assert cfg.profile_limit == 10
        assert cfg.profile_first_turn_only is False

    # ---- for_run() reset ----

    @pytest.mark.asyncio
    async def test_for_run_resets_profile_injected(self, mock_client: AsyncMock) -> None:
        """for_run() resets _profile_injected to False."""
        cap = VikingCapability(mode="all", profile_enabled=True)
        cap._client = mock_client
        cap._profile_injected = True  # Simulate after first turn

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap._profile_injected is False
        # Original is unchanged
        assert cap._profile_injected is True

    @pytest.mark.asyncio
    async def test_for_run_preserves_profile_config(self, mock_client: AsyncMock) -> None:
        """for_run() preserves profile_enabled and other profile config fields."""
        cap = VikingCapability(
            mode="all",
            profile_enabled=True,
            profile_max_tokens=500,
            profile_limit=10,
            profile_first_turn_only=False,
        )
        cap._client = mock_client

        ctx = _make_ctx()
        copy_cap = await cap.for_run(ctx)

        assert copy_cap.profile_enabled is True
        assert copy_cap.profile_max_tokens == 500
        assert copy_cap.profile_limit == 10
        assert copy_cap.profile_first_turn_only is False


# ---------------------------------------------------------------------------
# 7.5 — Handler Chain Integration Tests
# ---------------------------------------------------------------------------


class TestHandlerChain:
    """Tests for before_model_request handler chain wiring (Group 7).

    Verifies that all handlers are called in the correct order (D7),
    each handler checks its own enabled flag (D14), and the chained
    request_context modification works correctly.
    """

    async def test_all_handlers_disabled_returns_original(self, mock_client: AsyncMock) -> None:
        """All handlers disabled: before_model_request returns original context.

        Given: a VikingCapability with all features disabled (defaults).
        When: before_model_request is called.
        Then: the original request_context is returned unchanged.
        """
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(mode="all")
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap.before_model_request(ctx, rc)

        assert result is rc
        mock_client.search.assert_not_called()
        mock_client.find.assert_not_called()
        mock_client.create_session.assert_not_called()

    async def test_only_auto_recall_enabled(self, mock_client: AsyncMock) -> None:
        """Only auto_recall enabled: only recall handler fires.

        Given: a VikingCapability with auto_recall_enabled=True, all others False.
        When: before_model_request is called.
        Then: client.search is called (recall ran), but create_session and
            find are not (ingest and profile did not fire).
        """
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "uri": "viking://user/alice/memories/doc.md",
                        "score": 0.8,
                        "content": "relevant memory",
                        "context_type": "memory",
                    }
                ]
            }
        )

        cap = VikingCapability(mode="all", auto_recall_enabled=True)
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="hydraulic pressure")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap.before_model_request(ctx, rc)

        mock_client.search.assert_called_once()
        mock_client.create_session.assert_not_called()
        mock_client.find.assert_not_called()
        # Recall should have injected a system message
        assert result is not rc
        assert len(result.messages) == 2

    async def test_multiple_features_enabled_handler_order(self, mock_client: AsyncMock) -> None:
        """Multiple features enabled: handlers fire in D7 order.

        Given: auto_ingest + profile + auto_recall all enabled.
        When: before_model_request is called with messages from a
            previous turn plus a new user prompt.
        Then: all handlers fire (create_session for ingest, find for
            profile, search for recall), and the result contains a
            recall block (recall ran after ingest).
        """
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        mock_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "uri": "viking://user/alice/memories/doc.md",
                        "score": 0.8,
                        "content": "memory content",
                        "context_type": "memory",
                    }
                ]
            }
        )
        mock_client.find = AsyncMock(
            return_value={
                "hits": [
                    {
                        "uri": "viking://user/alice/memories/profile.md",
                        "content": "project context",
                        "context_type": "memory",
                        "score": 0.9,
                    }
                ]
            }
        )

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            profile_enabled=True,
            profile_first_turn_only=False,  # allow profile on non-first turns
            auto_recall_enabled=True,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        # Previous turn (user + assistant) + new user prompt
        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is X?")]),
            ModelResponse(parts=[TextPart(content="X is a thing.")]),
            ModelRequest(parts=[UserPromptPart(content="Tell me more about X")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        result = await cap.before_model_request(ctx, rc)

        # All handlers should have fired
        mock_client.create_session.assert_called_once()  # ingest
        mock_client.find.assert_called_once()  # profile
        mock_client.search.assert_called_once()  # recall

        # Ingest cursor should be updated (ingest ran)
        assert cap._last_ingested_idx == 3

        # Result should contain injected content (profile + recall)
        assert result is not rc
        # At least 2 new system messages should have been injected
        # (profile before the first user message, recall before the latest)
        assert len(result.messages) > 3

    async def test_feedback_loop_prevention_recall_then_ingest_sanitize(
        self, mock_client: AsyncMock
    ) -> None:
        """Feedback loop prevention: ingest sanitizes recall blocks.

        Given: auto_ingest + auto_recall both enabled with sanitize=True.
            Messages contain a user prompt with a <openviking-recall> block
            from a previous turn's recall injection.
        When: before_model_request is called.
        Then: the ingested content does NOT contain <openviking-recall>
            (sanitized before ingestion), preventing feedback loops.
        """
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        mock_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "uri": "viking://user/alice/memories/doc.md",
                        "score": 0.8,
                        "content": "new memory",
                        "context_type": "memory",
                    }
                ]
            }
        )

        cap = VikingCapability(
            mode="all",
            auto_ingest_enabled=True,
            auto_ingest_mode="sync",
            auto_ingest_sanitize=True,
            auto_recall_enabled=True,
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        # Previous turn has a recall block in the user message (from a prior recall injection)
        recall_block = (
            "<openviking-recall>\n"
            "  <hit uri='viking://user/alice/memories/old.md' score='0.9'/>\n"
            "  old recalled content\n"
            "</openviking-recall>"
        )
        messages = [
            ModelRequest(parts=[UserPromptPart(content=f"Question {recall_block}")]),
            ModelResponse(parts=[TextPart(content="Answer.")]),
            ModelRequest(parts=[UserPromptPart(content="Next question")]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        await cap.before_model_request(ctx, rc)

        # Ingest should have been called (create_session)
        mock_client.create_session.assert_called_once()

        # The first add_message call should contain the user message
        # with the recall block sanitized out
        add_calls = mock_client.add_message.call_args_list
        assert len(add_calls) >= 1
        first_user_content = add_calls[0].args[2]
        assert "<openviking-recall>" not in first_user_content
        assert "[recalled context omitted]" in first_user_content
        assert "old recalled content" not in first_user_content

        # Recall should also have fired (search called)
        mock_client.search.assert_called_once()

    async def test_auto_recall_runs_without_multimodal_bridge(self, mock_client: AsyncMock) -> None:
        """D14 regression: auto_recall fires even when multimodal_bridge=False.

        Given: a VikingCapability with auto_recall_enabled=True and
            multimodal_bridge=False (the default).
        When: before_model_request is called.
        Then: client.search is called — the handler chain runs despite
            multimodal_bridge being disabled. This is the D14 fix: the
            old early return ``if not self.multimodal_bridge ...`` would
            have skipped all handlers.
        """
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        mock_client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "uri": "viking://user/alice/memories/doc.md",
                        "score": 0.8,
                        "content": "memory",
                        "context_type": "memory",
                    }
                ]
            }
        )

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            multimodal_bridge=False,  # explicitly disabled
        )
        cap._client = mock_client
        cap._identity = VikingIdentity(account_id="acct", user_id="alice", role="user")

        msg = ModelRequest(parts=[UserPromptPart(content="test query")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap.before_model_request(ctx, rc)

        # Recall should have fired despite multimodal_bridge=False
        mock_client.search.assert_called_once()
        assert result is not rc  # context was modified by recall

    async def test_client_none_early_return(self) -> None:
        """Client is None: before_model_request returns original context.

        Given: a VikingCapability with auto_recall_enabled=True but
            _client is None (not initialized).
        When: before_model_request is called.
        Then: the original request_context is returned immediately
            without running any handlers.
        """
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        cap = VikingCapability(
            mode="all",
            auto_recall_enabled=True,
            auto_ingest_enabled=True,
            profile_enabled=True,
        )
        cap._client = None  # not initialized

        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        rc = _make_request_context([msg])
        ctx = _make_ctx()

        result = await cap.before_model_request(ctx, rc)

        assert result is rc
