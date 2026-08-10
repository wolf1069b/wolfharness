"""Unit + integration tests for AgentFactory modality capability resolution.

Tests cover:
- 5.5: image_output mapping to pydantic-ai Model profile
- 5.6: User-configured ModalityFilterCapability is populated with resolved caps
- FallbackModelConfig intersection (pessimistic) for capability resolution
- _model_config_names and _intersect_capabilities helpers
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness.agents.native_agent.agent import (
    Agent,
    _intersect_capabilities,
    _model_config_names,
)
from wolfharness.capabilities.modality_filter import ModalityFilterCapability
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.model_configs import (
    FallbackModelConfig,
    OpenAIModelConfig,
    StringModelConfig,
    TestModelConfig as _TestModelConfig,
)
from wolfharness_config.model_capabilities import ModelCapabilities


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _caps(
    *,
    image_input: bool | None = None,
    audio_input: bool | None = None,
    video_input: bool | None = None,
    document_input: bool | None = None,
    image_output: bool | None = None,
) -> ModelCapabilities:
    """Build a ModelCapabilities instance."""
    return ModelCapabilities(
        image_input=image_input,
        audio_input=audio_input,
        video_input=video_input,
        document_input=document_input,
        image_output=image_output,
    )


def _all_true_caps() -> ModelCapabilities:
    """Capabilities with all modalities enabled."""
    return ModelCapabilities(
        image_input=True,
        audio_input=True,
        video_input=True,
        document_input=True,
        image_output=True,
    )


def _text_only_caps() -> ModelCapabilities:
    """Capabilities with all multimodal inputs disabled."""
    return ModelCapabilities(
        image_input=False,
        audio_input=False,
        video_input=False,
        document_input=False,
        image_output=False,
    )


# ---------------------------------------------------------------------------
# 5.5 — image_output mapping to pydantic-ai Model profile
# ---------------------------------------------------------------------------


async def test_image_output_true_flows_to_model_profile() -> None:
    """image_output=True should set supports_image_output in model profile."""
    agent = Agent(name="test", model="test")
    caps = _caps(image_output=True)
    model = TestModel()
    result = agent._apply_image_output_profile(model, caps)
    assert result is model
    assert result.profile.get("supports_image_output") is True


async def test_image_output_false_flows_to_model_profile() -> None:
    """image_output=False should set supports_image_output=False in profile."""
    agent = Agent(name="test", model="test")
    caps = _caps(image_output=False)
    model = TestModel()
    result = agent._apply_image_output_profile(model, caps)
    assert result is model
    assert result.profile.get("supports_image_output") is False


async def test_image_output_none_does_not_modify_profile() -> None:
    """image_output=None should not modify the model profile."""
    agent = Agent(name="test", model="test")
    caps = _caps(image_output=None)
    model = TestModel()
    original_profile = model._profile
    result = agent._apply_image_output_profile(model, caps)
    assert result is model
    assert result._profile is original_profile


async def test_image_output_merges_with_existing_profile() -> None:
    """image_output should merge into an existing dict profile."""
    agent = Agent(name="test", model="test")
    caps = _caps(image_output=True)
    model = TestModel()
    # Set an existing profile dict.
    model._profile = {"supports_thinking": True}
    result = agent._apply_image_output_profile(model, caps)
    assert result.profile.get("supports_thinking") is True
    assert result.profile.get("supports_image_output") is True


# ---------------------------------------------------------------------------
# 5.6 — User-configured ModalityFilterCapability populated with resolved caps
# ---------------------------------------------------------------------------


async def test_user_config_modality_filter_populated_with_resolved_caps() -> None:
    """When user pre-configures a ModalityFilterCapability, it should be populated.

    The user must explicitly add a ModalityFilterCapability to the agent's
    capabilities list. The factory populates it with resolved capabilities
    but does NOT auto-inject one if the user hasn't configured it.
    """
    user_filter = ModalityFilterCapability(
        capabilities=_caps(image_input=False),
        image_strategy="drop",
    )
    config = NativeAgentConfig(
        model=_TestModelConfig(capabilities=_text_only_caps()),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[user_filter],
    )
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    filter_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, ModalityFilterCapability)
    ]
    # Should have exactly one (the user's instance, not auto-injected).
    assert len(filter_caps) == 1
    # The user's strategy settings are preserved.
    assert filter_caps[0].image_strategy == "drop"
    # Capabilities were populated by the factory.
    assert filter_caps[0].capabilities.image_input is False
    assert filter_caps[0].capabilities.audio_input is False
    assert filter_caps[0].capabilities.video_input is False
    assert filter_caps[0].capabilities.document_input is False


async def test_no_modality_filter_when_not_configured() -> None:
    """get_agentlet should NOT inject ModalityFilterCapability when not configured.

    Even if the model lacks input modalities, the capability is only
    activated when the user explicitly configures ``type: modality_filter``
    in the agent's capabilities list.
    """
    config = NativeAgentConfig(
        model=_TestModelConfig(capabilities=_text_only_caps()),
    )
    agent = Agent(name="test", model="test", agent_config=config)
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    filter_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, ModalityFilterCapability)
    ]
    assert len(filter_caps) == 0


async def test_no_inject_when_no_config() -> None:
    """get_agentlet should NOT inject when no agent_config is provided."""
    agent = Agent(name="test", model="test")
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    filter_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, ModalityFilterCapability)
    ]
    assert len(filter_caps) == 0


# ---------------------------------------------------------------------------
# VikingCapability.model_capabilities population
# ---------------------------------------------------------------------------


async def test_viking_cap_populated_with_resolved_caps() -> None:
    """VikingCapability.model_capabilities is populated by the factory.

    A user-configured VikingCapability must receive the resolved model
    capabilities so ``viking_read`` can auto-detect image-byte returns.
    """
    from wolfharness.capabilities.viking import VikingCapability

    viking = VikingCapability(mode="all")
    config = NativeAgentConfig(
        model=_TestModelConfig(capabilities=_all_true_caps()),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[viking],
    )
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    viking_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, VikingCapability)
    ]
    assert len(viking_caps) == 1
    assert viking_caps[0].model_capabilities is not None
    assert viking_caps[0].model_capabilities.image_input is True
    assert viking_caps[0]._should_return_image_bytes() is True


async def test_viking_cap_text_only_model_returns_false() -> None:
    """Text-only model resolution prevents image-byte returns in viking_read."""
    from wolfharness.capabilities.viking import VikingCapability

    viking = VikingCapability(mode="all")
    config = NativeAgentConfig(
        model=_TestModelConfig(capabilities=_text_only_caps()),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[viking],
    )
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    viking_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, VikingCapability)
    ]
    assert len(viking_caps) == 1
    assert viking_caps[0]._should_return_image_bytes() is False


async def test_viking_cap_no_model_name_gets_all_none_caps() -> None:
    """No resolvable model name injects ModelCapabilities() (all None fields).

    _should_return_image_bytes must degrade to text-only (safe default),
    even though model_capabilities is a non-None object.
    """
    from wolfharness.capabilities.viking import VikingCapability

    viking = VikingCapability(mode="all")
    agent = Agent(
        name="test",
        model="test",
        capabilities=[viking],
    )
    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    viking_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, VikingCapability)
    ]
    assert len(viking_caps) == 1
    # Injected (non-None), but all fields None → text-only degradation.
    assert viking_caps[0].model_capabilities is not None
    assert viking_caps[0].model_capabilities.image_input is None
    assert viking_caps[0]._should_return_image_bytes() is False


async def test_viking_injected_caps_feed_multimodal_bridge() -> None:
    """Injected ModelCapabilities also drive multimodal bridge modality checks.

    _supports_modality must reflect the resolved model: True for a vision
    model (bridge keeps images as HTTP URLs), False for text-only (bridge
    replaces images with viking:// text links).
    """
    from wolfharness.capabilities.viking import VikingCapability

    for caps in (_all_true_caps(), _text_only_caps()):
        viking = VikingCapability(mode="all", multimodal_bridge=True)
        config = NativeAgentConfig(
            model=_TestModelConfig(capabilities=caps),
        )
        agent = Agent(
            name="test",
            model="test",
            agent_config=config,
            capabilities=[viking],
        )
        pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
        viking_caps = [
            cap
            for cap in pydantic_agent.root_capability.capabilities
            if isinstance(cap, VikingCapability)
        ]
        assert len(viking_caps) == 1
        cap = viking_caps[0]
        # Bridge modality check follows resolved capabilities.
        assert cap._supports_modality("image/png") is bool(caps.image_input)

        # A for_run() copy must preserve the injected caps so the bridge
        # sees them inside a run.
        from unittest.mock import MagicMock

        copy_cap = await cap.for_run(MagicMock())  # type: ignore[arg-type]
        assert copy_cap._supports_modality("image/png") is bool(caps.image_input)


# ---------------------------------------------------------------------------
# 5.6 — FallbackModelConfig intersection (pessimistic)
# ---------------------------------------------------------------------------


def test_intersect_capabilities_pessimistic() -> None:
    """Intersection: False wins over True for each modality."""
    caps_a = ModelCapabilities(
        image_input=True,
        audio_input=True,
        video_input=False,
        document_input=True,
        image_output=True,
    )
    caps_b = ModelCapabilities(
        image_input=False,
        audio_input=True,
        video_input=False,
        document_input=True,
        image_output=False,
    )
    result = _intersect_capabilities([caps_a, caps_b])
    assert result.image_input is False  # a=True, b=False -> False
    assert result.audio_input is True  # both True
    assert result.video_input is False  # both False
    assert result.document_input is True
    assert result.image_output is False


def test_intersect_capabilities_all_true() -> None:
    """Intersection: all True -> all True."""
    caps_a = _all_true_caps()
    caps_b = _all_true_caps()
    result = _intersect_capabilities([caps_a, caps_b])
    assert result.image_input is True
    assert result.audio_input is True
    assert result.video_input is True
    assert result.document_input is True
    assert result.image_output is True


# ---------------------------------------------------------------------------
# 5.6 — _model_config_names helper
# ---------------------------------------------------------------------------


def test_model_config_names_string_config() -> None:
    """StringModelConfig should return its identifier."""
    config = StringModelConfig(identifier="openai:gpt-4o")
    names = _model_config_names(config)
    assert names == ["openai:gpt-4o"]


def test_model_config_names_openai_config() -> None:
    """OpenAIModelConfig should return its identifier."""
    config = OpenAIModelConfig(identifier="gpt-5-pro")
    names = _model_config_names(config)
    assert names == ["gpt-5-pro"]


def test_model_config_names_test_config() -> None:
    """_TestModelConfig has no identifier -> empty list."""
    config = _TestModelConfig()
    names = _model_config_names(config)
    assert names == []


def test_model_config_names_fallback_with_strings() -> None:
    """FallbackModelConfig with plain string sub-models."""
    config = FallbackModelConfig(models=["openai:gpt-4o", "anthropic:claude-sonnet-4-5"])
    names = _model_config_names(config)
    assert names == ["openai:gpt-4o", "anthropic:claude-sonnet-4-5"]


def test_model_config_names_fallback_with_configs() -> None:
    """FallbackModelConfig with nested BaseModelConfig sub-models."""
    config = FallbackModelConfig(
        models=[
            StringModelConfig(identifier="openai:gpt-4o"),
            OpenAIModelConfig(identifier="gpt-5-pro"),
        ],
    )
    names = _model_config_names(config)
    assert names == ["openai:gpt-4o", "gpt-5-pro"]


# ---------------------------------------------------------------------------
# 5.6 — Integration: resolve_capabilities called in get_agentlet
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_resolve_capabilities_called_for_string_model() -> None:
    """get_agentlet should call resolve_capabilities with the model name."""
    config = NativeAgentConfig(
        model=StringModelConfig(
            identifier="openai:gpt-4o",
            capabilities=_caps(image_input=False),
        ),
    )
    agent = Agent(name="test", model="test", agent_config=config)

    # Mock resolve_capabilities to avoid real tokonomics calls.
    mock_caps = _text_only_caps()
    with patch(
        "wolfharness.host.stubs.resolve_capabilities",
        new_callable=AsyncMock,
        return_value=mock_caps,
    ) as mock_resolve:
        await agent.get_agentlet(model=None, output_type=str)
        mock_resolve.assert_called_once()
        # First arg should be the model name.
        call_args = mock_resolve.call_args
        assert call_args[0][0] == "openai:gpt-4o"


@pytest.mark.integration
async def test_fallback_model_does_not_call_resolve_per_sub_model() -> None:
    """FallbackModelConfig should NOT call resolve_capabilities per sub-model.

    Fallback models use declared capabilities directly without per-model
    cache lookups or intersection.
    """
    config = NativeAgentConfig(
        model=FallbackModelConfig(
            models=["openai:gpt-4o", "anthropic:claude-sonnet-4-5"],
            capabilities=_caps(),
        ),
    )
    agent = Agent(name="test", model="test", agent_config=config)

    with patch(
        "wolfharness.host.stubs.resolve_capabilities",
        new_callable=AsyncMock,
    ) as mock_resolve:
        await agent.get_agentlet(model=None, output_type=str)
        # FallbackModelConfig returns declared caps directly — no per-model resolution.
        assert mock_resolve.call_count == 0


@pytest.mark.integration
async def test_fallback_model_user_filter_gets_declared_caps() -> None:
    """Fallback with user-configured modality_filter: gets declared caps directly.

    No per-model intersection is computed. When declared caps are all None
    (default), the ModalityFilterCapability receives all-None capabilities.
    _is_modality_supported() treats None as unsupported (text-only behavior).
    """
    user_filter = ModalityFilterCapability(
        capabilities=None,
        image_strategy="describe",
    )
    config = NativeAgentConfig(
        model=FallbackModelConfig(
            models=["openai:gpt-4o", "openai:gpt-3.5-turbo"],
            capabilities=_caps(),
        ),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[user_filter],
    )

    with patch(
        "wolfharness.host.stubs.resolve_capabilities",
        new_callable=AsyncMock,
    ):
        pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
        filter_caps = [
            cap
            for cap in pydantic_agent.root_capability.capabilities
            if isinstance(cap, ModalityFilterCapability)
        ]
        assert len(filter_caps) == 1
        # Declared caps are all None (from _caps()) — no intersection.
        assert filter_caps[0].capabilities is not None
        assert filter_caps[0].capabilities.image_input is None


# ---------------------------------------------------------------------------
# Cache-only resolution: no tokonomics queries at agent creation
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_agent_creation_no_tokonomics_query_for_custom_model() -> None:
    """Agent creation with a custom model should not initiate tokonomics queries.

    No error log, no network call — defaults to text-only caps.
    """
    config = NativeAgentConfig(
        model=StringModelConfig(identifier="wolf-ai:kimi-k2"),
    )
    agent = Agent(name="test", model="test", agent_config=config)

    with patch(
        "wolfharness.host.stubs.CapabilityCache.get_capability",
        new_callable=AsyncMock,
    ) as mock_get:
        pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
        # get_capability (async, initiates queries) must NOT be called.
        mock_get.assert_not_called()
        # No ModalityFilterCapability configured, so none should exist.
        filter_caps = [
            cap
            for cap in pydantic_agent.root_capability.capabilities
            if isinstance(cap, ModalityFilterCapability)
        ]
        assert len(filter_caps) == 0


@pytest.mark.integration
async def test_agent_creation_uses_cached_values() -> None:
    """Agent creation should use cached capability values when available."""
    from wolfharness.host.stubs import _get_default_cache

    cache = _get_default_cache()
    # Pre-populate cache for the model.
    cache._cache["openai:gpt-4o:image_input"] = True
    cache._cache["openai:gpt-4o:audio_input"] = True
    cache._cache["openai:gpt-4o:video_input"] = False
    cache._cache["openai:gpt-4o:document_input"] = False
    cache._cache["openai:gpt-4o:image_output"] = False

    config = NativeAgentConfig(
        model=StringModelConfig(identifier="openai:gpt-4o"),
    )
    user_filter = ModalityFilterCapability(capabilities=None)
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        capabilities=[user_filter],
    )

    pydantic_agent = await agent.get_agentlet(model=None, output_type=str)
    filter_caps = [
        cap
        for cap in pydantic_agent.root_capability.capabilities
        if isinstance(cap, ModalityFilterCapability)
    ]
    assert len(filter_caps) == 1
    assert filter_caps[0].capabilities.image_input is True
    assert filter_caps[0].capabilities.audio_input is True
    assert filter_caps[0].capabilities.video_input is False


# ---------------------------------------------------------------------------
# Model variant capability resolution
# ---------------------------------------------------------------------------


def test_get_declared_capabilities_from_model_variant() -> None:
    """_get_declared_capabilities reads from resolved_model_config (variant)."""
    variant_caps = ModelCapabilities(image_input=True, audio_input=False)
    variant_config = StringModelConfig(
        identifier="openai:gpt-4o",
        capabilities=variant_caps,
    )
    config = NativeAgentConfig(
        model=StringModelConfig(identifier="my_variant"),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        resolved_model_config=variant_config,
    )
    caps = agent._get_declared_capabilities()
    assert caps is not None
    assert caps.image_input is True
    assert caps.audio_input is False


def test_get_declared_capabilities_variant_without_caps() -> None:
    """Variant without capabilities returns None."""
    variant_config = StringModelConfig(identifier="openai:gpt-4o")
    config = NativeAgentConfig(
        model=StringModelConfig(identifier="my_variant"),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        resolved_model_config=variant_config,
    )
    caps = agent._get_declared_capabilities()
    assert caps is None


def test_get_declared_capabilities_falls_back_to_config_model() -> None:
    """When _resolved_model_config is None, falls back to self.config.model."""
    config = NativeAgentConfig(
        model=StringModelConfig(
            identifier="openai:gpt-4o",
            capabilities=ModelCapabilities(image_input=True),
        ),
    )
    agent = Agent(
        name="test",
        model="test",
        agent_config=config,
        resolved_model_config=None,
    )
    caps = agent._get_declared_capabilities()
    assert caps is not None
    assert caps.image_input is True


# ---------------------------------------------------------------------------
# _apply_image_output_profile with text-only defaults
# ---------------------------------------------------------------------------


async def test_apply_image_output_profile_false_no_crash() -> None:
    """_apply_image_output_profile with image_output=False should not crash."""
    agent = Agent(name="test", model="test")
    caps = ModelCapabilities(image_output=False)
    model = TestModel()
    result = agent._apply_image_output_profile(model, caps)
    assert result is model
    assert result.profile.get("supports_image_output") is False
