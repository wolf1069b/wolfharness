"""Shared model utilities for AgentPool servers.

This module provides helper functions for extracting provider information,
building provider lists from tokonomics discovery, and merging configured
variants across ACP and OpenCode servers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness.log import get_logger
from wolfharness.models.model_configs import (
    AnthropicModelConfig,
    AnyModelConfig,
    FallbackModelConfig,
    GeminiModelConfig,
    OpenAIModelConfig,
    StringModelConfig,
)
from wolfharness_server.shared.constants import (
    DEFAULT_MODEL_CONTEXT_LIMIT,
    DEFAULT_MODEL_INPUT_COST,
    DEFAULT_MODEL_OUTPUT_COST,
    DEFAULT_MODEL_OUTPUT_LIMIT,
)


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from acp.schema import SessionModelState
    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness_server.acp_server.provider_router import ProviderRouter
    from wolfharness_server.opencode_server.models import Provider

logger = get_logger(__name__)


def _extract_provider_from_identifier(identifier: str) -> str:
    """Extract provider name from a model identifier string.

    Args:
        identifier: Model identifier string (e.g., "openai:gpt-4o")

    Returns:
        Provider name extracted from identifier (e.g., "openai"), or "unknown"
        if no provider prefix found.
    """
    if ":" in identifier:
        return identifier.split(":", 1)[0]
    return "unknown"


def _extract_provider(config: AnyModelConfig) -> str:  # noqa: PLR0911
    """Extract provider name from AnyModelConfig.

    Priority:
    1. Explicit ``config.provider`` field (set in YAML)
    2. Known config types (Anthropic/OpenAI/Gemini)
    3. StringModelConfig: Extract from identifier (e.g., "openai:gpt-4o" -> "openai")
    4. FallbackModelConfig: Provider of first model in chain

    Args:
        config: Model configuration to extract provider from.

    Returns:
        Provider name as a string.
    """
    # Prefer explicit provider field when set
    if config.provider:
        return config.provider

    match config:
        case StringModelConfig(identifier=identifier):
            return _extract_provider_from_identifier(str(identifier))

        case AnthropicModelConfig():
            return "anthropic"

        case OpenAIModelConfig():
            return "openai"

        case GeminiModelConfig():
            return "google"

        case FallbackModelConfig(models=models) if models:
            first = models[0]
            if isinstance(first, str):
                return _extract_provider_from_identifier(first)
            if isinstance(
                first,
                StringModelConfig
                | AnthropicModelConfig
                | OpenAIModelConfig
                | GeminiModelConfig
                | FallbackModelConfig,
            ):
                return _extract_provider(first)
            return "unknown"

        case _:
            return "unknown"


def _resolve_variant_identifier(config: AnyModelConfig, variant_name: str) -> str:
    """Resolve a model variant config to its underlying model identifier.

    Returns the identifier in ``{system}:{model_name}`` format matching
    ``agent.model_name``, so the current model can be matched against
    configured variants.

    For StringModelConfig, resolves through ``config.get_model()`` to get
    the canonical system name (e.g., ``"openai-chat"`` → ``"openai"``).
    Falls back to the raw identifier if resolution fails.

    Args:
        config: Model variant configuration.
        variant_name: The variant name to use as fallback.

    Returns:
        Resolved model identifier string matching agent.model_name format.
    """
    if isinstance(config, StringModelConfig):
        try:
            model = config.get_model()
        except Exception:  # noqa: BLE001
            return config.identifier
        else:
            return f"{model.system}:{model.model_name}"
    return variant_name


def _find_variant_name(
    model_variants: dict[str, AnyModelConfig],
    resolved_model_name: str,
) -> str | None:
    """Find the variant name matching a resolved agent model name.

    Reverse-maps the pydantic-ai model identifier (e.g., ``"openai-chat:svc-v1"``)
    back to the configured variant name (e.g., ``"my_custom"``).

    Args:
        model_variants: Dict of variant name → config from manifest.
        resolved_model_name: The agent's model_name (pydantic-ai resolved identifier).

    Returns:
        Variant name if found, ``None`` otherwise.
    """
    for variant_name, config in model_variants.items():
        try:
            resolved = _resolve_variant_identifier(config, variant_name)
            if resolved == resolved_model_name:
                return variant_name
        except Exception:  # noqa: BLE001
            continue
    return None


def resolve_model_info_from_response(
    model_name: str | None,
    provider_name: str | None,
    model_variants: dict[str, AnyModelConfig],
) -> tuple[str, str]:
    """Resolve (model_id, provider_id) from a pydantic-ai API response.

    Reconstructs the combined identifier that ``_find_variant_name()`` expects
    (``"{provider_name}:{model_name}"``), reverse-maps it to the configured
    variant name, and returns the variant name + configured provider.

    When no variant matches (or inputs are ``None``), falls back to raw names
    with ``"unknown"`` / ``"wolfharness"`` defaults for ``None`` values.

    Args:
        model_name: Raw model name from ``result.response.model_name``
            (e.g. ``"svc/kimi-k2"``).
        provider_name: Raw provider name from ``result.response.provider_name``
            (e.g. ``"openai-chat"``).
        model_variants: Dict of variant name → config from manifest.

    Returns:
        Tuple of ``(model_id, provider_id)`` using variant names when matched.
    """
    if model_name and provider_name and model_variants:
        full_id = f"{provider_name}:{model_name}"
        variant_name = _find_variant_name(model_variants, full_id)
        if variant_name:
            config = model_variants[variant_name]
            return variant_name, _extract_provider(config)
        logger.debug("No variant match for %s", full_id)
    return model_name or "unknown", provider_name or "wolfharness"


def _build_providers_from_tokonomics(toko_models: list[TokoModelInfo]) -> list[Provider]:
    """Build providers list from tokonomics discovery results.

    Groups models by (provider, provider_display_name) and creates Provider
    objects with their associated models.

    Args:
        toko_models: List of tokonomics ModelInfo objects from discovery.

    Returns:
        List of Provider objects with models converted using Model.from_tokonomics().
    """
    from wolfharness_server.opencode_server.models import Model, Provider

    providers_by_name: dict[str, Provider] = {}

    for info in toko_models:
        # Skip embedding models
        if info.is_embedding:
            continue

        provider_id = info.provider

        if provider_id not in providers_by_name:
            providers_by_name[provider_id] = Provider(
                id=provider_id,
                name=provider_id.title(),
                models={},
            )

        model_id = info.id_override or info.id
        providers_by_name[provider_id].models[model_id] = Model.from_tokonomics(info)

    return list(providers_by_name.values())


def _apply_configured_variants(
    providers: list[Provider],
    configured_variants: dict[str, dict[str, Any]],
) -> None:
    """Merge configured variants into providers list.

    Configured variants with matching IDs override discovered models.
    New configured variants are added to their respective providers.

    Args:
        providers: List of Provider objects to modify in place.
        configured_variants: Dictionary mapping variant names to their
            configuration dictionaries. Each config dict should have a
            "provider" key indicating which provider the variant belongs to.

    Note:
        This function modifies the providers list in place. New providers
        are created if a configured variant references a non-existent provider.
    """
    from wolfharness_server.opencode_server.models import (
        Model,
        ModelCost,
        ModelLimit,
        Provider,
        ProviderCapabilities,
    )

    # Build lookup for provider name -> Provider object
    provider_lookup: dict[str, Provider] = {}
    for provider in providers:
        provider_lookup[provider.id.lower()] = provider

    for variant_name, variant_config in configured_variants.items():
        provider_name = variant_config.get("provider", "unknown").lower()

        if provider_name not in provider_lookup:
            # Create new provider entry for this variant
            provider_lookup[provider_name] = Provider(
                id=provider_name,
                name=provider_name.title(),
                models={},
            )
            providers.append(provider_lookup[provider_name])

        provider = provider_lookup[provider_name]

        # Use configured context_length or fall back to defaults
        ctx = variant_config.get("context_length")
        context_limit = float(ctx) if ctx is not None else DEFAULT_MODEL_CONTEXT_LIMIT

        # Check if model with this ID already exists
        if variant_name in provider.models:
            # Override existing (configured takes precedence)
            existing = provider.models[variant_name]
            existing.name = variant_name
            existing.capabilities.attachment = True  # Enable multimodal support
            if ctx is not None:
                existing.limit.context = context_limit
            # Note: variant-specific settings (temp, thinking) not exposed to client
        else:
            # Add new model - use a minimal Model creation
            provider.models[variant_name] = Model(
                id=variant_name,
                name=variant_name,
                capabilities=ProviderCapabilities(attachment=True),
                cost=ModelCost(
                    input=DEFAULT_MODEL_INPUT_COST,
                    output=DEFAULT_MODEL_OUTPUT_COST,
                ),
                limit=ModelLimit(
                    context=context_limit,
                    output=DEFAULT_MODEL_OUTPUT_LIMIT,
                ),
            )


async def build_model_state_for_acp(
    agent: BaseAgent[Any, Any],
    provider_router: ProviderRouter | None,
) -> SessionModelState | None:
    """Build SessionModelState for ACP with configured-first, tokonomics-fallback logic.

    1. Checks agent's pool manifest for configured model_variants
    2. Builds ACPModelInfo list from configured variants
    3. Filters out disabled providers via provider_router
    4. If configured list is non-empty → returns SessionModelState
    5. If empty → falls back to agent.get_available_models() (tokonomics)
    6. If both empty → returns None

    Args:
        agent: The agent to build model state for.
        provider_router: Optional provider router for disable filtering.

    Returns:
        SessionModelState with available models, or None if no models found.
    """
    from acp.schema import ModelInfo as ACPModelInfo, SessionModelState

    # Phase 1: Configured variants from manifest (configured-first)
    configured_models: list[ACPModelInfo] = []
    agent_pool = agent.host_context
    manifest = agent_pool.manifest if agent_pool else None

    if manifest and manifest.model_variants:
        for variant_name, config in manifest.model_variants.items():
            provider_name = _extract_provider(config)

            # Skip if provider is disabled
            if provider_router and provider_router.is_provider_disabled(provider_name):
                continue

            configured_models.append(
                ACPModelInfo(
                    model_id=variant_name,
                    name=variant_name,
                )
            )

    if configured_models:
        current_model = agent.model_name
        # Reverse-map agent's resolved pydantic-ai name to variant name
        if current_model and manifest:
            matched = _find_variant_name(manifest.model_variants, current_model)
            if matched:
                current_model_id = matched
            else:
                # Not a configured variant — add as standalone
                model_info = ACPModelInfo(
                    model_id=current_model,
                    name=current_model,
                    description="Currently configured model",
                )
                configured_models.insert(0, model_info)
                current_model_id = current_model
        else:
            current_model_id = configured_models[0].model_id
        return SessionModelState(
            available_models=configured_models,
            current_model_id=current_model_id,
        )

    # Phase 2: Tokonomics fallback
    try:
        toko_models = await agent.get_available_models()
    except Exception:
        logger.exception("Failed to get available models from agent")
        return None

    if not toko_models:
        return None

    # Filter disabled providers from raw tokonomics models (more accurate than parsing model_id)
    if provider_router:
        toko_models = [
            toko for toko in toko_models if not provider_router.is_provider_disabled(toko.provider)
        ]

    if not toko_models:
        return None

    acp_models_from_tokonomics = [
        ACPModelInfo(
            model_id=toko.id_override if toko.id_override else toko.id,
            name=toko.name,
            description=toko.description or "",
        )
        for toko in toko_models
    ]

    if not acp_models_from_tokonomics:
        return None

    all_ids = [m.model_id for m in acp_models_from_tokonomics]
    current_model = agent.model_name
    current_model_id = current_model if current_model and current_model in all_ids else all_ids[0]

    return SessionModelState(
        available_models=acp_models_from_tokonomics,
        current_model_id=current_model_id,
    )
