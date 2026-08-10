"""Config and provider routes."""

from __future__ import annotations

from collections import defaultdict
import contextlib
from datetime import timedelta
import os
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

from wolfharness.log import get_logger
from wolfharness.models.manifest import AgentsManifest
from wolfharness_server.opencode_server.dependencies import StateDep
from wolfharness_server.opencode_server.models import (
    Config,
    Mode,
    Model,
    ModelCost,
    ModelLimit,
    Provider,
    ProviderCapabilities,
    ProviderListResponse,
    ProvidersResponse,
)
from wolfharness_server.shared.constants import (
    DEFAULT_MODEL_CONTEXT_LIMIT,
    DEFAULT_MODEL_INPUT_COST,
    DEFAULT_MODEL_OUTPUT_COST,
    DEFAULT_MODEL_OUTPUT_LIMIT,
)
from wolfharness_server.shared.model_utils import (
    _build_providers_from_tokonomics,
    _extract_provider,
)


logger = get_logger(__name__)


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo


router = APIRouter(tags=["config"])

DEFAULT_IGNORE = ["node_modules/**", "__pycache__/**", ".venv/**", "*.pyc", ".mypy_cache/**"]
# Provider display names and environment variable mappings
PROVIDER_INFO: dict[str, tuple[str, list[str]]] = {
    "anthropic": ("Anthropic", ["ANTHROPIC_API_KEY"]),
    "openai": ("OpenAI", ["OPENAI_API_KEY"]),
    "google": ("Google", ["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
    "mistral": ("Mistral", ["MISTRAL_API_KEY"]),
    "groq": ("Groq", ["GROQ_API_KEY"]),
    "deepseek": ("DeepSeek", ["DEEPSEEK_API_KEY"]),
    "xai": ("xAI", ["XAI_API_KEY"]),
    "together": ("Together AI", ["TOGETHER_API_KEY"]),
    "perplexity": ("Perplexity", ["PERPLEXITY_API_KEY"]),
    "cohere": ("Cohere", ["COHERE_API_KEY"]),
    "fireworks": ("Fireworks AI", ["FIREWORKS_API_KEY"]),
    "openrouter": ("OpenRouter", ["OPENROUTER_API_KEY"]),
    "bedrock": ("AWS Bedrock", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
    "azure": ("Azure OpenAI", ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]),
    "vertex": ("Google Vertex AI", ["GOOGLE_APPLICATION_CREDENTIALS"]),
}


def _group_models_by_provider(models: list[TokoModelInfo]) -> dict[str, list[TokoModelInfo]]:
    """Group models by their provider."""
    grouped: dict[str, list[TokoModelInfo]] = defaultdict(list)
    for model in models:
        # Skip embedding models - OpenCode is for chat/agent models
        if model.is_embedding:
            continue
        grouped[model.provider].append(model)
    return grouped


def _build_providers(models: list[TokoModelInfo]) -> list[Provider]:
    """Build Provider list from tokonomics models."""
    grouped = _group_models_by_provider(models)
    providers: list[Provider] = []
    for provider_id, provider_models in sorted(grouped.items()):
        # Get provider display info
        display_name, env_vars = PROVIDER_INFO.get(
            provider_id, (provider_id.title(), [f"{provider_id.upper()}_API_KEY"])
        )
        # Convert models to OpenCode format
        models_dict = {i.id_override or i.id: Model.from_tokonomics(i) for i in provider_models}
        provider = Provider(id=provider_id, name=display_name, env=env_vars, models=models_dict)
        providers.append(provider)

    return providers


async def _get_available_models() -> list[TokoModelInfo]:
    """Fetch available models using tokonomics."""
    from tokonomics.model_discovery import get_all_models

    max_age = timedelta(days=7)  # Cache for a week
    return await get_all_models(max_age=max_age)


async def _get_configured_variants(
    manifest: AgentsManifest | None,
) -> dict[str, dict[str, Any]]:
    """Get model variants from manifest configuration.

    Returns variants dict or manifest or model_variants is None/empty.

    Args:
        manifest: The agents manifest containing model_variants configuration.

    Returns:
        Dictionary mapping variant names to their config dicts with provider info.
    """
    variants: dict[str, dict[str, Any]] = {}

    # Check manifest model_variants
    if manifest and manifest.model_variants:
        for name, config in manifest.model_variants.items():
            # Prefer explicit provider field, fall back to identifier parse
            provider = config.provider or _extract_provider(config)
            variants[name] = {
                "provider": provider,
                "context_length": config.context_length,
            }

    return variants


def _build_providers_from_configured(
    configured: dict[str, dict[str, Any]],
) -> list[Provider]:
    """Build providers list from configured variants.

    Args:
        configured: Dictionary mapping variant names to their config dicts.

    Returns:
        List of Provider objects with models grouped by provider.
    """
    providers_by_name: dict[str, Provider] = {}

    for variant_name, variant_config in configured.items():
        provider_name = variant_config.get("provider", "unknown")

        if provider_name not in providers_by_name:
            providers_by_name[provider_name] = Provider(
                id=provider_name.lower(),
                name=provider_name.title(),
                models={},
            )

        # Use configured context_length or fall back to defaults
        ctx = variant_config.get("context_length")
        context_limit = float(ctx) if ctx is not None else DEFAULT_MODEL_CONTEXT_LIMIT

        providers_by_name[provider_name].models[variant_name] = Model(
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

    return list(providers_by_name.values())


def _build_providers_from_variants(
    variants: dict[str, dict[str, object]],
) -> list[Provider]:
    """Build providers list from agent variant modes.

    For agents with thought_level modes (Codex, Claude Code), creates
    a single provider with all variants as models.

    Args:
        variants: Dictionary mapping variant names to their config dicts.

    Returns:
        List of Provider objects containing the variant models.
    """
    # For agent-specific modes, create a single provider with all variants
    return [
        Provider(
            id="agent",
            name="Agent Modes",
            models={
                name: Model(
                    id=name,
                    name=name,
                    capabilities=ProviderCapabilities(attachment=True),
                    cost=ModelCost(
                        input=DEFAULT_MODEL_INPUT_COST,
                        output=DEFAULT_MODEL_OUTPUT_COST,
                    ),
                    limit=ModelLimit(
                        context=DEFAULT_MODEL_CONTEXT_LIMIT,
                        output=DEFAULT_MODEL_OUTPUT_LIMIT,
                    ),
                )
                for name in variants
            },
        )
    ]


async def _build_providers_with_fallback(
    manifest: AgentsManifest | None,
    agent: object | None = None,
) -> list[Provider]:
    """Build providers list with fallback hierarchy.

    1. Primary: Use configured variants from manifest
    2. Secondary: Dynamically discover via tokonomics
    3. Tertiary: Get agent modes (Codex/Claude thought levels)
    4. Last resort: Return empty list with warning

    Args:
        manifest: The agents manifest containing model_variants configuration.
        agent: Optional agent instance to get agent-specific modes from.

    Returns:
        List of Provider objects following the fallback hierarchy.
    """
    # Primary: Configured variants
    configured = await _get_configured_variants(manifest)
    if configured:
        logger.info("Using configured variants from manifest", count=len(configured))
        return _build_providers_from_configured(configured)

    # Secondary: Tokonomics discovery
    try:
        toko_models = await _get_available_models()
        if toko_models:
            logger.debug("Using models from tokonomics discovery", count=len(toko_models))
            return _build_providers_from_tokonomics(toko_models)
    except Exception as e:  # noqa: BLE001
        logger.warning("Tokonomics discovery failed", error=str(e))

    # Tertiary: Agent-specific modes
    if agent:
        agent_variants = await _get_variants_from_agent(agent)
        if agent_variants:
            logger.debug("Using variants from agent modes", count=len(agent_variants))
            return _build_providers_from_variants(agent_variants)

    # Last resort: Empty with warning
    logger.warning("No model variants configured and no models available from discovery")
    logger.warning("No model variants configured and no models available from discovery")
    return []


def _get_dummy_providers() -> list[Provider]:
    """Return a single dummy provider for testing."""
    dummy_model = Model(
        id="gpt-4o",
        name="GPT-4o",
        capabilities=ProviderCapabilities(
            attachment=True,
            reasoning=False,
            temperature=True,
            tool_call=True,
        ),
        cost=ModelCost(input=5.0, output=15.0),
        limit=ModelLimit(context=128000.0, output=4096.0),
        release_date="2024-05-13",
    )
    dummy_provider = Provider(
        id="openai",
        name="OpenAI",
        env=["OPENAI_API_KEY"],
        models={"gpt-4o": dummy_model},
    )
    return [dummy_provider]


def _infer_default_model(state: StateDep) -> str | None:
    """Derive the default ``provider/model`` string from the manifest.

    Looks up the ``default_agent``'s model in the manifest.  If the model is
    a variant name defined in ``model_variants``, returns
    ``"<provider_id>/<variant_name>"`` so the TUI selects the right entry.
    """
    try:
        manifest = state.pool.manifest
    except (AttributeError, RuntimeError):
        return None
    if not manifest or not manifest.model_variants:
        return None

    # Find the default agent's model setting.
    default_name = manifest.default_agent
    agent_cfg = (manifest.agents or {}).get(default_name or "")
    if agent_cfg is None:
        return None

    # agent_cfg.model may be a raw str or a structured config (e.g.
    # StringModelConfig with identifier).  The identifier itself can be
    # either a variant name ("glm47") or a full model id ("openai-chat:svc/glm-4.7").
    from wolfharness.models.agents import NativeAgentConfig

    if not isinstance(agent_cfg, NativeAgentConfig):
        return None
    agent_model = agent_cfg.model
    variant_name: str | None = None

    raw_id: str | None = None
    if isinstance(agent_model, str):
        raw_id = agent_model
    elif hasattr(agent_model, "identifier"):
        raw_id = str(agent_model.identifier)

    if raw_id is not None:
        if raw_id in manifest.model_variants:
            # Identifier is a variant name (most common case: model: glm47).
            variant_name = raw_id
        else:
            # Identifier is a full model id — reverse-lookup in variants.
            for vname, vcfg in manifest.model_variants.items():
                if hasattr(vcfg, "identifier") and str(vcfg.identifier) == raw_id:
                    variant_name = vname
                    break

    if variant_name is None:
        return None

    provider_id = _extract_provider(manifest.model_variants[variant_name])
    return f"{provider_id}/{variant_name}"


@router.get("/config")
async def get_config(state: StateDep) -> Config:
    """Get server configuration."""
    from wolfharness_server.opencode_server.models.config import Keybinds, WatcherConfig

    # Initialize config if not yet set
    if state.config is None:
        state.config = Config()

    # Ensure keybinds are set with defaults
    if state.config.keybinds is None:
        state.config.keybinds = Keybinds()

    # Ensure watcher config is set with sensible defaults
    if state.config.watcher is None:
        state.config.watcher = WatcherConfig(ignore=DEFAULT_IGNORE)

    # Set a default model if not already configured.
    # Priority: default agent's model variant > tokonomics discovery.
    if state.config.model is None:
        state.config.model = _infer_default_model(state)

        if state.config.model is None:
            try:
                toko_models = await state.agent.get_available_models()
                if toko_models:
                    providers = _build_providers(toko_models)
                    for provider in providers:
                        if any(os.environ.get(env) for env in provider.env) and provider.models:
                            first_model = next(iter(provider.models.keys()))
                            state.config.model = f"{provider.id}/{first_model}"
                            break
            except Exception:  # noqa: BLE001
                pass

    return state.config


@router.patch("/config")
async def update_config(state: StateDep, config_update: Config) -> Config:
    """Update server configuration.

    Only updates fields that are provided (non-None).
    Returns the complete updated config.
    """
    # Initialize config if not yet set
    if state.config is None:
        state.config = Config()

    # Update only the fields that were provided
    update_data = config_update.model_dump(exclude_unset=True)

    # Sync model change to agents if provided
    if "model" in update_data and update_data["model"] is not None:
        new_model = update_data["model"]
        logger.info("PATCH /config received model update", model=new_model)
        if state.agent is not None:
            try:
                # Update the shared/template agent — no agent_lock needed with
                # per-session agents: each session has its own agent instance.
                logger.info("Calling agent.set_model", model=new_model)
                await state.agent.set_model(new_model)
                logger.info("Agent model successfully updated", model=new_model)
            except Exception as e:
                logger.warning("Failed to update agent model", error=str(e))
                logger.exception("Traceback while updating agent model")
        else:
            logger.warning("state.agent is None, cannot update model")

    for field_name, value in update_data.items():
        setattr(state.config, field_name, value)

    return state.config


async def _get_variants_from_agent(agent: object) -> dict[str, dict[str, object]]:
    """Get variants from agent's thought_level modes.

    Only supported for Codex and Claude Code agents which have static,
    known thought_level modes.

    Args:
        agent: The agent to get modes from

    Returns:
        Dict mapping variant names to empty config dicts (config is agent-internal)
    """
    return {}


@router.get("/global/config")
async def get_global_config(state: StateDep) -> Config:
    """Get server configuration (global alias for OpenCode 1.4.4+ compat)."""
    return await get_config(state)


@router.patch("/global/config")
async def update_global_config(state: StateDep, config_update: Config) -> Config:
    """Update server configuration (global alias for OpenCode 1.4.4+ compat)."""
    return await update_config(state, config_update)


@router.get("/config/providers")
async def get_providers(state: StateDep) -> ProvidersResponse:
    """Get available providers and models from agent."""
    # Get manifest from agent pool (may be None if not loaded)
    manifest: AgentsManifest | None = None
    with contextlib.suppress(AttributeError, RuntimeError):
        manifest = state.pool.manifest

    # Build providers using fallback hierarchy
    providers = await _build_providers_with_fallback(manifest, state.agent)

    # Build default models map: use first model for each connected provider
    default_models: dict[str, str] = {}
    connected_providers = [
        provider.id for provider in providers if any(os.environ.get(env) for env in provider.env)
    ]

    for provider in providers:
        if provider.id in connected_providers and provider.models:
            # Simply use the first available model
            default_models[provider.id] = next(iter(provider.models.keys()))

    return ProvidersResponse(providers=providers, default=default_models)


@router.get("/provider")
async def list_providers(state: StateDep) -> ProviderListResponse:
    """List all providers."""
    # Get manifest from agent pool (may be None if not loaded)
    manifest: AgentsManifest | None = None
    with contextlib.suppress(AttributeError, RuntimeError):
        manifest = state.pool.manifest

    # Build providers using fallback hierarchy
    providers = await _build_providers_with_fallback(manifest, state.agent)

    # Determine which providers are "connected" based on env vars
    connected = [
        provider.id for provider in providers if any(os.environ.get(env) for env in provider.env)
    ]

    # Build default models map: use first model for each connected provider
    default_models: dict[str, str] = {}
    for provider in providers:
        if provider.id in connected and provider.models:
            # Simply use the first available model
            default_models[provider.id] = next(iter(provider.models.keys()))

    return ProviderListResponse(
        all=providers,
        default=default_models,
        connected=connected,
    )


@router.get("/mode")
async def list_modes(state: StateDep) -> list[Mode]:
    """List available modes dynamically from agent."""
    if state.agent is None:
        return [Mode(name="default", tools={})]
    try:
        mode_categories = await state.agent.get_modes()
    except Exception:
        logger.exception("Failed to get modes from agent")
        return [Mode(name="default", tools={})]

    if not mode_categories:
        return [Mode(name="default", tools={})]

    # Find the mode category (permissions/behavior modes, not model selectors)
    category = next(
        (c for c in mode_categories if c.id == "mode" or c.category == "mode"),
        None,
    )
    if not category or not category.available_modes:
        return [Mode(name="default", tools={})]

    return [
        Mode(
            name=mode.id,
            tools={},
        )
        for mode in category.available_modes
    ]
