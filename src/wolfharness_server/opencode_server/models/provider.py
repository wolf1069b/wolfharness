"""Provider, model, and mode related models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, model_validator

from wolfharness_server.opencode_server.models.base import OpenCodeBaseModel
from wolfharness_server.opencode_server.models.common import ModelRef  # noqa: TC001


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from wolfharness_config.model_capabilities import ModelCapabilities

logger = logging.getLogger(__name__)


class CostCache(OpenCodeBaseModel):
    """Cache cost information."""

    read: float = 0.0
    write: float = 0.0


class ModelCost(OpenCodeBaseModel):
    """Cost information for a model."""

    input: float
    output: float
    cache: CostCache = Field(default_factory=CostCache)


class ProviderModalities(OpenCodeBaseModel):
    """Modalities supported by a model (boolean flags).

    Matches opencode's ProviderModalities schema where each modality
    is a boolean flag rather than a list of strings.
    """

    text: bool = True
    audio: bool = False
    image: bool = False
    video: bool = False
    pdf: bool = False


class ProviderApiInfo(OpenCodeBaseModel):
    """API connection information for a model's provider."""

    id: str = ""
    url: str = ""
    npm: str = ""


class ProviderCapabilities(OpenCodeBaseModel):
    """Model capabilities.

    Matches opencode's ProviderCapabilities schema with nested
    input/output modalities as boolean-flag objects and an
    interleaved field for thinking interleaving support.
    """

    attachment: bool = False
    reasoning: bool = False
    temperature: bool = True
    tool_call: bool = Field(default=True, alias="toolcall")
    input: ProviderModalities = Field(default_factory=ProviderModalities)
    output: ProviderModalities = Field(default_factory=ProviderModalities)
    interleaved: bool = False


class ModelLimit(OpenCodeBaseModel):
    """Limit information for a model."""

    context: float
    output: float


class Model(OpenCodeBaseModel):
    """Model information."""

    id: str
    name: str
    provider_id: str = ""
    api: ProviderApiInfo = Field(default_factory=ProviderApiInfo)
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    cost: ModelCost
    limit: ModelLimit
    status: str = "active"
    options: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    release_date: str = ""
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Model variants for reasoning/thinking levels.

    Maps variant names (e.g., 'low', 'medium', 'high', 'max') to
    provider-specific configuration options. The TUI uses this to
    let users cycle through thinking effort levels.
    """

    @classmethod
    def from_tokonomics(
        cls,
        model: TokoModelInfo,
        *,
        capabilities_override: ModelCapabilities | None = None,
    ) -> Self:
        """Convert a tokonomics ModelInfo to an OpenCode Model.

        Args:
            model: The tokonomics ModelInfo to convert.
            capabilities_override: Optional config-driven capabilities that
                override tokonomics-derived modality values. Each field set
                to ``True`` or ``False`` takes precedence; ``None`` fields
                defer to tokonomics runtime discovery.
        """
        from tokonomics.model_discovery.model_info import ModelPricing

        pricing = model.pricing or ModelPricing()
        cost = ModelCost(
            input=(pricing.prompt * 1_000_000) if pricing.prompt else 0.0,
            output=(pricing.completion * 1_000_000) if pricing.completion else 0.0,
            cache=CostCache(
                read=(pricing.input_cache_read * 1_000_000) if pricing.input_cache_read else 0.0,
                write=(pricing.input_cache_write * 1_000_000) if pricing.input_cache_write else 0.0,
            ),
        )
        # Convert limits
        context = float(model.context_window) if model.context_window else 128000.0
        output = float(model.max_output_tokens) if model.max_output_tokens else 4096.0
        # Build modalities from tokonomics data (convert to boolean flags)
        input_mods = [str(m) for m in model.input_modalities] if model.input_modalities else []
        output_mods = [str(m) for m in model.output_modalities] if model.output_modalities else []
        # Start with tokonomics-derived values
        input_image = "image" in input_mods
        input_audio = "audio" in input_mods
        input_video = "video" in input_mods
        input_pdf = "pdf" in input_mods or "file" in input_mods
        output_image = "image" in output_mods
        output_audio = "audio" in output_mods
        output_video = "video" in output_mods
        output_pdf = "pdf" in output_mods or "file" in output_mods
        # Apply overrides from ModelCapabilities (only when explicitly set)
        if capabilities_override is not None:
            if capabilities_override.image_input is not None:
                input_image = capabilities_override.image_input
            if capabilities_override.audio_input is not None:
                input_audio = capabilities_override.audio_input
            if capabilities_override.video_input is not None:
                input_video = capabilities_override.video_input
            if capabilities_override.document_input is not None:
                input_pdf = capabilities_override.document_input
            if capabilities_override.image_output is not None:
                output_image = capabilities_override.image_output
        # Use id_override if available (e.g., "opus" for Claude Code SDK)
        instance = cls(
            id=model.id_override or model.id,
            name=model.name,
            capabilities=ProviderCapabilities(
                attachment=False,
                reasoning="reasoning" in output_mods or "thinking" in model.name.lower(),
                temperature=True,
                input=ProviderModalities(
                    text=True,
                    audio=input_audio,
                    image=input_image,
                    video=input_video,
                    pdf=input_pdf,
                ),
                output=ProviderModalities(
                    text=True,
                    audio=output_audio,
                    image=output_image,
                    video=output_video,
                    pdf=output_pdf,
                ),
            ),
            cost=cost,
            limit=ModelLimit(context=context, output=output),
            release_date=model.created_at.strftime("%Y-%m-%d") if model.created_at else "",
        )

        # Passively populate the CapabilityCache from this ModelInfo.
        # Zero additional network requests — data is already available.
        try:
            from wolfharness.host.stubs import _get_default_cache

            _get_default_cache().populate_cache_from_model_info(model)
        except Exception:  # noqa: BLE001
            logger.debug("populate_cache_failed: %s", model.id)

        return instance


class Provider(OpenCodeBaseModel):
    """Provider information.

    Matches opencode's Provider.Info schema which requires source and options.
    """

    id: str
    name: str
    source: str = "config"
    """Provider source: 'env', 'config', 'custom', or 'api'."""

    env: list[str] = Field(default_factory=list)
    key: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Model] = Field(default_factory=dict)
    api: str | None = None
    npm: str | None = None

    @model_validator(mode="after")
    def _populate_model_refs(self) -> Self:
        """Auto-populate provider_id and api on all models.

        Updates fields individually to preserve existing data like url.
        """
        for model in self.models.values():
            model.provider_id = self.id
            # Only populate missing fields to avoid overwriting existing data
            if not model.api.id:
                model.api.id = self.api or ""
            if not model.api.npm:
                model.api.npm = self.npm or ""
        return self


class ProvidersResponse(OpenCodeBaseModel):
    """Response for /config/providers endpoint."""

    providers: list[Provider]
    default: dict[str, str] = Field(default_factory=dict)


class ProviderListResponse(OpenCodeBaseModel):
    """Response for /provider endpoint."""

    all: list[Provider]
    default: dict[str, str] = Field(default_factory=dict)
    connected: list[str] = Field(default_factory=list)


class Mode(OpenCodeBaseModel):
    """Agent mode configuration."""

    name: str
    tools: dict[str, bool] = Field(default_factory=dict)
    model: ModelRef | None = None
    prompt: str | None = None
    temperature: float | None = None
