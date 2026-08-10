"""Model capabilities configuration.

Defines the ``ModelCapabilities`` Pydantic model that declares which
multimodal input/output modalities a model supports. When attached to a
``BaseModelConfig``, it allows explicit override of capabilities that
would otherwise be discovered at runtime via tokonomics.

A field set to ``None`` (the default) means "not specified — query
tokonomics at runtime". A field set to ``True`` or ``False`` is an
explicit override that bypasses runtime discovery.

Example YAML::

    agents:
      my_agent:
        type: native
        model:
          type: openai
          model: gpt-4o
          capabilities:
            image_input: true
            audio_input: false
"""

from __future__ import annotations

from pydantic import ConfigDict, Field
from schemez import Schema


class ModelCapabilities(Schema):
    """Declare multimodal input/output capabilities for a model.

    Each field is a tri-state: ``None`` (not specified, defer to
    tokonomics runtime discovery), ``True`` (explicitly supported),
    or ``False`` (explicitly unsupported).

    Attributes:
        image_input: Whether the model accepts image inputs.
        audio_input: Whether the model accepts audio inputs.
        video_input: Whether the model accepts video inputs.
        document_input: Whether the model accepts document inputs.
        image_output: Whether the model can produce image outputs.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "x-icon": "octicon:eye-16",
            "x-doc-title": "Model Capabilities",
        },
    )

    image_input: bool | None = Field(
        default=None,
        title="Image input",
        description="Whether the model accepts image inputs.",
    )
    """Whether the model accepts image inputs."""

    audio_input: bool | None = Field(
        default=None,
        title="Audio input",
        description="Whether the model accepts audio inputs.",
    )
    """Whether the model accepts audio inputs."""

    video_input: bool | None = Field(
        default=None,
        title="Video input",
        description="Whether the model accepts video inputs.",
    )
    """Whether the model accepts video inputs."""

    document_input: bool | None = Field(
        default=None,
        title="Document input",
        description="Whether the model accepts document inputs.",
    )
    """Whether the model accepts document inputs."""

    image_output: bool | None = Field(
        default=None,
        title="Image output",
        description="Whether the model can produce image outputs.",
    )
    """Whether the model can produce image outputs."""


__all__ = ["ModelCapabilities"]
