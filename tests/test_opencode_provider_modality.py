"""Tests for Model.from_tokonomics() modality override via ModelCapabilities."""

from __future__ import annotations

from typing import Any

import pytest
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing

from wolfharness_config.model_capabilities import ModelCapabilities
from wolfharness_server.opencode_server.models.provider import Model


pytestmark = pytest.mark.unit


def _make_model_info(
    *,
    input_modalities: set[str] | None = None,
    output_modalities: set[str] | None = None,
    **kwargs: Any,
) -> ModelInfo:
    """Create a minimal ModelInfo for testing."""
    defaults: dict[str, Any] = {
        "id": "test-model",
        "name": "Test Model",
        "provider": "test-provider",
        "pricing": ModelPricing(prompt=0.0, completion=0.0),
    }
    defaults.update(kwargs)
    if input_modalities is not None:
        defaults["input_modalities"] = input_modalities
    if output_modalities is not None:
        defaults["output_modalities"] = output_modalities
    return ModelInfo(**defaults)


def test_from_tokonomics_no_override_uses_tokonomics_values() -> None:
    """Without override, modalities come from tokonomics input_modalities."""
    model_info = _make_model_info(
        input_modalities={"text", "image"},
        output_modalities={"text", "image"},
    )
    model = Model.from_tokonomics(model_info)
    assert model.capabilities.input.image is True
    assert model.capabilities.input.audio is False
    assert model.capabilities.output.image is True


def test_from_tokonomics_image_input_false_override() -> None:
    """image_input=False override disables input.image even if tokonomics says True."""
    model_info = _make_model_info(input_modalities={"text", "image"})
    override = ModelCapabilities(image_input=False)
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.input.image is False


def test_from_tokonomics_audio_input_false_override() -> None:
    """audio_input=False override disables input.audio."""
    model_info = _make_model_info(input_modalities={"text", "audio"})
    override = ModelCapabilities(audio_input=False)
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.input.audio is False


def test_from_tokonomics_video_input_true_override() -> None:
    """video_input=True override enables input.video even if tokonomics says False."""
    model_info = _make_model_info(input_modalities={"text"})
    override = ModelCapabilities(video_input=True)
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.input.video is True


def test_from_tokonomics_document_input_true_override() -> None:
    """document_input=True override enables input.pdf."""
    model_info = _make_model_info(input_modalities={"text"})
    override = ModelCapabilities(document_input=True)
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.input.pdf is True


def test_from_tokonomics_document_input_maps_file_modality() -> None:
    """Tokonomics 'file' modality maps to input.pdf even without override."""
    model_info = _make_model_info(input_modalities={"text", "file"})
    model = Model.from_tokonomics(model_info)
    assert model.capabilities.input.pdf is True


def test_from_tokonomics_image_output_true_override() -> None:
    """image_output=True override enables output.image even if tokonomics says False."""
    model_info = _make_model_info(
        input_modalities={"text"},
        output_modalities={"text"},
    )
    override = ModelCapabilities(image_output=True)
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.output.image is True


def test_from_tokonomics_none_fields_use_tokonomics_defaults() -> None:
    """When override fields are None, tokonomics values are preserved."""
    model_info = _make_model_info(
        input_modalities={"text", "image", "audio"},
        output_modalities={"text", "image"},
    )
    override = ModelCapabilities()  # all fields default to None
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.input.image is True
    assert model.capabilities.input.audio is True
    assert model.capabilities.output.image is True


def test_from_tokonomics_capabilities_override_none_same_as_no_override() -> None:
    """Passing capabilities_override=None is identical to not passing it."""
    model_info = _make_model_info(
        input_modalities={"text", "image"},
        output_modalities={"text"},
    )
    model_no_override = Model.from_tokonomics(model_info)
    model_none_override = Model.from_tokonomics(model_info, capabilities_override=None)
    assert model_no_override.capabilities == model_none_override.capabilities


def test_from_tokonomics_mixed_overrides() -> None:
    """Mix of True/False/None overrides only apply non-None fields."""
    model_info = _make_model_info(
        input_modalities={"text", "image", "audio"},
        output_modalities={"text"},
    )
    override = ModelCapabilities(
        image_input=False,  # override: False (was True)
        audio_input=None,  # no override (stays True)
        video_input=True,  # override: True (was False)
        document_input=False,  # override: False (was False)
        image_output=True,  # override: True (was False)
    )
    model = Model.from_tokonomics(model_info, capabilities_override=override)
    assert model.capabilities.input.image is False
    assert model.capabilities.input.audio is True
    assert model.capabilities.input.video is True
    assert model.capabilities.input.pdf is False
    assert model.capabilities.output.image is True
