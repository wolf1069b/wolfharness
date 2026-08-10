"""Unit tests for ``ModalityFilterCapabilityConfig`` YAML parsing and building.

Tests cover:
- Default strategy values
- Explicit strategy overrides
- ``build_capability()`` returns a ``ModalityFilterCapability``
- Built capability has correct strategy values
- Built capability has placeholder ``ModelCapabilities()`` (all None)
- ``KNOWN_CAPABILITY_TYPES`` includes ``"modality_filter"``
- Invalid strategy value raises validation error
"""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError
import pytest

from wolfharness_config.capabilities import (
    KNOWN_CAPABILITY_TYPES,
    BuiltinCapabilityConfig,
    ModalityFilterCapabilityConfig,
    build_capability,
)


@pytest.mark.unit
def test_modality_filter_config_defaults() -> None:
    """YAML parsing with default strategies."""
    config = TypeAdapter(BuiltinCapabilityConfig).validate_python({"type": "modality_filter"})
    assert isinstance(config, ModalityFilterCapabilityConfig)
    assert config.image_strategy == "describe"
    assert config.audio_strategy == "describe"
    assert config.video_strategy == "describe"
    assert config.document_strategy == "describe"


@pytest.mark.unit
def test_modality_filter_config_explicit_strategies() -> None:
    """YAML parsing with explicit strategies."""
    config = TypeAdapter(BuiltinCapabilityConfig).validate_python({
        "type": "modality_filter",
        "image_strategy": "drop",
        "audio_strategy": "pass",
    })
    assert isinstance(config, ModalityFilterCapabilityConfig)
    assert config.image_strategy == "drop"
    assert config.audio_strategy == "pass"
    assert config.video_strategy == "describe"
    assert config.document_strategy == "describe"


@pytest.mark.unit
def test_build_capability_returns_modality_filter() -> None:
    """``build_capability()`` returns a ``ModalityFilterCapability`` instance."""
    config = ModalityFilterCapabilityConfig()
    capability = build_capability(config)
    from wolfharness.capabilities.modality_filter import ModalityFilterCapability

    assert isinstance(capability, ModalityFilterCapability)


@pytest.mark.unit
def test_build_capability_has_correct_strategies() -> None:
    """Built capability has the strategy values from config."""
    config = ModalityFilterCapabilityConfig(
        image_strategy="drop",
        audio_strategy="pass",
        video_strategy="describe",
        document_strategy="drop",
    )
    capability = build_capability(config)
    assert capability.image_strategy == "drop"
    assert capability.audio_strategy == "pass"
    assert capability.video_strategy == "describe"
    assert capability.document_strategy == "drop"


@pytest.mark.unit
def test_build_capability_has_none_capabilities() -> None:
    """Built capability has ``capabilities=None`` (factory populates later)."""
    config = ModalityFilterCapabilityConfig()
    capability = build_capability(config)

    assert capability.capabilities is None


@pytest.mark.unit
def test_known_capability_types_includes_modality_filter() -> None:
    """``KNOWN_CAPABILITY_TYPES`` includes ``"modality_filter"``."""
    assert "modality_filter" in KNOWN_CAPABILITY_TYPES


@pytest.mark.unit
def test_invalid_strategy_raises_validation_error() -> None:
    """Invalid strategy value raises ``ValidationError``."""
    with pytest.raises(ValidationError):
        TypeAdapter(BuiltinCapabilityConfig).validate_python({
            "type": "modality_filter",
            "image_strategy": "invalid",
        })
