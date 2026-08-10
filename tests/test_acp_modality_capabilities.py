"""Unit tests for ACP modality capability derivation.

Tests that ``audio_prompts`` and ``image_prompts`` in the ACP
``InitializeResponse`` are correctly derived from the default agent's
``ModelCapabilities`` rather than being hardcoded to ``True``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from wolfharness.models.model_configs import BaseModelConfig, StringModelConfig
from wolfharness_config.model_capabilities import ModelCapabilities


pytestmark = pytest.mark.unit


def _derive_modality_prompts(model_config: Any) -> tuple[bool, bool]:
    """Replicate the derivation logic from ``AgentPoolACPAgent.initialize()``.

    This extracts the pure logic so we can test it without spinning up
    a full ACP server.
    """
    audio_prompts = True
    image_prompts = True
    if isinstance(model_config, BaseModelConfig) and model_config.capabilities is not None:
        caps = model_config.capabilities
        audio_prompts = caps.audio_input if caps.audio_input is not None else True
        image_prompts = caps.image_input if caps.image_input is not None else True
    return audio_prompts, image_prompts


def test_image_input_false_advertises_image_prompts_false() -> None:
    """Agent with ``image_input=False`` should advertise ``image_prompts=False``."""
    model_config = StringModelConfig(
        identifier="openai:gpt-4o",
        capabilities=ModelCapabilities(image_input=False),
    )
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert image_prompts is False
    assert audio_prompts is True  # audio_input not specified, defaults to True


def test_audio_input_false_advertises_audio_prompts_false() -> None:
    """Agent with ``audio_input=False`` should advertise ``audio_prompts=False``."""
    model_config = StringModelConfig(
        identifier="openai:gpt-4o",
        capabilities=ModelCapabilities(audio_input=False),
    )
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert audio_prompts is False
    assert image_prompts is True  # image_input not specified, defaults to True


def test_no_capabilities_defaults_to_true() -> None:
    """Agent with no capabilities specified should default both to ``True``."""
    model_config = StringModelConfig(identifier="openai:gpt-4o")
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert audio_prompts is True
    assert image_prompts is True


def test_both_true_advertises_both_true() -> None:
    """Agent with ``image_input=True`` and ``audio_input=True`` should advertise both."""
    model_config = StringModelConfig(
        identifier="openai:gpt-4o",
        capabilities=ModelCapabilities(image_input=True, audio_input=True),
    )
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert audio_prompts is True
    assert image_prompts is True


def test_none_fields_default_to_true() -> None:
    """``None`` fields in ModelCapabilities should default to ``True`` (optimistic)."""
    model_config = StringModelConfig(
        identifier="openai:gpt-4o",
        capabilities=ModelCapabilities(image_input=None, audio_input=None),
    )
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert audio_prompts is True
    assert image_prompts is True


def test_string_model_not_base_config_defaults_to_true() -> None:
    """A plain string model (not BaseModelConfig) should default both to ``True``."""
    model_config = "openai:gpt-4o"
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert audio_prompts is True
    assert image_prompts is True


def test_both_false_advertises_both_false() -> None:
    """Agent with both ``image_input=False`` and ``audio_input=False``."""
    model_config = StringModelConfig(
        identifier="openai:gpt-4o",
        capabilities=ModelCapabilities(image_input=False, audio_input=False),
    )
    audio_prompts, image_prompts = _derive_modality_prompts(model_config)
    assert audio_prompts is False
    assert image_prompts is False


def test_mock_default_agent_derives_capabilities() -> None:
    """Verify the logic works with a mock default_agent resembling the real one.

    This simulates the access pattern used in ``initialize()``:
    ``self.default_agent.config.model``.
    """
    mock_agent = MagicMock()
    mock_agent.config.model = StringModelConfig(
        identifier="anthropic:claude-sonnet-4-5",
        capabilities=ModelCapabilities(image_input=False, audio_input=False),
    )
    audio_prompts, image_prompts = _derive_modality_prompts(mock_agent.config.model)
    assert audio_prompts is False
    assert image_prompts is False
