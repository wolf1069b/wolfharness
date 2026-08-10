"""Unit tests for ModelCapabilities serialization and subtype inheritance."""

from __future__ import annotations

import pytest

from wolfharness.models.model_configs import (
    AnthropicModelConfig,
    BaseModelConfig,
    FallbackModelConfig,
    FunctionModelConfig,
    GeminiModelConfig,
    ImportModelConfig,
    InputModelConfig,
    OpenAIModelConfig,
    StringModelConfig,
    TestModelConfig,
)
from wolfharness_config.model_capabilities import ModelCapabilities


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Task 1.3 — ModelCapabilities serialization
# ---------------------------------------------------------------------------


def test_all_fields_default_to_none() -> None:
    """All fields are None when ModelCapabilities is constructed without args."""
    caps = ModelCapabilities()
    assert caps.image_input is None
    assert caps.audio_input is None
    assert caps.video_input is None
    assert caps.document_input is None
    assert caps.image_output is None


def test_image_input_false_round_trips() -> None:
    """ModelCapabilities(image_input=False) round-trips through serialization."""
    caps = ModelCapabilities(image_input=False)
    dumped = caps.model_dump()
    assert dumped == {
        "image_input": False,
        "audio_input": None,
        "video_input": None,
        "document_input": None,
        "image_output": None,
    }
    restored = ModelCapabilities.model_validate(dumped)
    assert restored == caps
    assert restored.image_input is False


def test_partial_specification() -> None:
    """Only image_input=True is set; all other fields remain None."""
    caps = ModelCapabilities(image_input=True)
    assert caps.image_input is True
    assert caps.audio_input is None
    assert caps.video_input is None
    assert caps.document_input is None
    assert caps.image_output is None


def test_all_fields_set() -> None:
    """All five fields can be set simultaneously."""
    caps = ModelCapabilities(
        image_input=True,
        audio_input=True,
        video_input=False,
        document_input=True,
        image_output=False,
    )
    assert caps.image_input is True
    assert caps.audio_input is True
    assert caps.video_input is False
    assert caps.document_input is True
    assert caps.image_output is False


def test_json_round_trip() -> None:
    """ModelCapabilities survives JSON serialization and deserialization."""
    original = ModelCapabilities(image_input=True, audio_input=False)
    json_str = original.model_dump_json()
    restored = ModelCapabilities.model_validate_json(json_str)
    assert restored == original


def test_nested_in_openai_model_config() -> None:
    """Capabilities nests correctly inside OpenAIModelConfig."""
    config = OpenAIModelConfig(
        identifier="gpt-5-codex",
        capabilities=ModelCapabilities(image_input=False),
    )
    assert config.capabilities is not None
    assert config.capabilities.image_input is False
    assert config.capabilities.audio_input is None


def test_nested_in_model_config_from_dict() -> None:
    """Capabilities parses correctly from a plain dict (YAML-style)."""
    config = OpenAIModelConfig.model_validate(
        {"type": "openai", "identifier": "gpt-5-codex", "capabilities": {"image_input": False}},
    )
    assert config.capabilities is not None
    assert config.capabilities.image_input is False


def test_capabilities_none_by_default_in_model_config() -> None:
    """Capabilities is None when not specified in model config."""
    config = OpenAIModelConfig(identifier="gpt-5-codex")
    assert config.capabilities is None


def test_capabilities_excluded_from_dump_when_none() -> None:
    """When capabilities is None, model_dump excludes it by default."""
    config = OpenAIModelConfig(identifier="gpt-5-codex")
    dumped = config.model_dump()
    assert dumped["capabilities"] is None


# ---------------------------------------------------------------------------
# Task 1.4 — All 9 model config subtypes inherit capabilities
# ---------------------------------------------------------------------------

ALL_SUBTYPE_FACTORIES: list[tuple[str, type[BaseModelConfig], dict[str, object]]] = [
    ("StringModelConfig", StringModelConfig, {"identifier": "openai:gpt-4o"}),
    ("OpenAIModelConfig", OpenAIModelConfig, {"identifier": "gpt-5-codex"}),
    ("AnthropicModelConfig", AnthropicModelConfig, {"identifier": "claude-sonnet-4-5"}),
    ("GeminiModelConfig", GeminiModelConfig, {"identifier": "gemini-2.0-flash"}),
    ("FallbackModelConfig", FallbackModelConfig, {"models": ["openai:gpt-4o"]}),
    ("TestModelConfig", TestModelConfig, {}),
    ("ImportModelConfig", ImportModelConfig, {"model": "os:getcwd"}),
    ("FunctionModelConfig", FunctionModelConfig, {"function": "os:getcwd"}),
    ("InputModelConfig", InputModelConfig, {"handler": "os:getcwd"}),
]


@pytest.mark.parametrize(
    ("name", "config_cls", "kwargs"),
    ALL_SUBTYPE_FACTORIES,
    ids=[entry[0] for entry in ALL_SUBTYPE_FACTORIES],
)
def test_subtype_has_capabilities_field(
    name: str,
    config_cls: type[BaseModelConfig],
    kwargs: dict[str, object],
) -> None:
    """Each of the 9 model config subtypes has the capabilities field."""
    assert "capabilities" in config_cls.model_fields, f"{name} is missing the 'capabilities' field"


@pytest.mark.parametrize(
    ("name", "config_cls", "kwargs"),
    ALL_SUBTYPE_FACTORIES,
    ids=[entry[0] for entry in ALL_SUBTYPE_FACTORIES],
)
def test_subtype_accepts_capabilities(
    name: str,
    config_cls: type[BaseModelConfig],
    kwargs: dict[str, object],
) -> None:
    """Each subtype can be constructed with capabilities=ModelCapabilities(...)."""
    caps = ModelCapabilities(image_input=True)
    config = config_cls(**kwargs, capabilities=caps)
    assert config.capabilities is not None
    assert config.capabilities.image_input is True


@pytest.mark.parametrize(
    ("name", "config_cls", "kwargs"),
    ALL_SUBTYPE_FACTORIES,
    ids=[entry[0] for entry in ALL_SUBTYPE_FACTORIES],
)
def test_subtype_capabilities_defaults_to_none(
    name: str,
    config_cls: type[BaseModelConfig],
    kwargs: dict[str, object],
) -> None:
    """Each subtype defaults capabilities to None when not specified."""
    config = config_cls(**kwargs)
    assert config.capabilities is None
