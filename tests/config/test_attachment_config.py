"""Tests for AttachmentImageConfig."""

from __future__ import annotations

import pytest
import yaml

from wolfharness_config.attachment import AttachmentImageConfig


pytestmark = pytest.mark.unit


def test_attachment_image_defaults():
    """AttachmentImageConfig defaults: auto_resize=True, 2000x2000, 5MB base64."""
    config = AttachmentImageConfig()

    assert config.auto_resize is True
    assert config.max_width == 2000
    assert config.max_height == 2000
    assert config.max_base64_bytes == 5 * 1024 * 1024


def test_attachment_image_custom_values():
    """AttachmentImageConfig accepts explicit overrides."""
    config = AttachmentImageConfig(
        auto_resize=False,
        max_width=4096,
        max_height=3072,
        max_base64_bytes=10 * 1024 * 1024,
    )

    assert config.auto_resize is False
    assert config.max_width == 4096
    assert config.max_height == 3072
    assert config.max_base64_bytes == 10 * 1024 * 1024


def test_attachment_image_rejects_non_positive_dimensions():
    """AttachmentImageConfig rejects max_width/max_height < 1."""
    with pytest.raises(ValueError, match="max_width"):
        AttachmentImageConfig(max_width=0)


def test_attachment_image_serialize_as_dict():
    """AttachmentImageConfig serializes to the expected dict shape."""
    config = AttachmentImageConfig()

    d = config.model_dump()

    assert d == {
        "auto_resize": True,
        "max_width": 2000,
        "max_height": 2000,
        "max_base64_bytes": 5 * 1024 * 1024,
    }


def test_attachment_image_yaml_round_trip():
    """AttachmentImageConfig round-trips through YAML."""
    config = AttachmentImageConfig(auto_resize=False, max_width=1024, max_height=768)

    loaded = AttachmentImageConfig.model_validate(
        yaml.safe_load(yaml.safe_dump(config.model_dump()))
    )

    assert loaded.auto_resize is False
    assert loaded.max_width == 1024
    assert loaded.max_height == 768
