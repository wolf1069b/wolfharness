"""Test that deprecated feature flag fields have been removed from OpenCodeConfig."""

import pytest

from wolfharness_config.session_pool import OpenCodeConfig


pytestmark = pytest.mark.unit


# The 8 deprecated feature flags that must no longer exist on OpenCodeConfig
DEPRECATED_FLAGS = [
    "use_session_pool",
    "use_session_pool_for_commands",
    "use_session_pool_for_skills",
    "use_session_pool_for_init",
    "use_session_pool_for_summarize",
    "use_session_pool_for_mcp",
    "use_session_pool_for_messages",
    "use_session_pool_for_status",
]


def test_deprecated_flags_removed() -> None:
    """Assert all 8 deprecated feature flag fields are absent from OpenCodeConfig."""
    model_fields = set(OpenCodeConfig.model_fields.keys())
    present = [f for f in DEPRECATED_FLAGS if f in model_fields]
    assert not present, (
        f"Expected {len(DEPRECATED_FLAGS)} deprecated flags to be removed, "
        f"but {len(present)} are still present: {present}"
    )


def test_deprecated_method_should_use_session_pool_for_removed() -> None:
    """Assert should_use_session_pool_for method is absent from OpenCodeConfig."""
    assert not hasattr(OpenCodeConfig, "should_use_session_pool_for"), (
        "OpenCodeConfig.should_use_session_pool_for should have been removed"
    )
