"""Tests for CustomToolsetConfig parameter passing support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness_config.toolsets import CustomToolsetConfig


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pydantic_ai.capabilities import AbstractCapability


class MockProvider(FunctionToolsetCapability):
    """Mock provider that accepts arbitrary kwargs."""

    def __init__(self, name: str, owner: str | None = None, **kwargs) -> None:
        """Initialize mock provider with arbitrary parameters."""
        super().__init__(name=name, owner=owner)
        self.custom_params = kwargs

    def get_capabilities(self) -> AbstractCapability | None:
        """Return a pydantic-ai capability for this provider.

        Returns:
            A pydantic-ai AbstractCapability instance, or None.
        """
        return None


class StrictProvider(FunctionToolsetCapability):
    """Mock provider with strict parameter requirements."""

    def __init__(self, name: str, required_arg: int):
        """Initialize strict provider with required argument."""
        super().__init__(name=name)
        self.required_arg = required_arg


async def test_custom_toolset_parameters():
    """Test that CustomToolsetConfig passes kw_args to provider constructor."""
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.MockProvider",
        kw_args={"key": "value", "another": 123},
    )
    provider = config.get_provider()

    # Provider should have received custom parameters
    assert hasattr(provider, "custom_params")
    assert provider.custom_params == {"key": "value", "another": 123}


async def test_custom_toolset_name_collision():
    """Test that kw_args can override default name."""
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.MockProvider",
        kw_args={"name": "new_name"},
    )
    provider = config.get_provider()

    # Provider should use the overridden name, not the class name
    assert provider.name == "new_name"


async def test_custom_toolset_invalid_parameters():
    """Test that invalid parameters raise appropriate errors with helpful messages."""
    import pytest

    # Missing required_arg should raise TypeError with helpful message
    config = CustomToolsetConfig(
        import_path="tests.toolsets.test_custom_toolset.StrictProvider",
        kw_args={"unknown_param": "value"},  # Missing required_arg
    )
    with pytest.raises(TypeError) as exc_info:
        config.get_provider()

    # Verify error message includes useful context
    error_msg = str(exc_info.value)
    assert "tests.toolsets.test_custom_toolset.StrictProvider" in error_msg
    assert "unknown_param" in error_msg
    assert "value" in error_msg
    assert "Original error:" in error_msg

    def get_capabilities(self) -> AbstractCapability | None:
        """Return a pydantic-ai capability for this provider.

        Returns:
            A pydantic-ai AbstractCapability instance, or None.
        """
        return None
