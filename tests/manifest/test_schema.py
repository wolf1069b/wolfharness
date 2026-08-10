from __future__ import annotations

import pytest

from wolfharness import AgentsManifest


pytestmark = pytest.mark.unit


def test_schema_generation():
    """Test that JSON schema can be generated from config models."""
    try:
        _schema = AgentsManifest.model_json_schema()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Failed to generate schema: {exc}")


if __name__ == "__main__":
    pytest.main([__file__])
