"""Test suite for import corrections.

Tests that all imports use the canonical pydantic_ai.usage module location.
"""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_runusage_imports():
    """Test that RunUsage is imported from pydantic_ai.usage, not pydantic_ai."""
    files_to_check = [
        "src/wolfharness_storage/sql_provider/sql_provider.py",
        "src/wolfharness_storage/file_provider/provider.py",
        "tests/test_history.py",
        "tests/mcp_client/test_client_conversion.py",
        "src/wolfharness_storage/sql_provider/utils.py",
        "src/wolfharness_storage/claude_provider/converters.py",
    ]

    root = Path(__file__).parent.parent.parent

    for file_path in files_to_check:
        full_path = root / file_path
        if not full_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        with full_path.open("r") as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content)

        # Find all import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if module == "pydantic_ai":
                    # Check if RunUsage is imported from pydantic_ai
                    names = [alias.name for alias in node.names]
                    if "RunUsage" in names:
                        print(f"✗ FAILED: {file_path} imports RunUsage from pydantic_ai")
                        print("  Expected: from pydantic_ai.usage import RunUsage")
                        print("  Actual:   from pydantic_ai import RunUsage")
                        raise AssertionError(
                            f"{file_path} should import RunUsage from pydantic_ai.usage"
                        )
                elif module == "pydantic_ai.usage":
                    names = [alias.name for alias in node.names]
                    if "RunUsage" in names:
                        print(f"✓ PASS: {file_path} imports RunUsage from pydantic_ai.usage")

    print("\n✓ All RunUsage imports are from pydantic_ai.usage")


def test_runusage_functionality():
    """Test that RunUsage can be imported and used correctly."""
    from pydantic_ai.usage import RunUsage

    # Test basic instantiation
    usage = RunUsage(
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=5,
        cache_write_tokens=3,
    )

    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.cache_read_tokens == 5
    assert usage.cache_write_tokens == 3

    print("✓ RunUsage instantiation works correctly")


if __name__ == "__main__":
    print("Testing RunUsage import corrections...\n")
    test_runusage_imports()
    print()
    test_runusage_functionality()
    print("\n✓ All import tests passed!")
