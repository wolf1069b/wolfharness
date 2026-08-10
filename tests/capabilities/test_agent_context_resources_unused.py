"""Integration test verifying AgentContext.resources field is unused.

Confirms that no code reads .resources.list(), .resources.read(),
.resources.exists(), or .resources.on_change() on AgentContext instances.
This validates that Phase 4 removal (task 4.14) is safe.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest


pytestmark = pytest.mark.unit


SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _grep_source(pattern: str) -> list[str]:
    """Grep source files for a pattern, excluding AGENTS.md and __pycache__."""
    results: list[str] = []
    regex = re.compile(pattern)
    for py_file in SRC_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
        except Exception:  # noqa: BLE001
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                results.append(f"{py_file}:{i}: {line.strip()}")
    return results


def test_no_agent_context_resources_reads() -> None:
    """No code reads .resources.list(), .resources.read(), etc. on AgentContext.

    The resources field is SET at run.py and agent.py but never READ.
    This test greps for any .resources. method call patterns that would
    indicate a read, excluding:
    - Import statements
    - Comments/docstrings
    - The wolfharness_sync module (unrelated 'resources' attribute)
    - Dataclass field definitions
    """
    # Look for patterns like `.resources.list()`, `.resources.read(`, etc.
    # on objects that could be AgentContext instances.
    # We specifically look for agent_context.resources or ctx.resources
    # or deps.resources access patterns.
    patterns = [
        r"agent_context\.resources\.",
        r"ctx\.resources\.",
        r"deps\.resources\.",
        r"run_ctx\.resources\.",
    ]
    all_matches: list[str] = []
    for pattern in patterns:
        matches = _grep_source(pattern)
        all_matches.extend(matches)

    # Filter out known false positives (comments, docstrings)
    real_matches = [
        m
        for m in all_matches
        if not m.strip().startswith("#") and '"""' not in m and "'''" not in m
    ]

    assert not real_matches, (
        f"Found {len(real_matches)} read(s) of AgentContext.resources:\n" + "\n".join(real_matches)
    )
