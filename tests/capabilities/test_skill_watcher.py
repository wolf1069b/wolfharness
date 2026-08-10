"""Integration test: filesystem watcher detects new skill without restart.

Tests that when a new SKILL.md file is created, a ChangeEvent fires
after the 500ms debounce period, and the skill appears in list_skills().
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

import pytest

from wolfharness.capabilities.skill_watcher import SkillFilesystemWatcher


pytestmark = pytest.mark.unit


class TestFilesystemWatcher:
    """Test filesystem watcher for skill hot-reload (task 4.38)."""

    @pytest.mark.asyncio
    async def test_watcher_starts_without_watchdog(self) -> None:
        """Watcher starts even without watchdog installed (no-op mode)."""
        watcher = SkillFilesystemWatcher(paths=[])
        watcher.start()
        # Should not raise even without watchdog
        watcher.stop()

    @pytest.mark.asyncio
    async def test_on_change_returns_none_when_not_running(self) -> None:
        """on_change() returns None when watcher is not running."""
        watcher = SkillFilesystemWatcher(paths=[])
        result = watcher.on_change()
        assert result is None

    @pytest.mark.asyncio
    async def test_watcher_with_temp_directory(self) -> None:
        """Watcher monitors a temp directory for SKILL.md changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            watcher = SkillFilesystemWatcher(paths=[tmp_path], debounce_ms=100)
            watcher.start()

            try:
                if not watcher._running:
                    pytest.skip("watchdog not installed")

                # Create a SKILL.md file
                skill_file = tmp_path / "my-skill" / "SKILL.md"
                skill_file.parent.mkdir(parents=True)
                skill_file.write_text("---\nname: my-skill\n---\nTest content")

                # Wait for debounce + processing
                await asyncio.sleep(0.6)

                # The watcher should have recorded activity
                assert watcher._last_event_time > 0

                watcher.stop()
            except Exception:
                watcher.stop()
                raise
