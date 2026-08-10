"""Filesystem watcher for skill hot-reload.

Uses ``watchdog`` to monitor skill directories for changes. When a
``SKILL.md`` file is created, modified, or deleted, a ``ChangeEvent``
is emitted after a 500ms debounce period.

If ``watchdog`` is not installed, the watcher is a no-op.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from wolfharness.capabilities.change_event import ChangeEvent


from wolfharness.log import get_logger


logger = get_logger(__name__)

DEBOUNCE_MS = 500
"""Debounce time in milliseconds before emitting a change event."""


class SkillFilesystemWatcher:
    """Watch skill directories for changes and emit ``ChangeEvent``.

    Monitors one or more directories for ``SKILL.md`` file changes.
    Uses a 500ms debounce to avoid emitting multiple events for
    rapid file system changes (e.g., editor save sequences).

    If ``watchdog`` is not installed, the watcher operates in
    no-op mode — ``start()`` does nothing and ``stop()`` is a no-op.
    """

    def __init__(
        self,
        paths: list[Path],
        debounce_ms: int = DEBOUNCE_MS,
    ) -> None:
        """Initialize the filesystem watcher.

        Args:
            paths: List of directories to watch for SKILL.md changes.
            debounce_ms: Debounce time in milliseconds. Default: 500.
        """
        self._paths = list(paths)
        self._debounce_ms = debounce_ms
        self._observer: Any | None = None
        self._change_queue: asyncio.Queue[ChangeEvent] = asyncio.Queue()
        self._last_event_time: float = 0.0
        self._debounce_task: asyncio.Task[None] | None = None
        self._running = False

    def start(self) -> None:
        """Start watching the configured directories.

        If ``watchdog`` is not installed, this is a no-op.
        """
        if self._running:
            return

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.debug(
                "watchdog not installed — skill filesystem watcher is disabled. "
                "Install with: uv sync --extra watchdog"
            )
            return

        class _SkillFileHandler(FileSystemEventHandler):
            """Handle filesystem events for SKILL.md files."""

            def __init__(self, watcher: SkillFilesystemWatcher) -> None:
                self._watcher = watcher

            def on_any_event(self, event: Any) -> None:
                """Handle any filesystem event.

                Args:
                    event: The watchdog filesystem event.
                """
                # Only care about SKILL.md files
                src_path = getattr(event, "src_path", "")
                if not src_path:
                    return
                if not src_path.endswith("SKILL.md"):
                    return
                self._watcher._schedule_debounce()

        self._observer = Observer()
        handler = _SkillFileHandler(self)
        for path in self._paths:
            if path.exists() and path.is_dir():
                self._observer.schedule(handler, str(path), recursive=True)

        self._observer.start()
        self._running = True
        logger.debug(
            "Skill filesystem watcher started",
            extra={"paths": [str(p) for p in self._paths]},
        )

    def stop(self) -> None:
        """Stop watching directories."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            self._debounce_task = None
        self._running = False

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        """Return an async iterator of change events.

        Returns:
            An async iterator yielding ``ChangeEvent(kind="skills_changed")``
            when skill files change, or ``None`` if the watcher is not running.
        """
        if not self._running:
            return None
        return self._iterate_changes()

    async def _iterate_changes(self) -> AsyncIterator[ChangeEvent]:
        """Yield change events from the queue.

        Yields:
            ``ChangeEvent`` with ``kind="skills_changed"``.
        """
        while self._running:
            event = await self._change_queue.get()
            if event is None:
                break
            yield event

    def _schedule_debounce(self) -> None:
        """Schedule a debounced change event emission."""
        self._last_event_time = time.monotonic()
        if self._debounce_task is None or self._debounce_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop — can't schedule debounce task.
                # The event will be picked up on next check.
                return
            self._debounce_task = loop.create_task(self._fire_after_debounce())

    async def _fire_after_debounce(self) -> None:
        """Fire a change event after the debounce period.

        Waits for the debounce period to elapse. If another event
        arrives during the wait, the timer resets.
        """
        debounce_seconds = self._debounce_ms / 1000.0
        while True:
            elapsed = time.monotonic() - self._last_event_time
            remaining = debounce_seconds - elapsed
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)

        from wolfharness.capabilities.change_event import ChangeEvent

        await self._change_queue.put(
            ChangeEvent(
                capability_name="skill_filesystem_watcher",
                kind="skills_changed",
            )
        )
