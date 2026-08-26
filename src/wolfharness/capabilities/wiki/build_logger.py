"""Structured build event logger — writes JSONL to ``logs/``.

Tracks every meaningful event during a wiki build:

- ``entity_created`` — new entity file (did not exist before)
- ``entity_updated`` — existing entity overwritten (merged or new content)
- ``folder_created`` — a new concept/class directory was created
- ``entity_merged`` — multiple raw entities consolidated into one
- ``opa_generated`` — OPA record created (schema violation, conflict, quality issue)
- ``ops_stored``   — OPS solution record persisted

Usage::

    logger = WikiBuildLogger()
    WikiBuildTools(..., build_logger=logger)

All events are written as newline-delimited JSON (``.jsonl``) to
``logs/wiki_build_{timestamp}.jsonl``.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Self


logger = logging.getLogger(__name__)

# Re-export for convenience.
UTC = UTC


class WikiBuildLogger:
    """Structured, file-backed build event logger.

    Each call appends one JSON line to a timestamped ``.jsonl`` file
    under *log_dir*.  Call ``close()`` when the build finishes, or use
    as a context manager.
    """

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self._path = self.log_dir / f"wiki_build_{timestamp}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        logger.info("Build log → %s", self._path)

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """The active log file path."""
        return self._path

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── core ──────────────────────────────────────────────────────────────

    def _log(self, event: str, **data: object) -> None:
        record: dict[str, object] = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        record.update(data)
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()

    # ── events ────────────────────────────────────────────────────────────

    def entity_created(
        self,
        uri: str,
        concept: str,
        object_name: str,
        char_count: int,
        *,
        class_name: str | None = None,
        folder_created: bool = False,
    ) -> None:
        """A brand-new entity was written to disk (file did not exist)."""
        self._log(
            "entity_created",
            uri=uri,
            concept=concept,
            class_name=class_name or "",
            object_name=object_name,
            char_count=char_count,
            folder_created=folder_created,
        )

    def entity_updated(
        self,
        uri: str,
        concept: str,
        object_name: str,
        char_count: int,
        char_count_before: int,
        *,
        class_name: str | None = None,
        reason: str = "",
    ) -> None:
        """An existing entity was overwritten (merge or new content)."""
        self._log(
            "entity_updated",
            uri=uri,
            concept=concept,
            class_name=class_name or "",
            object_name=object_name,
            char_count=char_count,
            char_count_before=char_count_before,
            reason=reason,
        )

    def folder_created(self, path: str) -> None:
        """A new concept or class directory was created."""
        self._log("folder_created", path=path)

    def entity_moved(
        self,
        src_path: str,
        dst_path: str,
        new_uri: str,
    ) -> None:
        """An entity file was moved to a new location."""
        self._log(
            "entity_moved",
            src_path=src_path,
            dst_path=dst_path,
            new_uri=new_uri,
        )

    def entity_merged(
        self,
        target_uri: str,
        source_count: int,
        concept: str,
        object_name: str,
        char_count_before: int,
        char_count_after: int,
        source_count_total: int,
        *,
        class_name: str | None = None,
    ) -> None:
        """Multiple raw entities were consolidated into one."""
        self._log(
            "entity_merged",
            target_uri=target_uri,
            source_count=source_count,
            concept=concept,
            class_name=class_name or "",
            object_name=object_name,
            char_count_before=char_count_before,
            char_count_after=char_count_after,
            source_count_total=source_count_total,
        )

    def opa_generated(
        self,
        opa_id: str,
        title: str,
        category: str,
        reason: str,
        *,
        target_uri: str = "",
        scope: str = "",
    ) -> None:
        """An OPA (Open Problem Annotation) was created.

        Reasons include: schema validation violation, entity content
        conflict across sources, cross-concept numerical inconsistency,
        ADP parsing quality issue, or LLM-detected contradiction.
        """
        self._log(
            "opa_generated",
            opa_id=opa_id,
            title=title,
            category=category,
            reason=reason,
            target_uri=target_uri,
            scope=scope,
        )

    def ops_stored(
        self,
        ops_id: str,
        opa_id: str,
        solution_summary: str,
        *,
        target_uri: str = "",
    ) -> None:
        """An OPS (Open Problem Solution) was persisted."""
        self._log(
            "ops_stored",
            ops_id=ops_id,
            opa_id=opa_id,
            target_uri=target_uri,
            solution_summary=solution_summary,
        )

    def mcp_call(
        self,
        tool_name: str,
        args: dict,
        result_summary: str,
        duration_ms: float,
        *,
        error: str = "",
    ) -> None:
        """An MCP tool was called by the organizer agent."""
        self._log(
            "mcp_call",
            tool_name=tool_name,
            args_summary=str(args)[:500],
            result_summary=result_summary[:500],
            duration_ms=round(duration_ms, 1),
            error=error,
        )

    def phase_timing(self, phase: str, elapsed_ms: float) -> None:
        """Record one evaluated build phase duration."""
        self._log("phase_timing", phase=phase, elapsed_ms=round(elapsed_ms, 1))

    def source_packet_recorded(self, packet_id: str, doc_id: str, source_count: int) -> None:
        """A source packet was persisted for a document."""
        self._log(
            "source_packet_recorded", packet_id=packet_id, doc_id=doc_id, source_count=source_count
        )

    def mutation_attempt(self, uri: str, op: str) -> None:
        """An entity write was attempted."""
        self._log("mutation_attempt", uri=uri, op=op)

    def mutation_applied(self, uri: str, op: str) -> None:
        """An entity write completed."""
        self._log("mutation_applied", uri=uri, op=op)
