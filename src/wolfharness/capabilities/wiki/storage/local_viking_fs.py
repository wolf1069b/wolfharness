"""Local filesystem backend that preserves the viking:// URI scheme.

Stores files on local disk (zero HTTP during build) but reports
``root_uri`` as ``viking://resources/{namespace}``. At finalize,
``finalize_upload()`` uploads each file individually via ``client.write``
with ``processing_mode='semantic_and_vectors'`` so entities become both
vector-embedded and semantically searchable while preserving the exact
canonical URI — no server-side splitting. Each file upload runs in a
worker thread with a hard timeout so a slow Viking write never hangs the
finalize tool call (the wiki is durable locally regardless).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openviking_sdk.errors import AlreadyExistsError  # type: ignore[import-untyped]

from .backend import FSBackend
from .local_fs import LocalFS


if TYPE_CHECKING:
    from pathlib import Path

    from .viking_fs import VikingClient

logger = logging.getLogger(__name__)

# Long entity/OP filenames exceed URI length limits on the Viking server.
# Filenames must already be short at GENERATION time (id builders clip UTF-8
# bytes), because renaming here would silently break every URI reference that
# points at the original filename. This constant is a hard validation gate.
_MAX_FILENAME_BYTES = 200
# client.write(semantic_and_vectors) writes the file then waits for the
# server to process it; on a busy server that can take 1-3min per file.
# Longer than this and we give up on this file (the write likely still
# landed server-side).
_UPLOAD_TIMEOUT_S = 180


class LocalVikingFS(FSBackend):
    """Local storage with viking:// URI identity.

    Delegates all I/O to an inner :class:`LocalFS` but overrides
    ``root_uri`` to return ``viking://resources/{namespace}``.
    """

    def __init__(self, namespace: str, local_root: Path) -> None:
        self.namespace = namespace
        self._local = LocalFS(local_root / namespace)

    @property
    def root_uri(self) -> str:
        return f"viking://resources/{self.namespace}"

    def read_text(self, key: str) -> str | None:
        return self._local.read_text(key)

    def write_text(self, key: str, content: str, *, overwrite: bool = True) -> None:
        self._local.write_text(key, content, overwrite=overwrite)

    def write_text_durable(self, key: str, content: str) -> None:
        self._local.write_text_durable(key, content)

    def write_many(self, writes: list[tuple[str, str]]) -> None:
        self._local.write_many(writes)

    def commit_many(self, writes: list[tuple[str, str]]) -> None:
        self._local.commit_many(writes)

    def exists(self, key: str) -> bool:
        return self._local.exists(key)

    def is_dir(self, key: str) -> bool:
        return self._local.is_dir(key)

    def list_dir(self, key: str, *, recursive: bool = False) -> list[str]:
        return self._local.list_dir(key, recursive=recursive)

    def list_entries(self, key: str, *, recursive: bool = False) -> list[str]:
        return self._local.list_entries(key, recursive=recursive)

    def mtime_ns(self, key: str) -> int | None:
        return self._local.mtime_ns(key)

    def size(self, key: str) -> int | None:
        return self._local.size(key)

    def mkdir_p(self, key: str) -> None:
        self._local.mkdir_p(key)

    def delete(self, key: str) -> None:
        self._local.delete(key)

    def move(self, src: str, dst: str) -> None:
        self._local.move(src, dst)

    def sync_native_relations(self, pairs: list[tuple[str, list[str]]]) -> dict[str, object]:
        """No-op: local backends have no remote graph API to sync.

        Relations are stored in entity frontmatter only; the native
        OpenViking link/unlink graph is irrelevant for local builds.
        """
        return {"linked": 0, "unlinked": 0, "errors": []}

    def finalize_upload(self, client: VikingClient) -> dict[str, object]:
        """Upload each file under the wiki root to Viking individually.

        Uses ``client.write`` per file (not ``add_resource`` per directory)
        because ``add_resource`` with ``parse_mode: no_split`` is broken
        server-side — the MarkdownParser rejects every file. Per-file
        ``write(processing_mode='semantic_and_vectors')`` preserves the
        exact canonical URI, creates embeddings, and avoids splitting.

        Each file upload runs in a worker thread with a hard timeout so a
        slow Viking write never hangs the finalize tool call.

        Returns:
            ``{"status": ..., "uploads": [{"path": str, "status": str}]}``
            where status is ``completed``, ``partial``, or ``failed``.
        """
        uploads: list[dict[str, object]] = []
        try:
            root = self._local.root
            all_files = self._collect_files(root)
            self._ensure_dirs(client, all_files, root)

            # --- upload each file sequentially ---
            for fp in all_files:
                rel = fp.relative_to(root).as_posix()
                uri = f"{self.root_uri}/{rel}"
                content = fp.read_text(encoding="utf-8")
                try:
                    client.write(
                        uri,
                        content,
                        mode="create",
                        wait=False,
                        processing_mode="semantic_and_vectors",
                        timeout=_UPLOAD_TIMEOUT_S,
                    )
                    uploads.append({"path": rel, "status": "ok"})
                except AlreadyExistsError:
                    try:
                        client.write(
                            uri,
                            content,
                            mode="replace",
                            wait=False,
                            processing_mode="semantic_and_vectors",
                            timeout=_UPLOAD_TIMEOUT_S,
                        )
                        uploads.append({"path": rel, "status": "ok"})
                    except Exception:
                        logger.exception("replace write %s failed", rel)
                        uploads.append({"path": rel, "status": "failed"})
                except Exception:
                    logger.exception("write %s failed", rel)
                    uploads.append({"path": rel, "status": "failed"})

            ok = [u for u in uploads if u["status"] == "ok"]
            failed = [u for u in uploads if u["status"] == "failed"]
            if not ok:
                status = "failed"
            elif failed:
                status = "partial"
            else:
                status = "completed"
        except Exception:
            logger.exception("finalize_upload failed; local finalize stands")
            return {"status": "failed", "uploads": uploads}
        return {"status": status, "uploads": uploads}

    def _collect_files(self, root: Path) -> list[Path]:
        """Walk ``root`` recursively, returning uploadable files.

        Skips ``index/``, ``source_packets/``, and hidden files. Validates
        filename byte length against ``_MAX_FILENAME_BYTES``.
        """
        all_files: list[Path] = []
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            rel = fp.relative_to(root)
            if any(p in {"index", "source_packets"} or p.startswith(".") for p in rel.parts):
                continue
            if len(fp.name.encode("utf-8")) > _MAX_FILENAME_BYTES:
                raise ValueError(
                    f"filename {fp.name!r} is {len(fp.name.encode('utf-8'))} bytes "
                    f"(limit {_MAX_FILENAME_BYTES}); id builders must clip at generation time",
                )
            all_files.append(fp)
        return all_files

    def _ensure_dirs(self, client: VikingClient, files: list[Path], root: Path) -> None:
        """Create the root namespace dir and all parent dirs on Viking."""
        try:
            client.mkdir(self.root_uri)
        except Exception:
            logger.debug("mkdir %s failed (may already exist)", self.root_uri, exc_info=True)

        parent_dirs: set[str] = set()
        for fp in files:
            rel = fp.relative_to(root)
            for i in range(1, len(rel.parts)):
                parent_dirs.add("/".join(rel.parts[:i]))
        for d in sorted(parent_dirs, key=lambda p: p.count("/")):
            uri = f"{self.root_uri}/{d}"
            try:
                client.mkdir(uri)
            except Exception:
                logger.debug("mkdir %s failed (may already exist)", uri, exc_info=True)
