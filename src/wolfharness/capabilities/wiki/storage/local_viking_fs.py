"""Local filesystem backend that preserves the viking:// URI scheme.

Stores files on local disk (zero HTTP during build) but reports
``root_uri`` as ``viking://resources/{namespace}``. At finalize, ``finalize_upload()``
uploads each top-level directory via ``add_resource(semantic_and_vectors)`` so
entities become both vector-embedded and semantically searchable; per-directory
uploads run in a worker thread with a hard timeout so a slow Viking upload never
hangs or 500s the finalize tool call (the wiki is durable locally regardless).

Long entity filenames (OP records exceed 255 bytes) would blow Viking's unzip
path limit (Errno 36), so each directory is copied to a temp tree with basenames
capped before upload.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

from .backend import FSBackend
from .local_fs import LocalFS


if TYPE_CHECKING:
    from .viking_fs import VikingClient


logger = logging.getLogger(__name__)

# Viking unzips uploads into a temp dir; a long basename can push the unzip path
# over the 255-byte limit (Errno 36). Entity/OP filenames must already be short
# at GENERATION time (id builders clip UTF-8 bytes), because renaming here would
# silently break every URI reference that points at the original filename.
# This constant is a hard validation gate, never a renamer.
_MAX_FILENAME_BYTES = 200
# add_resource(semantic_and_vectors) uploads the zip then waits for the queue to
# accept the job; on a busy server that can take 1-3min. Longer than this and we
# give up on this directory (the submit likely still landed server-side).
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
        """Upload each top-level directory under the wiki root to Viking.

        Per-directory ``add_resource(semantic_and_vectors)`` runs in a worker
        thread with a hard timeout. A slow/failing upload must never hang the
        finalize tool call — the wiki is already durable locally and the caller
        reports the result best-effort.

        Returns:
            ``{"status": "completed"|"failed", "uploads": [{"dir": str, "status": "ok"|"failed"}]}``
        """
        uploads: list[dict[str, object]] = []
        try:
            for child in sorted(self._local.root.iterdir()):
                if child.is_dir():
                    if child.name.startswith(".") or child.name in {"index", "source_packets"}:
                        continue
                    try:
                        safe = self._copy_dir(child)
                        executor = ThreadPoolExecutor(max_workers=1)
                        try:
                            r = executor.submit(
                                client.add_resource,
                                path=str(safe),
                                to=f"viking://resources/{self.namespace}",
                                processing_mode="semantic_and_vectors",
                                wait=False,
                            ).result(timeout=_UPLOAD_TIMEOUT_S)
                        finally:
                            executor.shutdown(wait=False)
                        uploads.append({"dir": child.name, "status": "ok", "detail": str(r)[:200]})
                    except Exception:
                        logger.exception("upload dir %s failed; local content stands", child.name)
                        uploads.append({"dir": child.name, "status": "failed"})
                elif child.is_file():
                    try:
                        uri = f"{self.root_uri}/{child.name}"
                        content = child.read_text(encoding="utf-8")
                        executor = ThreadPoolExecutor(max_workers=1)
                        try:
                            executor.submit(
                                client.write,
                                uri,
                                content,
                                mode="create",
                                wait=True,
                                timeout=_UPLOAD_TIMEOUT_S,
                            ).result(timeout=_UPLOAD_TIMEOUT_S)
                        finally:
                            executor.shutdown(wait=False)
                        uploads.append({"dir": child.name, "status": "ok"})
                    except Exception:
                        logger.exception("upload file %s failed", child.name)
                        uploads.append({"dir": child.name, "status": "failed"})
            status = (
                "completed" if uploads and all(u["status"] == "ok" for u in uploads) else "failed"
            )
        except Exception:
            logger.exception("finalize_upload failed; local finalize stands")
            return {"status": "failed", "uploads": uploads}
        return {"status": status, "uploads": uploads}

    @staticmethod
    def _copy_dir(src: Path) -> Path:
        """Copy one directory to a temp tree for upload.

        Filenames must already be byte-capped at generation time; if any still
        exceeds the limit this raises instead of renaming, because renaming
        would orphan every URI reference that points at the original name.
        """
        tmp = Path(tempfile.mkdtemp(prefix="ov_up_"))
        dst = tmp / src.name
        dst.mkdir(parents=True)
        for file_path in src.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(src)
            target = dst / rel
            if len(target.name.encode("utf-8")) > _MAX_FILENAME_BYTES:
                raise ValueError(
                    f"filename {target.name!r} is {len(target.name.encode('utf-8'))} bytes "
                    f"(limit {_MAX_FILENAME_BYTES}); id builders must clip at generation time"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
        return tmp
