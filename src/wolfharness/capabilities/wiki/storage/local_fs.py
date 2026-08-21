"""Local-directory backend.  Read side doubles as the historical-lib fallback."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from .backend import FSBackend, _strip_control_chars


class LocalFS(FSBackend):
    """Filesystem backend rooted at a local directory."""

    def __init__(self, root: Path, *, root_uri: str | None = None) -> None:
        self.root = Path(root).resolve()
        self._root_uri_override = root_uri

    @property
    def root_uri(self) -> str:
        if self._root_uri_override is not None:
            return self._root_uri_override
        return f"file://{self.root}"

    def _p(self, key: str) -> Path:
        path = self.root.joinpath(*key.split("/"))
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"path escapes backend root: {key!r}")
        return resolved

    def read_text(self, key: str) -> str | None:
        try:
            p = self._p(key)
        except ValueError:
            return None
        if not p.is_file():
            return None
        return _strip_control_chars(p.read_text(encoding="utf-8"))

    def write_text(self, key: str, content: str, *, overwrite: bool = True) -> None:
        p = self._p(key)
        if not overwrite and p.is_file():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_name = tmp.name
        Path(tmp_name).replace(p)

    def write_text_durable(self, key: str, content: str) -> None:
        self.write_text(key, content)

    def commit_many(self, writes: list[tuple[str, str]]) -> None:
        """Commit local publication writes with rollback on process errors.

        LocalFS is a development backend without a server-side transaction.
        Staging every replacement before the first rename prevents preparation
        failures from changing visible files; retained in-memory originals
        allow rollback if a rename fails.
        """
        if not writes:
            return
        staged: list[tuple[Path, Path]] = []
        originals: dict[Path, bytes | None] = {}
        try:
            for key, content in writes:
                target = self._p(key)
                target.parent.mkdir(parents=True, exist_ok=True)
                originals[target] = target.read_bytes() if target.is_file() else None
                with NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as tmp:
                    tmp.write(content)
                    tmp.flush()
                    staged.append((Path(tmp.name), target))
            for temporary, target in staged:
                temporary.replace(target)
        except OSError:
            for target, original in originals.items():
                if original is None:
                    if target.is_file():
                        target.unlink()
                else:
                    target.write_bytes(original)
            raise
        finally:
            for temporary, _target in staged:
                if temporary.is_file():
                    temporary.unlink()

    def exists(self, key: str) -> bool:
        try:
            return self._p(key).is_file()
        except ValueError:
            return False

    def is_dir(self, key: str) -> bool:
        try:
            return self._p(key).is_dir()
        except ValueError:
            return False

    def list_dir(self, key: str, *, recursive: bool = False) -> list[str]:
        base = self._p(key)
        if not base.is_dir():
            return []
        pattern = "**/*" if recursive else "*"
        out: list[str] = []
        for p in sorted(base.glob(pattern)):
            if not p.is_file():
                continue
            out.append(str(p.relative_to(self.root)))
        return out

    def list_entries(self, key: str, *, recursive: bool = False) -> list[str]:
        """List immediate child path-keys (files *and* directories) in one level."""
        base = self._p(key)
        if not base.is_dir():
            return []
        out: list[str] = []
        for p in sorted(base.glob("*")):
            out.append(str(p.relative_to(self.root)))
        return out

    def mtime_ns(self, key: str) -> int | None:
        p = self._p(key)
        try:
            st = p.stat()
        except OSError:
            return None
        return st.st_mtime_ns

    def size(self, key: str) -> int | None:
        p = self._p(key)
        try:
            st = p.stat()
        except OSError:
            return None
        return st.st_size

    def mkdir_p(self, key: str) -> None:
        self._p(key).mkdir(parents=True, exist_ok=True)

    def delete(self, key: str) -> None:
        p = self._p(key)
        if p.is_file():
            p.unlink()

    def move(self, src: str, dst: str) -> None:
        sp = self._p(src)
        dp = self._p(dst)
        dp.parent.mkdir(parents=True, exist_ok=True)
        if sp.is_file():
            sp.replace(dp)
