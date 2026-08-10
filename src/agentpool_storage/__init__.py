"""agentpool_storage — backward-compatible shim for wolfharness_storage.

This package has been renamed to ``wolfharness_storage``. Importing ``agentpool_storage``
is deprecated and will be removed in a future release.

Submodule imports are forwarded to the corresponding ``wolfharness_storage`` submodule.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib import abc, machinery

from wolfharness_storage import *  # noqa: F403

warnings.warn(
    "``agentpool_storage`` has been renamed to ``wolfharness_storage``. "
    "Update your imports: `from wolfharness_storage import ...`",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> object:
    """Forward top-level attribute access to wolfharness_storage."""
    import wolfharness_storage

    return getattr(wolfharness_storage, name)


class _ShimFinder(abc.MetaPathFinder):
    """Redirect agentpool_storage.X submodule imports to wolfharness_storage.X."""

    _prefix = "agentpool_storage."
    _target = "wolfharness_storage"

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> machinery.ModuleSpec | None:
        if not fullname.startswith(self._prefix):
            return None

        wolf_name = self._target + fullname[len(self._prefix) - 1 :]
        try:
            wolf_mod = importlib.import_module(wolf_name)
        except ModuleNotFoundError:
            return None

        sys.modules[fullname] = wolf_mod
        return machinery.ModuleSpec(fullname, loader=None)


if not any(isinstance(f, _ShimFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ShimFinder())
