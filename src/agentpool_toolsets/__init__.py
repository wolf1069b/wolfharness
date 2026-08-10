"""agentpool_toolsets — backward-compatible shim for wolfharness_toolsets.

This package has been renamed to ``wolfharness_toolsets``. Importing ``agentpool_toolsets``
is deprecated and will be removed in a future release.

Submodule imports are forwarded to the corresponding ``wolfharness_toolsets`` submodule.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib import abc, machinery

from wolfharness_toolsets import *  # noqa: F403

warnings.warn(
    "``agentpool_toolsets`` has been renamed to ``wolfharness_toolsets``. "
    "Update your imports: `from wolfharness_toolsets import ...`",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> object:
    """Forward top-level attribute access to wolfharness_toolsets."""
    import wolfharness_toolsets

    return getattr(wolfharness_toolsets, name)


class _ShimFinder(abc.MetaPathFinder):
    """Redirect agentpool_toolsets.X submodule imports to wolfharness_toolsets.X."""

    _prefix = "agentpool_toolsets."
    _target = "wolfharness_toolsets"

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
