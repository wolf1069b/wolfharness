"""agentpool_cli — backward-compatible shim for wolfharness_cli.

This package has been renamed to ``wolfharness_cli``. Importing ``agentpool_cli``
is deprecated and will be removed in a future release.

Submodule imports are forwarded to the corresponding ``wolfharness_cli`` submodule.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib import abc, machinery

from wolfharness_cli import *  # noqa: F403

warnings.warn(
    "``agentpool_cli`` has been renamed to ``wolfharness_cli``. "
    "Update your imports: `from wolfharness_cli import ...`",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> object:
    """Forward top-level attribute access to wolfharness_cli."""
    import wolfharness_cli

    return getattr(wolfharness_cli, name)


class _ShimFinder(abc.MetaPathFinder):
    """Redirect agentpool_cli.X submodule imports to wolfharness_cli.X."""

    _prefix = "agentpool_cli."
    _target = "wolfharness_cli"

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
