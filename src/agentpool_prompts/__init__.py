"""agentpool_prompts — backward-compatible shim for wolfharness_prompts.

This package has been renamed to ``wolfharness_prompts``. Importing ``agentpool_prompts``
is deprecated and will be removed in a future release.

Submodule imports are forwarded to the corresponding ``wolfharness_prompts`` submodule.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib import abc, machinery

from wolfharness_prompts import *  # noqa: F403
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

warnings.warn(
    "``agentpool_prompts`` has been renamed to ``wolfharness_prompts``. "
    "Update your imports: `from wolfharness_prompts import ...`",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> object:
    """Forward top-level attribute access to wolfharness_prompts."""
    import wolfharness_prompts

    return getattr(wolfharness_prompts, name)


class _AliasLoader(abc.Loader):
    """Loader that returns an already-imported target module as-is.

    ``create_module`` hands back the ``wolfharness`` module object itself,
    so the import machinery registers it under the alias name without
    re-executing the module code. Re-execution would produce duplicate
    module and class objects, breaking identity-based checks such as
    ``isinstance`` and ``ann is SomeClass``.
    """

    def __init__(
        self,
        module: types.ModuleType,
        original_spec: machinery.ModuleSpec | None,
    ) -> None:
        self._module = module
        self._original_spec = original_spec

    def create_module(self, spec: machinery.ModuleSpec) -> types.ModuleType:
        return self._module

    def exec_module(self, module: types.ModuleType) -> None:
        # ``_init_module_attrs`` overwrites ``__spec__`` with the alias spec
        # before this runs. Restore the target module's own spec so that
        # introspection and ``importlib.reload`` keep seeing the real module.
        if self._original_spec is not None:
            module.__spec__ = self._original_spec


class _ShimFinder(abc.MetaPathFinder):
    """Redirect agentpool_prompts.X submodule imports to wolfharness_prompts.X."""

    _prefix = "agentpool_prompts."
    _target = "wolfharness_prompts"

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

        # Do not pre-set ``sys.modules[fullname]`` here: if the alias is
        # already present in ``sys.modules`` when ``_find_spec`` returns,
        # CPython reuses the target module's own spec and ``_load_unlocked``
        # re-executes the module file, duplicating every class defined in it.
        return machinery.ModuleSpec(fullname, loader=_AliasLoader(wolf_mod, wolf_mod.__spec__))


if not any(isinstance(f, _ShimFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ShimFinder())
