"""Regression tests: deprecated shim imports must not duplicate modules.

The ``agentpool*`` shim packages redirect submodule imports to the
corresponding ``wolfharness*`` modules. A previous implementation pre-set
``sys.modules[alias]`` inside ``find_spec`` and returned a dummy
``ModuleSpec(name, loader=None)``. CPython's ``_find_spec`` then reused the
target module's own spec, so ``_load_unlocked`` re-executed the module file
and duplicated every class defined in it. Identity-based checks (such as
``ann is AgentContext`` in ``wrap_tool_for_pydantic_ai``) silently failed
across the alias boundary, bypassing tool argument injection and producing
runtime errors like::

    tool_callable() missing 1 required positional argument: 'agent_ctx'

Each check runs in a fresh subprocess so the import order can be controlled
deterministically; in-process module caches would mask the ordering that
triggers the bug.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit

_TARGET_ALIAS_PAIRS = [
    ("wolfharness.agents.context", "agentpool.agents.context", "AgentContext"),
    ("wolfharness.mcp_server.client", "agentpool.mcp_server.client", "MCPClient"),
]


@pytest.mark.parametrize(("wolf", "alias", "cls"), _TARGET_ALIAS_PAIRS)
@pytest.mark.parametrize("order", ["wolf_first", "alias_first"])
def test_shim_import_shares_module_and_class_objects(
    wolf: str, alias: str, cls: str, order: str
) -> None:
    """Importing via the shim must yield the same module and class objects."""
    first, second = (wolf, alias) if order == "wolf_first" else (alias, wolf)
    code = (
        "import sys\n"
        f"import {first}\n"
        f"import {second}\n"
        f"alias_mod = sys.modules[{alias!r}]\n"
        f"wolf_mod = sys.modules[{wolf!r}]\n"
        "assert alias_mod is wolf_mod, 'shim created a duplicate module object'\n"
        f"assert alias_mod.{cls} is wolf_mod.{cls}, 'shim duplicated {cls}'\n"
        f"from {alias} import {cls} as alias_cls\n"
        f"from {wolf} import {cls} as wolf_cls\n"
        "assert alias_cls is wolf_cls, 'from-import produced different classes'\n"
        "spec = wolf_mod.__spec__\n"
        f"assert spec is not None and spec.name == {wolf!r}, spec\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.parametrize(("wolf", "alias", "cls"), _TARGET_ALIAS_PAIRS)
def test_shim_module_spec_is_preserved(wolf: str, alias: str, cls: str) -> None:
    """The target module's ``__spec__`` must not be clobbered by the alias spec."""
    code = (
        "import sys\n"
        f"import {alias}\n"
        f"wolf_mod = sys.modules[{wolf!r}]\n"
        "spec = wolf_mod.__spec__\n"
        f"assert spec is not None and spec.name == {wolf!r}, spec\n"
        "assert spec.loader is not None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
