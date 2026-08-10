"""CodeModeCapability — wraps all agent tools into a single ``execute_code`` meta-tool.

Replaces ``CodeModeCapability`` with a native ``AbstractCapability``
that exposes one Python-execution tool. The LLM writes an ``async def main()``
function that can call any wrapped tool as a regular async function.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
import inspect
from typing import TYPE_CHECKING, Any

import anyenv
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from schemez.code_generation.namespace_callable import NamespaceCallable

from wolfharness.capabilities.codemode.default_prompt import USAGE
from wolfharness.capabilities.codemode.helpers import (
    tools_to_codegen,
    validate_code,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from wolfharness.tools.base import Tool


_CODE_MODE_INSTRUCTIONS = (
    """\
You have access to a single `execute_code` tool that lets you run Python code.
All available tools are exposed as async functions inside the code environment.

Write an `async def main()` function that calls the provided tool functions
and returns a result. Do NOT call `main()` yourself — the runtime does that.

Example:
```
async def main():
    result = await some_tool(arg1="value")
    return result
```
"""
    + USAGE
)


class CodeModeCapability(AbstractCapability[AgentDepsT]):
    """Capability that wraps all agent tools into a single ``execute_code`` meta-tool.

    The meta-tool accepts Python source code as a string. The code must define
    an ``async def main()`` function that can call any wrapped tool as a regular
    async function. The runtime executes the code, calls ``main()``, and returns
    the result.

    Tools are wrapped once at construction time and remain static for the
    capability's lifetime.
    """

    def __init__(
        self,
        tools: Sequence[Tool[Any]],
        *,
        include_docstrings: bool = True,
        toolset_id: str = "code_mode",
    ) -> None:
        """Initialize the code mode capability.

        Args:
            tools: AgentPool ``Tool`` instances to wrap into the meta-tool.
            include_docstrings: Include function docstrings in generated code.
            toolset_id: Identifier for the produced ``FunctionToolset``.
        """
        self._tools = list(tools)
        self._include_docstrings = include_docstrings
        self._toolset_id = toolset_id
        self._toolset: FunctionToolset[AgentDepsT] | None = None

    async def __aenter__(self) -> CodeModeCapability[AgentDepsT]:
        """Enter async context — no-op (no resources to acquire)."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit async context — no-op (no resources to release)."""

    def on_change(self) -> AsyncIterator[Any] | None:
        """Return change notifications, or ``None`` if tools are static.

        Tools are fixed at construction time, so there are never changes.
        """
        return None

    def get_instructions(self) -> str | None:
        """Return code mode usage instructions for the system prompt."""
        return _CODE_MODE_INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a single-tool ``FunctionToolset`` with the ``execute_code`` meta-tool."""
        if self._toolset is None:
            self._toolset = FunctionToolset(
                [self._execute_code],
                id=self._toolset_id,
            )
        return self._toolset

    async def _execute_code(
        self,
        ctx: RunContext[AgentDepsT],
        python_code: str,
        title: str,
    ) -> Any:
        """Execute Python code with all wrapped tools available as functions.

        Args:
            ctx: The run context providing agent dependencies.
            python_code: Python code to execute. Must define ``async def main()``.
            title: Short descriptive title for this script (3-4 words).
        """
        toolset_generator = tools_to_codegen(
            tools=self._tools,
            include_docstrings=self._include_docstrings,
        )
        namespace = toolset_generator.generate_execution_namespace()

        for value in namespace.values():
            if isinstance(value, NamespaceCallable):
                original_callable = value.callable
                if "agent_ctx" in inspect.signature(original_callable).parameters:
                    value.callable = partial(original_callable, agent_ctx=ctx)

        validate_code(python_code)
        start_time = datetime.now(UTC)
        exit_code = 0
        error_msg: str | None = None
        result_value: Any = None

        try:
            exec(python_code, namespace)
            result_value = namespace["main"]()
            if inspect.isawaitable(result_value):
                result_value = await result_value
            if not result_value:
                result_value = "Code executed successfully"
        except Exception as e:  # noqa: BLE001
            exit_code = 1
            error_msg = f"{e!s}"
            result_value = f"Error executing code: {error_msg}"
        finally:
            end_time = datetime.now(UTC)
            duration = (end_time - start_time).total_seconds()
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")

            script_path = f"codemode/scripts/{timestamp}_{title}.py"
            metadata = {
                "title": title,
                "timestamp": start_time.isoformat(),
                "exit_code": exit_code,
                "duration": duration,
                "result": str(result_value),
                "error": error_msg,
            }
            metadata_path = f"codemode/scripts/{timestamp}_{title}.json"
            metadata_json = anyenv.dump_json(metadata, indent=True).encode("utf-8")
            script_bytes = python_code.encode("utf-8")

            # Write script and metadata to an in-memory filesystem.
            # In production, agents have a persistent _internal_fs, but
            # AgentContext doesn't expose it directly. The script history
            # is best-effort — if lost, it doesn't affect execution.
            from fsspec.implementations.memory import MemoryFileSystem

            fs = MemoryFileSystem()
            fs.pipe(script_path, script_bytes)
            fs.pipe(metadata_path, metadata_json)

        return result_value


__all__ = ["CodeModeCapability"]
