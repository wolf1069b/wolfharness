"""Harness validation for the OPA → OPS → OPL proposal flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT


if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext, ToolDefinition


class OPFlowGuardCapability(AbstractCapability[AgentDepsT]):
    """Reject empty or directly publishable OP records before the tool call."""

    def get_toolset(self) -> None:
        return

    def get_instructions(self) -> str | None:
        return None

    def validate(self, tool_name: str, args: object) -> str:
        if not isinstance(args, dict):
            return ""
        if tool_name.endswith("create_ops"):
            if str(args.get("status", "unconfirmed")) != "unconfirmed":
                return "OPS proposals must be created with status=unconfirmed."
            missing = [
                name
                for name in ("parent_opa", "title", "retrieval_query", "analysis", "solution")
                if not str(args.get(name, "")).strip()
            ]
            if not isinstance(args.get("retrieved_uris"), list):
                missing.append("retrieved_uris")
            if not isinstance(args.get("evidence_uris"), list) or not args["evidence_uris"]:
                missing.append("evidence_uris")
            return "OPS is incomplete: " + ", ".join(missing) if missing else ""
        if tool_name.endswith("create_opl"):
            if str(args.get("status", "unconfirmed")) != "unconfirmed":
                return "OPL proposals must remain status=unconfirmed."
            missing = [
                name
                for name in ("parent_opa", "title", "proposal", "rationale")
                if not str(args.get(name, "")).strip()
            ]
            if not isinstance(args.get("ops_uris"), list) or not args["ops_uris"]:
                missing.append("ops_uris")
            if not isinstance(args.get("evidence_uris"), list) or not args["evidence_uris"]:
                missing.append("evidence_uris")
            return "OPL is incomplete: " + ", ".join(missing) if missing else ""
        return ""

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        feedback = self.validate(call.tool_name, args)
        if feedback:
            raise ModelRetry(f"OP flow harness rejected the call:\n\n{feedback}")
        return await handler(args)
