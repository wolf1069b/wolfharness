"""EntityWriteValidationCapability — pre-write content validation hooks.

Splits the content-validation responsibility out of the monolithic
``WikiHarnessCapability``: this capability wraps entity write tools,
auto-cleans ``open_gap`` placeholders, runs the deterministic hook chain, and
raises ``ModelRetry`` before the filesystem handler when structural errors
are found.
"""

from __future__ import annotations

from datetime import UTC, datetime
import tempfile
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT

from wolfharness.capabilities.agent_context import AgentContextDeps, resolve_agent_context_from_deps
from wolfharness.capabilities.file_team_state import FileTeamState
from wolfharness.capabilities.wiki.auto_repair import clean_open_gap
from wolfharness.capabilities.wiki.quality import parse_frontmatter
from wolfharness.capabilities.wiki.validation import (
    ENTITY_VALIDATION_HOOKS,
    FORMAL_WRITE_EXCLUDED_HOOKS,
    run_entity_validation,
    validation_feedback,
)


if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext, ToolDefinition

_TOOLS_THAT_TRIGGER_HOOKS: frozenset[str] = frozenset(
    {
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
    },
)

# Tools where we auto-clean open_gap placeholders before the write hits disk.
# write tools receive full ``content``; patch tools receive ``operations``
# (not full content) so cleaning is deferred to the batch tool.
_TOOLS_THAT_AUTO_CLEAN: frozenset[str] = frozenset(
    {
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
    },
)

_TASK_SCOPED_MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
        "patch_entity",
        "patch_entities_batch",
        "patch_symptom_profile",
        "merge_entity",
        "delete_entity",
        "move_entity",
        "create_subdir",
        "register_bom_component",
        "register_bom_identity_batch",
        "register_no_entity_chapters",
        "record_source_packet",
        "create_opa",
        "create_ops",
        "update_ops",
        "apply_ops",
        "ingest_external_ops",
        "create_opl",
        "apply_opl",
        "resolve_opa",
        "refine_opa_reason_code",
        "apply_opa",
    },
)


class EntityWriteValidationCapability(AbstractCapability[AgentDepsT]):
    """Run validation hooks before entity writes and append warning feedback.

    A validation error raises ``ModelRetry`` before ``handler`` is called,
    guaranteeing that a rejected write has not touched the filesystem.
    """

    def get_toolset(self) -> None:
        """Return ``None`` — this capability provides no tools."""
        return

    def get_instructions(self) -> str | None:
        """Return ``None`` — instructions come from the agent's system prompt."""
        return None

    @staticmethod
    def _wiki_tool_name(tool_name: str) -> str | None:
        if tool_name in _TOOLS_THAT_TRIGGER_HOOKS:
            return tool_name
        matches = [
            candidate
            for candidate in _TOOLS_THAT_TRIGGER_HOOKS
            if tool_name.endswith(f"_{candidate}")
        ]
        return max(matches, key=len) if matches else None

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Validate entity content before the write handler runs."""
        agent_ctx = resolve_agent_context_from_deps(
            ctx.deps,
            capability_name="EntityWriteValidationCapability",
        )
        task_feedback = self.validate_task_scope(call.tool_name, agent_ctx)
        if task_feedback:
            raise ModelRetry(task_feedback)
        tool_name = self._wiki_tool_name(call.tool_name)

        # Auto-clean open_gap in content before validation and write.
        if tool_name in _TOOLS_THAT_AUTO_CLEAN and isinstance(args, dict):
            content = args.get("content", "")
            if isinstance(content, str) and content:
                cleaned = clean_open_gap(content)
                if cleaned != content:
                    args = {**args, "content": cleaned}

        if tool_name not in _TOOLS_THAT_TRIGGER_HOOKS:
            return await handler(args)

        if tool_name == "write_entities_batch" and isinstance(args, dict):
            items = args.get("entities", [])
            if not isinstance(items, list):
                raise ModelRetry("Entity batch REJECTED: `entities` must be a list.")
            feedback_parts: list[str] = []
            has_batch_errors = False
            cleaned_items: list[object] = []
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ModelRetry(f"Entity batch REJECTED: entities[{index}] must be an object.")
                cleaned_item = item
                content = item.get("content", "")
                if isinstance(content, str):
                    cleaned = clean_open_gap(content)
                    if cleaned != content:
                        cleaned_item = {**item, "content": cleaned}
                feedback, has_errors = self._run_hooks("write_entity", cleaned_item)
                if feedback:
                    feedback_parts.append(f"entities[{index}]:\n{feedback}")
                has_batch_errors = has_batch_errors or has_errors
                cleaned_items.append(cleaned_item)
            if has_batch_errors:
                raise ModelRetry(
                    "Entity batch REJECTED by validation hooks:\n\n" + "\n\n".join(feedback_parts),
                )
            result = await handler({**args, "entities": cleaned_items})
            if feedback_parts:
                return f"{result!s}\n\n" + "\n\n".join(feedback_parts)
            return result

        hook_feedback, has_errors = self._run_hooks(tool_name, args)
        if has_errors:
            raise ModelRetry(
                f"Entity write REJECTED by validation hooks:\n\n{hook_feedback}\n\nPlease fix the issues above and retry the write.",
            )

        result = await handler(args)
        if not hook_feedback:
            return result
        return f"{result!s}\n\n{hook_feedback}"

    @staticmethod
    def validate_task_scope(tool_name: str, agent_ctx: AgentContextDeps) -> str:
        """Require team members to own active work before mutating the Wiki."""
        canonical = next(
            (
                candidate
                for candidate in _TASK_SCOPED_MUTATION_TOOLS
                if tool_name == candidate or tool_name.endswith(f"_{candidate}")
            ),
            None,
        )
        metadata = agent_ctx.session.metadata
        if canonical is None or metadata.get("team_role") != "member":
            return ""
        team_id = str(metadata.get("team_id", ""))
        member_name = str(metadata.get("team_member_name", ""))
        base_dir = str(metadata.get("team_base_dir", ""))
        if not base_dir and agent_ctx.team_mode_config is not None:
            base_dir = agent_ctx.team_mode_config.effective_base_dir
        if not base_dir:
            # Match AgentPool's FileTeamState runtime fallback.  Older spawned
            # sessions did not persist team_base_dir even though their shared
            # task board was correctly rooted in the process temp directory.
            base_dir = tempfile.gettempdir()
        if not team_id or not member_name:
            return "Wiki mutation REJECTED: member task identity is incomplete."
        tasks = FileTeamState(base_dir).list_tasks(team_id)
        active = [
            task
            for task in tasks
            if task.get("owner") == member_name and task.get("status") == "in_progress"
        ]
        if active:
            lease_tokens: dict[str, str] = metadata.setdefault("_task_lease_tokens", {})
            lease_seconds = int(getattr(agent_ctx.team_mode_config, "lease_ttl_seconds", 300))
            for task in active:
                task_id = str(task.get("task_id", ""))
                lease_token = str(task.get("lease_token", ""))
                lease_expires_at = str(task.get("lease_expires_at", ""))
                if not lease_token:
                    continue  # Backward-compatible read of pre-lease task files.
                held_token = str(lease_tokens.get(task_id, ""))
                if held_token and held_token != lease_token:
                    return "Wiki mutation REJECTED: task lease was replaced; do not write with a stale worker session."
                try:
                    expires = datetime.fromisoformat(lease_expires_at)
                except ValueError:
                    return "Wiki mutation REJECTED: task lease metadata is invalid."
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
                if expires <= datetime.now(UTC):
                    try:
                        reclaimed = FileTeamState(base_dir).update_task(
                            team_id,
                            task_id,
                            {"status": "in_progress"},
                            expected_lease_token=held_token or lease_token,
                            lease_owner=member_name,
                            claim=True,
                            lease_seconds=lease_seconds,
                        )
                    except (FileNotFoundError, OSError, ValueError):
                        return "Wiki mutation REJECTED: task lease expired or was replaced; reclaim the task before writing."
                    refreshed_token = str(reclaimed.get("lease_token", ""))
                    if not refreshed_token:
                        return "Wiki mutation REJECTED: task lease renewal did not return a token."
                    lease_tokens[task_id] = refreshed_token
            return ""
        return (
            "Wiki mutation REJECTED: team member "
            f"{member_name!r} must own an in_progress task before calling {canonical}. "
            "Call task_list(mine_only=True), then task_update the exact assigned task to in_progress."
        )

    def _run_hooks(self, tool_name: str, args: Any) -> tuple[str, bool]:
        """Run all validation hooks against entity content from tool args.

        Returns ``(feedback_text, has_errors)`` where ``has_errors``
        indicates whether any hook returned severity ``"error"``.
        """
        if not isinstance(args, dict):
            return "", False

        content = args.get("content", "")
        if not content or not isinstance(content, str):
            return "", False

        concept = args.get("concept", "")
        class_name = args.get("class_name", "")
        object_name = args.get(
            "object_name",
            args.get("name", args.get("symptom_name", "")),
        )
        # Workers commonly put class_name/object_name in the content frontmatter
        # rather than as explicit tool args.  Fall back to the frontmatter so a
        # structurally complete write isn't falsely rejected for "no class_name".
        fm = parse_frontmatter(content)
        if not class_name and isinstance(fm.get("class_name"), str):
            class_name = fm["class_name"]
        if not object_name and isinstance(fm.get("object_name"), str):
            object_name = fm["object_name"]
        # Tool args may carry non-str extras (Pydantic ``Unknown``); coerce to
        # strings so the validation chain always receives plain strings.
        class_name = str(class_name) if class_name else ""
        object_name = str(object_name) if object_name else ""

        results = run_entity_validation(
            content=content,
            concept=concept,
            class_name=class_name,
            object_name=object_name,
        )
        feedback, has_errors = validation_feedback(results)

        # For confirmed/machine-validated content, re-run with the formal
        # hook subset (excluding hooks that are not in the formal write gate)
        # so the capability layer matches the MCP server's _FORMAL_WRITE_HOOKS.
        # For non-confirmed/draft content, downgrade errors to warnings so
        # drafts can land on disk and be fixed in later passes.  The finalize
        # audit (not promotion-time re-validation) is the strict backstop.
        is_downgraded = False
        if has_errors:
            status = str(fm.get("status", "")).strip()
            pub_state = str(fm.get("publication_state", "")).strip()
            val_state = str(fm.get("validation_state", "")).strip()
            is_formal = (
                status in {"confirmed", "deprecated"}
                or pub_state == "published"
                or val_state == "machine_validated"
            )
            if is_formal:
                formal_hooks = tuple(
                    h for h in ENTITY_VALIDATION_HOOKS if h.name not in FORMAL_WRITE_EXCLUDED_HOOKS
                )
                results = run_entity_validation(
                    content=content,
                    concept=concept,
                    class_name=class_name,
                    object_name=object_name,
                    hooks=formal_hooks,
                )
                feedback, has_errors = validation_feedback(results)
            else:
                has_errors = False
                is_downgraded = True

        if is_downgraded and "❌" in feedback:
            feedback = feedback.replace("❌", "⚠️").replace(
                "(write blocked)", "(draft — non-blocking)"
            )

        return feedback, has_errors
