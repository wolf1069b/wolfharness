"""TeamWorkflowGuardCapability — conductor team-orchestration guardrails.

Splits the team-workflow validation responsibility out of the monolithic
``WikiHarnessCapability``: this capability validates ``task_create`` calls
made by the conductor before they are dispatched, rejecting under-specified
or oversized extraction tasks via ``ModelRetry``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT

from wolfharness.capabilities.agent_context import AgentContextDeps, resolve_agent_context_from_deps
from wolfharness.capabilities.file_team_state import FileTeamState


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic_ai.capabilities.abstract import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext, ToolDefinition

_PROCEDURE_ONLY_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "diagnosis",
        "inspection",
        "measurement",
        "maintenance",
        "removal",
        "installation",
        "replacement",
        "adjustment",
        "repair",
        "test",
    },
)
_COMMON_INVALID_PROCEDURE_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "assembly",
        "calibration",
        "disassembly",
    },
)
_CHAPTER_REF_RE = re.compile(r"\bch\d{4}\b")
_CHAPTER_COUNT_RE = re.compile(
    r"(?:\bchapter_count\s*[:=]\s*(\d+)"
    r"|\b(?:chapter|chapters|章节)\s*(?:count|数量|数)?\s*[:=]?\s*(\d+)"
    r"|\b(\d+)\s*(?:chapters?|章节))",
    re.IGNORECASE,
)
_CLASS_LINE_RE = re.compile(r"\bclass(?:_name)?\s*[:：]\s*([^\n]+)", re.IGNORECASE)
_PACKET_ID_RE = re.compile(r"^[a-zA-Z0-9_]+$")
_PACKET_DECL_RE = re.compile(
    r"\bpacket_id\s*[:=]\s*['\"]?([A-Za-z0-9_]+)(?=$|[\s,.;。；，}\]])",
    re.IGNORECASE,
)
_PROFILE_DECL_RE = re.compile(r"\baudit_profile\s*[:=]\s*['\"]?(manual|case)\b", re.IGNORECASE)
_EXPECTED_ARTIFACTS_RE = re.compile(r"\bexpected_artifacts?\s*[:=]", re.IGNORECASE)
_PHASE_1A_RE = re.compile(
    r"(?:\b1a\b|phase\s*1a|source[- ]analysis|source\s+packet)", re.IGNORECASE
)
_PHASE_0_RE = re.compile(r"(?:\bphase\s*[:=]?\s*(?:phase)?0\b|\bphase0\b)", re.IGNORECASE)
_PHASE_0_OPERATION_RE = re.compile(
    r"\bphase0_operation\s*[:=]\s*['\"]?"
    r"(component_write|device_write|no_entity_register)\b",
    re.IGNORECASE,
)
_PHASE_0_IDENTITY_OPERATION_RE = re.compile(
    r"\bphase0_operation\s*[:=]\s*['\"]?identity_plan\b",
    re.IGNORECASE,
)
_BOM_SOURCE_COUNT_RE = re.compile(r"\bbom_source_count\s*[:=]\s*(\d+)\b", re.IGNORECASE)
_RESOLVED_COMPONENT_COUNT_RE = re.compile(
    r"\bresolved_component_count\s*[:=]\s*(\d+)\b",
    re.IGNORECASE,
)
_EXTRACTION_TASK_RE = re.compile(
    r"(?:phase\s*[:=]?\s*1[ab]|source[- ]analysis|source\s+packet|entity_type\s*[:=])",
    re.IGNORECASE,
)
_CHUNK_DECL_RE = re.compile(r"(?:^|\s)chunk_id\s*[:=]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)


def _chunk_id_of(description: str) -> str:
    """Extract the declared ``chunk_id`` from a task description, if any."""
    match = _CHUNK_DECL_RE.search(description)
    return match.group(1).strip() if match is not None else ""


_PHASE_DONE_RE = re.compile(r'"phase"\s*:\s*"done"')
_WORKER_ROLE_RE = re.compile(
    r"\bworker_role\s*[:=]\s*['\"]?"
    r"(wiki_(?:(?:extraction|relation|opa|ops|opl)_worker|file_operator))\b",
    re.IGNORECASE,
)
_OP_SUBJECT_RE = re.compile(r"\b(OPA|OPS|OPL)\b", re.IGNORECASE)
_OP_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "wiki_opa_worker": ("target_uri", "evidence_uris", "expected_artifacts"),
    "wiki_ops_worker": (
        "parent_opa",
        "retrieval_query",
        "evidence_uris",
        "expected_artifacts",
    ),
    "wiki_opl_worker": ("parent_opa", "ops_uris", "evidence_uris", "expected_artifacts"),
}


class TeamWorkflowGuardCapability(AbstractCapability[AgentDepsT]):
    """Validate conductor team-tool calls (single and batch creation).

    Only fires for the ``wiki_conductor`` role; other roles pass through
    untouched.  Violations raise ``ModelRetry`` before the task is created.
    """

    def __init__(
        self,
        *,
        role: str = "wiki_conductor",
        max_build_state_bytes: int = 65536,
        max_phase0_components_per_task: int = 5,
    ) -> None:
        if max_build_state_bytes < 1024:
            raise ValueError("max_build_state_bytes must be at least 1024")
        if max_phase0_components_per_task < 1:
            raise ValueError("max_phase0_components_per_task must be positive")
        self._role = role
        self._max_build_state_bytes = max_build_state_bytes
        self._max_phase0_components_per_task = max_phase0_components_per_task

    def get_toolset(self) -> None:
        """Return ``None`` — this capability provides no tools."""
        return

    def get_instructions(self) -> str | None:
        """Return ``None`` — team conventions live in the workflow skill."""
        return None

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Reject under-specified extraction tasks before dispatch."""
        feedback = self.validate(call.tool_name, args)
        if not feedback and call.tool_name.endswith("write_blackboard"):
            feedback = self.validate_build_state_write(args)
        if feedback:
            raise ModelRetry(
                f"Team workflow action REJECTED by wiki harness:\n\n{feedback}\n\nPlease fix the orchestration issue and retry.",
            )
        agent_ctx = resolve_agent_context_from_deps(
            ctx.deps,
            capability_name="TeamWorkflowGuardCapability",
        )
        assignment_feedback = ""
        if call.tool_name.endswith("task_create") or call.tool_name.endswith("task_create_batch"):
            assignment_feedback = self._validate_runtime_assignment(args, agent_ctx)
        if assignment_feedback:
            raise ModelRetry(
                f"OP task assignment REJECTED by wiki harness:\n\n{assignment_feedback}\n\nAssign the task to a member running the declared OP worker role.",
            )
        is_done_write = call.tool_name.endswith("write_blackboard") and self._is_done_write(args)
        if not is_done_write:
            # Never let a previous successful finalize authorize a done write
            # after another tool call (including a failed finalize attempt).
            agent_ctx.session.metadata.pop("_wiki_finalize_receipt", None)
        if is_done_write:
            done_feedback = self.validate_done_transition(
                args,
                agent_ctx.session.metadata.get("_wiki_finalize_receipt"),
            )
            if done_feedback:
                raise ModelRetry(
                    f"Build completion REJECTED by wiki harness:\n\n{done_feedback}\n\nCall finalize_wiki successfully and persist its receipt before retrying.",
                )

        result = await handler(args)
        if call.tool_name.endswith("finalize_wiki") and isinstance(result, dict):
            if (
                result.get("status") in ("finalized", "finalized_local")
                and result.get("op_flow_passed") is True
            ):
                agent_ctx.session.metadata["_wiki_finalize_receipt"] = {
                    "audit_profile": str(result.get("audit_profile", "")),
                    "source_snapshot_id": str(result.get("source_snapshot_id", "")),
                }
        elif is_done_write:
            agent_ctx.session.metadata.pop("_wiki_finalize_receipt", None)
        return result

    def _validate_checkpoint_build(self, args: Any) -> str:
        """Require an explicit build_id on every conductor checkpoint write.

        A checkpoint_build call that omits build_id lets the server mint a
        fresh auto-generated id over an in-progress build, silently detaching
        every plan and source-packet owner (all chapters stay pending).
        The conductor must always pass the canonical build identity; recovery
        paths that legitimately omit it are server-internal and never flow
        through the conductor tool.  Rejecting here turns the root-cause bug
        into a prompt-time error instead of silent data corruption.
        """
        if not isinstance(args, dict):
            return "checkpoint_build requires an argument object."
        build_id = str(args.get("build_id", "")).strip()
        if not build_id:
            return (
                "checkpoint_build must declare an explicit `build_id` matching this build. "
                "Omitting it auto-generates a new id that detaches every plan and source "
                "packet (all chapters stay pending forever). Pass the build_id used at "
                "attach time (see inspect_build_checkpoint / build_state)."
            )
        return ""

    def _validate_runtime_assignment(self, args: Any, agent_ctx: AgentContextDeps) -> str:
        """Bind specialized task contracts to the owner's registered agent."""
        if not isinstance(args, dict):
            return ""
        tasks_value = args.get("tasks")
        raw_tasks: list[Any] = tasks_value if isinstance(tasks_value, list) else [args]
        errors: list[str] = []
        team_id = str(agent_ctx.session.metadata.get("team_id", ""))
        base_dir = str(agent_ctx.session.metadata.get("team_base_dir", ""))
        if not base_dir and agent_ctx.team_mode_config is not None:
            base_dir = agent_ctx.team_mode_config.effective_base_dir
        team_state = FileTeamState(base_dir) if team_id and base_dir else None
        existing_tasks = team_state.list_tasks(team_id) if team_state else []
        roster = team_state.list_member_names(team_id) if team_state else set()
        for index, task in enumerate(raw_tasks, 1):
            if not isinstance(task, dict):
                continue
            subject = str(task.get("subject", ""))
            owner = str(task.get("owner", task.get("assigned_to", ""))).strip()
            description = str(task.get("description", ""))
            new_chunk_id = _chunk_id_of(description)
            # Reject tasks for workers that were never created via team_add_member.
            # task_create/task_create_batch is only for existing idle workers with
            # live sessions; a conductor that missed team_add_member would spin
            # forever creating tasks nobody can claim.
            if team_state and owner and owner not in roster:
                errors.append(
                    f"Task {index}: owner '{owner}' is not a registered team member. "
                    f"Use team_add_member(initial_task=...) to create this worker "
                    f"before assigning tasks."
                )
            if team_state:
                same_subject = [
                    existing
                    for existing in existing_tasks
                    if existing.get("owner") == owner
                    and str(existing.get("subject", "")).strip() == subject.strip()
                    and existing.get("status") not in {"completed", "cancelled", "deleted"}
                ]
                duplicate = [
                    existing
                    for existing in same_subject
                    if not new_chunk_id
                    or _chunk_id_of(str(existing.get("description", ""))) == new_chunk_id
                ]
                if duplicate:
                    sample_id = str(duplicate[0].get("task_id", "?"))
                    sample_status = str(duplicate[0].get("status", "?"))
                    errors.append(
                        f"Task {index}: duplicate dispatch — an active task with the same "
                        f"owner and subject already exists ({sample_id}, status={sample_status}). "
                        "Recheck the task board before re-dispatching; reuse the existing task_id.",
                    )
            text = f"{subject}\n{description}"
            role_match = _WORKER_ROLE_RE.search(text)
            if role_match is None:
                if _OP_SUBJECT_RE.search(subject):
                    errors.append(f"Task {index}: OP tasks must declare `worker_role`. ")
                elif _EXTRACTION_TASK_RE.search(text):
                    owner = str(task.get("owner", task.get("assigned_to", ""))).strip()
                    owner_agent = (
                        team_state.get_member_agent(team_id, owner)
                        if team_state and owner
                        else None
                    )
                    if owner_agent != "wiki_extraction_worker":
                        errors.append(
                            f"Task {index}: extraction task owner {owner!r} runs {owner_agent or 'unknown'!r}, expected 'wiki_extraction_worker'.",
                        )
                continue
            role = role_match.group(1).lower()
            owner = str(task.get("owner", task.get("assigned_to", ""))).strip()
            owner_agent = (
                team_state.get_member_agent(team_id, owner) if team_state and owner else None
            )
            if not owner:
                errors.append(f"Task {index}: {role} task must have an explicit owner.")
            elif owner_agent != role:
                errors.append(
                    f"Task {index}: owner {owner!r} runs {owner_agent or 'unknown'!r}, expected {role!r}.",
                )
            if role in _OP_REQUIRED_FIELDS:
                missing = [field for field in _OP_REQUIRED_FIELDS[role] if field not in text]
                if missing:
                    errors.append(f"Task {index}: {role} task is missing {', '.join(missing)}.")
                if role in {"wiki_ops_worker", "wiki_opl_worker"} and "blocked_by" not in task:
                    # OP DAG roots legitimately declare blocked_by: [] (no
                    # predecessor); only a missing field is a contract breach.
                    errors.append(
                        f"Task {index}: {role} task must declare `blocked_by` for the OP DAG."
                    )
        return "\n".join(errors)

    def _validate_op_assignment(self, args: Any, agent_ctx: AgentContextDeps) -> str:
        """Backward-compatible alias for the expanded runtime assignment guard."""
        return self._validate_runtime_assignment(args, agent_ctx)

    @staticmethod
    def _is_done_write(args: Any) -> bool:
        """Return whether a blackboard write requests the terminal phase."""
        if not isinstance(args, dict) or str(args.get("key", "")) != "build_state":
            return False
        return _PHASE_DONE_RE.search(str(args.get("value", ""))) is not None

    def validate_build_state_write(self, args: Any) -> str:
        """Keep the durable recovery state bounded and structurally readable."""
        if not isinstance(args, dict) or str(args.get("key", "")) != "build_state":
            return ""
        if str(args.get("mode", "overwrite")) != "overwrite":
            return "`build_state` is a snapshot and must use mode='overwrite', never append."
        value = args.get("value", "")
        if not isinstance(value, str):
            return "`build_state` must be serialized as one JSON object string."
        size = len(value.encode("utf-8"))
        if size > self._max_build_state_bytes:
            return (
                f"`build_state` is {size} bytes, exceeding the configured "
                f"{self._max_build_state_bytes}-byte recovery-state limit. "
                "Keep only phase, snapshot ids, counts, task ids, packet ids, and blockers."
            )
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return "`build_state` must be valid JSON so patrol recovery is deterministic."
        if not isinstance(parsed, dict):
            return "`build_state` must be a JSON object."
        return ""

    @staticmethod
    def validate_done_transition(args: Any, receipt: object) -> str:
        """Require a matching, same-session finalize receipt for ``done``."""
        if not isinstance(args, dict) or not isinstance(receipt, dict):
            return "`build_state.phase=done` requires a successful same-session finalize receipt."
        value = str(args.get("value", ""))
        try:
            build_state = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return "The terminal build_state must be a JSON object."
        if not isinstance(build_state, dict):
            return "The terminal build_state must be a JSON object."
        audit_profile = str(receipt.get("audit_profile", ""))
        source_snapshot_id = str(receipt.get("source_snapshot_id", ""))
        if not audit_profile or not source_snapshot_id:
            return "Finalize receipt is incomplete; audit profile and source snapshot are required."
        if (
            build_state.get("audit_profile") != audit_profile
            or build_state.get("source_snapshot_id") != source_snapshot_id
        ):
            return "The done build_state must persist the finalize audit profile and source snapshot id."
        return ""

    def validate(self, tool_name: str, args: Any) -> str:
        """Return rejection feedback for an invalid team-tool call, else ``""``."""
        if self._role != "wiki_conductor":
            return ""
        if tool_name.endswith("checkpoint_build"):
            return self._validate_checkpoint_build(args)
        is_batch = tool_name.endswith("task_create_batch")
        if not is_batch and not tool_name.endswith("task_create"):
            return ""
        if not isinstance(args, dict):
            return ""

        if is_batch:
            raw_tasks = args.get("tasks", args.get("items", args.get("task_specs", [])))
            if not isinstance(raw_tasks, list):
                return "Batch task creation must provide a list under `tasks`."
            errors: list[str] = []
            for index, task in enumerate(raw_tasks, 1):
                feedback = self.validate("task_create", task)
                if feedback:
                    errors.append(f"Task {index}:\n{feedback}")
            return "\n".join(errors)

        owner = str(args.get("owner", args.get("assigned_to", "")))
        subject = str(args.get("subject", ""))
        description = str(args.get("description", ""))
        text = f"{subject}\n{description}"
        if owner.startswith(("file_op", "file_operator")) and _PHASE_0_RE.search(text):
            return self._validate_phase0_file_operation(text)
        if not owner.startswith(("extraction_worker", "extraction_")):
            return ""
        if _PHASE_0_RE.search(text):
            return self._validate_phase0_identity_operation(args, text)

        is_phase_1a = _PHASE_1A_RE.search(text) is not None
        is_procedure_task = "Procedure" in text or "1B-2" in text

        errors: list[str] = []
        packet_id = str(args.get("packet_id", "")).strip()
        if not packet_id:
            packet_match = _PACKET_DECL_RE.search(text)
            packet_id = packet_match.group(1) if packet_match is not None else ""
        if not packet_id:
            errors.append(
                "Extraction tasks must declare the stable source `packet_id` they consume or produce."
            )
        elif is_phase_1a and not _PACKET_ID_RE.fullmatch(packet_id):
            errors.append(
                "Phase 1A `packet_id` must use only ASCII letters, digits, and `_`; use a stable source-domain slug or content-derived identifier.",
            )
        if _PROFILE_DECL_RE.search(text) is None:
            errors.append(
                "Extraction tasks must declare the immutable `audit_profile` (`manual` or `case`)."
            )
        if _EXPECTED_ARTIFACTS_RE.search(text) is None:
            errors.append(
                "Extraction tasks must declare `expected_artifacts` for completion checks."
            )
        chapter_refs = set(_CHAPTER_REF_RE.findall(text))
        if is_phase_1a:
            declared_counts = {
                int(value)
                for match in _CHAPTER_COUNT_RE.finditer(text)
                for value in match.groups()
                if value is not None
            }
            if not declared_counts:
                errors.append(
                    "Phase 1A source-analysis tasks must declare `chapter_count` for bounded dispatch.",
                )
            if len(chapter_refs) > 6 or any(count > 6 for count in declared_counts):
                errors.append(
                    "Phase 1A source-analysis tasks must contain at most 6 chapters per source packet. "
                    f"This task references {len(chapter_refs)} explicit chapters and declares counts "
                    f"{sorted(declared_counts)}.",
                )
            return "\n".join(f"- {error}" for error in errors)
        is_symptom_task = "1B-4" in text or "Symptom/Profile" in text
        if not is_procedure_task and not is_symptom_task:
            return "\n".join(f"- {error}" for error in errors)
        if len(chapter_refs) > 6:
            errors.append(
                f"Procedure extraction tasks must be chunked to at most 6 explicit chapters. This task references {len(chapter_refs)} chapters: {', '.join(sorted(chapter_refs))}.",
            )

        if "chunk_id" not in text or "chunk_of" not in text:
            errors.append(
                "Procedure extraction tasks must declare `chunk_id` and `chunk_of` so conductor can track, retry, or reassign the chunk independently.",
            )

        if "heartbeat" not in text.lower() or "task_update" not in text:
            errors.append(
                "Procedure extraction tasks must require a first-turn `task_update` heartbeat before entity writes.",
            )

        if "depends_on_stage" not in text:
            errors.append("Procedure extraction tasks must declare `depends_on_stage: 1B-2`.")

        if is_procedure_task:
            invalid_classes: set[str] = set()
            for match in _CLASS_LINE_RE.finditer(text):
                class_text = match.group(1).lower()
                for raw_token in re.split(r"[^a-z_]+", class_text):
                    token = raw_token.strip("_")
                    if token in _COMMON_INVALID_PROCEDURE_CLASS_NAMES or (
                        token and token.endswith("ing") and token not in _PROCEDURE_ONLY_CLASS_NAMES
                    ):
                        invalid_classes.add(token)
            if invalid_classes:
                errors.append(
                    "Procedure class_name must use the canonical 10-class vocabulary; invalid values: "
                    + ", ".join(sorted(invalid_classes))
                    + ". Map 拆解/分解/装配内容 to removal/installation/replacement/repair as appropriate.",
                )

        # Symptom identity is part of the URI hash.  Allowing the worker to
        # write a Symptom with an empty/temporary class creates dangling Fault
        # and Profile links.  Fail the task before it starts so the conductor
        # chooses one source-backed class and reuses it for write_entity and
        # write_symptom_profile.
        if "1B-4" in text or "Symptom/Profile" in text:
            class_matches = re.findall(
                r"class_name\s*[:=]\s*['\"]?([^'\"\n,)]*)",
                text,
                flags=re.IGNORECASE,
            )
            normalized = {match.strip() for match in class_matches if match.strip()}
            if not normalized or any(value in {"", "none", "null"} for value in normalized):
                errors.append(
                    "Symptom/Profile tasks must declare one non-empty source-backed class_name and reuse it for write_entity and write_symptom_profile.",
                )
            elif len(normalized) > 1:
                errors.append(
                    f"Symptom/Profile task declares multiple class_name values ({', '.join(sorted(normalized))}); choose one stable class for the batch.",
                )
            if "same class_name" not in text and "稳定" not in text and "reuse" not in text.lower():
                errors.append(
                    "Symptom/Profile task must state that the same class_name is reused for all entity writes.",
                )

        return "\n".join(f"- {error}" for error in errors)

    @staticmethod
    def _validate_phase0_identity_operation(args: Mapping[str, object], text: str) -> str:
        """Require one resolved BOM input for the semantic identity planner."""
        errors: list[str] = []
        if _PHASE_0_IDENTITY_OPERATION_RE.search(text) is None:
            errors.append(
                "Phase 0 extraction tasks must declare `phase0_operation=identity_plan`; structural writes belong in separate file-operator tasks.",
            )
        packet_id = str(args.get("packet_id", "")).strip()
        if not packet_id:
            packet_match = _PACKET_DECL_RE.search(text)
            packet_id = packet_match.group(1) if packet_match is not None else ""
        if not packet_id or _PACKET_ID_RE.fullmatch(packet_id) is None:
            errors.append("Phase 0 identity plans require a stable ASCII `packet_id`.")
        if _PROFILE_DECL_RE.search(text) is None:
            errors.append("Phase 0 identity plans must declare the immutable `audit_profile`.")
        if _EXPECTED_ARTIFACTS_RE.search(text) is None:
            errors.append("Phase 0 identity plans must declare `expected_artifacts`.")
        if "target_model" not in text:
            errors.append("Phase 0 identity plans must declare the `target_model`.")
        if "config_scope" not in text:
            errors.append("Phase 0 identity plans must declare a bounded `config_scope`.")
        if "bom_source_uri" not in text:
            errors.append("Phase 0 identity plans require one resolved `bom_source_uri`.")
        source_count_match = _BOM_SOURCE_COUNT_RE.search(text)
        if source_count_match is None or int(source_count_match.group(1)) != 1:
            errors.append(
                "Phase 0 identity plans must declare `bom_source_count=1`; split multiple BOM files into independent packets.",
            )
        return "\n".join(f"- {error}" for error in errors)

    def _validate_phase0_file_operation(self, text: str) -> str:
        """Keep Phase 0 file tasks deterministic, resumable, and bounded."""
        errors: list[str] = []
        operation_match = _PHASE_0_OPERATION_RE.search(text)
        if operation_match is None:
            errors.append(
                "Phase 0 file tasks must declare one `phase0_operation`: "
                "`component_write`, `device_write`, or `no_entity_register`. "
                "BOM identity planning belongs in a persisted extraction packet before file writes; "
                "do not combine identity planning, Component writes, Device creation, and coverage registration.",
            )
            return "\n".join(f"- {error}" for error in errors)

        operation = operation_match.group(1).lower()
        if _PROFILE_DECL_RE.search(text) is None:
            errors.append("Phase 0 file tasks must declare the immutable `audit_profile`.")
        if _EXPECTED_ARTIFACTS_RE.search(text) is None:
            errors.append("Phase 0 file tasks must declare `expected_artifacts`.")

        if operation == "component_write":
            if "resolved_components" not in text:
                errors.append(
                    "`component_write` requires `resolved_components` from the persisted BOM identity packet.",
                )
            if "source_packet_uri" not in text:
                errors.append(
                    "`component_write` requires a real `source_packet_uri` for resumable input."
                )
            count_match = _RESOLVED_COMPONENT_COUNT_RE.search(text)
            if count_match is None:
                errors.append("`component_write` must declare `resolved_component_count`.")
            else:
                count = int(count_match.group(1))
                if count < 1 or count > self._max_phase0_components_per_task:
                    errors.append(
                        "Phase 0 component writes must contain at most "
                        f"{self._max_phase0_components_per_task} resolved Components; got {count}. "
                        "Split by target Component URI so each chunk can retry independently.",
                    )
        elif operation == "device_write":
            if "resolved_component_uris" not in text:
                errors.append("`device_write` requires `resolved_component_uris`.")
            if "resolved_system_chapter_uris" not in text:
                errors.append("`device_write` requires `resolved_system_chapter_uris`.")
            if "source_packet_uri" not in text:
                errors.append("`device_write` requires the Phase 0 identity `source_packet_uri`.")
        elif operation == "no_entity_register":
            if "resolved_source_uris" not in text:
                errors.append("`no_entity_register` requires explicit `resolved_source_uris`.")
            if "packet_id" not in text:
                errors.append("`no_entity_register` requires a stable ASCII `packet_id`.")

        return "\n".join(f"- {error}" for error in errors)
