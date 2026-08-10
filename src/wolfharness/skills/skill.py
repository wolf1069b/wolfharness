"""Claude Code Skill following the Agent Skills Spec."""

from __future__ import annotations

import html
import json
import logging
import os
from pathlib import PurePosixPath
from typing import Annotated, Any
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from upathtools import UPath

from wolfharness_config.skills import SkillMcpServerConfig, SkillToolConfig


# Type for skill paths - can be UPath (for local filesystem skills)
# or PurePosixPath (for virtual skill:// URIs from MCP providers)
SkillPathType = UPath | PurePosixPath


# Last synced with https://github.com/agentskills/agentskills
SPEC_SYNCED_COMMIT = "f019a02dbbb1302217c4b4a14557d6384d9ace9a"

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500


class Skill(BaseModel):
    """A Claude Code Skill with metadata and lazy-loaded instructions.

    Follows the Agent Skills Spec for field naming and validation rules.
    Frontmatter fields use ``extra="forbid"`` so unknown YAML keys are rejected.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Annotated[str, Field(max_length=MAX_SKILL_NAME_LENGTH)]
    description: Annotated[str, Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH)]
    skill_path: SkillPathType
    license: str | None = None
    compatibility: Annotated[str | None, Field(max_length=MAX_COMPATIBILITY_LENGTH)] = None
    allowed_tools: str | None = Field(default=None, alias="allowed-tools")
    metadata: dict[str, str] = Field(default_factory=dict)
    instructions: str | None = Field(default=None, exclude=True)

    # Model invocation control
    disable_model_invocation: bool = Field(default=False, alias="disable-model-invocation")

    # User invocation control
    user_invocable: bool = Field(default=True, alias="user-invocable")

    # Context preservation setting (e.g., "fork", "continue")
    context: str | None = None

    # Agent type compatibility (e.g., "general-purpose", "coding")
    agent: str | None = None

    # Argument hint for slash command completion
    argument_hint: str | None = Field(default=None, alias="argument-hint")

    # MCP server configurations for skill-provided tools
    mcp_servers: dict[str, SkillMcpServerConfig] | None = Field(default=None, alias="mcp-servers")

    # Tool configurations for skill-provided functionality
    tools: list[SkillToolConfig] | None = Field(default=None)

    # Resolved reference path set by SkillURIResolver when the URI contains
    # a reference file path (e.g., skill://skill-name/path/to/ref.md).
    # Not a frontmatter field — set programmatically during resolution.
    resolved_reference_path: str | None = Field(default=None, exclude=True)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        name = unicodedata.normalize("NFKC", v.strip())
        # Normalize underscores to hyphens per Agent Skills Spec (kebab-case).
        # The spec mandates "lowercase letters, numbers, and hyphens only".
        name = name.replace("_", "-")
        if not name:
            raise ValueError("Skill name must be non-empty")
        if name != name.lower():
            raise ValueError(f"Skill name {name!r} must be lowercase")
        if name.startswith("-") or name.endswith("-"):
            raise ValueError("Skill name cannot start or end with a hyphen")
        if "--" in name:
            raise ValueError("Skill name cannot contain consecutive hyphens")
        if not all(c.isalnum() or c == "-" for c in name):
            msg = (
                f"Skill name {name!r} contains invalid characters. "
                "Only letters, digits, and hyphens are allowed."
            )
            raise ValueError(msg)
        return name

    @model_validator(mode="before")
    @classmethod
    def _normalize_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict) and "metadata" in data:
            meta = data["metadata"]
            if isinstance(meta, dict):
                data["metadata"] = {str(k): str(v) for k, v in meta.items()}
        return data

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _normalize_allowed_tools(cls, v: Any) -> str | None:
        """Normalize allowed_tools: accept list[str] in code, keep str from YAML.

        When constructing Skill objects programmatically, callers may pass
        a list of tool names for convenience. This validator normalizes
        those to the space-separated string format that matches what YAML
        frontmatter parsing produces.
        """
        if isinstance(v, list):
            return " ".join(v)
        if isinstance(v, str) or v is None:
            return v
        return str(v)

    def parsed_allowed_tools(self) -> list[str]:
        """Parse allowed_tools into a list of individual tool names.

        Returns:
            List of tool name strings, split by comma or whitespace.
            Returns an empty list if allowed_tools is None.
        """
        if self.allowed_tools is None:
            return []
        return [
            token.strip() for token in self.allowed_tools.replace(",", " ").split() if token.strip()
        ]

    @property
    def safe_uri(self) -> str:
        """Return a safe ``skill://`` URI for external exposure.

        Returns the flat ``skill://{name}`` form for local filesystem skills.
        For virtual/MCP skills (PurePosixPath), returns the existing URI as-is.

        This property ensures that absolute filesystem paths are never leaked
        to LLM prompts, tool outputs, or API responses.
        """
        if not isinstance(self.skill_path, UPath):
            # Virtual/MCP skill — already has a skill:// or mcp:// URI
            return str(self.skill_path)
        # Local filesystem skill — generate a flat skill:// URI
        return f"skill://{self.name}"

    def load_instructions(self) -> str:
        """Lazy-load full instructions from SKILL.md.

        For local filesystem skills (UPath), loads from disk.
        For virtual skills (PurePosixPath like skill:// URIs), the instructions
        must be pre-set during skill creation or fetched via the provider.

        Raises:
            ValueError: If called on a virtual skill without pre-set instructions.
                For MCP-based skills, use provider.get_skill_instructions() instead.
        """
        if self.instructions is None:
            # Check for virtual paths (PurePosixPath that is not a UPath)
            # UPath registers as a virtual subclass of PurePosixPath, so
            # isinstance(x, PurePosixPath) is True for both. We use
            # isinstance(x, UPath) to distinguish real filesystem paths.
            if not isinstance(self.skill_path, UPath):
                raise ValueError(
                    f"Cannot load instructions for virtual skill '{self.name}'. "
                    "Instructions must be pre-set during skill creation or fetched "
                    "via provider.get_skill_instructions() for MCP-based skills."
                )

            # UPath represents actual filesystem paths
            skill_file = self.skill_path / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8")
                parts = content.split("---", 2)
                self.instructions = parts[2].strip() if len(parts) >= 3 else ""  # noqa: PLR2004
            else:
                self.instructions = ""
        return self.instructions

    @classmethod
    def from_skill_dir(cls, skill_dir: SkillPathType) -> Skill:
        """Parse a SKILL.md file from a directory and return a validated Skill.

        Args:
            skill_dir: Path to the skill directory containing SKILL.md.

        Returns:
            Validated Skill instance.

        Raises:
            FileNotFoundError: If SKILL.md is not found.
            ValueError: If frontmatter is missing or invalid YAML.
            ValidationError: If fields fail Pydantic validation.
        """
        # Check for virtual paths (PurePosixPath that is not a UPath)
        if not isinstance(skill_dir, UPath):
            raise TypeError(
                "Cannot load skill from virtual path using from_skill_dir. "
                "Use provider.get_skill_instructions() for MCP-based skills."
            )

        skill_file = find_skill_md(skill_dir)
        if skill_file is None:
            raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

        # find_skill_md returns UPath | None, and we already checked skill_dir is UPath
        if not isinstance(skill_file, UPath):
            raise TypeError("Virtual paths cannot be read from filesystem")
        content = skill_file.read_text("utf-8")
        metadata, _body = parse_frontmatter(content)

        # Load companion mcp.json (takes precedence over frontmatter mcp-servers)
        mcp_servers = _load_mcp_json(skill_dir)
        if mcp_servers is not None:
            metadata["mcp-servers"] = mcp_servers

        return cls(skill_path=skill_dir, **metadata)


def _expand_env_vars_in_value(value: Any) -> Any:
    """Recursively expand ``${VAR}`` environment variables in all string values.

    Args:
        value: A value that may contain strings needing env var expansion.

    Returns:
        The value with all strings having env vars expanded.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env_vars_in_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars_in_value(item) for item in value]
    return value


def _load_mcp_json(
    skill_dir: SkillPathType,
) -> dict[str, SkillMcpServerConfig] | None:
    """Load companion ``mcp.json`` file from a skill directory.

    Looks for a ``mcp.json`` file in the skill directory that follows the
    Claude Desktop MCP server configuration format:

    .. code-block:: json

        {
            "mcpServers": {
                "server-name": {
                    "command": "npx",
                    "args": ["-y", "@playwright/mcp"]
                }
            }
        }

    Environment variables (``${VAR}`` syntax) are expanded in all string values.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        Dictionary of server name to ``SkillMcpServerConfig``, or ``None``
        if no ``mcp.json`` file exists or it cannot be parsed.
    """
    # Only load mcp.json from filesystem paths (UPath), not virtual paths
    if not isinstance(skill_dir, UPath):
        return None

    mcp_json_path = skill_dir / "mcp.json"
    if not mcp_json_path.exists():
        return None

    try:
        raw = json.loads(mcp_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logging.getLogger(__name__).warning("Failed to parse mcp.json in %s: %s", skill_dir, exc)
        return None

    servers_raw = raw.get("mcpServers")
    if not isinstance(servers_raw, dict):
        return None

    result: dict[str, SkillMcpServerConfig] = {}
    for name, entry in servers_raw.items():
        if not isinstance(entry, dict):
            continue
        # Expand env vars in all string values within the entry
        expanded = _expand_env_vars_in_value(entry)
        result[name] = SkillMcpServerConfig(**expanded)

    return result


def find_skill_md(skill_dir: SkillPathType) -> SkillPathType | None:
    """Find the SKILL.md file in a skill directory.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        Path to the SKILL.md file, or None if not found.
    """
    # Check for virtual paths (PurePosixPath that is not a UPath)
    if not isinstance(skill_dir, UPath):
        return skill_dir / "SKILL.md"

    # UPath represents actual filesystem paths
    path = skill_dir / "SKILL.md"
    return path if path.exists() else None


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md content.

    Args:
        content: Raw content of a SKILL.md file.

    Returns:
        Tuple of (metadata dict, markdown body).

    Raises:
        ValueError: If frontmatter is missing or invalid.
    """
    import yamling

    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    parts = content.split("---", 2)
    if len(parts) < 3:  # noqa: PLR2004
        raise ValueError("SKILL.md frontmatter not properly closed with ---")
    try:
        metadata = yamling.load_yaml(parts[1])
    except yamling.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}") from e

    if not isinstance(metadata, dict):
        raise TypeError("SKILL.md frontmatter must be a YAML mapping")

    return metadata, parts[2].strip()


def to_prompt(skills: list[Skill]) -> str:
    """Generate the ``<available_skills>`` XML block for agent system prompts.

    This XML format is what Anthropic uses and recommends for Claude models.

    Args:
        skills: List of skills to include.

    Returns:
        XML string with ``<available_skills>`` block.
    """
    if not skills:
        return "<available_skills>\n</available_skills>"

    lines = ["<available_skills>"]
    for skill in skills:
        # Skip skills that disable model invocation
        if skill.disable_model_invocation:
            continue

        # Build optional metadata attributes
        attrs: list[str] = []
        if not skill.user_invocable:
            attrs.append('user-invocable="false"')
        if skill.context:
            attrs.append(f'context="{html.escape(skill.context)}"')
        if skill.agent:
            attrs.append(f'agent="{html.escape(skill.agent)}"')

        attr_str = " " + " ".join(attrs) if attrs else ""

        lines.append(f"<skill{attr_str}>")
        lines.append(f"<name>{html.escape(skill.name)}</name>")
        lines.append(f"<description>{html.escape(skill.description)}</description>")

        # Add argument hint if present
        if skill.argument_hint:
            lines.append(f"<argument-hint>{html.escape(skill.argument_hint)}</argument-hint>")

        lines.append(f"<uri>{html.escape(skill.safe_uri)}</uri>")
        lines.append("</skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)
