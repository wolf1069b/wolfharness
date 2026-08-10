"""Parsing logic for file-based agent definitions.

Supports loading agents from markdown files with YAML frontmatter in various formats:
- OpenCode: https://github.com/sst/opencode
- AgentPool (native): Full NativeAgentConfig fields in frontmatter
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, assert_never

from upathtools import to_upath

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.file_agents import FileAgentConfig


logger = get_logger(__name__)
AgentFileFormat = Literal["opencode", "native"]

# Fields that pass through directly to NativeAgentConfig
PASSTHROUGH_FIELDS = {
    "toolsets",
    "session",
    "output_type",
    "retries",
    "output_retries",
    "end_strategy",
    "avatar",
    "config_file_path",
    "knowledge",
    "workers",
    "debug",
    "usage_limits",
    "tool_mode",
    "display_name",
    "triggers",
}


def extract_frontmatter(content: str, file_path: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from markdown content.

    Args:
        content: Full markdown file content
        file_path: Path for error messages

    Returns:
        Tuple of (metadata dict, system prompt body)

    Raises:
        ValueError: If frontmatter is missing or invalid
    """
    import yamling

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No YAML frontmatter found in {file_path}")

    try:
        metadata = yamling.load_yaml(frontmatter_match.group(1))
    except yamling.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter in {file_path}: {e}") from e

    if not isinstance(metadata, dict):
        raise ValueError(f"YAML frontmatter must be a dictionary in {file_path}")  # noqa: TRY004

    system_prompt = content[frontmatter_match.end() :].strip()
    return metadata, system_prompt


def detect_format(metadata: dict[str, Any]) -> AgentFileFormat:
    """Detect the file format based on frontmatter content.

    Args:
        metadata: Parsed YAML frontmatter

    Returns:
        Detected format: "opencode" or "native"
    """
    # OpenCode indicators
    is_opencode = (
        any(key in metadata for key in ["mode", "temperature", "maxSteps", "disable"])
        or ("tools" in metadata and isinstance(metadata["tools"], dict))
        or ("permission" in metadata and isinstance(metadata["permission"], dict))
    )
    if is_opencode:
        return "opencode"

    # Default to native format
    return "native"


def parse_opencode_format(
    metadata: dict[str, Any],
    system_prompt: str,
    file_path: str,
) -> dict[str, Any]:
    """Parse OpenCode format frontmatter.

    Args:
        metadata: Parsed YAML frontmatter
        system_prompt: Markdown body content
        file_path: Path for logging

    Returns:
        Dict of NativeAgentConfig kwargs
    """
    config_kwargs: dict[str, Any] = {}

    # Description
    if description := metadata.get("description"):
        config_kwargs["description"] = description

    # Model handling
    if (model := metadata.get("model")) and model != "inherit":
        config_kwargs["model"] = model

    # Temperature (logged, not directly supported)
    if temperature := metadata.get("temperature"):
        logger.debug(
            "OpenCode temperature not directly supported",
            temperature=temperature,
            file_path=file_path,
        )

    # MaxSteps (logged, not directly supported)
    if max_steps := metadata.get("maxSteps"):
        logger.debug(
            "OpenCode maxSteps not directly supported", max_steps=max_steps, file_path=file_path
        )

    # Disable (logged, not directly supported)
    if disable := metadata.get("disable"):
        logger.debug(
            "OpenCode disable not directly supported", disable=disable, file_path=file_path
        )

    # Mode (informational only)
    if mode := metadata.get("mode"):
        logger.debug("OpenCode mode informational only", mode=mode, file_path=file_path)

    # Permission handling (granular per-tool)
    if permission := metadata.get("permission"):
        edit_perm = permission.get("edit") if isinstance(permission, dict) else None
        if edit_perm in ("deny", "ask"):
            config_kwargs["requires_tool_confirmation"] = (
                "always" if edit_perm == "ask" else "never"
            )
        logger.debug(
            "OpenCode permission partial mapping", permission=permission, file_path=file_path
        )

    # Tools dict format (not yet supported)
    if (tools := metadata.get("tools")) and isinstance(tools, dict):
        logger.debug("OpenCode tools dict not yet supported", tools=tools, file_path=file_path)

    # System prompt from markdown body
    if system_prompt:
        config_kwargs["system_prompt"] = system_prompt

    # Pass through wolfharness specific fields
    for field in PASSTHROUGH_FIELDS:
        if field in metadata:
            config_kwargs[field] = metadata[field]

    return config_kwargs


def parse_native_format(metadata: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """Parse native format frontmatter.

    This format allows full NativeAgentConfig fields in the frontmatter,
    with the markdown body used as system prompt.

    Args:
        metadata: Parsed YAML frontmatter
        system_prompt: Markdown body content

    Returns:
        Dict of NativeAgentConfig kwargs
    """
    # Start with all metadata (it's already in NativeAgentConfig format)
    config_kwargs = dict(metadata)

    # Add system prompt from body if present and not already defined
    if system_prompt and "system_prompt" not in config_kwargs:
        config_kwargs["system_prompt"] = system_prompt

    return config_kwargs


def parse_agent_file(
    file_path: str,
    *,
    file_format: AgentFileFormat | Literal["auto"] = "auto",
    skills_registry: Any | None = None,
) -> NativeAgentConfig:
    """Parse agent markdown file to NativeAgentConfig.

    Supports OpenCode and native formats with auto-detection.
    Also supports local and remote paths via UPath.

    Args:
        file_path: Path to .md file with YAML frontmatter (local or remote)
        file_format: File format to use ("auto" for detection, or explicit format)
        skills_registry: Optional skills registry for loading skills

    Returns:
        Parsed NativeAgentConfig

    Raises:
        ValueError: If file cannot be parsed
    """
    from wolfharness.models.agents import NativeAgentConfig

    path = to_upath(file_path)
    content = path.read_text("utf-8")

    # Extract frontmatter and body
    metadata, system_prompt = extract_frontmatter(content, file_path)

    # Detect or use specified format
    detected_format = detect_format(metadata) if file_format == "auto" else file_format
    match detected_format:
        case "opencode":
            config_kwargs = parse_opencode_format(metadata, system_prompt, file_path)
        case "native":
            config_kwargs = parse_native_format(metadata, system_prompt)
        case _ as unreachable:
            assert_never(unreachable)

    return NativeAgentConfig(**config_kwargs)


def parse_file_agent_reference(
    reference: str | FileAgentConfig,
    *,
    skills_registry: Any | None = None,
) -> NativeAgentConfig:
    """Parse a file agent reference (path string or config) to NativeAgentConfig.

    Args:
        reference: Either a path string (auto-detect format) or FileAgentConfig
            with explicit type discriminator
        skills_registry: Optional skills registry for loading skills

    Returns:
        Parsed NativeAgentConfig
    """
    if isinstance(reference, str):
        # Simple path string: auto-detect format
        return parse_agent_file(reference, skills_registry=skills_registry)

    # Explicit config: use type as format (claude maps to native)
    fmt: AgentFileFormat = "native" if reference.type == "claude" else reference.type
    return parse_agent_file(
        reference.path,
        file_format=fmt,
        skills_registry=skills_registry,
    )
