"""Utility for loading tool schema from YAML or JSON files."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from schemez.functionschema import OpenAIFunctionDefinition
import yaml


if TYPE_CHECKING:
    from pydantic_ai import Tool


def load_tool_schema(path: str | Path | None) -> OpenAIFunctionDefinition | None:
    """Load tool schema from a YAML or JSON file.

    Args:
        path: Path to the schema file. Can be a string or Path object.
            If None, returns None.

    Returns:
        The parsed schema as a dictionary, or None if path is None.

    Raises:
        FileNotFoundError: If path is provided but the file doesn't exist.
        ValueError: If the file cannot be parsed as valid YAML or JSON.

    Examples:
        >>> schema = load_tool_schema("tools/my_tool.yaml")
        >>> schema = load_tool_schema(Path("tools/my_tool.json"))
        >>> schema = load_tool_schema(None)  # Returns None
    """
    if path is None:
        return None

    # Convert to Path object if string
    file_path = Path(path)

    # Check if file exists - fail fast
    if not file_path.exists():
        raise FileNotFoundError(f"Tool schema file not found: {file_path}")

    # Read the file content
    content = file_path.read_text(encoding="utf-8")

    # Try to parse based on file extension
    suffix = file_path.suffix.lower()

    try:
        if suffix in (".yaml", ".yml"):
            return _build_definition(yaml.safe_load(content))
        if suffix == ".json":
            return _build_definition(json.loads(content))
        # Try JSON first, then YAML if extension is unclear
        try:
            return _build_definition(json.loads(content))
        except json.JSONDecodeError:
            return _build_definition(yaml.safe_load(content))
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse tool schema file {file_path}: {e}") from e


def _build_definition(data: Any) -> OpenAIFunctionDefinition:
    """Construct an ``OpenAIFunctionDefinition`` from parsed JSON/YAML data.

    Args:
        data: The parsed data (must be a dict with ``name``, ``description``,
            and ``parameters`` keys).

    Returns:
        A validated ``OpenAIFunctionDefinition``.

    Raises:
        ValueError: If ``data`` is not a mapping, or is missing the required
            ``name`` key.
    """
    try:
        raw = cast(dict[str, Any], data)
        name = cast(str, raw["name"])
    except (KeyError, TypeError) as e:
        if not isinstance(data, dict):
            msg = f"Tool schema must be a mapping with a 'name' key, got {type(data).__name__}"
        else:
            msg = "Tool schema is missing required key 'name'"
        raise ValueError(msg) from e
    return OpenAIFunctionDefinition(
        name=name,
        description=cast(str, raw.get("description", "")),
        parameters=cast(Any, raw.get("parameters", {})),
    )


def apply_params_schema(
    tool: Tool[object],
    schema: OpenAIFunctionDefinition | None,
) -> Tool[object]:
    """Override ``tool.function_schema.json_schema`` with YAML schema's ``parameters``.

    Args:
        tool: The pydantic-ai Tool to update.
        schema: Optional schema dictionary from ``load_tool_schema``. If ``None``
            or missing a ``"parameters"`` key, the tool is returned unchanged.

    Returns:
        The tool with its ``function_schema.json_schema`` overridden, or the
        original tool if no parameters were found.
    """
    if schema is not None:
        params = schema.get("parameters")
        if params is not None:
            tool.function_schema = replace(
                tool.function_schema,
                json_schema=dict[str, Any](params),
            )
    return tool
