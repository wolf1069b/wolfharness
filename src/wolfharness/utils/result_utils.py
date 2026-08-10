"""Response utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai.output import ToolOutput
from schemez import InlineSchemaDef


if TYPE_CHECKING:
    from wolfharness_config.output_types import StructuredResponseConfig


def to_type(
    output_type: Any,
    # output_type: str | InlineSchemaDef | type | None,
    responses: dict[str, StructuredResponseConfig] | None = None,
) -> type[BaseModel | str] | ToolOutput[Any]:
    match output_type:
        case str() if responses and output_type in responses:
            defn = responses[output_type]
            schema_type = defn.response_schema.get_schema()
            if defn.result_tool_name != "final_result" or defn.result_tool_description:
                return ToolOutput(
                    schema_type,
                    name=defn.result_tool_name,
                    description=defn.result_tool_description,
                )
            return schema_type
        case str():
            raise ValueError(f"Missing responses dict for response type: {output_type!r}")
        case InlineSchemaDef():
            return output_type.get_schema()
        case None:
            return str
        case type() as model if issubclass(model, BaseModel | str):
            return model
        case ToolOutput():
            return output_type
        case _:
            raise TypeError(f"Invalid output_type: {type(output_type)}")
