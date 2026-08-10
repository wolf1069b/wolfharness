"""Test enum elicitation response handling in ACP server."""

from __future__ import annotations

from mcp import types
import pytest

from acp import RequestPermissionResponse
from wolfharness_server.acp_server.input_provider import (
    _create_array_enum_elicitation_options,
    _create_oneof_elicitation_options,
    _handle_array_enum_elicitation_response,
    _handle_enum_elicitation_response,
    _handle_oneof_elicitation_response,
    _is_array_enum_schema,
    _is_enum_schema,
    _is_oneof_schema,
)


pytestmark = pytest.mark.integration


def test_enum_elicitation_response_returns_dict_format():
    """Test that enum elicitation response returns content in correct dict format.

    According to MCP spec and OpenCode server implementation,
    content should be wrapped in a dict with a "value" key.

    This test reproduces the bug where the ACP server returns
    a plain string instead of {"value": "..."}
    """
    # Create a mock schema with enum options
    schema = {
        "type": "string",
        "enum": ["option_a", "option_b", "option_c"],
    }

    # Simulate user selecting the first option
    response = RequestPermissionResponse.model_validate({
        "outcome": {
            "outcome": "selected",
            "optionId": "enum_0_option_a",
        }
    })

    # Call the function
    result = _handle_enum_elicitation_response(response, schema)

    # Verify the result is an ElicitResult (not ErrorData)
    assert isinstance(result, types.ElicitResult)

    # According to MCP spec and OpenCode implementation,
    # content should be a dict with "value" key
    assert result.action == "accept"
    assert isinstance(result.content, dict), (
        f"Expected content to be dict, got {type(result.content)}"
    )
    assert "value" in result.content, (
        f"Expected content to have 'value' key, got keys: {result.content.keys()}"
    )
    assert result.content["value"] == "option_a", (
        f"Expected content['value'] to be 'option_a', got '{result.content.get('value')}'"
    )


def test_enum_elicitation_response_handles_cancel():
    """Test that the cancel option is correctly handled."""
    schema = {
        "type": "string",
        "enum": ["option1", "option2"],
    }

    response = RequestPermissionResponse.model_validate({
        "outcome": {
            "outcome": "selected",
            "optionId": "cancel",
        }
    })

    result = _handle_enum_elicitation_response(response, schema)

    # Verify the result is an ElicitResult (not ErrorData)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "cancel"


def test_enum_elicitation_response_handles_multiple_options():
    """Test that different enum options are correctly handled."""
    schema = {
        "type": "string",
        "enum": ["option_x", "option_y", "option_z"],
    }

    # Test selecting the second option
    response = RequestPermissionResponse.model_validate({
        "outcome": {
            "outcome": "selected",
            "optionId": "enum_1_option_y",
        }
    })

    result = _handle_enum_elicitation_response(response, schema)

    # Verify the result is an ElicitResult (not ErrorData)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert isinstance(result.content, dict)
    assert result.content["value"] == "option_y"

    # Test selecting the third option
    response2 = RequestPermissionResponse.model_validate({
        "outcome": {
            "outcome": "selected",
            "optionId": "enum_2_option_z",
        }
    })

    result2 = _handle_enum_elicitation_response(response2, schema)

    # Verify the result is an ElicitResult (not ErrorData)
    assert isinstance(result2, types.ElicitResult)
    assert result2.action == "accept"
    assert isinstance(result2.content, dict)
    assert result2.content["value"] == "option_z"


# ─── oneOf schema detection ───


def test_is_oneof_schema_detects_valid_oneof():
    """Test that _is_oneof_schema detects oneOf schemas with const entries."""
    schema = {
        "type": "string",
        "oneOf": [
            {"const": "A", "title": "Option A"},
            {"const": "B", "title": "Option B"},
        ],
    }
    assert _is_oneof_schema(schema) is True


def test_is_oneof_schema_rejects_without_const():
    """Test that _is_oneof_schema rejects oneOf entries without const."""
    schema = {
        "type": "string",
        "oneOf": [
            {"title": "Option A"},
            {"title": "Option B"},
        ],
    }
    assert _is_oneof_schema(schema) is False


def test_is_oneof_schema_rejects_plain_enum():
    """Test that _is_oneof_schema does not match plain enum schemas."""
    schema = {"type": "string", "enum": ["A", "B"]}
    assert _is_oneof_schema(schema) is False


def test_is_oneof_schema_rejects_non_string_type():
    """Test that _is_oneof_schema rejects non-string types."""
    schema = {
        "type": "number",
        "oneOf": [{"const": 1}],
    }
    assert _is_oneof_schema(schema) is False


# ─── array-enum schema detection ───


def test_is_array_enum_schema_detects_valid_array_enum():
    """Test that _is_array_enum_schema detects array schemas with enum items."""
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": ["A", "B", "C"]},
    }
    assert _is_array_enum_schema(schema) is True


def test_is_array_enum_schema_rejects_plain_string():
    """Test that _is_array_enum_schema rejects plain string schemas."""
    schema = {"type": "string", "enum": ["A", "B"]}
    assert _is_array_enum_schema(schema) is False


def test_is_array_enum_schema_rejects_empty_enum():
    """Test that _is_array_enum_schema rejects empty enum arrays."""
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": []},
    }
    assert _is_array_enum_schema(schema) is False


# ─── oneOf option extraction ───


def test_create_oneof_elicitation_options_extracts_const_and_title():
    """Test that oneOf options are extracted with title as label and const as value."""
    schema = {
        "type": "string",
        "oneOf": [
            {"const": "A", "title": "Option A"},
            {"const": "B", "title": "Option B"},
        ],
    }
    options = _create_oneof_elicitation_options(schema)
    assert options is not None
    assert len(options) == 3  # 2 options + cancel
    assert options[0].option_id == "oneof_0_A"
    assert options[0].name == "Option A"
    assert options[1].option_id == "oneof_1_B"
    assert options[1].name == "Option B"
    assert options[2].option_id == "cancel"


def test_create_oneof_elicitation_options_fallback_to_str_const():
    """Test that oneOf options fall back to const string when title is absent."""
    schema = {
        "type": "string",
        "oneOf": [
            {"const": "X"},
        ],
    }
    options = _create_oneof_elicitation_options(schema)
    assert options is not None
    assert options[0].name == "X"


def test_create_oneof_elicitation_options_returns_none_for_invalid():
    """Test that oneOf extractor returns None for schemas without const entries."""
    schema = {
        "type": "string",
        "oneOf": [
            {"title": "No const here"},
        ],
    }
    assert _create_oneof_elicitation_options(schema) is None


# ─── array-enum option extraction ───


def test_create_array_enum_elicitation_options_extracts_enum_values():
    """Test that array-enum options are extracted from items.enum."""
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": ["Red", "Green", "Blue"]},
    }
    options = _create_array_enum_elicitation_options(schema)
    assert options is not None
    assert len(options) == 4  # 3 options + cancel
    assert options[0].option_id == "array_enum_0_Red"
    assert options[0].name == "Red"
    assert options[1].name == "Green"


def test_create_array_enum_elicitation_options_uses_descriptions():
    """Test that array-enum options use x-option-descriptions for labels."""
    schema = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["A", "B"],
            "x-option-descriptions": {"A": "Alpha", "B": "Beta"},
        },
    }
    options = _create_array_enum_elicitation_options(schema)
    assert options is not None
    assert options[0].name == "Alpha"
    assert options[1].name == "Beta"


def test_create_array_enum_elicitation_options_returns_none_for_invalid():
    """Test that array-enum extractor returns None for schemas without enum."""
    schema = {
        "type": "array",
        "items": {"type": "string"},
    }
    assert _create_array_enum_elicitation_options(schema) is None


# ─── oneOf response handling ───


def test_handle_oneof_elicitation_response_returns_const_value():
    """Test that oneOf response returns the const value, not the title."""
    schema = {
        "type": "string",
        "oneOf": [
            {"const": "val_a", "title": "Option A"},
            {"const": "val_b", "title": "Option B"},
        ],
    }
    response = RequestPermissionResponse.model_validate({
        "outcome": {
            "outcome": "selected",
            "optionId": "oneof_0_val_a",
        }
    })
    result = _handle_oneof_elicitation_response(response, schema)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"value": "val_a"}


def test_handle_oneof_elicitation_response_handles_cancel():
    """Test that oneOf cancel is handled correctly."""
    schema = {"type": "string", "oneOf": [{"const": "X"}]}
    response = RequestPermissionResponse.model_validate({
        "outcome": {"outcome": "selected", "optionId": "cancel"}
    })
    result = _handle_oneof_elicitation_response(response, schema)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "cancel"


# ─── array-enum response handling ───


def test_handle_array_enum_elicitation_response_returns_list():
    """Test that array-enum response returns value as a single-element list."""
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": ["A", "B"]},
    }
    response = RequestPermissionResponse.model_validate({
        "outcome": {
            "outcome": "selected",
            "optionId": "array_enum_1_B",
        }
    })
    result = _handle_array_enum_elicitation_response(response, schema)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"value": ["B"]}


def test_handle_array_enum_elicitation_response_handles_cancel():
    """Test that array-enum cancel is handled correctly."""
    schema = {"type": "array", "items": {"enum": ["X"]}}
    response = RequestPermissionResponse.model_validate({
        "outcome": {"outcome": "selected", "optionId": "cancel"}
    })
    result = _handle_array_enum_elicitation_response(response, schema)
    assert isinstance(result, types.ElicitResult)
    assert result.action == "cancel"


# ─── Backward compatibility ───


def test_is_enum_schema_still_works_for_legacy():
    """Test that existing enum schema detection is unchanged."""
    schema = {"type": "string", "enum": ["A", "B"]}
    assert _is_enum_schema(schema) is True
    assert _is_oneof_schema(schema) is False
    assert _is_array_enum_schema(schema) is False
