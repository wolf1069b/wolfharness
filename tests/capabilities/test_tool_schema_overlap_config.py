"""Unit tests for `ToolSchemaOverlapConfig` construction-time validation.

Covers task 7.1 (config model validation, sentinel semantics, optionals,
construction-time collision checks) plus the construction-time halves of
adversarial tasks 7.9 (parameter rename collision detected before listing)
and 7.11 (enum/default incompatibility rejected at construction).
"""

from __future__ import annotations

import copy

from pydantic import ValidationError
import pytest

from wolfharness.capabilities.tool_schema_overlap_config import (
    UNDEFINED,
    ParamOverride,
    ToolOverride,
    ToolSchemaOverlapConfig,
)


pytestmark = pytest.mark.unit


class TestUndefinedSentinel:
    """Behavior of the UNDEFINED sentinel meaning "leave the default unchanged"."""

    def test_is_falsy(self) -> None:
        assert not UNDEFINED
        assert bool(UNDEFINED) is False

    def test_equality_is_identity(self) -> None:
        sentinel = UNDEFINED
        assert sentinel == UNDEFINED
        assert UNDEFINED != None  # noqa: E711  # explicit None comparison is the point
        assert UNDEFINED != ""
        assert UNDEFINED != 0

    def test_copy_and_deepcopy_return_singleton(self) -> None:
        assert copy.copy(UNDEFINED) is UNDEFINED
        assert copy.deepcopy(UNDEFINED) is UNDEFINED

    def test_repr_is_sentinel_name(self) -> None:
        assert repr(UNDEFINED) == "UNDEFINED"

    def test_explicit_none_default_is_preserved(self) -> None:
        override = ParamOverride(default=None)
        assert override.default is None
        assert override.default is not UNDEFINED

    def test_omitted_default_is_sentinel(self) -> None:
        assert ParamOverride().default is UNDEFINED


class TestParamOverride:
    """Construction-time validation of single-parameter overrides."""

    def test_minimal_construction(self) -> None:
        override = ParamOverride()
        assert override.name is None
        assert override.description is None
        assert override.type is None
        assert override.enum is None
        assert override.required is None
        assert override.default is UNDEFINED

    def test_default_outside_enum_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not one of the configured enum values"):
            ParamOverride(enum=["json", "xml"], default="yaml")

    def test_default_inside_enum_accepted(self) -> None:
        override = ParamOverride(enum=["json", "xml"], default="json")
        assert override.default == "json"

    def test_default_type_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="does not match the configured type"):
            ParamOverride(type="integer", default="not-a-number")

    def test_boolean_default_rejected_for_integer(self) -> None:
        # bool subclasses int in Python but is not a JSON number.
        with pytest.raises(ValidationError, match="does not match the configured type"):
            ParamOverride(type="integer", default=True)

    def test_default_none_removal_marker_accepted(self) -> None:
        # `default: null` meaningfully differs from an omitted default.
        override = ParamOverride(type="string", default=None)
        assert override.default is None

    def test_enum_values_checked_against_type(self) -> None:
        with pytest.raises(
            ValidationError, match=r"enum value .* does not match the configured type"
        ):
            ParamOverride(type="integer", enum=["1", "2"])

    def test_number_accepts_int_and_float_defaults(self) -> None:
        assert ParamOverride(type="number", default=1).default == 1
        assert ParamOverride(type="number", default=1.5).default == 1.5

    def test_unknown_type_is_not_checked(self) -> None:
        override = ParamOverride(type="custom-format", default={"any": "thing"})
        assert override.default == {"any": "thing"}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParamOverride(bogus="value")  # type: ignore[call-arg]


class TestToolOverride:
    """Construction-time validation of per-tool override consistency."""

    def test_minimal_construction(self) -> None:
        override = ToolOverride()
        assert override.name is None
        assert override.param_names == {}
        assert override.param_descriptions == {}
        assert override.param_overrides == {}
        assert override.param_additions == {}
        assert override.param_removals == set()

    def test_param_rename_collision_with_addition_rejected(self) -> None:
        with pytest.raises(ValidationError, match="produced by both"):
            ToolOverride(
                param_names={"location": "city"},
                param_additions={"city": ParamOverride(type="string")},
            )

    def test_param_rename_collision_with_visible_addition_name_rejected(self) -> None:
        # The collision check runs on the visible name (po.name), not the key.
        with pytest.raises(ValidationError, match="produced by both"):
            ToolOverride(
                param_names={"location": "city"},
                param_additions={"city_key": ParamOverride(name="city")},
            )

    def test_two_renames_of_same_parameter_rejected(self) -> None:
        with pytest.raises(ValidationError, match="renamed to both"):
            ToolOverride(
                param_names={"location": "city"},
                param_overrides={"location": ParamOverride(name="place")},
            )

    def test_redundant_duplicate_rename_rejected(self) -> None:
        # Same target via both mechanisms is ambiguous about the source of
        # truth, so the single-source-of-truth rule rejects it too.
        with pytest.raises(ValidationError, match="produced by both"):
            ToolOverride(
                param_names={"location": "city"},
                param_overrides={"location": ParamOverride(name="city")},
            )

    def test_removed_and_renamed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both removed and renamed"):
            ToolOverride(param_names={"location": "city"}, param_removals={"location"})

    def test_removed_and_added_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both removed and added"):
            ToolOverride(
                param_additions={"units": ParamOverride(type="string")},
                param_removals={"units"},
            )

    def test_removal_with_default_override_allowed(self) -> None:
        # Removal combined with param_overrides defaults is the documented
        # "remove required parameter but inject a default" pattern.
        override = ToolOverride(
            param_removals={"api_key"},
            param_overrides={"api_key": ParamOverride(default="sk-default")},
        )
        assert override.param_removals == {"api_key"}

    def test_param_descriptions_accept_empty_values(self) -> None:
        override = ToolOverride(param_descriptions={"location": ""})
        assert override.param_descriptions == {"location": ""}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolOverride(bogus="value")  # type: ignore[call-arg]


class TestToolSchemaOverlapConfig:
    """Validation of the top-level servers/global_overrides mapping."""

    def test_empty_config_valid(self) -> None:
        config = ToolSchemaOverlapConfig()
        assert config.servers == {}
        assert config.global_overrides == {}

    def test_nested_round_trip(self) -> None:
        config = ToolSchemaOverlapConfig.model_validate({
            "servers": {
                "weather": {
                    "get_weather": {
                        "name": "fetch_weather",
                        "param_names": {"location": "city"},
                        "param_removals": ["notes"],
                    }
                }
            },
            "global_overrides": {"search": {"description": "Web search."}},
        })
        override = config.servers["weather"]["get_weather"]
        assert isinstance(override, ToolOverride)
        assert override.name == "fetch_weather"
        assert override.param_names == {"location": "city"}
        assert override.param_removals == {"notes"}
        assert config.global_overrides["search"].description == "Web search."

    def test_duplicate_tool_rename_targets_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"tool rename target .* is used by both"):
            ToolSchemaOverlapConfig(
                servers={
                    "web-a": {"search": ToolOverride(name="web_search")},
                    "web-b": {"query": ToolOverride(name="web_search")},
                }
            )

    def test_global_and_server_rename_conflict_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"tool rename target .* is used by both"):
            ToolSchemaOverlapConfig(
                servers={"web-a": {"search": ToolOverride(name="web_search")}},
                global_overrides={"find": ToolOverride(name="web_search")},
            )

    def test_identical_shadowing_not_a_config_error(self) -> None:
        # A server-scoped entry and a global entry for the same tool name may
        # both configure the same rename target: the server entry shadows the
        # global one, so there is no collision between distinct sources.
        config = ToolSchemaOverlapConfig(
            servers={"weather": {"get_weather": ToolOverride(name="fetch_weather")}},
            global_overrides={"get_weather": ToolOverride(name="fetch_weather")},
        )
        assert config.servers["weather"]["get_weather"].name == "fetch_weather"

    def test_empty_server_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSchemaOverlapConfig.model_validate({"servers": {"": {"tool": {}}}})

    def test_empty_tool_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSchemaOverlapConfig.model_validate({"global_overrides": {"": {}}})

    def test_unknown_top_level_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSchemaOverlapConfig.model_validate({"name_overrides": {}})
