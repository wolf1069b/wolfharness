"""Tests for OpenCode message models.

Tests the MessageWithParts class, UserMessage model nesting, and
backward-compatible variant deserialization.
"""

from __future__ import annotations

import pytest

from wolfharness_server.opencode_server.models import (
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageWithParts,
    ModelRef,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.models.message import (
    AssistantMessage,
    TextPartInput,
)


pytestmark = pytest.mark.integration


class TestMessageWithPartsRole:
    """Tests for MessageWithParts.role property."""

    def test_user_message_has_role_user(self):
        """User message should have role='user'."""
        msg = MessageWithParts.user(
            message_id="test-user-1",
            session_id="test-session",
            time=TimeCreated(created=1234567890),
            agent_name="test-agent",
        )

        assert msg.role == "user"
        assert msg.info.role == "user"

    def test_assistant_message_has_role_assistant(self):
        """Assistant message should have role='assistant'."""
        msg = MessageWithParts.assistant(
            message_id="test-assistant-1",
            session_id="test-session",
            time=MessageTime(created=1234567890),
            agent_name="test-agent",
            model_id="gpt-4",
            parent_id="",
            provider_id="openai",
            path=MessagePath(cwd="/test", root="/test"),
        )

        assert msg.role == "assistant"
        assert msg.info.role == "assistant"

    def test_role_property_matches_info_role(self):
        """Role property should match info.role for both message types."""
        user_msg = MessageWithParts.user(
            message_id="user-1",
            session_id="session-1",
            time=TimeCreated(created=1000),
            agent_name="agent",
        )

        assistant_msg = MessageWithParts.assistant(
            message_id="assistant-1",
            session_id="session-1",
            time=MessageTime(created=1000),
            agent_name="agent",
            model_id="gpt-4",
            parent_id="parent-1",
            provider_id="openai",
            path=MessagePath(cwd="/home", root="/home"),
        )

        # Verify role property delegates to info.role
        assert user_msg.role == user_msg.info.role
        assert assistant_msg.role == assistant_msg.info.role


class TestMessageWithPartsFactories:
    """Tests for MessageWithParts factory methods."""

    def test_user_factory_creates_user_message(self):
        """MessageWithParts.user() should create a user message."""
        msg = MessageWithParts.user(
            message_id="msg-1",
            session_id="session-1",
            time=TimeCreated(created=1234567890000),
            agent_name="test-agent",
        )

        assert msg.info.id == "msg-1"
        assert msg.info.session_id == "session-1"
        assert msg.info.agent == "test-agent"
        assert msg.role == "user"
        assert msg.parts == []

    def test_assistant_factory_creates_assistant_message(self):
        """MessageWithParts.assistant() should create an assistant message."""
        from wolfharness_server.opencode_server.models import AssistantMessage

        msg = MessageWithParts.assistant(
            message_id="msg-1",
            session_id="session-1",
            time=MessageTime(created=1234567890000, completed=1234567900000),
            agent_name="test-agent",
            model_id="gpt-4o",
            parent_id="parent-1",
            provider_id="openai",
            path=MessagePath(cwd="/workspace", root="/workspace"),
            mode="default",
            cost=0.001,
            finish="stop",
        )

        # Verify common fields
        assert msg.info.id == "msg-1"
        assert msg.info.session_id == "session-1"
        assert msg.info.agent == "test-agent"
        assert msg.role == "assistant"
        assert msg.parts == []

        # Verify assistant-specific fields (narrow type first)
        assert isinstance(msg.info, AssistantMessage)
        info: AssistantMessage = msg.info
        assert info.model_id == "gpt-4o"
        assert info.parent_id == "parent-1"
        assert info.provider_id == "openai"


class TestMessageWithPartsParts:
    """Tests for MessageWithParts parts manipulation."""

    def test_add_text_part(self):
        """Should be able to add text parts to a message."""
        msg = MessageWithParts.user(
            message_id="msg-1",
            session_id="session-1",
            time=TimeCreated(created=1234567890000),
            agent_name="agent",
        )

        part = msg.add_text_part("Hello, world!")

        assert len(msg.parts) == 1
        assert msg.parts[0] == part
        assert part.text == "Hello, world!"
        assert part.message_id == "msg-1"
        assert part.session_id == "session-1"

    def test_add_multiple_parts(self):
        """Should be able to add multiple parts to a message."""
        msg = MessageWithParts.assistant(
            message_id="msg-1",
            session_id="session-1",
            time=MessageTime(created=1234567890000),
            agent_name="agent",
            model_id="gpt-4",
            parent_id="",
            provider_id="openai",
            path=MessagePath(cwd="/test", root="/test"),
        )

        msg.add_step_start_part()
        msg.add_text_part("Processing...")

        assert len(msg.parts) == 2


class TestModelRefVariant:
    """Tests for ModelRef with optional variant field."""

    def test_model_ref_without_variant(self):
        """ModelRef should work without variant (backward compat)."""
        ref = ModelRef(provider_id="openai", model_id="gpt-4o")
        assert ref.provider_id == "openai"
        assert ref.model_id == "gpt-4o"
        assert ref.variant is None

    def test_model_ref_with_variant(self):
        """ModelRef should accept variant field."""
        ref = ModelRef(provider_id="openai", model_id="gpt-4o", variant="high")
        assert ref.variant == "high"

    def test_model_ref_with_only_variant(self):
        """ModelRef should accept variant-only (no provider/model)."""
        ref = ModelRef(variant="medium")
        assert ref.variant == "medium"
        assert ref.provider_id is None
        assert ref.model_id is None

    def test_model_ref_serialization_with_variant(self):
        """ModelRef should serialize variant as camelCase under by_alias."""
        ref = ModelRef(provider_id="openai", model_id="gpt-4o", variant="medium")
        data = ref.model_dump(by_alias=True, exclude_none=True)
        assert data == {"providerID": "openai", "modelID": "gpt-4o", "variant": "medium"}

    def test_model_ref_serialization_without_variant(self):
        """ModelRef without variant should not include variant in output."""
        ref = ModelRef(provider_id="openai", model_id="gpt-4o")
        data = ref.model_dump(by_alias=True, exclude_none=True)
        assert "variant" not in data
        assert data == {"providerID": "openai", "modelID": "gpt-4o"}

    def test_model_ref_serialization_variant_only(self):
        """ModelRef with only variant should only include variant in output."""
        ref = ModelRef(variant="low")
        data = ref.model_dump(by_alias=True, exclude_none=True)
        assert data == {"variant": "low"}


class TestUserMessageVariantNesting:
    """Tests for UserMessage with variant nested under model."""

    def test_user_message_with_model_variant(self):
        """UserMessage should accept variant via model object."""
        msg = UserMessage(
            id="msg-1",
            session_id="session-1",
            time=TimeCreated(created=1234567890),
            model=ModelRef(provider_id="openai", model_id="gpt-4o", variant="high"),
        )
        assert msg.model is not None
        assert msg.model.variant == "high"

    def test_user_message_without_model(self):
        """UserMessage should work without model at all."""
        msg = UserMessage(
            id="msg-1",
            session_id="session-1",
            time=TimeCreated(created=1234567890),
        )
        assert msg.model is None

    def test_user_message_no_top_level_variant_in_output(self):
        """Serialized UserMessage should NOT have top-level variant."""
        msg = UserMessage(
            id="msg-1",
            session_id="session-1",
            time=TimeCreated(created=1234567890),
            model=ModelRef(provider_id="openai", model_id="gpt-4o", variant="high"),
        )
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert "variant" not in data
        assert data["model"] == {"providerID": "openai", "modelID": "gpt-4o", "variant": "high"}

    def test_backward_compat_top_level_variant_no_model(self):
        """Old JSON with top-level variant and no model should deserialize.

        The variant should be migrated into a model object with just variant.
        """
        msg = UserMessage.model_validate({
            "id": "msg-1",
            "sessionID": "session-1",
            "time": {"created": 1234567890},
            "variant": "medium",
        })
        assert msg.model is not None
        assert msg.model.variant == "medium"
        assert msg.model.provider_id is None
        assert msg.model.model_id is None

    def test_backward_compat_top_level_variant_with_existing_model(self):
        """Old JSON with top-level variant AND model should merge variant into model."""
        msg = UserMessage.model_validate({
            "id": "msg-1",
            "sessionID": "session-1",
            "time": {"created": 1234567890},
            "model": {"providerID": "openai", "modelID": "gpt-4o"},
            "variant": "low",
        })
        assert msg.model is not None
        assert msg.model.provider_id == "openai"
        assert msg.model.model_id == "gpt-4o"
        assert msg.model.variant == "low"

    def test_new_format_variant_in_model_no_migration(self):
        """New JSON with variant inside model should work without migration."""
        msg = UserMessage.model_validate({
            "id": "msg-1",
            "sessionID": "session-1",
            "time": {"created": 1234567890},
            "model": {"providerID": "openai", "modelID": "gpt-4o", "variant": "max"},
        })
        assert msg.model is not None
        assert msg.model.variant == "max"
        assert msg.model.provider_id == "openai"
        assert msg.model.model_id == "gpt-4o"

    def test_no_variant_no_model(self):
        """JSON without variant or model should work."""
        msg = UserMessage.model_validate({
            "id": "msg-1",
            "sessionID": "session-1",
            "time": {"created": 1234567890},
        })
        assert msg.model is None

    def test_variant_not_in_serialized_output(self):
        """After migration, top-level variant should NOT appear in serialized output."""
        msg = UserMessage.model_validate({
            "id": "msg-1",
            "sessionID": "session-1",
            "time": {"created": 1234567890},
            "variant": "medium",
        })
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert "variant" not in data
        assert data["model"]["variant"] == "medium"


class TestMessageRequestVariantNesting:
    """Tests for MessageRequest with variant nested under model."""

    def test_message_request_with_model_variant(self):
        """MessageRequest should accept variant via model object."""
        req = MessageRequest(
            parts=[TextPartInput(text="hello")],
            model=ModelRef(provider_id="openai", model_id="gpt-4o", variant="high"),
        )
        assert req.model is not None
        assert req.model.variant == "high"

    def test_backward_compat_top_level_variant_no_model(self):
        """Old JSON with top-level variant should migrate into model."""
        req = MessageRequest.model_validate({
            "parts": [{"type": "text", "text": "hello"}],
            "variant": "medium",
        })
        assert req.model is not None
        assert req.model.variant == "medium"

    def test_backward_compat_top_level_variant_with_model(self):
        """Old JSON with both top-level variant and model should merge."""
        req = MessageRequest.model_validate({
            "parts": [{"type": "text", "text": "hello"}],
            "model": {"providerID": "openai", "modelID": "gpt-4o"},
            "variant": "low",
        })
        assert req.model is not None
        assert req.model.variant == "low"
        assert req.model.provider_id == "openai"

    def test_no_variant_in_serialized_output(self):
        """MessageRequest should not have top-level variant in output."""
        req = MessageRequest(
            parts=[TextPartInput(text="hello")],
            model=ModelRef(provider_id="openai", model_id="gpt-4o", variant="high"),
        )
        data = req.model_dump(by_alias=True, exclude_none=True)
        assert "variant" not in data
        assert data["model"]["variant"] == "high"


class TestAssistantMessageVariantMigration:
    """Tests for AssistantMessage variant removal."""

    def test_assistant_message_ignores_top_level_variant(self):
        """Old JSON with top-level variant should silently drop it."""
        msg = AssistantMessage.model_validate({
            "id": "msg-1",
            "sessionID": "session-1",
            "parentID": "parent-1",
            "modelID": "gpt-4o",
            "providerID": "openai",
            "path": {"cwd": "/test", "root": "/test"},
            "time": {"created": 1234567890},
            "variant": "high",
        })
        # AssistantMessage no longer has a variant field
        assert not hasattr(msg, "variant")

    def test_assistant_message_serialization_no_variant(self):
        """AssistantMessage should not include variant in output."""
        msg = AssistantMessage(
            id="msg-1",
            session_id="session-1",
            parent_id="parent-1",
            model_id="gpt-4o",
            provider_id="openai",
            path=MessagePath(cwd="/test", root="/test"),
            time=MessageTime(created=1234567890),
        )
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert "variant" not in data
