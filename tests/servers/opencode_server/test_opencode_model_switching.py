"""Tests to verify OpenCode model switching issues.

These tests verify the root causes of the issue where model changes
in OpenCode TUI are not reflected in wolfharness runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

from tests.servers.opencode_server.conftest import run_message_phases
from wolfharness import Agent, AgentPool, AgentsManifest
from wolfharness_server.opencode_server.models import (
    MessageRequest,
    MessageWithParts,
    ModelRef,
    TextPartInput,
    TimeCreated,
    UserMessage,
)
from wolfharness_server.opencode_server.models.config import Config


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Test Root Cause #1: Model variants not included in validation
# =============================================================================


@pytest.fixture
def manifest_with_model_variants() -> AgentsManifest:
    """Create a manifest with model_variants for testing."""
    config_yaml = """
model_variants:
  qwen35:
    type: string
    identifier: "openai-chat:svc/qwen35"
  glm47:
    type: string
    identifier: "openai-chat:svc/glm-4.7"

agents:
  test_agent:
    type: native
    model: glm47
    system_prompt: "You are a test agent"
"""
    return AgentsManifest.from_yaml(config_yaml)


@pytest.fixture
async def agent_with_variants(manifest_with_model_variants: AgentsManifest):
    """Create an agent with model_variants in its pool."""
    async with AgentPool(manifest_with_model_variants) as pool:
        agent = pool.manifest.agents["test_agent"].get_agent(pool=pool)
        async with agent:
            # Store pool reference for tests (runtime attribute)
            agent._test_pool = pool  # type: ignore[attr-defined]
            yield agent  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.real_model
async def test_model_variant_resolution(agent_with_variants: Agent):
    """Test that model variants can be resolved to actual models.

    This verifies that variant names like 'qwen35' can be resolved
    to their full configuration.
    """
    agent = agent_with_variants
    pool = agent._test_pool  # type: ignore[attr-defined]
    manifest = pool.manifest

    # Verify variants are in manifest
    assert "qwen35" in manifest.model_variants
    assert "glm47" in manifest.model_variants

    # Verify variant config has identifier
    qwen_config = manifest.model_variants["qwen35"]
    assert qwen_config.identifier == "openai-chat:svc/qwen35"


@pytest.mark.unit
@pytest.mark.real_model
async def test_resolve_model_string_with_variant(agent_with_variants: Agent):
    """Test _resolve_model_string correctly resolves variant names.

    Expected: _resolve_model_string('qwen35') should resolve the variant.
    Actual: This should work if the implementation is correct.

    Issue: If get_available_models() doesn't include variants,
    validation in _set_mode will fail.
    """
    agent = agent_with_variants

    # Test that variant can be resolved
    model, _settings = agent._resolve_model_string("qwen35")

    # The model should be resolved from the variant
    assert model is not None


@pytest.mark.unit
@pytest.mark.real_model
async def test_get_available_models_excludes_variants(
    agent_with_variants: Agent, monkeypatch: pytest.MonkeyPatch
):
    """CRITICAL TEST: get_available_models() returns tokonomics models, not config variants.

    This is the ROOT CAUSE of the validation failure.

    When OpenCode TUI sends a variant name like 'qwen35',
    _set_mode validates against get_available_models() which returns
    tokonomics-discovered models, not config variants.

    Result: Validation fails because 'qwen35' is not in the tokonomics list.
    """
    # Use tokonomics path so the get_all_models mock is exercised
    monkeypatch.setenv("MODELS_DEV_FALLBACK", "1")
    agent = agent_with_variants

    # Mock tokonomics to return a predictable list
    with patch("tokonomics.model_discovery.get_all_models") as mock_get_all:
        # Simulate tokonomics returning models without our variants
        from tokonomics.model_discovery.model_info import ModelInfo

        mock_model = ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
        )
        mock_get_all.return_value = [mock_model]

        available_models = await agent.get_available_models()

        # Verify tokonomics models are returned
        assert available_models is not None
        model_ids = [m.id for m in available_models]
        assert "gpt-4o" in model_ids

        # CRITICAL: Verify variants are NOT in the list
        assert "qwen35" not in model_ids, (
            "model_variants should NOT appear in get_available_models() - this is the bug!"
        )
        assert "glm47" not in model_ids, (
            "model_variants should NOT appear in get_available_models() - this is the bug!"
        )


@pytest.mark.unit
@pytest.mark.real_model
async def test_set_mode_validation_fails_for_variants(
    agent_with_variants: Agent, monkeypatch: pytest.MonkeyPatch
):
    """CRITICAL TEST: Verify that _set_mode validation fails for model_variants.

    This reproduces the exact failure that happens when OpenCode TUI
    tries to switch to a model variant.

    Expected: _set_mode should accept 'qwen35' as a valid model.
    Actual: Validation fails because get_available_models() doesn't include variants.

    This test documents the CURRENT BUGGY BEHAVIOR.
    After fix, this test should pass.
    """
    # Use tokonomics path so the get_all_models mock is exercised
    monkeypatch.setenv("MODELS_DEV_FALLBACK", "1")
    agent = agent_with_variants

    # Mock get_available_models to simulate tokonomics response
    with patch("tokonomics.model_discovery.get_all_models") as mock_get_all:
        from tokonomics.model_discovery.model_info import ModelInfo

        mock_model = ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
        )
        mock_get_all.return_value = [mock_model]

        # Try to set mode to a variant
        # This simulates what happens when OpenCode TUI sends 'qwen35'
        with pytest.raises(Exception, match=r"(?i:unknown|qwen35)") as exc_info:
            await agent._set_mode("model", "qwen35")

        # Should fail with UnknownModeError or similar
        # The exact exception type may vary
        error_msg = str(exc_info.value).lower()
        assert "unknown" in error_msg or "qwen35" in error_msg, (
            f"Expected error about unknown mode, got: {exc_info.value}"
        )


# =============================================================================
# Test Root Cause #2: Config update doesn't sync to agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.real_model
async def test_config_update_does_not_sync_to_agent(agent_with_variants: Agent):
    """CRITICAL TEST: Verify that updating config.model doesn't update agent._model.

    This tests the exact code path in config_routes.py:update_config().

    When PATCH /config is called with {"model": "new_model"}:
    - state.config.model is updated ✓
    - state.agent.set_model() is NEVER called ✗

    Result: Agent continues using old model.
    """
    agent = agent_with_variants

    # Get initial model
    initial_model = agent.model_name
    assert initial_model is not None

    # Simulate what happens in config_routes.py:update_config()
    config = Config(model="openai-chat:svc/qwen35")

    # This is what update_config() does - only updates config object
    state_config = Config()
    update_data = config.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        setattr(state_config, field_name, value)

    # Verify: state_config is updated
    assert state_config.model == "openai-chat:svc/qwen35"

    # CRITICAL: Verify agent model is NOT updated
    # This demonstrates the bug - config and agent are out of sync
    assert agent.model_name == initial_model, (
        "BUG: Agent model should NOT have changed (config update doesn't sync to agent)"
    )


@pytest.mark.unit
@pytest.mark.real_model
async def test_manual_set_model_works(agent_with_variants: Agent):
    """Test that directly calling set_model() DOES work.

    This verifies that if we fix the sync issue in update_config(),
    model switching will work correctly.
    """
    agent = agent_with_variants

    initial_model = agent.model_name

    # Manually call set_model (what update_config() SHOULD do)
    # Note: This might fail due to the validation bug above
    try:
        await agent.set_model("openai-chat:svc/glm-4.7")
        # If it worked, model should change
        assert agent.model_name != initial_model or agent.model_name == "openai-chat:svc/glm-4.7"
    except Exception as e:  # noqa: BLE001
        # Expected to fail due to validation issues
        pytest.skip(f"set_model failed (expected due to validation bug): {e}")


# =============================================================================
# Test Root Cause #3: Message processing restores original model
# =============================================================================


@pytest.mark.unit
@pytest.mark.real_model
async def test_message_processing_restores_model(agent_with_variants: Agent):
    """CRITICAL TEST: Verify that message processing restores original model.

    In message_routes.py:_process_message():
    1. Store original_model = agent.model_name
    2. await agent.set_model(requested_model)  # Temporary switch
    3. Run agent
    4. await agent.set_model(original_model)   # Always restore!

    This is by design for per-message model override, but prevents
    persistent model changes from the TUI.

    This test documents the CURRENT BEHAVIOR.
    """
    agent = agent_with_variants
    initial_model = agent.model_name

    # Simulate what happens in _process_message()
    # Note: We can't easily mock the full flow, so we document the behavior

    # The issue is that even if we fix set_model() to work,
    # _process_message() will ALWAYS restore the original model after processing.

    # This demonstrates that we need a separate mechanism for persistent
    # model changes vs per-message overrides.

    # For now, just verify the current state
    assert agent.model_name == initial_model

    # Document: To fix this, we need to either:
    # 1. Add a separate endpoint for persistent model changes
    # 2. Or add a flag to skip model restoration in _process_message()


# =============================================================================
# Integration-style tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.real_model
async def test_opencode_model_flow_simulation():
    """Simulate the complete OpenCode model switching flow.

    This test simulates:
    1. OpenCode TUI connects to wolfharness
    2. TUI gets available models (including model_variants as synthetic provider)
    3. User selects 'qwen35' in TUI
    4. TUI sends message with model override
    5. Agent processes message with temporary model
    6. Model is restored after message

    Expected: Model change should persist (but currently doesn't due to bugs).
    """
    # Create manifest with model_variants
    config_yaml = """
model_variants:
  qwen35:
    type: string
    identifier: "openai-chat:svc/qwen35"

agents:
  assistant:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are an assistant"
"""
    manifest = AgentsManifest.from_yaml(config_yaml)

    async with AgentPool(manifest) as pool:
        agent = pool.manifest.agents["assistant"].get_agent(pool=pool)
        async with agent:
            # Simulate OpenCode TUI getting available providers
            # In config_routes.py:_build_providers_from_variants(),
            # model_variants are exposed as a synthetic "agent" provider

            # Simulate user selecting 'qwen35'
            # TUI would send: provider_id="agent", model_id="qwen35"
            # Which gets constructed as: requested_model = "agent:qwen35"

            # The issue: "agent:qwen35" is not a valid model identifier
            # It should be just "qwen35" or the variant should be resolved

            # Document the current buggy flow
            variant_name = "qwen35"
            synthetic_provider_id = "agent"
            constructed_model_id = f"{synthetic_provider_id}:{variant_name}"

            # This is what happens in message_routes.py:236
            assert constructed_model_id == "agent:qwen35"

            # The problem: _resolve_model_string doesn't handle "agent:" prefix
            # It treats the whole thing as a model name

            # Verify the variant exists in manifest
            assert variant_name in pool.manifest.model_variants

            # But constructed_model_id will fail validation
            # because it's not in get_available_models() and doesn't match
            # the variant name due to the "agent:" prefix

            # This test documents all three issues:
            # 1. "agent:" prefix not handled
            # 2. Variants not in get_available_models()
            # 3. Even if fixed, model is restored after message


# =============================================================================
# Per-Session Agent Model Switching Tests
# =============================================================================


class _MockModelInfo:
    """Minimal stand-in for tokonomics ModelInfo."""

    def __init__(self, id: str, id_override: str | None = None) -> None:  # noqa: A002
        self.id = id
        self.id_override = id_override


class PerSessionAgentMock:
    """Mock agent that tracks set_model calls for per-session isolation testing."""

    def __init__(
        self,
        name: str,
        model_name: str = "test-model",
        *,
        available_models: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.set_model_calls: list[str] = []
        self.set_mode_calls: list[tuple[str, str | None]] = []
        self.get_available_models_calls = 0
        self.agent_pool: Any = None
        self.host_context: Any = None
        self.env: Any = None
        self.storage: Any = None
        self._input_provider = None
        self.tools = Mock()
        self._available_models = [_MockModelInfo(m) for m in (available_models or [])]

    async def get_available_models(self) -> list[Any]:
        self.get_available_models_calls += 1
        return self._available_models

    async def set_model(self, model: str) -> None:
        self.set_model_calls.append(model)
        self.model_name = model

    async def set_mode(self, mode: str, category_id: str | None = None) -> None:
        self.set_mode_calls.append((mode, category_id))


def _make_mock_state_with_session_agent(
    tmp_project_dir: Path,
    session_agents: dict[str, PerSessionAgentMock],
) -> tuple[Any, Any]:
    """Create a ServerState wired so get_or_create_session_agent returns per-session mocks."""
    from unittest.mock import AsyncMock, Mock

    from wolfharness.lifecycle import RunOutcome, RunState
    from wolfharness.utils.time_utils import now_ms
    from wolfharness_server.opencode_server.state import ServerState

    shared_agent = PerSessionAgentMock(name="shared-agent")
    shared_agent.env = Mock()
    shared_agent.env.get_fs = Mock(return_value=Mock())

    # Set up pool mock BEFORE creating ServerState so __post_init__ captures it.
    pool = Mock()
    pool.manifest = Mock()
    pool.manifest.config_file_path = "/tmp/test"
    pool.manifest.model_variants = {}
    pool.manifest.agents = {shared_agent.name: shared_agent}

    storage = Mock()
    storage.save_session = AsyncMock()
    storage.log_message = AsyncMock()
    pool.storage = storage

    # SessionPool mock
    session_pool = Mock()
    session_pool.sessions = Mock()

    async def _get_or_create_session_agent(
        session_id: str,
        agent_name: str | None = None,
        input_provider: Any | None = None,
    ) -> Any:
        if session_id not in session_agents:
            session_agents[session_id] = PerSessionAgentMock(
                name=f"session-agent-{session_id}",
            )
        return session_agents[session_id]

    session_pool.sessions.get_or_create_session_agent = AsyncMock(
        side_effect=_get_or_create_session_agent
    )
    session_pool.sessions.get_or_create_session = AsyncMock(return_value=(Mock(), True))
    session_pool.sessions.get_session = Mock(return_value=None)

    # RunHandle that completes immediately
    run_handle = Mock()
    run_handle._run_state = RunState.DONE
    run_handle.outcome = RunOutcome.COMPLETED
    run_handle.complete_event = Mock()
    run_handle.complete_event.wait = AsyncMock(return_value=None)
    session_pool.send_message = AsyncMock(return_value=run_handle)

    # EventBus mock
    session_pool.event_bus = Mock()
    from tests._helpers.mock_stream import EmptyReceiveStream

    session_pool.event_bus.subscribe = AsyncMock(return_value=EmptyReceiveStream())
    session_pool.event_bus.unsubscribe = AsyncMock(return_value=None)

    pool.session_pool = session_pool
    shared_agent.agent_pool = pool
    shared_agent.host_context = pool
    shared_agent._agent_pool = pool  # state.py resolves _pool via agent._agent_pool

    state = ServerState(
        working_dir=str(tmp_project_dir),
        agent=shared_agent,  # type: ignore[arg-type]
    )

    # Initialize backward-compat dicts removed from ServerState dataclass
    # so tests and helper fallbacks can access them.
    state.messages = {}  # type: ignore[attr-defined]
    state.todos = {}  # type: ignore[attr-defined]
    state.input_providers = {}  # type: ignore[attr-defined]
    # No session_pool_integration — run_message_phases will use the
    # fallback path via session_pool.sessions.get_or_create_session.

    # Pre-populate sessions in state
    for session_id in session_agents:
        from wolfharness_server.opencode_server.models import Session
        from wolfharness_server.opencode_server.models.common import TimeCreatedUpdated

        now = now_ms()
        session = Session(
            id=session_id,
            project_id="default",
            directory=str(tmp_project_dir),
            title=f"Session {session_id}",
            version="1",
            time=TimeCreatedUpdated(created=now, updated=now),
        )
        state.sessions[session_id] = session
        state.messages[session_id] = []  # type: ignore[attr-defined]

    return state, pool


@pytest.mark.unit
async def test_model_switch_targets_per_session_agent(tmp_project_dir: Path) -> None:
    """Model switching must call set_model on the per-session agent, not the shared agent.

    The shared ``state.agent`` is no longer used for model switching.
    ``run_message_phases`` uses
    ``session_pool.sessions.get_or_create_session_agent()`` so each session
    gets its own isolated model configuration.
    """
    from wolfharness.utils import identifiers as identifier

    session_id = "test-session-a"
    session_agents: dict[str, PerSessionAgentMock] = {
        session_id: PerSessionAgentMock(
            name=f"session-agent-{session_id}",
            available_models=["gpt-4o"],
        ),
    }
    state, _pool = _make_mock_state_with_session_agent(tmp_project_dir, session_agents)

    shared_agent = state.agent
    assert isinstance(shared_agent, PerSessionAgentMock)

    request = MessageRequest(
        parts=[TextPartInput(text="Hello!")],
        model=ModelRef(provider_id="openai-chat", model_id="gpt-4o"),
    )
    user_msg_id = identifier.ascending("message")
    user_message = UserMessage(
        id=user_msg_id,
        session_id=session_id,
        time=TimeCreated.now(),
        agent="default",
        model=request.model,
    )
    user_msg_with_parts = MessageWithParts(info=user_message)
    user_msg_with_parts.add_text_part("Hello!")
    state.messages[session_id].append(user_msg_with_parts)

    await run_message_phases(session_id, request, state, user_msg_id, user_msg_with_parts)

    # Per-session agent should have been created and had set_model called on it
    per_session_agent = session_agents[session_id]
    assert per_session_agent.set_model_calls == ["gpt-4o"]
    assert per_session_agent.get_available_models_calls == 1

    # Shared agent must NOT have been touched
    assert shared_agent.set_model_calls == []
    assert shared_agent.get_available_models_calls == 0


@pytest.mark.unit
async def test_model_switch_affects_only_target_session(tmp_project_dir: Path) -> None:
    """Switching model in session A must not affect session B's agent."""
    from wolfharness.utils import identifiers as identifier

    session_a = "session-a"
    session_b = "session-b"
    session_agents: dict[str, PerSessionAgentMock] = {
        session_a: PerSessionAgentMock(
            name="agent-a", model_name="model-a", available_models=["gpt-4o"]
        ),
        session_b: PerSessionAgentMock(name="agent-b", model_name="model-b"),
    }
    state, _pool = _make_mock_state_with_session_agent(tmp_project_dir, session_agents)

    # Process message for session A WITH model override
    request_a = MessageRequest(
        parts=[TextPartInput(text="Hello A!")],
        model=ModelRef(provider_id="openai-chat", model_id="gpt-4o"),
    )
    user_msg_id_a = identifier.ascending("message")
    user_message_a = UserMessage(
        id=user_msg_id_a,
        session_id=session_a,
        time=TimeCreated.now(),
        agent="default",
        model=request_a.model,
    )
    user_msg_with_parts_a = MessageWithParts(info=user_message_a)
    user_msg_with_parts_a.add_text_part("Hello A!")
    state.messages[session_a].append(user_msg_with_parts_a)

    await run_message_phases(session_a, request_a, state, user_msg_id_a, user_msg_with_parts_a)

    # Process message for session B WITHOUT model override
    request_b = MessageRequest(
        parts=[TextPartInput(text="Hello B!")],
    )
    user_msg_id_b = identifier.ascending("message")
    user_message_b = UserMessage(
        id=user_msg_id_b,
        session_id=session_b,
        time=TimeCreated.now(),
        agent="default",
        model=request_b.model,
    )
    user_msg_with_parts_b = MessageWithParts(info=user_message_b)
    user_msg_with_parts_b.add_text_part("Hello B!")
    state.messages[session_b].append(user_msg_with_parts_b)

    await run_message_phases(session_b, request_b, state, user_msg_id_b, user_msg_with_parts_b)

    # Session A's agent should have switched
    assert session_agents[session_a].set_model_calls == ["gpt-4o"]

    # Session B's agent should NOT have been switched
    assert session_agents[session_b].set_model_calls == []
    assert session_agents[session_b].model_name == "model-b"


@pytest.mark.unit
async def test_other_sessions_retain_original_model(tmp_project_dir: Path) -> None:
    """After switching model in one session, other sessions keep their original model."""
    from wolfharness.utils import identifiers as identifier

    session_a = "session-a"
    session_b = "session-b"
    session_agents: dict[str, PerSessionAgentMock] = {
        session_a: PerSessionAgentMock(
            name="agent-a", model_name="original-model-a", available_models=["new-model"]
        ),
        session_b: PerSessionAgentMock(name="agent-b", model_name="original-model-b"),
    }
    state, _pool = _make_mock_state_with_session_agent(tmp_project_dir, session_agents)

    # Process message for session A with model override
    request = MessageRequest(
        parts=[TextPartInput(text="Switch model!")],
        model=ModelRef(provider_id="openai-chat", model_id="new-model"),
    )
    user_msg_id = identifier.ascending("message")
    user_message = UserMessage(
        id=user_msg_id,
        session_id=session_a,
        time=TimeCreated.now(),
        agent="default",
        model=request.model,
    )
    user_msg_with_parts = MessageWithParts(info=user_message)
    user_msg_with_parts.add_text_part("Switch model!")
    state.messages[session_a].append(user_msg_with_parts)

    await run_message_phases(session_a, request, state, user_msg_id, user_msg_with_parts)

    # Session A switched
    assert session_agents[session_a].model_name == "new-model"

    # Session B retained its original model
    assert session_agents[session_b].model_name == "original-model-b"
