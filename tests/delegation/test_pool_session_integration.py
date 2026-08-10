"""Integration tests for AgentPool SessionPool integration.

Tests cover SessionPool lifecycle, configuration, create_session convenience,
and protocol feature flags.
"""

from __future__ import annotations

import pytest

from wolfharness import Agent, AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.orchestrator import SessionPool
from wolfharness_config.session_pool import SessionPoolConfig


@pytest.fixture
def basic_manifest() -> AgentsManifest:
    """Create a minimal manifest with one agent."""
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    return AgentsManifest(agents={"test_agent": agent_config})


# =============================================================================
# SessionPool Lifecycle
# =============================================================================


class TestSessionPoolLifecycle:
    """Test SessionPool initialization and shutdown within AgentPool."""

    @pytest.mark.integration
    async def test_session_pool_always_initialized(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """SessionPool should always be initialized."""
        async with AgentPool(basic_manifest) as pool:
            assert pool.session_pool is not None
            assert isinstance(pool.session_pool, SessionPool)

    @pytest.mark.integration
    async def test_session_pool_initialized_when_enabled(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """SessionPool should be initialized when enable_session_pool=True."""
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool:
            assert pool.session_pool is not None
            assert isinstance(pool.session_pool, SessionPool)

    @pytest.mark.integration
    async def test_session_pool_shutdown_on_exit(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """SessionPool should be shut down when AgentPool exits."""
        pool = AgentPool(basic_manifest, enable_session_pool=True)
        async with pool:
            assert pool.session_pool is not None
        assert pool._session_pool is None

    @pytest.mark.integration
    async def test_multiple_enter_exit_cycles(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """AgentPool should support multiple enter/exit cycles with SessionPool."""
        pool = AgentPool(basic_manifest, enable_session_pool=True)

        for _ in range(3):
            async with pool:
                assert pool.session_pool is not None
            assert pool._session_pool is None


# =============================================================================
# SessionPool Configuration
# =============================================================================


class TestSessionPoolConfiguration:
    """Test SessionPool configuration propagation."""

    @pytest.mark.integration
    async def test_default_session_pool_config(self) -> None:
        """Default SessionPoolConfig should have expected defaults."""
        cfg = SessionPoolConfig()
        assert cfg.enable_auto_resume is True
        assert cfg.enable_event_bus is True
        assert cfg.session_ttl_seconds == 3600.0
        assert cfg.max_auto_resume == 10
        assert cfg.max_queue_size == 1000
        assert cfg.mcp_max_processes == 100

    @pytest.mark.integration
    async def test_custom_session_pool_config(self) -> None:
        """Custom SessionPoolConfig should propagate to SessionPool."""
        cfg = SessionPoolConfig(
            enable_auto_resume=False,
            enable_event_bus=False,
            session_ttl_seconds=1800.0,
            max_auto_resume=5,
            max_queue_size=500,
            mcp_max_processes=50,
        )
        manifest = AgentsManifest(
            agents={
                "test_agent": NativeAgentConfig(
                    name="test_agent",
                    model="test",
                    system_prompt="You are a test agent",
                )
            },
            session_pool=cfg,
        )

        async with AgentPool(manifest, enable_session_pool=True) as pool:
            sp = pool.session_pool
            assert sp is not None
            assert sp._enable_auto_resume is False
            assert sp._enable_event_bus is False
            assert sp.sessions._session_ttl_seconds == 1800.0
            assert sp.sessions._mcp_max_processes == 50
            assert sp.event_bus._max_queue_size == 500

    @pytest.mark.integration
    async def test_explicit_config_overrides_manifest(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """Explicit session_pool_config parameter should override manifest."""
        explicit_cfg = SessionPoolConfig(max_auto_resume=99)

        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
            session_pool_config=explicit_cfg,
        ) as pool:
            sp = pool.session_pool
            assert sp is not None
            assert pool._session_pool_config.max_auto_resume == 99


# =============================================================================
# create_session Convenience Method
# =============================================================================


class TestCreateSession:
    """Test AgentPool.create_session() convenience method."""

    @pytest.mark.integration
    async def test_create_session_returns_state(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """create_session should return a SessionState."""
        async with AgentPool(basic_manifest) as pool:
            state = await pool.create_session("test-session", agent_name="test_agent")
            assert state.session_id == "test-session"
            assert state.agent_name == "test_agent"

    @pytest.mark.integration
    async def test_create_session_with_metadata(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """create_session should pass metadata through to SessionPool."""
        async with AgentPool(basic_manifest) as pool:
            state = await pool.create_session(
                "test-session",
                agent_name="test_agent",
                custom_key="custom_value",
            )
            assert state.metadata.get("custom_key") == "custom_value"


# =============================================================================
# Protocol Feature Flags
# =============================================================================


class TestProtocolFeatureFlags:
    """Test per-protocol session pool feature flags on AgentsManifest."""

    def test_acp_config_default(self) -> None:
        """ACPConfig.use_session_pool should default to True."""
        manifest = AgentsManifest()
        assert manifest.acp.use_session_pool is True

    def test_acp_config_from_yaml(self) -> None:
        """ACP config should parse from YAML."""
        manifest = AgentsManifest.from_yaml("""
acp:
  use_session_pool: true
""")
        assert manifest.acp.use_session_pool is True

    def test_session_pool_config_from_yaml(self) -> None:
        """SessionPool config should parse from YAML."""
        manifest = AgentsManifest.from_yaml("""
session_pool:
  enable_auto_resume: false
  session_ttl_seconds: 7200.0
  max_auto_resume: 20
""")
        assert manifest.session_pool.enable_auto_resume is False
        assert manifest.session_pool.session_ttl_seconds == 7200.0
        assert manifest.session_pool.max_auto_resume == 20


# =============================================================================
# Group 3.7: AgentPool + SessionPool Integration
# =============================================================================


class TestAgentPoolSessionPoolIntegration:
    """Test AgentPool and SessionPool work together end-to-end."""

    @pytest.mark.integration
    async def test_create_session_returns_proper_session_id(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """create_session should return a SessionState with the correct session_id."""
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool:
            state = await pool.create_session("my-session-123")
            assert state.session_id == "my-session-123"

    @pytest.mark.integration
    async def test_session_pool_property_returns_active_pool(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """AgentPool.session_pool should return the initialized SessionPool."""
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool:
            sp = pool.session_pool
            assert sp is not None
            assert isinstance(sp, SessionPool)
            assert sp.pool is pool

    @pytest.mark.integration
    async def test_create_session_with_agent_name_and_metadata(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """create_session should propagate agent_name and metadata to SessionPool."""
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool:
            state = await pool.create_session(
                "session-with-meta",
                agent_name="test_agent",
                project="test-project",
                version="1.0",
            )
            assert state.agent_name == "test_agent"
            assert state.metadata.get("project") == "test-project"
            assert state.metadata.get("version") == "1.0"

    @pytest.mark.integration
    async def test_multiple_sessions_can_be_created(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """Multiple sessions should coexist in the SessionPool."""
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool:
            state1 = await pool.create_session("session-1")
            state2 = await pool.create_session("session-2")
            state3 = await pool.create_session("session-3")

            assert state1.session_id == "session-1"
            assert state2.session_id == "session-2"
            assert state3.session_id == "session-3"

            # All should be tracked by the SessionController
            sp = pool.session_pool
            assert sp is not None
            assert sp.sessions.get_session("session-1") is not None
            assert sp.sessions.get_session("session-2") is not None
            assert sp.sessions.get_session("session-3") is not None


# =============================================================================
# Group 3.8: Mixed-mode tests (SessionPool enabled/disabled)
# =============================================================================


class TestMixedMode:
    """Test agents work consistently with or without SessionPool."""

    @pytest.mark.integration
    async def test_agent_run_with_session_pool_enabled(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """Agent should produce output when SessionPool is enabled."""
        from pydantic_ai.models.test import TestModel

        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool:
            sp = pool.session_pool
            assert sp is not None
            # Create session and get per-session agent so TestModel
            # is set on the agent that session_pool.run_stream() will use.
            await sp.create_session("ses_test", agent_name="test_agent")
            agent = await sp.sessions.get_or_create_session_agent("ses_test")
            assert isinstance(agent, Agent)
            await agent.set_model(TestModel(custom_output_text="enabled"))
            result = await agent.run("hello", session_id="ses_test")
            assert result.data == "enabled"

    @pytest.mark.integration
    async def test_agent_run_with_session_pool_disabled(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """Agent should produce output when SessionPool is disabled.

        Since SessionPool is always enabled now, this tests the direct
        execution path by creating an agent without a pool reference.
        """
        from pydantic_ai.models.test import TestModel

        async with AgentPool(basic_manifest) as pool:
            # Create agent without pool reference to bypass SessionPool
            agent = pool.manifest.agents["test_agent"].get_agent()
            assert isinstance(agent, Agent)
            await agent.set_model(TestModel(custom_output_text="disabled"))
            result = await agent.run("hello", session_id="ses_test")
            assert result.data == "disabled"

    @pytest.mark.integration
    async def test_same_agent_api_in_both_modes(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """Agent API should behave identically regardless of SessionPool mode."""
        from pydantic_ai.models.test import TestModel

        # With SessionPool enabled
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool_enabled:
            sp = pool_enabled.session_pool
            assert sp is not None
            await sp.create_session("ses_test", agent_name="test_agent")
            agent_enabled = await sp.sessions.get_or_create_session_agent("ses_test")
            assert isinstance(agent_enabled, Agent)
            await agent_enabled.set_model(TestModel(custom_output_text="same"))
            result_enabled = await agent_enabled.run("hello", session_id="ses_test")

        # With SessionPool disabled (direct execution)
        async with AgentPool(basic_manifest) as pool_disabled:
            agent_disabled = pool_disabled.manifest.agents["test_agent"].get_agent()
            assert isinstance(agent_disabled, Agent)
            await agent_disabled.set_model(TestModel(custom_output_text="same"))
            result_disabled = await agent_disabled.run("hello", session_id="ses_test")

        assert result_enabled.data == result_disabled.data
        assert result_enabled.data == "same"

    @pytest.mark.integration
    async def test_get_agent_returns_same_type_in_both_modes(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """get_agent should return the same agent type regardless of SessionPool."""
        async with AgentPool(
            basic_manifest,
            enable_session_pool=True,
        ) as pool_enabled:
            agent_enabled = pool_enabled.manifest.agents["test_agent"].get_agent(pool=pool_enabled)

        async with AgentPool(basic_manifest) as pool_disabled:
            agent_disabled = pool_disabled.manifest.agents["test_agent"].get_agent(
                pool=pool_disabled
            )

        assert type(agent_enabled) is Agent
        assert type(agent_disabled) is Agent


# =============================================================================
# Group 3.9: Rollback tests (feature flag off after being on)
# =============================================================================


class TestRestart:
    """Test restarting AgentPool maintains SessionPool."""

    @pytest.mark.integration
    async def test_restart_maintains_session_pool(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """SessionPool should be available after restarting AgentPool."""
        pool = AgentPool(basic_manifest)

        # First run
        async with pool:
            assert pool.session_pool is not None
            state = await pool.create_session("session-1")
            assert state.session_id == "session-1"

        # Second run: SessionPool still available
        pool2 = AgentPool(basic_manifest)
        async with pool2:
            assert pool2.session_pool is not None
            state2 = await pool2.create_session("session-2")
            assert state2.session_id == "session-2"

    @pytest.mark.integration
    async def test_agent_functionality_preserved_after_restart(
        self,
        basic_manifest: AgentsManifest,
    ) -> None:
        """Agent should still work after restarting AgentPool."""
        from pydantic_ai.models.test import TestModel

        pool = AgentPool(basic_manifest)

        async with pool:
            sp = pool.session_pool
            assert sp is not None
            await sp.create_session("ses_test", agent_name="test_agent")
            agent = await sp.sessions.get_or_create_session_agent("ses_test")
            assert isinstance(agent, Agent)
            await agent.set_model(TestModel(custom_output_text="before"))
            result_before = await agent.run("hello", session_id="ses_test")
            assert result_before.data == "before"

        pool_after = AgentPool(basic_manifest)
        async with pool_after:
            sp_after = pool_after.session_pool
            assert sp_after is not None
            await sp_after.create_session("ses_test", agent_name="test_agent")
            agent_after = await sp_after.sessions.get_or_create_session_agent("ses_test")
            assert isinstance(agent_after, Agent)
            await agent_after.set_model(TestModel(custom_output_text="after"))
            result_after = await agent_after.run("hello", session_id="ses_test")
            assert result_after.data == "after"
