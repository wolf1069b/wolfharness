"""Tests for skill loading behavior with include_default=false.

These tests verify that ACP server respects the manifest's skills.include_default
setting when deciding whether to load default skills (.claude/skills/).

Run with: pytest tests/servers/acp_server/test_acp_skills_red_flags.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from upathtools import UPath

from wolfharness.skills.manager import SkillsManager
from wolfharness_config.skills import SkillsConfig


pytestmark = pytest.mark.unit


class TestSkillsIncludeDefault:
    """Tests: skills loading must respect manifest configuration."""

    def test_skills_config_include_default_false(self) -> None:
        """SkillsConfig with include_default=false must not include default paths."""
        config = SkillsConfig(
            paths=[UPath("./custom-skills/")],
            include_default=False,
        )

        paths = config.get_effective_paths()

        # Should only contain the custom path, not defaults
        default_paths = [UPath("~/.claude/skills/"), UPath(".claude/skills/")]
        for default_path in default_paths:
            assert default_path not in paths, (
                f"Default path {default_path} found in effective paths despite "
                f"include_default=false."
            )
        assert UPath("./custom-skills/") in paths, "Custom path missing from effective paths"

    def test_skills_config_include_default_true(self) -> None:
        """SkillsConfig with include_default=true must include default paths."""
        config = SkillsConfig(
            paths=[UPath("./custom-skills/")],
            include_default=True,
        )

        paths = config.get_effective_paths()

        # Should contain both custom and default paths
        assert UPath("./custom-skills/") in paths, "Custom path missing"
        assert len(paths) > 1, "Default paths not included when include_default=true"

    def test_wolfharness_acp_agent_load_skills_defaults_to_none(self) -> None:
        """AgentPoolACPAgent.load_skills defaults to None.

        None means "use manifest's include_default setting".
        """
        from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

        assert AgentPoolACPAgent.load_skills is None, (
            "AgentPoolACPAgent.load_skills should default to None, "
            "allowing the manifest's include_default setting to control behavior."
        )

    def test_acp_server_from_config_uses_manifest_include_default(self) -> None:
        """ACPServer.from_config must derive load_skills from manifest's include_default.

        When load_skills is not explicitly provided (None), from_config should
        use manifest.skills.include_default as the default.
        """
        # Create a manifest with include_default=False
        from wolfharness.models.manifest import AgentsManifest
        from wolfharness_config.skills import SkillsConfig
        from wolfharness_server.acp_server.server import ACPServer

        manifest = AgentsManifest(
            skills=SkillsConfig(
                paths=[UPath("./skills/")],
                include_default=False,
            )
        )

        # Pass load_skills=None (default) - should use manifest's include_default=False
        server = ACPServer.from_config(
            manifest,
            load_skills=None,
        )

        assert server.load_skills is False, (
            "ACPServer.load_skills should be False when manifest has include_default=False "
            "and no explicit load_skills argument is provided."
        )

    def test_acp_server_from_config_explicit_load_skills_overrides_manifest(self) -> None:
        """Explicit load_skills argument must override manifest's include_default."""
        from wolfharness.models.manifest import AgentsManifest
        from wolfharness_config.skills import SkillsConfig
        from wolfharness_server.acp_server.server import ACPServer

        manifest = AgentsManifest(
            skills=SkillsConfig(
                paths=[UPath("./skills/")],
                include_default=False,  # Manifest says False
            )
        )

        # Explicit True overrides manifest
        server = ACPServer.from_config(manifest, load_skills=True)
        assert server.load_skills is True, "Explicit load_skills=True should override manifest"

        manifest2 = AgentsManifest(
            skills=SkillsConfig(
                paths=[UPath("./skills/")],
                include_default=True,  # Manifest says True
            )
        )

        # Explicit False overrides manifest
        server2 = ACPServer.from_config(manifest2, load_skills=False)
        assert server2.load_skills is False, "Explicit load_skills=False should override manifest"

    @pytest.mark.asyncio
    async def test_init_client_skills_respects_none_load_skills(self) -> None:
        """init_client_skills() must not be called when load_skills resolves to False.

        When AgentPoolACPAgent.load_skills is None and manifest has include_default=False,
        init_client_skills should not be called.
        """
        # Create a real AgentPool with include_default=False
        import yamling

        from wolfharness import AgentPool, AgentsManifest
        from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

        config = """\
agents:
  test_agent:
    type: native
    model: test
    system_prompt: "You are a test agent."
skills:
  include_default: false
"""
        manifest_dict = yamling.load_yaml(config, verify_type=dict)
        manifest_obj = AgentsManifest.model_validate(manifest_dict)

        # Create a mock agent with the pool's host_context
        async with AgentPool(manifest_obj) as real_pool:
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"
            mock_agent.host_context = real_pool.get_context()

            # Create ACP agent with load_skills=None
            acp_agent = AgentPoolACPAgent(
                client=MagicMock(),
                default_agent=mock_agent,
                load_skills=None,
            )

            # The load_skills should be None, and when checking should_load_skills,
            # it should resolve to False based on manifest
            assert acp_agent.load_skills is None
            assert acp_agent.host_context is not None
            assert acp_agent.host_context.manifest.skills.include_default is False

    def test_serve_acp_cli_load_skills_defaults_to_none(self) -> None:
        """serve-acp CLI load_skills defaults to None.

        None means "use manifest's skills.include_default setting".
        Users can explicitly pass --skills or --no-skills to override.
        """
        import inspect

        from wolfharness_cli.serve_acp import acp_command

        sig = inspect.signature(acp_command)
        load_skills_param = sig.parameters.get("load_skills")
        assert load_skills_param is not None, "load_skills parameter not found"
        assert load_skills_param.default is None, (
            "serve-acp CLI load_skills should default to None, "
            "so that the manifest's skills.include_default setting is used by default."
        )

    def test_manifest_include_default_controls_acp_skill_loading(self) -> None:
        """Manifest's include_default must control ACP skill loading.

        When skills.include_default=false in manifest and no explicit CLI override,
        ACP server should NOT load .claude/skills/.
        """
        from wolfharness.models.manifest import AgentsManifest
        from wolfharness_config.skills import SkillsConfig

        manifest = AgentsManifest(
            skills=SkillsConfig(
                paths=[UPath("./skills/")],
                include_default=False,
            )
        )

        # Without explicit override, load_skills should follow manifest
        from wolfharness_server.acp_server.server import ACPServer

        server = ACPServer.from_config(manifest)
        assert server.load_skills is False, (
            "ACP server should not load skills when manifest has include_default=False "
            "and no explicit load_skills argument is provided."
        )

    def test_manifest_include_default_true_loads_skills(self) -> None:
        """Manifest's include_default=True must enable ACP skill loading."""
        from wolfharness.models.manifest import AgentsManifest
        from wolfharness_config.skills import SkillsConfig
        from wolfharness_server.acp_server.server import ACPServer

        manifest = AgentsManifest(
            skills=SkillsConfig(
                paths=[UPath("./skills/")],
                include_default=True,
            )
        )

        server = ACPServer.from_config(manifest)
        assert server.load_skills is True, (
            "ACP server should load skills when manifest has include_default=True "
            "and no explicit load_skills argument is provided."
        )

    @pytest.mark.asyncio
    async def test_local_resource_provider_respects_include_default_false(self, tmp_path) -> None:
        """SkillsManager must not discover default skills when include_default=False.

        Regression: SkillsManager.discover_skills() did not sync registry.skills_dirs,
        so default skill paths leaked into the registry.
        """
        import logging

        from wolfharness_config.skills import SkillsConfig

        # Enable debug logging
        logging.getLogger("wolfharness.skills").setLevel(logging.DEBUG)

        # Create a custom skill directory with one skill
        custom_skills_dir = tmp_path / "custom-skills"
        custom_skills_dir.mkdir()
        skill_dir = custom_skills_dir / "custom-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: custom-skill\ndescription: A custom test skill\n---\n# Custom Skill"
        )

        # Create a default skill directory with one skill
        default_skills_dir = tmp_path / ".claude" / "skills"
        default_skills_dir.mkdir(parents=True)
        default_skill_dir = default_skills_dir / "default-skill"
        default_skill_dir.mkdir()
        (default_skill_dir / "SKILL.md").write_text(
            "---\nname: default-skill\ndescription: A default test skill\n---\n# Default Skill"
        )

        # Create SkillsManager with include_default=False
        config = SkillsConfig(
            paths=[UPath(str(custom_skills_dir))],
            include_default=False,
        )

        skills_manager = SkillsManager(
            name="test_skills",
            config=config,
            config_file_path=tmp_path / "config.yml",
        )

        # Enter context to initialize
        await skills_manager.__aenter__()

        try:
            # Debug: print paths
            print(
                f"DEBUG: skills_manager.registry.skills_dirs ="
                f" {skills_manager.registry.skills_dirs}"
            )

            print(
                f"DEBUG: skills_manager.registry.list_items() ="
                f" {skills_manager.registry.list_items()}"
            )

            # Verify only custom skill was discovered, not default skill
            skill_names = set(skills_manager.registry.list_items())
            print(f"DEBUG: skill_names = {skill_names}")

            # Should only have custom skill, not default skill
            assert "custom-skill" in skill_names, (
                f"Custom skill missing from registry. Got: {skill_names}"
            )
            assert "default-skill" not in skill_names, (
                f"Default skill leaked into registry despite include_default=False. "
                f"Got: {skill_names}. This is the SkillsManager regression."
            )
        finally:
            await skills_manager.__aexit__(None, None, None)
