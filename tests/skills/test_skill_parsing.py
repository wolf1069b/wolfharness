"""Tests for Skill model field parsing: allowed_tools, mcp.json loading, mcp_servers/tools."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
from upathtools import UPath

from wolfharness.skills.skill import Skill


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# allowed_tools parsing
# ---------------------------------------------------------------------------


def test_allowed_tools_string_format(tmp_path: Path) -> None:
    """allowed-tools: "bash, read" → parsed_allowed_tools() returns ["bash", "read"]."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: A skill with allowed tools
        allowed-tools: "bash, read"
        ---

        # Instructions
        """),
        encoding="utf-8",
    )
    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.parsed_allowed_tools() == ["bash", "read"]


def test_allowed_tools_list_normalized() -> None:
    """Skill(allowed_tools=["bash", "read"]) → normalized to string, parsed correctly."""
    # Direct construction triggers the list→str before-validator
    skill = Skill(
        name="my-skill",
        description="A skill with allowed tools",
        skill_path=UPath("/virtual/my-skill"),
        allowed_tools=["bash", "read"],
    )
    assert isinstance(skill.allowed_tools, str)
    assert skill.parsed_allowed_tools() == ["bash", "read"]


def test_allowed_tools_none(tmp_path: Path) -> None:
    """allowed_tools=None → parsed_allowed_tools() returns []."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: No allowed tools
        ---

        # Instructions
        """),
        encoding="utf-8",
    )
    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.allowed_tools is None
    assert skill.parsed_allowed_tools() == []


def test_allowed_tools_comma_and_space_mixed(tmp_path: Path) -> None:
    """Mixed delimiters like 'bash, read grep' → parsed correctly."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: Mixed delimiters
        allowed-tools: "bash, read grep"
        ---

        # Instructions
        """),
        encoding="utf-8",
    )
    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.parsed_allowed_tools() == ["bash", "read", "grep"]


# ---------------------------------------------------------------------------
# mcp.json loading and precedence
# ---------------------------------------------------------------------------


def test_mcp_json_loads_and_takes_precedence(tmp_path: Path) -> None:
    """mcp.json in skill dir is loaded and overrides frontmatter mcp-servers."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    # SKILL.md with frontmatter mcp-servers
    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: Skill with MCP
        mcp-servers:
          frontmatter-server:
            command: "echo"
            args: ["frontmatter"]
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    # mcp.json companion file (should take precedence)
    (skill_dir / "mcp.json").write_text(
        json.dumps({
            "mcpServers": {
                "json-server": {
                    "command": "npx",
                    "args": ["-y", "@test/mcp"],
                },
            },
        }),
        encoding="utf-8",
    )

    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.mcp_servers is not None
    # Frontmatter server should NOT be present — mcp.json overrides
    assert "frontmatter-server" not in skill.mcp_servers
    # JSON server should be present
    assert "json-server" in skill.mcp_servers
    assert skill.mcp_servers["json-server"].command == "npx"
    assert skill.mcp_servers["json-server"].args == ["-y", "@test/mcp"]


def test_mcp_json_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """${VAR} in mcp.json is expanded from environment."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: Skill with env vars
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    # Set env var
    monkeypatch.setenv("MCP_TEST_PATH", "/custom/path")

    (skill_dir / "mcp.json").write_text(
        json.dumps({
            "mcpServers": {
                "env-server": {
                    "command": "python",
                    "args": ["${MCP_TEST_PATH}/server.py"],
                    "env": {"DATA_DIR": "${MCP_TEST_PATH}/data"},
                },
            },
        }),
        encoding="utf-8",
    )

    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.mcp_servers is not None
    server = skill.mcp_servers["env-server"]
    assert server.args == ["/custom/path/server.py"]
    assert server.env == {"DATA_DIR": "/custom/path/data"}


def test_mcp_json_absent_falls_back_to_frontmatter(tmp_path: Path) -> None:
    """Without mcp.json, frontmatter mcp-servers is used."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: Skill with frontmatter MCP
        mcp-servers:
          fm-server:
            command: "echo"
            args: ["hello"]
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    # No mcp.json — frontmatter should be preserved
    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.mcp_servers is not None
    assert "fm-server" in skill.mcp_servers
    assert skill.mcp_servers["fm-server"].command == "echo"


def test_mcp_json_missing_file(tmp_path: Path) -> None:
    """No mcp.json and no frontmatter mcp-servers → mcp_servers is None."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: No MCP at all
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.mcp_servers is None


@pytest.mark.skip(
    reason="Flaky in CI: caplog capture unreliable across logger propagation/handler ordering"
)
def test_mcp_json_invalid_json_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Invalid mcp.json is silently ignored (warning logged)."""
    import logging

    caplog.set_level(logging.WARNING)

    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: Skill with broken mcp.json
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    (skill_dir / "mcp.json").write_text("this is not json", encoding="utf-8")

    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.mcp_servers is None
    assert "Failed to parse mcp.json" in caplog.text


# ---------------------------------------------------------------------------
# tools parsing
# ---------------------------------------------------------------------------


def test_tools_parsed_from_frontmatter(tmp_path: Path) -> None:
    """Tools list in frontmatter is parsed into SkillToolConfig objects."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: Skill with Python tools
        tools:
          - type: python
            import_path: "os:getcwd"
          - type: python
            import_path: "os:listdir"
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.tools is not None
    assert len(skill.tools) == 2
    assert skill.tools[0].import_path == "os:getcwd"
    assert skill.tools[1].import_path == "os:listdir"


def test_tools_not_present(tmp_path: Path) -> None:
    """No 'tools' in frontmatter → tools is None."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
        ---
        name: my-skill
        description: No tools
        ---

        # Instructions
        """),
        encoding="utf-8",
    )

    skill = Skill.from_skill_dir(UPath(skill_dir))
    assert skill.tools is None
