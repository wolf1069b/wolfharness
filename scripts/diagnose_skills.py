"""Skills TUI display diagnostic script.

Checks every stage of the skill discovery pipeline to identify
why skills may not appear in the TUI.

Usage:
    uv run python scripts/diagnose_skills.py [config.yml]
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(name: str, passed: bool, detail: str = "") -> None:
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}")
    if detail:
        print(f"     {detail}")


async def diagnose(config_path: str | None = None) -> None:  # noqa: PLR0915
    """Run full diagnostic pipeline."""
    from upathtools import to_upath

    # ============================================================
    # Stage 1: Filesystem Discovery
    # ============================================================
    print_header("Stage 1: Filesystem Discovery")

    from wolfharness_config.skills import DEFAULT_SKILLS_PATHS

    # Check default skill directories
    for dp in DEFAULT_SKILLS_PATHS:
        resolved = to_upath(dp).expanduser()
        exists = resolved.exists()
        print_result(
            f"Default path: {dp} (resolved: {resolved})",
            exists,
            "Directory exists" if exists else "Directory NOT found",
        )
        if exists:
            # Check for SKILL.md in subdirectories
            subdirs = [p for p in resolved.iterdir() if p.is_dir()]
            if subdirs:
                for subdir in subdirs:
                    skill_md = subdir / "SKILL.md"
                    has_skill_md = skill_md.exists()
                    print_result(
                        f"  Subdir: {subdir.name}/",
                        has_skill_md,
                        "SKILL.md found" if has_skill_md else "SKILL.md NOT found",
                    )
                    if has_skill_md:
                        # Try to parse frontmatter
                        try:
                            content = skill_md.read_text()
                            if content.startswith("---"):
                                end = content.find("---", 3)
                                if end != -1:
                                    import yaml

                                    frontmatter = yaml.safe_load(content[3:end])
                                    has_name = "name" in (frontmatter or {})
                                    has_desc = "description" in (frontmatter or {})
                                    print_result(
                                        "  Frontmatter has 'name'",
                                        has_name,
                                        f"name={frontmatter.get('name')!r}"
                                        if has_name
                                        else "MISSING",
                                    )
                                    print_result(
                                        "  Frontmatter has 'description'",
                                        has_desc,
                                        f"description={frontmatter.get('description')!r}"
                                        if has_desc
                                        else "MISSING",
                                    )
                                    # Check for unknown keys
                                    from wolfharness.skills.skill import SkillMetadata

                                    try:
                                        SkillMetadata.model_validate(frontmatter)
                                        print_result(
                                            "  Frontmatter validation",
                                            True,
                                            "Passes strict validation",
                                        )
                                    except Exception as e:  # noqa: BLE001
                                        print_result(
                                            "  Frontmatter validation", False, f"FAILED: {e}"
                                        )
                                else:
                                    print_result(
                                        "  Frontmatter parsing", False, "No closing --- found"
                                    )
                            else:
                                print_result(
                                    "  Frontmatter parsing",
                                    False,
                                    "No YAML frontmatter (must start with ---)",
                                )
                        except Exception as e:  # noqa: BLE001
                            print_result("  Frontmatter parsing", False, str(e))
            else:
                print_result(
                    f"  No subdirectories in {resolved}", False, "Skills must be in SUBDIRECTORIES"
                )

    # Check custom paths from config
    if config_path:
        print("\n  --- Custom paths from config ---")
        try:
            from wolfharness.models.manifest import AgentsManifest

            manifest = AgentsManifest.from_file(config_path)
            if manifest.skills:
                for p in manifest.skills.paths:
                    resolved = to_upath(p).expanduser()
                    exists = resolved.exists()
                    print_result(
                        f"Custom path: {p} (resolved: {resolved})",
                        exists,
                        "Directory exists" if exists else "Directory NOT found",
                    )
                print_result(
                    "include_default=True",
                    manifest.skills.include_default,
                    "Default paths will also be searched"
                    if manifest.skills.include_default
                    else "Default paths will NOT be searched",
                )
            else:
                print_result("Skills config in manifest", False, "No skills section in config")
        except Exception as e:  # noqa: BLE001
            print_result("Config loading", False, str(e))

    # ============================================================
    # Stage 2: SkillsManager / SkillsRegistry
    print_header("Stage 2: SkillsManager / SkillsRegistry")

    from wolfharness.skills.manager import SkillsManager

    try:
        config_file_path = to_upath(config_path) if config_path else None
        from wolfharness_config.skills import SkillsConfig

        skills_config: SkillsConfig | None = None
        if config_path:
            from wolfharness.models.manifest import AgentsManifest

            manifest = AgentsManifest.from_file(config_path)
            skills_config = manifest.skills

        manager = SkillsManager(
            name="diagnostic",
            config=skills_config,
            config_file_path=config_file_path,
        )
        async with manager:
            skill_names = manager.registry.list_items()
            print_result(
                "SkillsManager discovered skills",
                len(skill_names) > 0,
                f"Found {len(skill_names)} skills: {skill_names}"
                if skill_names
                else "NO skills found!",
            )

    except Exception as e:  # noqa: BLE001
        print_result("SkillsManager initialization", False, str(e))

    # ============================================================
    # Stage 3: AgentPool initialization (if config provided)
    # ============================================================
    if config_path:
        print_header("Stage 3: AgentPool Initialization")

        try:
            from wolfharness.delegation import AgentPool

            async with AgentPool(config_path) as pool:
                # Check skill_provider
                sp = pool.skill_provider
                print_result(
                    "pool.skill_provider is not None",
                    sp is not None,
                    f"type={type(sp).__name__}" if sp else "Skill provider not initialized!",
                )

                if sp is not None:
                    try:
                        provider_skills = await sp.get_skills()
                        print_result(
                            "skill_provider returns skills",
                            len(provider_skills) > 0,
                            (
                                f"Found {len(provider_skills)} skills:"
                                f" {[s.name for s in provider_skills]}"
                            ),
                        )
                    except Exception as e:  # noqa: BLE001
                        print_result("skill_provider.get_skills()", False, str(e))

                # Check skills registry
                skills = pool.skills
                skill_list = skills.list_skills()
                print_result(
                    "pool.skills has skills",
                    len(skill_list) > 0,
                    f"Found {len(skill_list)} skills: {[s.name for s in skill_list]}",
                )

        except Exception as e:  # noqa: BLE001
            print_result("AgentPool initialization", False, str(e))
            import traceback

            traceback.print_exc()

    # ============================================================
    # Stage 4: OpenCode Server Bridge (simulated)
    # ============================================================
    print_header("Stage 4: OpenCode Server Bridge Check")

    if config_path:
        try:
            from wolfharness.delegation import AgentPool

            async with AgentPool(config_path) as pool:
                # Simulate what server.py does
                from wolfharness_server.opencode_server.skill_bridge import OpenCodeSkillBridge

                bridge = OpenCodeSkillBridge(skill_provider=pool.skill_provider)

                commands = bridge.get_commands()
                print_result(
                    "Bridge has commands after subscription",
                    len(commands) > 0,
                    f"Commands: {[c.name for c in commands]}"
                    if commands
                    else "NO commands in bridge!",
                )

                bridge_skill_commands = bridge.get_skill_commands()
                print_result(
                    "Bridge has skill_commands",
                    len(bridge_skill_commands) > 0,
                    f"Skills: {[c.name for c in bridge_skill_commands]}"
                    if bridge_skill_commands
                    else "NO skill commands!",
                )
        except Exception as e:  # noqa: BLE001
            print_result("Bridge simulation", False, str(e))
    else:
        print("  ⚠️  Skipped (no config path provided)")

    # ============================================================
    # Stage 5: HTTP API Check (if server is running)
    # ============================================================
    print_header("Stage 5: HTTP API Check")

    import json
    import urllib.request

    base_url = "http://127.0.0.1:4096"
    for endpoint in ["/skill", "/command"]:
        try:
            req = urllib.request.Request(f"{base_url}{endpoint}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                if isinstance(data, list):
                    print_result(
                        f"GET {endpoint}",
                        len(data) > 0,
                        f"Returns {len(data)} items: {[d.get('name', '?') for d in data]}",
                    )
                else:
                    print_result(f"GET {endpoint}", True, f"Returns: {data}")
        except Exception as e:  # noqa: BLE001
            print_result(f"GET {endpoint}", False, f"Server not reachable or error: {e}")

    # ============================================================
    # Summary
    # ============================================================
    print_header("Summary & Common Fixes")

    print("""
Common fixes for skills not showing in TUI:

1. SKILL.md must be in a SUBDIRECTORY of the skills dir:
   ✅  ~/.claude/skills/my-skill/SKILL.md
   ❌  ~/.claude/skills/SKILL.md

2. SKILL.md must have valid YAML frontmatter:
   ---
   name: my-skill
   description: What this skill does
   ---
   (No unknown keys allowed! Only: name, description)

3. Config must include the skills directory:
   skills:
     paths:
       - ./my-skills
     include_default: true

4. Relative paths resolve against the CONFIG FILE location, not CWD.

5. Check server logs with:
   OBSERVABILITY_ENABLED=true wolfharness serve-opencode config.yml

6. If pool.skill_provider is None, the OpenCode bridge won't have commands.
""")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else None
    if config:
        print(f"Using config: {config}")
    else:
        print("No config path provided. Some checks will be skipped.")
        print("Usage: uv run python scripts/diagnose_skills.py [config.yml]")

    asyncio.run(diagnose(config))
