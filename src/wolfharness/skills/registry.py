"""Claude Code Skills registry with auto-discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from fsspec import AbstractFileSystem
from upathtools import is_directory
from upathtools.helpers import to_upath, upath_to_fs

from wolfharness.log import get_logger
from wolfharness.skills.skill import Skill
from wolfharness.tools.exceptions import ToolError
from wolfharness.utils.baseregistry import BaseRegistry


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from upathtools import JoinablePathLike, UPath


logger = get_logger(__name__)


class SkillsRegistry(BaseRegistry[str, Skill]):
    """Registry for Claude Code Skills with auto-discovery."""

    DEFAULT_SKILL_PATHS: ClassVar = ["~/.claude/skills/", ".claude/skills/"]

    def __init__(self, skills_dirs: Sequence[JoinablePathLike] | None = None) -> None:
        """Initialize with custom skill directories or auto-detect."""
        super().__init__()
        if skills_dirs:
            self.skills_dirs = [to_upath(i).expanduser() for i in skills_dirs]
        else:
            self.skills_dirs = [to_upath(i).expanduser() for i in self.DEFAULT_SKILL_PATHS]

        # Event handlers for skill lifecycle changes
        self._skill_added_handlers: list[Callable[[str, Skill], None]] = []
        self._skill_removed_handlers: list[Callable[[str, None], None]] = []

    async def discover_skills(self) -> None:
        """Scan filesystem and register all found skills."""
        for skills_dir in self.skills_dirs:
            await self.register_skills_from_path(skills_dir)

    async def register_skills_from_path(
        self,
        skills_dir: JoinablePathLike | AbstractFileSystem,
        base_path: str | None = None,
        replace: bool = True,
        **storage_options: Any,
    ) -> None:
        """Register skills from a given path.

        Args:
            skills_dir: Path to the directory containing skills, or filesystem instance.
            base_path: When skills_dir is a filesystem, the path within that filesystem
                      to look for skills. Defaults to root_marker if not specified.
            replace: Whether to replace existing skills with same name.
            storage_options: Additional options to pass to the filesystem.
        """
        from upathtools.async_ops import to_async_fs

        if isinstance(skills_dir, AbstractFileSystem):
            fs = to_async_fs(skills_dir)
            search_path = base_path if base_path is not None else fs.root_marker
            original_skills_dir: UPath | None = None
        else:
            original_skills_dir = to_upath(skills_dir).expanduser()
            fs = upath_to_fs(original_skills_dir, **storage_options)
            search_path = fs.root_marker

        try:
            entries = await fs._ls(search_path, detail=True)
        except FileNotFoundError:
            logger.debug("Skills directory not found", path=search_path)
            return

        skill_dirs = [
            entry
            for entry in entries
            if await is_directory(fs, entry["name"], entry_type=entry.get("type"))
        ]
        if not skill_dirs:
            logger.info("No skills found", skills_dir=search_path)
            return
        logger.info("Found skills", skills=skill_dirs, skills_dir=search_path)
        for skill_entry in skill_dirs:
            entry_name = skill_entry["name"]
            if original_skills_dir is not None:
                skill_dir_path = original_skills_dir / entry_name
            else:
                skill_dir_path = to_upath(entry_name)

            fs_skill_md_path = f"{entry_name}/SKILL.md"
            try:
                await fs._cat_file(fs_skill_md_path)
            except FileNotFoundError:
                logger.debug("SKILL.md not found in skill directory", path=entry_name)
                continue

            try:
                skill = self._parse_skill(skill_dir_path)
                self.register(skill.name, skill, replace=replace)
                logger.debug("Successfully loaded skill", name=skill.name, path=str(skill_dir_path))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to parse skill",
                    path=str(skill_dir_path),
                    error=str(e),
                    error_type=type(e).__name__,
                )

    def _parse_skill(self, skill_dir_path: JoinablePathLike) -> Skill:
        """Parse a skill from its directory path.

        Args:
            skill_dir_path: Path to the skill directory containing SKILL.md

        Returns:
            Parsed Skill instance

        Raises:
            ToolError: If skill cannot be parsed
        """
        from upathtools import to_upath

        path = to_upath(skill_dir_path)

        try:
            # Use the Skill class method to properly parse SKILL.md with frontmatter
            return Skill.from_skill_dir(path)
        except FileNotFoundError as e:
            raise ToolError(f"SKILL.md not found in {path}") from e

    @property
    def _error_class(self) -> type[ToolError]:
        """Error class to use for this registry."""
        return ToolError

    def _validate_item(self, item: Any) -> Skill:
        """Validate and possibly transform item before registration."""
        if not isinstance(item, Skill):
            raise ToolError(f"Expected Skill instance, got {type(item)}")
        return item

    def get_skill_instructions(self, skill_name: str) -> str:
        """Lazy load full instructions for a skill."""
        skill = self.get(skill_name)
        return skill.load_instructions()

    def on_skill_added(self, callback: Callable[[str, Skill], None]) -> None:
        """Register a callback to be called when a skill is added.

        Args:
            callback: A callable that receives the skill name and skill instance.
        """
        self._skill_added_handlers.append(callback)

    def on_skill_removed(self, callback: Callable[[str, None], None]) -> None:
        """Register a callback to be called when a skill is removed.

        Args:
            callback: A callable that receives the skill name and None.
        """
        self._skill_removed_handlers.append(callback)

    def register(self, key: str, item: Skill | Any, replace: bool = False) -> None:
        """Register a skill and emit events to registered callbacks.

        Args:
            key: The skill name to register.
            item: The skill instance or data to register.
            replace: Whether to replace an existing skill with the same name.
        """
        super().register(key, item, replace)
        for handler in self._skill_added_handlers:
            handler(key, item)

    def __delitem__(self, key: str) -> None:
        """Remove a skill and emit events to registered callbacks.

        Args:
            key: The skill name to remove.
        """
        if key in self._items:
            for handler in self._skill_removed_handlers:
                handler(key, None)
            del self._items[key]
        else:
            raise self._error_class(f"Item not found: {key}")


if __name__ == "__main__":
    import os

    import anyio
    from upathtools import UPath

    from wolfharness.log import configure_logging

    configure_logging()

    async def main() -> None:
        reg = SkillsRegistry()
        p = UPath(
            "github://",
            token=os.getenv("GITHUB_TOKEN"),
            username="phil65",
            org="anthropics",
            repo="skills",
        )
        print("Repository contents:")
        print([f.name for f in p.iterdir()][:5])

        await reg.register_skills_from_path(
            p,
            token=os.getenv("GITHUB_TOKEN"),
            username="phil65",
            org="anthropics",
            repo="skills",
        )
        print(f"Found {len(reg)} skills:")
        for name, skill in reg.items():
            print(f"  - {name}: {skill.description[:60]}...")

    anyio.run(main)
