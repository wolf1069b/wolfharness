"""Skills manager for pool-wide management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, overload

from upathtools import JoinablePathLike, UPath, to_upath

from wolfharness.log import get_logger
from wolfharness.skills.registry import SkillsRegistry
from wolfharness_config.skills import SkillsConfig  # noqa: TC001


if TYPE_CHECKING:
    from types import TracebackType

    from fsspec import AbstractFileSystem

    from wolfharness.skills.skill import Skill


logger = get_logger(__name__)


class SkillsManager:
    """Pool-wide skills registry management.

    Owns the single skills registry for the pool. Skills can be discovered
    from multiple directories. The actual `load_skill` tool is provided
    separately via SkillsTools toolset.
    """

    def __init__(
        self,
        name: str = "skills",
        owner: str | None = None,
        skills_dirs: list[JoinablePathLike] | None = None,
        config: SkillsConfig | None = None,
        config_file_path: UPath | None = None,
    ) -> None:
        """Initialize the skills manager.

        Args:
            name: Name for this manager
            owner: Owner of this manager
            skills_dirs: Directories to search for skills
            config: Optional skills configuration from manifest
            config_file_path: Optional path to configuration file for resolving relative paths
        """
        self.name = name
        self.owner = owner
        self.registry = SkillsRegistry(skills_dirs)
        self._initialized = False
        self._config = config
        self._config_file_path = config_file_path

    def __repr__(self) -> str:
        skill_count = len(self.registry.list_items()) if self._initialized else "?"
        return f"SkillsManager(name={self.name!r}, skills={skill_count})"

    async def __aenter__(self) -> Self:
        """Initialize to skills manager and discover skills."""
        try:
            await self.discover_skills(self._config, self._config_file_path)
            self._initialized = True
            count = len(self.registry.list_items())
            logger.info("Skills manager initialized", name=self.name, skill_count=count)
        except Exception as e:
            msg = "Failed to initialize skills manager"
            logger.exception(msg, name=self.name, error=e)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up the skills manager."""

    @overload
    async def add_skills_directory(self, path: JoinablePathLike) -> None: ...

    @overload
    async def add_skills_directory(self, path: str, *, fs: AbstractFileSystem) -> None: ...

    async def add_skills_directory(
        self,
        path: JoinablePathLike,
        *,
        fs: AbstractFileSystem | None = None,
    ) -> None:
        """Add a new skills directory and discover its skills.

        Args:
            path: Path to directory containing skills.
            fs: Optional filesystem instance. When provided, path is interpreted
                as a path within this filesystem (e.g., ".claude/skills").
        """
        if fs is not None:
            # Pass filesystem directly to registry with the path
            await self.registry.register_skills_from_path(fs, base_path=str(path))
            logger.info(
                "Added skills directory from filesystem",
                protocol=fs.protocol,
                path=path,
            )
        else:
            upath = to_upath(path)
            if upath not in self.registry.skills_dirs:
                self.registry.skills_dirs.append(upath)
                await self.registry.register_skills_from_path(upath)
                logger.info("Added skills directory", path=str(path))

    async def discover_skills(
        self,
        config: SkillsConfig | None = None,
        config_file_path: UPath | None = None,
    ) -> None:
        """Discover skills from configured paths.

        Args:
            config: Optional skills configuration.
            config_file_path: Optional path to configuration file for resolving relative paths.
        """
        from wolfharness_config.skills import DEFAULT_SKILLS_PATHS

        base_path = config_file_path.parent if config_file_path is not None else UPath.cwd()
        logger.debug("Starting skills discovery", base_path=str(base_path))

        if config:
            paths = config.get_effective_paths(config_file_path)
            default_paths = [p.expanduser() for p in DEFAULT_SKILLS_PATHS]
        else:
            paths = self.registry.skills_dirs
            default_paths = [p.expanduser() for p in DEFAULT_SKILLS_PATHS]

        # Sync registry's skills_dirs to reflect actual effective paths.
        # This ensures downstream consumers (e.g. SkillManagerCap)
        # do not re-discover default paths when include_default=False.
        self.registry.skills_dirs = [to_upath(p).expanduser() for p in paths]

        logger.debug("Skills discovery paths", paths=[str(p) for p in paths])
        for path in reversed(paths):
            upath = to_upath(path).expanduser()
            if not upath.exists():
                if any(upath == dp for dp in default_paths):
                    logger.debug("Default skills directory not found", path=upath)
                else:
                    logger.warning("Custom skills directory not found", path=upath)
                continue
            logger.debug("Processing skills directory", path=str(upath))
            await self.registry.register_skills_from_path(upath, replace=True)
        logger.debug("Skills discovery completed", total_skills=len(self.registry.list_items()))

    async def refresh(self) -> None:
        """Force rediscovery of all skills."""
        await self.discover_skills()
        skill_count = len(self.registry.list_items())
        logger.info("Skills refreshed", name=self.name, skill_count=skill_count)

    def list_skills(self) -> list[Skill]:
        """Get all available skills."""
        return [self.registry.get(name) for name in self.registry.list_items()]

    def get_skill(self, name: str) -> Skill:
        """Get a skill by name."""
        return self.registry.get(name)

    def get_skill_instructions(self, skill_name: str) -> str:
        """Get full instructions for a specific skill."""
        return self.registry.get_skill_instructions(skill_name)
