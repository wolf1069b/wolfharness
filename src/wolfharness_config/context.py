"""Context variable management for config path resolution.

This module provides context-aware path resolution using ContextVars,
allowing config-relative paths to work correctly regardless of CWD.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from contextvars import Token
    from types import TracebackType
    from typing import Self

    from upathtools import JoinablePathLike, UPath
else:
    from upathtools import UPath


# Context variable storing the current config directory.
# This is set during manifest loading to enable config-relative path resolution.
CONFIG_DIR: ContextVar[UPath | None] = ContextVar("config_dir", default=None)

# Global module-level variable for config directory
# This persists even outside of with-blocks, allowing runtime access
_config_dir_global: UPath | None = None


def get_config_dir() -> UPath | None:
    """Get the current config directory for runtime access.

    This function returns the config directory from the most recent
    ConfigContextManager, even when called outside the with-block.
    Useful for Providers and other runtime components that need
    to resolve paths after initialization.

    Priority:
    1. Module-level global variable (_config_dir_global)
    2. ContextVar (CONFIG_DIR) - for backward compatibility
    3. None if no context is set

    Returns:
        UPath: The config directory path, or None if not set

    Example:
        >>> with ConfigContextManager("/project/config.yml"):
        ...     # Inside with block
        ...     dir1 = get_config_dir()
        ... # Outside with block - still accessible!
        ... dir2 = get_config_dir()  # Returns same path
    """
    if _config_dir_global is not None:
        return _config_dir_global
    return CONFIG_DIR.get()


class ConfigContextManager(AbstractContextManager["ConfigContextManager"]):
    """Context manager for setting config directory during manifest loading.

    This context manager temporarily sets the CONFIG_DIR context variable,
    enabling config-relative path resolution for all Pydantic models using
    ConfigPath fields.

    Example:
        >>> with ConfigContextManager("/path/to/config.yml"):
        ...     manifest = AgentsManifest.model_validate(yaml_data)
        ...     # All ConfigPath fields resolve relative to config directory
    """

    def __init__(self, config_path: JoinablePathLike | None) -> None:
        """Initialize with a config file path.

        Args:
            config_path: Path to the configuration file (or directory).
                If a file path, the parent directory is used as config dir.
                If None, no context is set (paths resolve to CWD).
        """
        self._config_dir: UPath | None = None
        self._token: Token[UPath | None] | None = None
        self._previous_dir: UPath | None = None

        if config_path is not None:
            path = UPath(config_path)
            # If path points to a file, use its parent directory
            # Otherwise use the path itself as config directory
            self._config_dir = path.parent if path.suffix else path

    def __enter__(self) -> Self:
        """Enter the context and set CONFIG_DIR."""
        if self._config_dir is not None:
            global _config_dir_global  # noqa: PLW0603
            self._previous_dir = _config_dir_global
            _config_dir_global = self._config_dir
            self._token = CONFIG_DIR.set(self._config_dir)
            import logging

            logging.getLogger("wolfharness.config").debug(
                "ConfigContextManager.__enter__: set _config_dir_global=%s, previous=%s",
                _config_dir_global,
                self._previous_dir,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context and reset CONFIG_DIR."""
        if self._token is not None:
            CONFIG_DIR.reset(self._token)
        # Restore previous global config dir (handles nested contexts)
        global _config_dir_global  # noqa: PLW0603
        old_value = _config_dir_global
        _config_dir_global = self._previous_dir
        import logging

        logging.getLogger("wolfharness.config").debug(
            "ConfigContextManager.__exit__: restored _config_dir_global=%s (was %s)",
            _config_dir_global,
            old_value,
        )
