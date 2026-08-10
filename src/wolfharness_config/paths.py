"""Config path resolution utilities.

Provides the ConfigPath type and resolve_config_path function for
config-relative path resolution with environment variable overrides
and backward compatibility.
"""

from __future__ import annotations

import os
from typing import Annotated

from pydantic import BeforeValidator
from upathtools import UPath


# Environment variable names
CONFIG_DIR_ENV_VAR = "AGENTPOOL_CONFIG_DIR"
LEGACY_PATHS_ENV_VAR = "AGENTPOOL_LEGACY_PATHS"


def resolve_config_path(path: str | UPath) -> UPath:
    """Resolve a configuration path using context-aware resolution.

    Resolution priority:
    1. Legacy mode (AGENTPOOL_LEGACY_PATHS=1) -> Return as-is
    2. Absolute path -> Return as-is
    3. Environment variable (AGENTPOOL_CONFIG_DIR)
    4. Module-level global variable (_config_dir_global from ConfigContextManager)
    5. ContextVar (CONFIG_DIR within with-block)
    6. Return relative path (resolves later against CWD)

    Args:
        path: The path to resolve (can be absolute or relative)

    Returns:
        UPath: Resolved absolute path, or original relative path if no context available
    """
    upath = UPath(path)

    # Priority 1: Legacy mode bypasses all resolution
    if os.environ.get(LEGACY_PATHS_ENV_VAR):
        return upath

    # Priority 2: Absolute paths are returned as-is
    if upath.is_absolute():
        return upath

    # Priority 3: Environment variable override
    config_dir_env = os.environ.get(CONFIG_DIR_ENV_VAR)
    if config_dir_env:
        return UPath(config_dir_env) / upath

    # Priority 4 & 5: Use get_config_dir() which checks both global and ContextVar
    from wolfharness_config.context import get_config_dir

    config_dir_ctx = get_config_dir()
    if config_dir_ctx is not None:
        return config_dir_ctx / upath

    # Fallback: Return relative path (caller can resolve against CWD if needed)
    return upath


# Pydantic type alias for config-relative paths.
# Use this as a field type to enable automatic path resolution.
ConfigPath = Annotated[UPath, BeforeValidator(resolve_config_path)]
"""Type alias for config-relative paths with automatic resolution.

This type can be used in Pydantic models to automatically resolve
paths relative to the config file location:

    class MyConfig(Schema):
        data_path: ConfigPath  # Resolves relative to config dir

Example:
    >>> with ConfigContextManager("/home/user/project/config.yml"):
    ...     config = MyConfig(data_path="./data")
    ...     str(config.data_path)  # "/home/user/project/data"
"""
