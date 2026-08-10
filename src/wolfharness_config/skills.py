"""Skills configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import warnings

from pydantic import ConfigDict, Field
from schemez import Schema
from upathtools import UPath

from wolfharness_config.paths import ConfigPath


if TYPE_CHECKING:
    from wolfharness_config.mcp_server import MCPServerConfig


DEFAULT_SKILLS_PATHS = [
    UPath("~/.claude/skills/"),
    UPath(".claude/skills/"),
]


class SkillsInstructionConfig(Schema):
    """Configuration for dynamic skills injection via AbstractCapability.

    Controls how skills are dynamically injected into agent prompts as
    instructions. This enables agents to discover and use skills without
    explicit tool calls, making skill usage more natural and context-aware.

    The static ``<available-skills>`` catalog (skill name + description) is
    ALWAYS emitted; the ``inject`` field governs whether full
    ``<skill_content>`` instructions are also added:

    Modes:
    - "description" (default): catalog only — NO full ``<skill_content>``.
      Full instructions are loaded on demand via ``load_skill``. This is the
      token-saving default.
    - "matcher": catalog + full instructions for skills selected by the
      runtime-injected ``matcher_fn`` (non-serialized). Falls back to
      ``description`` with a warning if no matcher is set.
    - "all": catalog + full instructions for every skill (opt-in heaviest
      mode; preserves legacy inject-all behavior).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:mortar-board-16",
            "x-doc-title": "Skills Instruction Configuration",
        }
    )

    inject: Literal["description", "matcher", "all"] = Field(
        default="description",
        title="Skill content injection mode",
        examples=["description", "matcher", "all"],
    )
    """Skill content injection mode.

    Controls whether full ``<skill_content>`` instructions are injected into
    the system prompt in addition to the static ``<available-skills>``
    catalog:
    - ``description`` (default): catalog only; full instructions on demand.
    - ``matcher``: catalog + full instructions for matcher-selected skills.
    - ``all``: catalog + full instructions for every skill (heaviest).
    """

    max_skills: int = Field(
        default=20,
        ge=1,
        le=100,
        title="Maximum skills",
        examples=[10, 20, 50],
    )
    """Maximum number of skills to inject.

    Limits the number of skills included in prompts to prevent
    excessive token usage. Skills are ranked by relevance when
    this limit is exceeded.
    """


class SkillsConfig(Schema):
    """Configuration for custom skill discovery paths.

    Skills are discovered from configured directories, allowing
    users to add custom skills from local paths. The discovery
    follows "first path wins" semantics - earlier paths in the list
    take precedence over later ones.

    Default paths (when include_default=True):
    - ~/.claude/skills/ (user home directory)
    - .claude/skills/ (relative to current directory)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:mortar-board-16",
            "x-doc-title": "Skills Configuration",
        }
    )

    paths: list[ConfigPath] = Field(
        default_factory=list,
        title="Custom skill paths",
        examples=[["/path/to/skills", "./my-skills", "s3://bucket/skills"]],
    )
    """List of custom paths to search for skills.

    Paths can be:
    - Absolute: /home/user/skills
    - Relative: ./my-skills (resolved against config file location or CWD)
    - Remote: s3://bucket/skills, github://org/repo/skills

    Earlier paths take precedence over later ones ("first path wins").

    Paths are automatically resolved relative to the config file location
    via ConfigPath validation.
    """

    include_default: bool = Field(
        default=True,
        title="Include default paths",
        examples=[True, False],
    )
    """Whether to include default skill paths in discovery.

    Default paths are appended after custom paths:
    - ~/.claude/skills/
    - .claude/skills/

    Set to False to disable default paths entirely.
    """

    instruction: SkillsInstructionConfig = Field(default_factory=SkillsInstructionConfig)
    """Configuration for dynamic skills injection via AbstractCapability."""

    def get_effective_paths(self, config_file_path: UPath | None = None) -> list[UPath]:
        """Get the effective list of paths for skill discovery.

        DEPRECATED: ConfigPath now handles path resolution automatically.
        This method is kept for backward compatibility but the paths
        field now contains pre-resolved UPath objects.

        Args:
            config_file_path: Path to the YAML configuration file.
                This parameter is now ignored as paths are resolved
                automatically during validation via ConfigPath.

        Returns:
            List of UPath objects for skill discovery, ordered by priority
            (custom paths first, then default paths if enabled).
        """
        warnings.warn(
            "get_effective_paths() is deprecated; paths are now resolved automatically "
            "via ConfigPath",
            DeprecationWarning,
            stacklevel=2,
        )

        # Paths are already resolved by ConfigPath validation
        result: list[UPath] = list(self.paths)

        # Append default paths if enabled
        if self.include_default:
            result.extend(DEFAULT_SKILLS_PATHS)

        return result


class SkillMcpServerConfig(Schema):
    """Configuration for an MCP server used by a skill tool.

    Specifies how to connect to an MCP (Model Context Protocol) server
    for providing skill-level tools. Either a local command-based server
    or a remote URL-based server can be configured.

    Attributes:
        command: Executable command to start the MCP server (e.g., "npx").
            Use None when connecting via url.
        args: Command-line arguments passed to the executable.
        url: Remote URL for connecting to an existing MCP server.
            Use None when starting a local server via command.
        headers: HTTP headers to include when connecting via url.
        env: Environment variables to set when launching the server process.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "x-icon": "octicon:tools-16",
            "x-doc-title": "Skill MCP Server",
        },
    )

    command: str | None = Field(
        default=None,
        title="MCP server command",
        examples=["npx", "uvx", "python"],
    )
    """Executable command to start the MCP server process.

    Set to None when connecting to a remote MCP server via url instead.
    """

    args: list[str] = Field(
        default_factory=list,
        title="Command arguments",
        examples=[["-y", "@playwright/mcp"]],
    )
    """Command-line arguments passed to the executable."""

    url: str | None = Field(
        default=None,
        title="Server URL",
        examples=["http://localhost:8080/mcp"],
    )
    """Remote URL for connecting to an existing MCP server.

    Set to None when starting a local server via command instead.
    """

    headers: dict[str, str] = Field(
        default_factory=dict,
        title="HTTP headers",
        examples=[{"Authorization": "Bearer token123"}],
    )
    """HTTP headers to include when connecting to a URL-based server."""

    env: dict[str, str] = Field(
        default_factory=dict,
        title="Environment variables",
        examples=[{"NODE_ENV": "production"}],
    )
    """Environment variables to set when launching the server process."""

    def to_mcp_server_config(self, name: str) -> MCPServerConfig:
        """Convert to a standard :class:`MCPServerConfig` subclass.

        Produces a :class:`StdioMCPServerConfig` when ``command`` is set,
        or a :class:`StreamableHTTPMCPServerConfig` when ``url`` is set.

        Args:
            name: Server name to assign in the resulting config.

        Returns:
            A ``MCPServerConfig`` instance suitable for the MCP manager.

        Raises:
            ValueError: If neither ``command`` nor ``url`` is specified.
        """
        from wolfharness_config.mcp_server import (
            StdioMCPServerConfig,
            StreamableHTTPMCPServerConfig,
        )

        if self.command:
            return StdioMCPServerConfig(
                name=name,
                command=self.command,
                args=list(self.args),
                env=dict(self.env) if self.env else None,
            )
        if self.url:
            from pydantic import HttpUrl

            return StreamableHTTPMCPServerConfig(
                name=name,
                url=HttpUrl(self.url),
                headers=dict(self.headers) if self.headers else None,
                env=dict(self.env) if self.env else None,
            )
        raise ValueError(
            f"SkillMcpServerConfig for {name!r} must specify either 'command' or 'url'"
        )


class SkillToolConfig(Schema):
    """Configuration for a skill tool implemented in Python.

    Defines a callable Python function that a skill can use as a tool.
    The function is referenced by its Python import path and invoked
    with arguments from the skill definition.

    Attributes:
        type: The tool implementation type. Currently only "python" is
            supported, which loads a callable via its Python import path.
        import_path: Dotted Python path to the callable, in the format
            "module:function" or "package.module:function".
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "x-icon": "octicon:code-16",
            "x-doc-title": "Skill Tool Configuration",
        },
    )

    type: Literal["python"] = Field(
        title="Tool type",
        examples=["python"],
    )
    """The tool implementation type.

    Currently only "python" is supported. Future types may include
    "docker", "subprocess", or other execution backends.
    """

    import_path: str = Field(
        title="Import path",
        examples=["mymodule:my_function", "package.module:ToolClass"],
    )
    """Python import path to the tool callable.

    Uses the format "module:function" (e.g., "os:getcwd").
    The callable is imported lazily when the tool is first invoked.
    """
