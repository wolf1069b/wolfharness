"""Agent and command models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from wolfharness_server.opencode_server.models.base import OpenCodeBaseModel
from wolfharness_server.opencode_server.models.common import ModelRef  # noqa: TC001


PermissionBehavior = Literal["ask", "allow", "deny"]
AgentMode = Literal["subagent", "primary", "all"]


class AgentPermission(OpenCodeBaseModel):
    """Agent permission settings."""

    edit: PermissionBehavior = "ask"
    bash: dict[str, PermissionBehavior] = Field(default_factory=dict)
    skill: dict[str, PermissionBehavior] = Field(default_factory=dict)
    webfetch: PermissionBehavior | None = None
    doom_loop: PermissionBehavior | None = None
    external_directory: PermissionBehavior | None = None


class Agent(OpenCodeBaseModel):
    """Agent information matching SDK type."""

    name: str
    display_name: str | None = None
    description: str | None = None
    mode: AgentMode = "primary"
    native: bool | None = None
    hidden: bool | None = None
    default: bool | None = None
    top_p: float | None = None
    temperature: float | None = None
    color: str | None = None
    permission: AgentPermission = Field(default_factory=AgentPermission)
    model: ModelRef | None = None
    prompt: str | None = None
    tools: dict[str, bool] = Field(default_factory=dict)
    options: dict[str, str] = Field(default_factory=dict)


class Command(OpenCodeBaseModel):
    """Slash command matching OpenCode SDK Command.Info type."""

    name: str
    description: str | None = None
    agent: str | None = None
    """Target agent name for this command."""
    model: str | None = None
    """Model identifier override for this command."""
    source: Literal["command", "mcp", "skill"] | None = "command"
    """Source of the command: built-in, MCP prompt, or skill."""
    template: str = ""
    """Template content for skill commands (SKILL.md body)."""
    subtask: bool = False
    """Whether this command runs as a subtask."""
    hints: list[str] = Field(default_factory=list)
    """Input hints extracted from template (e.g. $1, $2, $ARGUMENTS)."""


class SkillInfo(OpenCodeBaseModel):
    """Skill information."""

    name: str
    """Skill name."""

    description: str
    """Skill description."""

    location: str
    """File path where the skill is defined."""

    content: str
    """Skill content (e.g. SKILL.md body)."""


class ProviderAuthMethod(OpenCodeBaseModel):
    """Authentication method for a provider."""

    type: Literal["oauth", "api"]
    """Auth type."""

    label: str
    """Human-readable label for the auth method."""


class ProviderAuthAuthorization(OpenCodeBaseModel):
    """Response from starting a provider OAuth flow."""

    url: str
    """URL to open in browser for authorization."""

    method: Literal["auto", "code"]
    """Authorization method."""

    instructions: str
    """Instructions to display to the user."""


class WorktreeInfo(OpenCodeBaseModel):
    """Git worktree information."""

    name: str
    """Worktree name."""

    branch: str
    """Git branch name."""

    directory: str
    """Full path to the worktree directory."""


class WorktreeCreateRequest(OpenCodeBaseModel):
    """Request to create a new git worktree."""

    name: str | None = None
    """Optional worktree name. Auto-generated if not provided."""

    start_command: str | None = None
    """Optional startup script to run after creation."""


class WorktreeRemoveRequest(OpenCodeBaseModel):
    """Request to remove a git worktree."""

    directory: str
    """Worktree directory path to remove."""


class WorktreeResetRequest(OpenCodeBaseModel):
    """Request to reset a git worktree."""

    directory: str
    """Worktree directory path to reset."""


class WorkspaceInfo(OpenCodeBaseModel):
    """Workspace information matching OpenCode's experimental workspace API."""

    id: str
    """Workspace identifier."""

    type: str
    """Workspace backend type, such as ``worktree``."""

    branch: str | None = None
    """Workspace branch name."""

    name: str | None = None
    """Human-readable workspace name."""

    directory: str | None = None
    """Workspace directory path."""

    extra: object | None = None
    """Backend-specific metadata."""

    project_id: str
    """Project identifier owning this workspace."""


class WorkspaceCreateRequest(OpenCodeBaseModel):
    """Request to create a workspace through OpenCode's experimental API."""

    id: str | None = None
    """Optional workspace identifier."""

    type: str = "worktree"
    """Workspace backend type. AgentPool currently supports ``worktree``."""

    branch: str | None = None
    """Optional branch name requested by the client."""

    extra: object | None = None
    """Backend-specific metadata."""


class WorkspaceEventConnectionStatus(OpenCodeBaseModel):
    """Connection status for a workspace.

    Matches OpenCode SDK's ``WorkspaceEventConnectionStatus`` type used by
    ``/experimental/workspace/status``.
    """

    workspace_id: str
    """Workspace identifier (serialized as ``workspaceID``)."""

    status: Literal["connected", "connecting", "disconnected", "error"]
    """Connection status of the workspace event stream."""


class OpenCodeCapabilities(OpenCodeBaseModel):
    """Experimental server capabilities for OpenCode TUI.

    Matches the body OpenCode expects from ``/experimental/capabilities``.
    """

    background_subagents: bool
    """Whether the server supports background subagents."""


class AuthInfo(OpenCodeBaseModel):
    """Authentication credential info."""

    type: str = "api_key"
    """Auth type (e.g., 'api_key', 'oauth')."""

    token: str | None = None
    """API key or access token."""

    refresh: str | None = None
    """Refresh token (for OAuth)."""

    expires: int | None = None
    """Token expiry timestamp."""
