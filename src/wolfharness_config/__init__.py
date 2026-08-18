"""Core data models for AgentPool."""

from __future__ import annotations


from typing import Annotated
from pydantic import Field

from wolfharness_config.tools import ImportToolConfig, BaseToolConfig
from wolfharness_config.wolfharness_tools import AgentpoolToolConfig
from wolfharness_config.builtin_tools import BuiltinToolConfig

from wolfharness_config.attachment import AttachmentImageConfig
from wolfharness_config.capabilities import CapabilityConfig
from wolfharness_config.forward_targets import ForwardingTarget
from wolfharness_config.session import SessionQuery
from wolfharness_config.session_pool import ACPConfig, OpenCodeConfig, SessionPoolConfig
from wolfharness_config.teams import TeamConfig, TeamMemberConfig
from wolfharness_config.durable import CheckpointConfig, DeferredToolConfig
from wolfharness_config.mcp_server import (
    BaseMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
    MCPServerConfig,
    SSEMCPServerConfig,
)
from wolfharness_config.event_handlers import (
    BaseEventHandlerConfig,
    StdoutEventHandlerConfig,
    CallbackEventHandlerConfig,
    EventHandlerConfig,
    resolve_handler_configs,
)
from wolfharness_config.hooks import (
    BaseHookConfig,
    CallableHookConfig,
    CommandHookConfig,
    HookConfig,
    HooksConfig,
    PromptHookConfig,
)
from wolfharness_config.graph_config import (
    GraphConfig,
    GraphEdgeConfig,
    GraphJoinConfig,
    GraphStepConfig,
)
from wolfharness_config.graph_translation import (
    build_steps_from_agents,
    translate_config_to_graph,
    translate_connections_to_edges,
    translate_team_to_graph,
    translate_teams_to_graphs,
)
from wolfharness_config.toolsets import ToolsetConfig
from wolfharness_config.skills import SkillsConfig, DEFAULT_SKILLS_PATHS
from wolfharness_config.skill_commands import SkillSlashConfig, SkillCommandConfig
from wolfharness_config.resolution import (
    ConfigLayer,
    ConfigSource,
    ResolvedConfig,
    find_project_config,
    get_global_config_dir,
    get_global_config_path,
    resolve_config,
    resolve_config_for_server,
)


ToolConfig = Annotated[
    ImportToolConfig | AgentpoolToolConfig,
    Field(discriminator="type"),
]

NativeAgentToolConfig = Annotated[
    ToolConfig | BuiltinToolConfig,
    Field(discriminator="type"),
]

# Unified type for all tool configurations (single tools + toolsets)
AnyToolConfig = Annotated[
    NativeAgentToolConfig | ToolsetConfig,
    Field(discriminator="type"),
]
__all__ = [
    "DEFAULT_SKILLS_PATHS",
    "ACPConfig",
    "AnyToolConfig",
    "AttachmentImageConfig",
    "BaseEventHandlerConfig",
    "BaseHookConfig",
    "BaseMCPServerConfig",
    "BaseToolConfig",
    "CallableHookConfig",
    "CallbackEventHandlerConfig",
    "CapabilityConfig",
    "CheckpointConfig",
    "CommandHookConfig",
    "ConfigLayer",
    "ConfigSource",
    "DeferredToolConfig",
    "EventHandlerConfig",
    "ForwardingTarget",
    "GraphConfig",
    "GraphEdgeConfig",
    "GraphJoinConfig",
    "GraphStepConfig",
    "HookConfig",
    "HooksConfig",
    "MCPServerConfig",
    "NativeAgentToolConfig",
    "OpenCodeConfig",
    "PromptHookConfig",
    "ResolvedConfig",
    "SSEMCPServerConfig",
    "SessionPoolConfig",
    "SessionQuery",
    "SkillCommandConfig",
    "SkillSlashConfig",
    "SkillsConfig",
    "StdioMCPServerConfig",
    "StdoutEventHandlerConfig",
    "StreamableHTTPMCPServerConfig",
    "TeamConfig",
    "TeamMemberConfig",
    "ToolConfig",
    "ToolsetConfig",
    "build_steps_from_agents",
    "find_project_config",
    "get_global_config_dir",
    "get_global_config_path",
    "resolve_config",
    "resolve_config_for_server",
    "resolve_handler_configs",
    "translate_config_to_graph",
    "translate_connections_to_edges",
    "translate_team_to_graph",
    "translate_teams_to_graphs",
]
