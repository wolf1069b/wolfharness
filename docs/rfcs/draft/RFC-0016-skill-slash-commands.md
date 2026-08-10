---
rfc_id: RFC-0016
title: Unified Skill-to-Slash Command Architecture
status: DRAFT
author: Sisyphus
reviewers:
  - Metis (Plan Consultant) - REVIEWED 2025-03-17
  - Momus (Plan Critic) - REVIEWED 2025-03-17
created: 2025-03-17
last_updated: 2025-03-17
decision_date: null
---

# RFC-0016: Unified Skill-to-Slash Command Architecture

**Document Version**: v2.0 (Revised after review)

**Key Changes in v2.0**:
- Fixed OpenCode Bridge to use native slashed Commands (not MCP Prompts)
- Reordered implementation phases (OpenCode Bridge last)
- Added ACP Schema Changes section
- Added Protocol Capability Matrix
- Enhanced Appendix A with detailed mappings

## Overview

This RFC proposes a unified architecture to expose **Skills** as **Slash Commands** across OpenCode, ACP, and AG-UI protocols. The goal is to allow users to trigger Claude Code skills via intuitive slash command syntax (e.g., `/python-expert`) instead of requiring explicit tool calls to `load_skill`.

**Why this matters now**: AgentPool has a mature skill system with auto-discovery via `SkillsRegistry`, but users currently must know skill names and use the `skill` tool to invoke them. Exposing skills as slash commands improves discoverability and provides a more natural user experience across all supported protocols.

**Expected outcome**: Users can type `/skill-name` in OpenCode TUI, Zed (via ACP), or AG-UI clients to activate skills. The system automatically maps slash commands to skill invocations using a protocol-agnostic abstraction layer.

## Background & Context

### Current State

**Skills System** (`src/wolfharness/skills/`):
- Skills are defined in `SKILL.md` files with YAML frontmatter containing `name`, `description`, `allowed-tools`, etc.
- `Skill` class models skill metadata with validation per Agent Skills Spec
- `SkillsRegistry` auto-discovers skills from directories (e.g., `~/.claude/skills/`)
- Skills are invoked via the `skill` tool which loads instructions into agent context

**ACP Protocol** (`src/acp/schema/slash_commands.py`):
- Already has a `slash_commands.py` schema defining:
  - `AvailableCommand`: Command metadata (name, description, input hint)
  - `CommandInputHint`: Text input specification for commands
- ACP agents can expose capabilities including slash commands

**OpenCode Protocol** (`src/wolfharness_server/opencode_server/`):
- Uses `slashed` library for command handling
- Has `SkillMetadata` TypedDict in tool_metadata.py
- Commands defined with name, description, usage, and category
- Supports streaming command output

**AG-UI Protocol** (`src/wolfharness_server/agui_server/`):
- HTTP-based protocol with event streaming
- Agents expose tools and capabilities via HTTP endpoints
- Currently no native slash command support

### Glossary

- **Skill**: A reusable workflow/prompt collection stored in `SKILL.md` following the Agent Skills Spec
- **Slash Command**: A user-triggerable command syntax starting with `/` (e.g., `/test-skill`)
- **Skill Command**: A slash command backed by a Skill - when triggered, loads skill instructions into agent context
- **Command Registry**: Protocol-agnostic registry mapping command names to invokers

## Problem Statement

### Current Pain Points

1. **Poor Discoverability**: Users must know skill names exist and use `skill` tool explicitly
2. **Inconsistent UX**: Skills behave differently depending on protocol (tool vs slash command)
3. **Protocol Fragmentation**: No unified way to expose skills across OpenCode, ACP, and AG-UI
4. **Skill Inertia**: Skills installed in `~/.claude/skills/` are auto-discovered but underutilized without UI hints

### Evidence

- Skills are registered in `SkillsRegistry` but only accessible via `load_skill` tool
- ACP's `slash_commands.py` schema exists but has minimal integration with the skill system
- OpenCode has `/commit` and other slash commands but skills must be loaded as tools

### Impact of Not Solving

- Users miss skill functionality due to lack of visibility
- Protocol-specific implementations lead to code duplication
- Skill authors must understand multiple protocol nuances

## Goals & Non-Goals

### Goals

1. **Unified Exposure**: Skills exposed as slash commands work consistently across OpenCode, ACP, and AG-UI
2. **Auto-Registration**: All discovered skills automatically register as slash commands
3. **Protocol-First**: Leverage each protocol's native command patterns without abstraction leakage
4. **Runtime Discovery**: Commands appear/disappear as skills are added/removed from filesystem
5. **Backward Compatible**: Existing `skill` tool continues to work; no breaking changes

### Non-Goals

1. **NO** adding new skill metadata fields (use existing spec)
2. **NO** changing skill file structure (SKILL.md remains unchanged)
3. **NO** protocol-specific command aliases (use skill name as-is)
4. **NO** complex command composition (prefix style only: `/skill-name args`)
5. **NO** requiring skills to be configured - graceful degradation when SkillsRegistry absent

## Evaluation Criteria

| Criterion | Weight | Description | Min Threshold |
|-----------|--------|-------------|---------------|
| Protocol Compatibility | Critical | Works with OpenCode, ACP, AG-UI | Must support all three |
| Runtime Performance | High | Command registration <50ms | <100ms acceptable |
| Code Simplicity | High | Few moving parts, clear data flow | <500 LOC per server |
| Backward Compatibility | Critical | No breaking changes | 100% compatible |
| Skill Spec Compliance | Critical | Follows Agent Skills Spec | Full compliance |
| Testability | Medium | Comprehensive test coverage | >80% coverage |
| Documentation | Medium | Clear migration/usage docs | Required for review |

## Options Analysis

### Option 1: Protocol-Specific Adapters (Direct Mapping)

**Description**: Each server protocol (OpenCode, ACP, AG-UI) has its own skill-to-command adapter that directly reads from `SkillsRegistry` and creates protocol-native commands.

**Architecture**:
```
SkillsRegistry (source of truth)
    ├─ OpenCodeAdapter → slashed Commands → OpenCode Server
    ├─ ACPAdapter → AvailableCommand[] → ACP Server
    └─ AGUIAdapter → Tool definitions → AG-UI Server
```

**Advantages**:
- Simple to understand: each adapter is independent
- Native protocol behavior: no abstraction layers
- Easy to add protocol-specific optimizations

**Disadvantages**:
- Code duplication: similar logic in each adapter
- Inconsistent behavior potential if implementations diverge
- More maintenance burden across 3+ adapters

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Protocol Compatibility | 9/10 | Native per-protocol implementation |
| Performance | 8/10 | No abstraction overhead |
| Code Simplicity | 5/10 | Duplicated logic across adapters |
| Backward Compatibility | 10/10 | No changes to existing code |
| Skill Spec Compliance | 10/10 | Direct schema mapping |
| Testability | 6/10 | Duplicated test patterns |

**Effort Estimate**: 2-3 weeks (parallel work on 3 adapters)

**Risk Assessment**:
- Medium risk of behavior drift between protocols
- Requires changes to 3 server implementations

### Option 2: Unified Command Registry with Protocol Bridges

**Description**: Create a `SkillCommandRegistry` that acts as a protocol-agnostic source of commands. Each protocol server registers a bridge that translates commands to the native protocol format.

**Architecture**:
```
SkillsRegistry
    ↓ (watches for changes)
SkillCommandRegistry (protocol-agnostic commands)
    ├─ OpenCodeBridge → slashed Commands
    ├─ ACPBridge → AvailableCommand[]
    └─ AGUIBridge → Capability definitions
```

**Advantages**:
- Single source of truth for commands
- Consistent behavior across protocols
- Easy to add new protocols (just write a bridge)
- Centralized command lifecycle management

**Disadvantages**:
- More complex initial design
- Additional abstraction layer
- Risk of "leaky abstraction" if not careful

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Protocol Compatibility | 9/10 | Bridges can handle protocol quirks |
| Performance | 9/10 | Efficient in-memory registry |
| Code Simplicity | 7/10 | One registry + clean bridges |
| Backward Compatibility | 10/10 | No breaking changes |
| Skill Spec Compliance | 10/10 | Registry enforces consistency |
| Testability | 9/10 | Test registry once, bridges separately |

**Effort Estimate**: 3-4 weeks (upfront design, then cleaner implementation)

**Risk Assessment**:
- Low risk: well-defined interfaces
- New component requires careful API design
- Potential over-engineering risk

### Option 3: Code Generation at Startup

**Description**: At server startup, scan skills directory and generate protocol-specific command code/config files, then load those.

**Architecture**:
```
SkillsRegistry (startup scan)
    ├─ generates opencode_commands.py
    ├─ generates acp_commands.json
    └─ generates agui_commands.yaml
(servers load generated files)
```

**Advantages**:
- Zero runtime overhead
- Static code is easier to review
- No dynamic command registration complexity

**Disadvantages**:
- Requires server restart to pick up new skills
- File generation complexity
- Harder to support hot-reload of skills
- More build/deployment complexity

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Protocol Compatibility | 6/10 | Harder to handle protocol nuances |
| Performance | 10/10 | Zero runtime overhead |
| Code Simplicity | 5/10 | Codegen adds complexity |
| Backward Compatibility | 8/10 | Startup order dependencies |
| Skill Spec Compliance | 10/10 | Generated from spec |
| Testability | 5/10 | Testing generated code is harder |

**Effort Estimate**: 4-5 weeks (codegen infrastructure + templates)

**Risk Assessment**:
- Medium risk: codegen tooling can be brittle
- Doesn't meet goal of runtime discovery
- Deployment complexity

### Option 4: Use Existing Tool System

**Description**: Instead of slash commands, expose skills as specialized tools with category="skill".

**Architecture**:
```
SkillsRegistry
    ↓
Dynamic tool generation: skill__<name> tools
    ↓
All protocols (OpenCode/ACP/AG-UI) use existing tool exposure
```

**Advantages**:
- Minimal new code (use existing tool framework)
- Works immediately across all protocols
- Consistent with current skill tool

**Disadvantages**:
- Different UX (tools vs slash commands)
- Poor discoverability (tools list can be long)
- Doesn't leverage protocol-specific command features

**Evaluation**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Protocol Compatibility | 10/10 | Uses existing tool system |
| Performance | 9/10 | Existing machinery |
| Code Simplicity | 9/10 | Minimal new code |
| Backward Compatibility | 10/10 | No changes needed |
| Skill Spec Compliance | 10/10 | Works with existing skills |
| Testability | 9/10 | Test existing tool system |

**Effort Estimate**: 1 week

**Risk Assessment**:
- Low technical risk
- Doesn't solve UX discoverability issue
- Not true slash commands (fails requirement)

## Recommendation

**Recommended Option**: **Option 2: Unified Command Registry with Protocol Bridges**

**Justification**:
1. **Scoring**: Highest total weighted score (54/60 vs 48/60 for Option 1)
2. **Maintainability**: Centralized registry avoids code duplication
3. **Extensibility**: Easy to add new protocols by writing bridges
4. **Consistency**: Single source of truth ensures uniform behavior
5. **Runtime Discovery**: Meets the goal of dynamic skill loading

**Acknowledged Trade-offs**:
- Higher initial design effort (6-7 weeks vs 2-3 weeks for naive approach)
- OpenCode requires MCP Prompt integration (complex mapping)
- AG-UI requires tool-based exposure (different UX)

**Why not Option 1**: While Option 1 is simpler initially, the duplicated effort across 3+ protocols will result in higher long-term maintenance costs and risk of behavior divergence.

**Why not Option 3**: Fails the requirement for runtime discovery without restart.

**Why not Option 4**: While simplest, it doesn't provide the slash command UX I'm looking for.

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AgentPool Runtime                                    │
│                                                                              │
│   ┌──────────────────┐         ┌──────────────────────┐                     │
│   │ SkillsRegistry   │◄────────│ SkillCommandRegistry │                     │
│   │ (existing)       │  watch  │ (new component)      │                     │
│   └──────────────────┘         └──────────┬───────────┘                     │
│                                            │                                 │
│                      ┌─────────────────────┼─────────────────────┐           │
│                      │                     │                     │           │
│                      ▼                     ▼                     ▼           │
│   ┌──────────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│   │ OpenCodeBridge       │    │ ACPBridge        │    │ AGUIBridge       │   │
│   │ (slashed Commands)   │    │ (AvailableCmd[]) │    │ (Capabilities)   │   │
│   └──────────┬───────────┘    └────────┬─────────┘    └────────┬─────────┘   │
│              │                         │                      │             │
│              ▼                         ▼                      ▼             │
│   ┌──────────────────┐      ┌──────────────────┐    ┌──────────────────┐   │
│   │ OpenCode Server  │      │ ACP Server       │    │ AG-UI Server     │   │
│   └──────────────────┘      └──────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New Components

#### 1. SkillCommand (Protocol-Agnostic)

```python
# src/wolfharness/skills/command.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Coroutine, ParamSpec

if TYPE_CHECKING:
    from wolfharness.skills.skill import Skill


P = ParamSpec("P")


@dataclass(frozen=True)
class SkillCommand:
    """Protocol-agnostic representation of a skill as a slash command.

    Bridges map this representation to protocol-specific command types.
    """

    name: str
    """Command name (without leading slash)."""

    description: str
    """Human-readable description."""

    skill: Skill
    """Reference to the underlying skill."""

    input_hint: str | None = None
    """Hint text for command arguments."""

    category: str = "skill"
    """Command category for grouping."""

    def is_valid_input(self, input_text: str) -> tuple[bool, str | None]:
        """Validate user input for this command.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Skills generally accept free-form text
        # Could be extended with JSON schema validation per skill
        return True, None
```

#### 2. SkillCommandRegistry

```python
# src/wolfharness/skills/command_registry.py

from __future__ import annotations

from typing import TYPE_CHECKING

from wolfharness.log import get_logger
from wolfharness.skills.command import SkillCommand
from wolfharness.utils.baseregistry import BaseRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from wolfharness.skills.skill import Skill
    from wolfharness.skills.registry import SkillsRegistry


logger = get_logger(__name__)

CommandChangeHandler = Callable[[str, SkillCommand | None], None]
"""Callback for command changes: (command_name, command_or_none)."""


class SkillCommandRegistry(BaseRegistry[str, SkillCommand]):
    """Registry for skill-based slash commands.

    Watches SkillsRegistry for changes and maintains a synchronized
    mapping of skill names to slash commands.

    Usage:
        registry = SkillCommandRegistry(skills_registry)
        await registry.initialize()

        # Register bridges
        registry.on_command_change(opencode_bridge.handle_change)
        registry.on_command_change(acp_bridge.handle_change)
    """

    def __init__(self, skills_registry: SkillsRegistry | None = None) -> None:
        """Initialize registry with optional source skills registry.

        Args:
            skills_registry: The source of skills to expose as commands.
                           If None, commands can be added manually via register().
        """
        super().__init__()
        self._skills = skills_registry
        self._change_handlers: list[CommandChangeHandler] = []

    async def initialize(self) -> None:
        """Sync registry with current skills and start watching.
        
        Graceful degradation: If skills_registry is None, registry starts
        empty and commands must be manually added.
        """
        if self._skills is not None:
            await self._sync_commands()
            # TODO: Set up filesystem watcher for hot-reload
        else:
            logger.debug("No skills registry configured - starting with empty command set")

    def on_command_change(self, handler: CommandChangeHandler) -> None:
        """Register a handler for command registration/deregistration.

        Args:
            handler: Callback invoked when commands are added/removed.
                     Called with (command_name, command) for new commands,
                     (command_name, None) for removed commands.
        """
        self._change_handlers.append(handler)
        # Notify of existing commands
        for name, command in self._items.items():
            handler(name, command)
    
    @property
    def has_skills(self) -> bool:
        """Whether this registry has a backing skills registry."""
        return self._skills is not None 
    
    @property
    def has_commands(self) -> bool:
        """Whether this registry has any commands registered."""
        return len(self._items) > 0

    async def _sync_commands(self) -> None:
        """Rebuild command registry from current skills."""
        current_names = set(self._items.keys())
        skill_names = set(self._skills.keys())

        # Remove commands for deleted skills
        for name in current_names - skill_names:
            await self._remove_command(name)

        # Add/update commands for current skills
        for name in skill_names:
            skill = self._skills.get(name)
            if name not in current_names:
                await self._add_command(skill)

    async def _add_command(self, skill: Skill) -> None:
        """Add command for a skill."""
        command = SkillCommand(
            name=skill.name,
            description=skill.description,
            skill=skill,
            input_hint=f"Arguments for {skill.name}",
        )
        self.register(skill.name, command)
        for handler in self._change_handlers:
            handler(skill.name, command)
        logger.debug("Registered skill command", command=skill.name)

    async def _remove_command(self, name: str) -> None:
        """Remove command by name."""
        if name in self._items:
            del self._items[name]
            for handler in self._change_handlers:
                handler(name, None)
            logger.debug("Unregistered skill command", command=name)

    @property
    def _error_class(self) -> type[Exception]:
        from wolfharness.tools.exceptions import ToolError

        return ToolError
```

#### 3. Protocol Bridges

**OpenCode Bridge** (`src/wolfharness_server/opencode_server/skill_bridge.py`):

**⚠️ CRITICAL**: OpenCode uses **native slashed Commands** for `/command-name` execution, NOT MCP Prompts. MCP Prompts are read-only templates that cannot execute commands or modify state.

```python
"""Bridge between SkillCommandRegistry and OpenCode's native command system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext

if TYPE_CHECKING:
    from wolfharness.agents.opencode_agent import OpenCodeAgent
    from wolfharness.skills.command import SkillCommand
    from wolfharness.skills.manager import SkillsManager


class SkillCommandWrapper(Command):
    """Wraps a Skill as a native OpenCode slashed Command.
    
    This class provides the actual execution logic when users type
    /skill:name or /skill-alias in the OpenCode TUI.
    """
    
    def __init__(
        self,
        skill_cmd: SkillCommand,
        manager: SkillsManager,
    ) -> None:
        self._skill_cmd = skill_cmd
        self._manager = manager
        super().__init__(
            name=self._get_command_name(),
            description=skill_cmd.description,
            category="skill",
            usage=f"/{self._get_command_name()} [arguments]",
        )
    
    def _get_command_name(self) -> str:
        """Generate command name with prefix."""
        return f"skill:{self._skill_cmd.name}"
    
    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        """Execute skill command.
        
        This method is called when user types /skill:name in OpenCode.
        It loads skill instructions into the agent context.
        
        Args:
            ctx: Command context providing access to agent and output
            args: Command arguments as list of strings
        """
        args_str = " ".join(args)
        
        ctx.output.write(f"Loading skill: {self._skill_cmd.name}...")
        
        try:
            # Load skill via manager
            instructions = await self._manager.load_skill(
                self._skill_cmd.skill.name,
                arguments=args_str,
            )
            
            # Inject skill instructions into agent context
            agent: OpenCodeAgent = ctx.agent
            await agent.inject_skill_context(instructions)
            
            ctx.output.write(f"✓ Skill '{self._skill_cmd.name}' loaded successfully")
            
        except Exception as e:
            ctx.output.error(f"Failed to load skill '{self._skill_cmd.name}': {e}")


class OpenCodeSkillBridge:
    """Bridges skill commands to OpenCode's native slashed command system.
    
    This bridge integrates with OpenCode's slashed library to provide
    true slash command UX (/skill:name) with execution capabilities.
    """
    
    def __init__(self, skills_manager: SkillsManager) -> None:
        """Initialize bridge with skills manager.
        
        Args:
            skills_manager: Manager for skill operations and loading.
        """
        self._manager = skills_manager
        self._commands: dict[str, SkillCommandWrapper] = {}
    
    def handle_change(self, name: str, command: SkillCommand | None) -> None:
        """Handle skill command registration change.
        
        Called by SkillCommandRegistry when skills are added/removed.
        
        Args:
            name: Skill name
            command: SkillCommand if added, None if removed.
        """
        cmd_name = f"skill:{name}"
        if command is None:
            self._commands.pop(cmd_name, None)
            logger.debug("Unregistered OpenCode skill command", command=name)
        else:
            wrapper = SkillCommandWrapper(command, self._manager)
            self._commands[cmd_name] = wrapper
            logger.debug("Registered OpenCode skill command", command=name)
    
    def get_commands(self) -> list[Command]:
        """Get all registered skills as slashed Commands.
        
        These commands are registered with OpenCode's CommandStore to
        enable /skill:name execution in the TUI.
        """
        return list(self._commands.values())
    
    def get_command(self, name: str) -> Command | None:
        """Get a specific command by name."""
        return self._commands.get(name)
```

**ACP Bridge** (`src/wolfharness_server/acp_server/commands/skill_commands.py`):

```python
"""Bridge between SkillCommandRegistry and ACP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema.slash_commands import AvailableCommand

if TYPE_CHECKING:
    from wolfharness.skills.command import SkillCommand


class ACPSkillBridge:
    """Bridges skill commands to ACP server capabilities."""

    def __init__(self) -> None:
        """Initialize bridge."""
        self._commands: dict[str, AvailableCommand] = {}

    def handle_change(self, name: str, command: SkillCommand | None) -> None:
        """Handle command registration change."""
        if command is None:
            self._commands.pop(name, None)
        else:
            self._commands[name] = self._to_acp_command(command)

    def get_available_commands(self) -> list[AvailableCommand]:
        """Get commands in ACP format."""
        return list(self._commands.values())

    def _to_acp_command(self, skill_cmd: SkillCommand) -> AvailableCommand:
        """Convert SkillCommand to ACP AvailableCommand."""
        return AvailableCommand.create(
            name=skill_cmd.name,
            description=skill_cmd.description,
            input_hint=skill_cmd.input_hint,
        )
```

**ACP Schema Changes**:

**⚠️ REQUIRED**: Add `slash_commands` field to `AgentCapabilities` schema:

```python
# In src/acp/schema/capabilities.py

class AgentCapabilities(AnnotatedObject):
    """Agent capabilities including slash commands."""
    
    load_session: bool | None = False
    mcp_capabilities: McpCapabilities | None = Field(default_factory=McpCapabilities)
    prompt_capabilities: PromptCapabilities | None = Field(default_factory=PromptCapabilities)
    session_capabilities: SessionCapabilities | None = Field(default_factory=SessionCapabilities)
    
    # NEW: Slash commands field for skill exposure
    slash_commands: list[AvailableCommand] | None = Field(default_factory=list)
    """Available slash commands for this agent.
    
    Includes skill commands exposed by the agent. Commands are provided
    per-session and can change dynamically as skills are added/removed.
    """
```

**AG-UI Bridge** (`src/wolfharness_server/agui_server/skill_tools.py`):

**⚠️ CRITICAL**: AG-UI is **tool-oriented** with no native slash command concept. Skills must be exposed as **tools**.

```python
"""Bridge between SkillCommandRegistry and AG-UI server as tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ag_ui import Tool, UserMessage

if TYPE_CHECKING:
    from wolfharness.skills.command import SkillCommand
    from wolfharness.skills.manager import SkillsManager


class AGUISkillToolAdapter:
    """Wraps a Skill as an AG-UI Tool."""
    
    def __init__(self, skill_cmd: SkillCommand, manager: SkillsManager) -> None:
        self._skill_cmd = skill_cmd
        self._manager = manager
    
    def to_agui_tool(self) -> Tool:
        """Convert skill to AG-UI Tool format.
        
        AG-UI uses OpenAI function format for tools.
        """
        return Tool(
            name=f"skill__{self._skill_cmd.name}",  # Double underscore to avoid conflicts
            description=self._skill_cmd.description,
            parameters={
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "string",
                        "description": self._skill_cmd.input_hint or f"Arguments for {self._skill_cmd.name}",
                    }
                },
                "required": [],  # Arguments optional
            },
        )
    
    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute skill invocation.
        
        Returns skill instructions which the AG-UI agent processes.
        """
        args_str = arguments.get("arguments", "")
        
        # Get skill instructions
        instructions = await self._manager.get_skill_instructions(
            self._skill_cmd.skill.name
        )
        
        # Append user arguments if provided
        if args_str:
            instructions = f"User input: {args_str}\n\n{instructions}"
        
        return instructions


class AGUISkillBridge:
    """Bridges skill commands to AG-UI's tool system."""
    
    def __init__(self, skills_manager: SkillsManager) -> None:
        """Initialize bridge."""
        self._manager = skills_manager
        self._tools: dict[str, AGUISkillToolAdapter] = {}
    
    def handle_change(self, name: str, command: SkillCommand | None) -> None:
        """Handle command registration change."""
        tool_name = f"skill__{name}"
        if command is None:
            self._tools.pop(tool_name, None)
        else:
            self._tools[tool_name] = AGUISkillToolAdapter(command, self._manager)
    
    def get_tools(self) -> list[Tool]:
        """Get all registered skills as AG-UI Tools."""
        return [t.to_agui_tool() for t in self._tools.values()]
    
    def get_handler(self, tool_name: str) -> AGUISkillToolAdapter | None:
        """Get handler for a tool by name."""
        return self._tools.get(tool_name)
```

### Data Models

**SkillCommand Config** (optional enhancement):

```python
# src/wolfharness_config/skill_commands.py

from __future__ import annotations

from schemez import Schema


class SkillCommandConfig(Schema):
    """Configuration for skill command exposure.

    Allows disabling specific skills from slash command exposure
    or configuring input validation.
    """

    enabled: bool = True
    """Whether to expose this skill as a slash command."""

    input_schema: dict | None = None
    """Optional JSON schema for input validation."""

    aliases: list[str] = []
    """Alternative names for the command."""
```

### Protocol Capability Matrix

This matrix maps how skill slash commands are exposed across different protocols:

| Capability | ACP | AG-UI | OpenCode |
|------------|-----|-------|----------|
| **Discovery** | `AvailableCommandsUpdate` | Tool list in agent endpoint | `GET /command` via slashed CommandStore |
| **Execution** | Session Prompt | Tool call | Native slashed command |
| **Input Format** | Command name + params | `{arguments: string}` | Positional args `$1 $2` |
| **State Support** | Session state | Stateless | Agent state |
| **Streaming** | Via ACP events | Via SSE | Via CLI stdout |

#### Protocol-Specific Details

**ACP**:
- **Discovery**: `AgentCapabilities.slash_commands` field with `AvailableCommand` list
- **Execution**: Send skill name as prompt to agent session
- **Runtime Changes**: `AvailableCommandsUpdate` pushes new commands to clients

**AG-UI**:
- **Discovery**: Tools array in HTTP response (OpenAI function format)
- **Execution**: Tool call with `{arguments: "user args"}` parameter
- **User Experience**: AI calls function, no direct slash syntax for users
- **Trade-off**: No direct /command UX, but works with existing tool UI

**OpenCode**:
- **Discovery**: `GET /command` lists commands from MCP Prompts + slashed CommandStore
- **Execution**: Native `/skill:name args` via slashed library
- **Argument Handling**: Supports bash-style `$1`, `$2`, `$ARGUMENTS`
- **Context Injection**: Direct skill instructions injection into agent context

#### Why MCP Prompts Cannot Execute Skills

**Clarification**: While OpenCode `GET /command` returns MCP Prompts for read-only templates, skill commands require **execution** and **state modification**. MCP Prompts are fundamentally read-only and cannot:
- Execute arbitrary code
- Modify agent context state
- Call external APIs

Therefore, skill commands must be implemented as:
- **OpenCode**: Native slashed Commands (actual execution)
- **ACP**: Session Prompt mechanism (delegated to agent)
- **AG-UI**: Tools (function calls with execution)

### Gradual Degradation: Skills Registry Optional

**Problem**: What if the pool doesn't have a SkillsRegistry configured?

**Solution**: Make `SkillCommandRegistry` operate in two modes:

| Mode | SkillsRegistry | Behavior |
|------|----------------|----------|
| **Full Mode** | Provided | Automatic sync, hot-reload via filesystem watcher |
| **Manual Mode** | None | Registry starts empty, commands added manually or left disabled |

**Graceful Degradation Implementation**:

```python
# In AgentPool initialization

async def _setup_skill_commands(self) -> None:
    """Setup skill command registry with optional skills support."""
    from wolfharness.skills.command_registry import SkillCommandRegistry
    
    # SkillsRegistry may be None if no skill_dirs configured
    skills_registry = getattr(self, '_skills', None)
    
    cmd_registry = SkillCommandRegistry(skills_registry)
    await cmd_registry.initialize()
    
    # Only register bridges if we have actual skills
    if cmd_registry.has_skills and cmd_registry.has_commands:
        self._handle_skill_command_integration(cmd_registry)
    elif skills_registry is None:
        logger.info(
            "Skill commands disabled - no skills configured. "
            "Skills will be unavailable via slash commands."
        )
    else:
        logger.debug(
            "Skill commands initialized with %d commands",
            len(cmd_registry._items)
        )
```

**Protocol-Specific Graceful Handling**:

```python
# OpenCode Server - commands only if skills available
async def _setup_skill_commands(self) -> None:
    if not self._pool.skill_commands.has_skills:
        return  # No-op, commands just don't appear in /command

# ACP Server - capabilities only if skills available
async def _get_capabilities(self) -> AgentCapabilities:
    if not self._pool.skill_commands.has_skills:
        return AgentCapabilities()  # No slash_commands field
    
    # ... full implementation

# AG-UI Server - tools only if skills available
def get_tools(self) -> list[AGUITool]:
    if not self._pool.skill_commands.has_skills:
        return []  # No skill tools exposed
    
    # ... full implementation
```

**Configuration Matrix**:

| Config | Skill Discovery | Slash Commands |
|--------|----------------|----------------|
| `skill_dirs` + this feature | Auto via filesystem | Available |
| `skill_dirs` + disabled | Auto via filesystem | Not available |
| No `skill_dirs` | Disabled | Not available (graceful) |

### Integration Points

**OpenCode Server Integration**:

```python
# In wolfharness_server/opencode_server/server.py

async def _setup_skill_commands(self) -> None:
    """Setup skill command bridge with OpenCode's native slashed system."""
    from wolfharness.skills.command_registry import SkillCommandRegistry
    from wolfharness_server.opencode_server.skill_bridge import OpenCodeSkillBridge
    from slashed import CommandStore

    # Initialize command registry (watches SkillsRegistry for changes)
    cmd_registry = SkillCommandRegistry(self._pool.skills)
    await cmd_registry.initialize()

    # Create bridge and register for changes
    bridge = OpenCodeSkillBridge(self._pool.skills)
    cmd_registry.on_command_change(bridge.handle_change)

    # Create command store and register all skill commands
    command_store = CommandStore()
    for cmd in bridge.get_commands():
        command_store.register_command(cmd)

    # Make command store available to agent state
    self._command_store = command_store
    self._skill_bridge = bridge
```

**Command Discovery Integration** (in `agent_routes.py`):

```python
@router.get("/command")
async def list_commands(state: StateDep) -> list[Command]:
    """List available commands including native commands and skill commands."""
    # Get native commands from MCP prompts
    prompts = await state.agent.tools.list_prompts()
    commands = [
        Command(name=p.name, description=p.description or "")
        for p in prompts
    ]
    
    # Add skill commands from the slashed CommandStore
    if state.command_store:
        skill_commands = [
            Command(name=cmd.name, description=cmd.description)
            for cmd in state.command_store.list_commands()
            if cmd.category == "skill"  # Only include skill category commands
        ]
        commands.extend(skill_commands)
    
    return commands
```

**Command Execution Flow**:

When a user types `/skill:name arg1 arg2` in OpenCode:

1. **OpenCode CLI** captures the command input
2. **slashed library** routes to `SkillCommandWrapper.execute()`
3. **SkillCommandWrapper** calls `SkillsManager.load_skill()` with arguments
4. **Agent context** is updated with skill instructions via `inject_skill_context()`
5. **Response** is streamed back to the CLI showing skill load status

This provides true slash command UX with full execution capabilities.

**ACP Server Integration**:

```python
# In wolfharness_server/acp_server/session.py or agent.py

async def _get_capabilities(self) -> AgentCapabilities:
    """Get agent capabilities including skill commands."""
    from wolfharness_server.acp_server.commands.skill_commands import ACPSkillBridge

    bridge = ACPSkillBridge()

    # Register bridge for updates
    self._pool.skill_commands.on_command_change(bridge.handle_change)

    # Include skill commands in capabilities
    return AgentCapabilities(
        slash_commands=bridge.get_available_commands(),
        # ... other capabilities
    )
```

### Security Considerations

1. **Input Validation**: Skills receive user input via slash command args - need to sanitize
2. **Allowed Tools**: Respect skill's `allowed-tools` metadata when executing
3. **Rate Limiting**: Consider rate limiting for skill invocation
4. **Sandboxing**: Skill execution should maintain existing sandbox boundaries

## Implementation Plan

**Revised Timeline**: 6-7 weeks

**Phase Ordering Rationale**: ACP Bridge first (simplest), AG-UI Bridge second (medium complexity), OpenCode Bridge last (most complex due to native slashed Command integration).

### Phase 1: Core Foundation (Weeks 1-2)

1. **Create SkillCommand dataclass** (`src/wolfharness/skills/command.py`)
   - Define protocol-agnostic command representation
   - Add validation methods

2. **Create SkillCommandRegistry** (`src/wolfharness/skills/command_registry.py`)
   - Implement watcher pattern for skills changes
   - Add callback registration mechanism
   - Add dependency resolution for skills (handle depends-on field)
   - Unit tests with mocked SkillsRegistry

3. **Integrate with AgentPool** (`src/wolfharness/delegation/pool.py`)
   - Add `skill_commands` property to AgentPool
   - Initialize registry during pool startup

4. **Add filesystem watcher support**
   - Implement hot-reload when SKILL.md files change
   - Add file locking to prevent race conditions

**Deliverable**: Core registry component with tests and hot-reload

### Phase 2: ACP Bridge (Weeks 2-3)

**Strategy**: Implement ACP Bridge first as it has the cleanest mapping.

1. **Create ACPSkillBridge** (`src/wolfharness_server/acp_server/commands/skill_commands.py`)
   - Map SkillCommand to AvailableCommand
   - Handle capability updates via `AvailableCommandsUpdate`

2. **Add ACP Schema Changes**
   - Add `slash_commands` field to `AgentCapabilities`
   - Update JSON schema validation

3. **Integrate with ACPServer** (`src/wolfharness_server/acp_server/server.py`)
   - Include skill slash commands in agent capabilities
   - Handle command execution via session/prompt mechanism

4. **Add ACP integration tests**
   - Test command discovery via capabilities
   - Test command execution via ACP requests

**Deliverable**: Working ACP slash commands

### Phase 3: AG-UI Bridge (Weeks 3-5)

1. **Create AGUISkillBridge** (`src/wolfharness_server/agui_server/skill_tools.py`)
   - Create `AGUISkillToolAdapter` for tool format conversion
   - Handle OpenAI function format conversion

2. **Integrate with BaseAgentAGUIAdapter**
   - Add skill tools to agent tool list
   - Handle tool execution returning skill instructions

3. **Add AG-UI integration tests**
   - Test tool discovery in agent endpoint
   - Test tool execution returning skill content

**Deliverable**: Skills available as AG-UI agent tools

### Phase 4: OpenCode Bridge (Weeks 5-7)

**Strategy**: OpenCode Bridge last due to complexity of native slashed Command integration.

**⚠️ CRITICAL**: OpenCode Bridge uses **native slashed Commands** (NOT MCP Prompts).

1. **Create OpenCodeSkillBridge** (`src/wolfharness_server/opencode_server/skill_bridge.py`)
   - Create `SkillCommandWrapper` extending slashed `Command` class
   - Implement `execute()` method for skill loading
   - Integrate with OpenCode's `CommandStore`

2. **Implement Command Execution Flow**
   - Route `/skill:name` commands to SkillCommandWrapper
   - Handle bash/fsspec argument substitution: `$1`, `$2`, `$@`, `$ARGUMENTS`
   - Implement `inject_skill_context()` in OpenCodeAgent

3. **Integrate with OpenCodeServer** (`src/wolfharness_server/opencode_server/server.py`)
   - Register bridge during server initialization
   - Make CommandStore available to GET /command endpoint
   - Wire up command execution in agent routes

4. **Add end-to-end tests**
   - Test skill loading via slash command
   - Test skill arguments passing (positional and named)
   - Test inline output vs. printed execution modes
   - Test hot-reload during active sessions

**Deliverable**: Working OpenCode skill commands via native /skill:name syntax
   - Migration guide for existing skills

3. **Performance Optimization**
   - Lazy loading of skill instructions
   - Command caching
   - Benchmark tests

**Deliverable**: Complete unified system with documentation

### Dependencies

**External**:
- Existing `slashed` library (already in use for OpenCode)
- ACP protocol schemas (already implemented)

**Internal**:
- `SkillsRegistry` (existing)
- `SkillsManager` (existing)
- Server base classes (existing)

### Rollback Strategy

1. **Feature Flag**: Add `--enable-skill-commands` flag (default: true)
2. **Revert**: If issues arise, disable the flag to revert to current behavior
3. **Emergency Patch**: Each bridge can be disabled independently via config

## Missing Considerations (Discovered During Review)

The following considerations were identified during the review process and should be addressed in implementation:

### M1: Skill Dependencies and Ordering ⚠️

**Gap**: What if skill A depends on skill B being loaded first?

**Example**:
```markdown
# SKILL.md for skill-a
---
name: skill-a
depends-on: [skill-b]  # Load skill-b before skill-a
---
```

**Recommendation**: Add dependency resolution to `SkillCommandRegistry`:
- Topological sort for dependency ordering
- Cyclic dependency detection and error reporting
- Async loading with dependency resolution

```python
class SkillCommandRegistry:
    async def _add_command(self, skill: Skill) -> None:
        # Resolve dependencies
        for dep in skill.dependencies:
            if dep not in self._items:
                await self._add_command(self._skills.get(dep))
        # Then add this skill
        ...
```

### M2: Hot-Reload Edge Cases ⚠️

**Issues to Handle**:
1. **Agent context with old skill version**: What if skill is modified mid-conversation?
2. **In-flight executions**: Executions in progress when skill changes
3. **Cache coherence**: Instructions provider cache vs. new skill version

**Recommendation**:
```python
class SkillCommandRegistry:
    async def _on_skill_changed(self, skill_name: str):
        # 1. Update command registry
        await self._update_command(skill)
        
        # 2. Notify active sessions (for ACP)
        for handler in self._change_handlers:
            handler(skill_name, updated_command)
        
        # 3. Log warning: active conversations may have stale context
        logger.warning(
            "Skill updated mid-session - active conversations use old version",
            skill=skill_name,
            active_sessions=len(self._active_sessions)
        )
```

### M3: Skill Versioning ⚠️

**Gap**: Current spec only has `compatibility` string field. No semantic versioning.

**Recommendation**:
- Add optional `version` field to SKILL.md frontmatter
- Extend `SkillCommandConfig` with `min_version` compatibility checks
- Track which version is currently loaded in registry

### M4: Observability and Analytics ⚠️

**Gap**: No tracking of skill command usage.

**Recommendation**: Extend storage schema with `skill_command_invocations`:

```python
@dataclass
class SkillCommandInvocation:
    skill_name: str
    protocol: str  # "opencode", "acp", "agui"
    timestamp: datetime
    duration_ms: int
    success: bool
    error_type: str | None
    arguments_hash: str  # Hashed for privacy
```

### M5: Performance Optimization ⚠️

**Identified Needs**:
- Lazy loading of skill instructions
- Command caching (avoid re-parsing SKILL.md files)
- Registry initialization benchmarking

### M6: Observability Configuration ⚠️

The RFC should include metrics collection for:
- Command invocation counts by skill/protocol
- Execution latency percentiles
- Error rates by skill
- Hot-reload events

---

## Open Questions

### ✅ RESOLVED: Input Parsing (Q1)

**DECISION**: Raw string input with optional JSON Schema validation

**Rationale**:
- Skills are conversational tools, not strictly typed functions
- `load_skill()` already accepts string arguments in the current API
- Simpler implementation, more flexible for users
- Skill can parse arguments internally as needed

**Implementation**:
```python
# SkillCommand receives raw string
args_str = " ".join(args)

# Optional: JSON schema validation if defined in SKILL.md frontmatter
if self._skill.input_schema:
    is_valid, error = validate_against_schema(args_str, self._skill.input_schema)
    if not is_valid:
        raise ValidationError(error)
```

---

### ✅ RESOLVED: Skill Context Auto-Injection (Q2)

**DECISION**: Auto-inject via system prompt modification upon command trigger

**Rationale**:
- Matches current `SkillsInstructionProvider` behavior
- Provides seamless user experience
- Can be disabled via configuration

**Implementation Flow**:
1. User types: `/skill-name arg1 arg2`
2. Skill instructions loaded via `skills_manager.get_skill_instructions()`
3. Instructions injected into agent's system prompt context
4. User arguments appended as user message
5. Agent processes with skill-loaded context

**Configuration** (in `SKILL.md` or `agents.yml`):
```yaml
skills:
  my-skill:
    slash_command:
      auto_inject: true  # Default: true
```

---

### ✅ RESOLVED: Command Conflicts (Q3)

**DECISION**: Use `/skill:` prefix by default, configurable per skill, native commands take precedence

**Rules**:
| Rule | Behavior |
|------|----------|
| Default | Skills prefixed `/skill:skill-name` (e.g., `/skill:example-creator`) |
| Override | Config `prefix: ""` removes prefix entirely, bare name used |
| Conflict | Native commands take precedence; warning logged if conflict detected |
| Aliases | Skills can define aliases in SKILL.md metadata (optional) |

**Configuration** (in SKILL.md):
```yaml
---
name: my-skill
slash_command:
  prefix: ""  # Override to use bare "/my-skill"
aliases: ["myalias", "ms"]  # Alternative access names
---
```

---

### ✅ RESOLVED: Permission Model (Q4)

**DECISION**: Per-skill and global configuration in `agents.yml`

**Schema Addition**:
```python
# wolfharness_config/skill_commands.py
class SkillSlashConfig(Schema):
    """Configuration for skill slash command exposure."""
    
    enabled: bool = True
    """Whether this skill is exposed as a slash command."""
    
    allowed_agents: list[str] = []
    """Agent names that can use this skill via slash command (empty = all)."""
    
    require_confirmation: bool = False
    """Require user confirmation before invoking (for destructive skills)."""
```

**Global Defaults** (in `agents.yml`):
```yaml
skills:
  global_slash_config:
    enabled: true
    require_confirmation: false
    
  per_skill_config:
    dangerous-skill:
      require_confirmation: true
      allowed_agents: ["admin", "main"]
```

---

### ✅ RESOLVED: AG-UI Protocol (Q5)

**DECISION**: Skills exposed as AG-UI Tools, not commands

**Research Conclusion**:
- AG-UI is **stateless HTTP+SSE** protocol
- AG-UI uses **OpenAI function format** for tools
- No native "slash command" concept exists in AG-UI
- Tools are the primary extensibility mechanism

**Implementation**:
- Each skill becomes an AG-UI `Tool` with OpenAI function schema
- User invokes via tool call, not slash command syntax
- May require wrapper agent to translate `/` syntax to tool calls
- Potential future: AG-UI custom capability for slash commands

**AG-UI Usage**:
```json
// Tool definition returned in agent endpoint
{
  "name": "skill__my-skill",
  "description": "Skill description",
  "parameters": {
    "type": "object",
    "properties": {
      "arguments": {
        "type": "string",
        "description": "Arguments for the skill"
      }
    }
  }
}
```

## Decision Record

**Status**: REVISED AFTER REVIEW → AWAITING REVIEW

**Summary**: RFC revised based on Metis and Momus review. Critical correction: OpenCode Bridge now uses native slashed Commands instead of MCP Prompts.

**Key Revisions**:
- ✅ OpenCode Bridge: MCP Prompts → Native slashed Commands
- ✅ Phase Ordering: OpenCode Bridge moved to Phase 4 (last)
- ✅ ACP Schema: Added `slash_commands` field documentation
- ✅ Protocol Matrix: Added detailed capability comparison
- ✅ Implementation guidance: Bash/$ARGUMENTS syntax documented

**Reviewer Checklist**:
- [ ] Architecture review: Unified registry + bridges pattern
- [ ] Security review: Input validation and sandboxing
- [ ] Performance review: <50ms registration target
- [ ] Backward compatibility: No SKILL.md format changes required
- [ ] Documentation: Protocol Capability Matrix included

**Conditions for Approval**:
1. ✅ All Open Questions resolved (in document)
2. ✅ AG-UI approach clarified (Tools, not Commands)
3. Performance benchmark to be validated in Phase 1

---

## Appendix A: Quick Reference

### Skill Structure Reminder
```yaml
---
name: my-skill              # Becomes /my-skill
description: Does X         # Command description
allowed-tools: [read,grep]  # Enforced on invocation
---

# Instructions here become agent context after loading
```

### Protocol Command Mappings

| Protocol | Native Type | Implementation | Execution |
|----------|-------------|----------------|-----------|
| **OpenCode** | `slashed.Command` | `SkillCommandWrapper` extending `Command` | `/skill:name args` routs to `execute()` |
| **ACP** | `AvailableCommand` | Direct schema mapping via `ACPSkillBridge` | Session prompt mechanism |
| **AG-UI** | `Tool` (OpenAI format) | `AGUISkillToolAdapter` wrapping skills | Function call with `{arguments: str}` |

### User Experience by Protocol

| Protocol | User Input | Result |
|----------|------------|--------|
| **OpenCode** | `/skill:name arg1 arg2` | Skill loaded, instructions in context |
| **ACP** | `/name arg1 arg2` in prompt | Prompt sent to agent, skill invoked |
| **AG-UI** | Tool call via UI | AI calls `skill__name`, instructions returned |

### Key Implementation Notes

**OpenCode**:
- Commands are **native slashed Commands** (not MCP Prompts)
- Arguments support bash-style substitution: `$1`, `$2`, `$ARGUMENTS`
- CommandStore integrates with `GET /command` endpoint

**ACP**:
- Commands exposed via `AgentCapabilities.slash_commands` field
- Requires schema change to capabilities
- Runtime updates via `AvailableCommandsUpdate`

**AG-UI**:
- Skills exposed as tools (function calling interface)
- No native slash command syntax
- Best effort: May require wrapper for /command syntax

### File Paths Reference

```
src/wolfharness/skills/
├── command.py              # NEW: SkillCommand dataclass
├── command_registry.py     # NEW: SkillCommandRegistry
└── ...                     # existing files

src/wolfharness_server/opencode_server/
├── skill_bridge.py         # NEW: OpenCodeSkillBridge
└── server.py               # MODIFY: integration

src/wolfharness_server/acp_server/
└── commands/
    └── skill_commands.py   # NEW: ACPSkillBridge

src/wolfharness_config/
└── skill_commands.py       # NEW: Optional config schema
```
