---
rfc_id: RFC-0008
title: Dynamic Skills Injection via ResourceProvider Instructions
status: ACCEPTED
author: Sisyphus
reviewers: []
created: 2026-02-07
last_updated: 2026-02-09
decision_date: 2026-02-09
related_prds: []
related_rfcs:
  - RFC-0005: Skills Injection into System Prompts (SUPERSEDED)
  - RFC-0007: Dynamic Instructions for Resource Providers
---

# RFC-0008: Dynamic Skills Injection via ResourceProvider Instructions

## Overview

This RFC proposes a new approach for automatic skills injection into agent system prompts, superseding RFC-0005. Instead of static skill injection via `SystemPrompts`, we leverage RFC-0007's dynamic instruction mechanism through `ResourceProvider.get_instructions()`. This enables runtime context-aware skill selection and formatting.

**Motivation**: RFC-0007 introduced a powerful mechanism for dynamic, context-aware instructions through ResourceProviders. Rather than implementing a parallel static injection system (as proposed in RFC-0005), we should build skills injection on top of RFC-0007's infrastructure. This provides:
- Runtime skill selection based on conversation context
- Cleaner architecture with single instruction pathway
- Future extensibility for ML-based skill relevance scoring

**Relation to RFC-0005**: This RFC supersedes RFC-0005 by replacing its static `Skill.format_for_injection()` + `SystemPrompts` integration approach with a dynamic `ResourceProvider`-based approach. The core goals remain the same (automatic skills injection), but the implementation aligns with RFC-0007.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- [Migration from RFC-0005](#migration-from-rfc-0005)
- Implementation Plan
- [Open Questions](#open-questions)

---

## Background & Context

### Current State (Post RFC-0007)

AgentPool now has RFC-0007's dynamic instruction infrastructure:

1. **Instruction Function Types** (`src/wolfharness/prompts/instructions.py`):
   ```python
   InstructionFunc = (
       SimpleInstruction | AgentContextInstruction | 
       RunContextInstruction | BothContextsInstruction
   )
   ```

2. **ResourceProvider Extension** (`src/wolfharness/resource_providers/base.py`):
   ```python
   class ResourceProvider:
       async def get_instructions(self) -> list[InstructionFunc]:
           """Return dynamic instruction functions re-evaluated on each run."""
           return []
   ```

3. **NativeAgent Integration** (`src/wolfharness/agents/native_agent/agent.py`):
   ```python
   async def get_agentlet(self, ...):
       # Collect instructions from all providers
       for provider in self.tools.providers:
           provider_instructions = await provider.get_instructions()
           for fn in provider_instructions:
               wrapped = wrap_instruction(fn, fallback="")
               all_instructions.append(wrapped)
       
       return PydanticAgent(
           ...,
           instructions=all_instructions,  # Static + dynamic
       )
   ```

4. **Context Wrapping** (`src/wolfharness/utils/context_wrapping.py`):
   - `wrap_instruction()` adapts instruction functions to pydantic-ai's `(RunContext) -> str` signature
   - Automatically injects AgentContext and/or RunContext based on function signature

### Complete RFC-0005 Background

RFC-0005 proposed **static skill injection** into agent system prompts through multiple approaches. The recommended approach (Option 1: Prepare Hook Approach) was:

#### RFC-0005 Skill.format_for_injection()

```python
# src/wolfharness/skills/skill.py
class Skill:
    def format_for_injection(
        self,
        injection_mode: Literal["metadata", "full"] = "full",
    ) -> str:
        """Format skill content for injection into system prompt.
        
        Inspired by RFC-0002's prepare hook pattern.
        """
        if injection_mode == "metadata":
            return self._prepare_metadata()
        elif injection_mode == "full":
            return self._prepare_full()
    
    def _prepare_metadata(self) -> str:
        """Minimal skill description (~100 tokens)."""
        return f"### {self.name}\n\n{self.description}"
    
    def _prepare_full(self) -> str:
        """Complete skill with instructions."""
        if self._instructions is None:
            self._load_content()
        return f"### {self.name}\n\n{self.description}\n\n{self._instructions}"

# src/wolfharness/agents/sys_prompts.py
class SystemPrompts:
    def __init__(
        self,
        ...,
        inject_skills: Literal["off", "metadata", "full"] = "off",
        skills_registry: SkillsRegistry | None = None,
    ) -> None:
        self.inject_skills = inject_skills
        self.skills_registry = skills_registry
    
    async def format_system_prompt(self, agent: BaseAgent) -> str:
        result = ...  # Build base prompt
        if self.inject_skills != "off" and self.skills_registry:
            skills_section = await self._build_skills_section()
            result += "\n\n" + skills_section
        return result.strip()
    
    async def _build_skills_section(self) -> str:
        """Build skills section from registry (MARKDOWN FORMAT)."""
        skills = await self.skills_registry.list_items_async()
        lines = ["## Available Skills\n"]
        for skill in skills.values():
            content = skill.format_for_injection(injection_mode=self.inject_skills)
            lines.append(content)
        return "\n\n".join(lines)
```

#### RFC-0005 Configuration

```yaml
# config.yml
skills:
  paths:
    - ./skills
  injection_mode: metadata  # Pool-wide default

agents:
  coder:
    type: native
    model: openai:gpt-4o
    system_prompt: "You are an expert developer."
    skills_injection: full  # Override pool default
```

#### RFC-0005 Limitations Addressed by RFC-0008

1. **Static Injection**: Skills formatted once at agent creation, not per-run.
2. **No Runtime Context**: Cannot adapt based on conversation state.
3. **Markdown Format**: Less structured than XML for LLM parsing.
4. **Bloated Prompts**: No mechanism for selective injection or metadata-only views.

### Why RFC-0007's Mechanism is Better for Skills

| Aspect | RFC-0005 (Static) | RFC-0007 (Dynamic) | RFC-0008 (Enhanced) |
|--------|-------------------|-------------------|---------------------|
| **Timing** | Once at agent creation | Every run | Every run |
| **Context access** | None | AgentContext + RunContext | AgentContext + RunContext |
| **Format** | Markdown | Markdown | **XML** |
| **Architecture** | Parallel system | Unified with other instructions | **Dedicated Provider** |
| **Future ML** | Hard to integrate | Natural extension point | Natural extension point |

---

## Problem Statement

### The Problem

1. **RFC-0005's Static Approach is Suboptimal**: Static skill injection cannot adapt to runtime context (conversation history, current task, user preferences).

2. **Markdown Format Limitations**: RFC-0005 uses markdown concatenation. **XML format provides clearer element boundaries** with explicit open/close tags, which may improve parsing reliability.

3. **Separation of Concerns**: RFC-0005 mixed prompt construction with registry management. RFC-0008 uses a dedicated `SkillsInstructionProvider`.

4. **Token Efficiency**: Static injection always includes all skills. Dynamic injection can selectively include only relevant skills based on metadata.

---

## Goals & Non-Goals

### Goals (In Scope)

1. **Implement skills injection via RFC-0007's ResourceProvider mechanism**.
2. **Support dynamic, context-aware skill selection and formatting**.
3. **Provide injection modes**: off, metadata-only, full.
4. **Use structured XML format** for clearer element boundaries.
5. **Dedicated instruction provider**: Create separate `SkillsInstructionProvider` class to keep `SkillsTools` focused on tool provision.
6. **Maintain backward compatibility** with existing agent configurations.
7. **Enable future extensibility** for ML-based skill relevance scoring.

### Non-Goals (Out of Scope)

1. **ML-based skill selection** (foundation for future RFC).
2. **Slash command integration** (planned for future enhancement).
3. **Skill versioning or conflict resolution**.
4. **Modifying RFC-0007's core mechanism**.
5. **Non-native agent support**.

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| **RFC Alignment** | High | Leverages RFC-0007 infrastructure | Uses `get_instructions()` |
| **Implementation Simplicity** | High | Clean, maintainable code | <150 LOC new code |
| **Flexibility** | High | Supports multiple injection strategies | 3 modes (off/metadata/full) |
| **Backward Compatibility** | High | Existing configs work unchanged | 100% compatibility |
| **Token Efficiency** | Medium | Optimized context usage | Selective injection ready |
| **Future Extensibility** | Medium | Easy to add ML-based selection | Clear extension points |

---

## Options Analysis

### Option 1: Dedicated SkillsInstructionProvider with XML Injection (RECOMMENDED)

**Description**

Create a dedicated `SkillsInstructionProvider` class (`src/wolfharness/resource_providers/skills_instruction.py`) that implements RFC-0007's dynamic instruction mechanism. This approach:
- Keeps concern separated from tool provision (`SkillsTools` remains unchanged).
- Uses **structured XML format** for skill representation.
- Supports runtime selection and formatting based on `AgentContext`.

**Key Improvements Over RFC-0005:**
1. **Dedicated instruction provider** - Separate class for single responsibility.
2. **XML format** - Clearer element boundaries for LLM parsing.
3. **Dynamic per-run** - Instructions regenerated on each agent run.
4. **Backward compatible** - Existing configurations continue to work.

**Configuration Models**:

```python
class SkillsInstructionConfig(BaseModel):
    """Configuration for skills injection via ResourceProvider."""
    
    mode: Literal["off", "metadata", "full"] = "metadata"
    max_skills: int | None = None


class SkillsConfig(BaseModel):
    """Extended skills configuration."""
    
    paths: list[UPath | str] = Field(default_factory=list)
    include_default: bool = Field(default=True)
    instruction: SkillsInstructionConfig | None = Field(
        default=None,
        description="Skills injection configuration."
    )
```


```python
# src/wolfharness_config/toolsets.py (extend existing SkillsToolsConfig)
class SkillsToolsConfig(ToolsetConfig):
    """Configuration for skills tools toolset.
    
    This config allows overriding pool-wide injection settings 
    for a specific agent.
    """
    
    type: Literal["skills"] = "skills"
    
    # Injection overrides (None = use pool-wide config)
    injection_mode: Literal["off", "metadata", "full"] | None = None
    max_skills: int | None = None
```

```yaml
# Example configuration
skills:
  paths:
    - ./skills
  include_default: true

  # Pool-wide skill injection defaults
  instruction:
    mode: metadata  # Default is "off", enable injection with metadata or full
    max_skills: 10

agents:
  coder:
    type: native
    model: openai:gpt-4o
    tools:
      - type: skills
      # Uses pool-wide defaults (metadata, 10 skills)

  expert:
    type: native
    model: openai:gpt-4o
    tools:
      - type: skills
        injection_mode: full  # Override to full for this agent
        max_skills: 5
```

**Advantages**

- **RFC Alignment**: Fully leverages RFC-0007 infrastructure.
- **Dedicated Provider**: Keeps prompt injection logic separate from tool provision.
- **Dynamic Context**: Instruction functions receive AgentContext for runtime decisions.
- **Clean Architecture**: Unified instruction pathway via RFC-0007.
- **Backward Compatible**: Default behavior is "off"; existing configs work unchanged.

- **Future-Ready**: Easy to add context-aware skill selection
- **Token Efficient**: Can filter skills based on conversation (future)

**Disadvantages**

- **XML Verbosity**: XML is more verbose than markdown (but provides better structure)
- **Requires RFC-0007 Knowledge**: Developers need to understand RFC-0007's instruction mechanism
- **Async Only**: Must use async instruction functions

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| RFC Alignment | 10/10 | Direct use of RFC-0007 |
| Implementation Simplicity | 9/10 | Reuses toolset pattern, extends SkillsTools class |
| Flexibility | 10/10 | Dynamic, XML format, CommandStore integration |
| Backward Compatibility | 10/10 | Opt-in, default off |
| Token Efficiency | 8/10 | Ready for selective injection |

| Performance | 9/10 | Minimal overhead |

---

### Option 2: Hybrid Static-Dynamic (RFC-0005 + RFC-0007)

**Description**

Keep RFC-0005's static injection in `SystemPrompts` AND add RFC-0007 dynamic instructions. Users can choose between approaches.

**Implementation**:

```python
class SystemPrompts:
    def __init__(
        self,
        ...,
        # RFC-0005 approach
        inject_skills: Literal["off", "metadata", "full"] = "off",
        skills_registry: SkillsRegistry | None = None,
        # RFC-0007 approach (separate)
        dynamic_skill_provider: SkillInstructionProvider | None = None,
    ):
        ...
```

**Advantages**

- **Flexibility**: Users can choose approach
- **Backward Compatibility**: Existing RFC-0005 code still works

**Disadvantages**

- **Technical Debt**: Two parallel systems to maintain
- **User Confusion**: Which approach to use?
- **Complexity**: More code, more tests, more docs

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| RFC Alignment | 5/10 | Duplicates functionality |
| Implementation Simplicity | 4/10 | Two systems |
| Flexibility | 6/10 | Choice is complexity |
| Backward Compatibility | 7/10 | Maintains RFC-0005 |
| Token Efficiency | 5/10 | Static approach limited |
| Future Extensibility | 4/10 | Two paths to extend |
| Performance | 6/10 | Overhead of both |

---

### Option 3: Agent-Level Instruction Configuration

**Description**

Instead of a dedicated `SkillInstructionProvider`, allow agents to reference skills directly in their instruction configuration using RFC-0007's `ProviderInstructionConfig`.

**Implementation**:

```yaml
agents:
  coder:
    type: native
    model: openai:gpt-4o
    instructions:
      - "You are a helpful assistant."
      - type: skills  # NEW instruction type
        mode: metadata
        max_skills: 10
```

**Advantages**

- **Explicit**: Skills are part of instruction configuration
- **Flexible**: Per-agent control

**Disadvantages**

- **No Provider**: Loses ResourceProvider benefits (change signals, lifecycle)
- **Manual Setup**: Users must add to each agent
- **Inconsistent**: Different pattern than other resources

**Evaluation Against Criteria**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| RFC Alignment | 6/10 | Uses instructions but not provider |
| Implementation Simplicity | 6/10 | New config type |
| Flexibility | 5/10 | Manual per-agent setup |
| Backward Compatibility | 5/10 | Requires config changes |
| Token Efficiency | 6/10 | Limited selection |
| Future Extensibility | 5/10 | Not integrated with provider system |
| Performance | 8/10 | Direct, no provider overhead |

---

### Options Comparison Summary

| Criterion | Weight | Option 1 (Toolset) | Option 2 (Hybrid) | Option 3 (Agent-Level) |
|-----------|--------|--------------------|-------------------|------------------------|
| RFC Alignment | High | 10 | 5 | 6 |
| Implementation Simplicity | High | 9 | 4 | 6 |
| Flexibility | High | 10 | 6 | 5 |
| Backward Compatibility | High | 10 | 7 | 5 |
| Token Efficiency | Medium | 8 | 5 | 6 |
| Future Extensibility | Medium | 10 | 4 | 5 |
| Performance | Low | 9 | 6 | 8 |
| **Simple Average** | | **9.4/10** | **5.0/10** | **5.9/10** |

---

## Recommendation

### Recommended Option

**Option 1: Dedicated SkillsInstructionProvider with XML Injection**

### Justification

Option 1 is recommended because:

1. **Separation of Concerns**: Creates a dedicated `SkillsInstructionProvider` instead of bloating `SkillsTools`, maintaining a clean architecture.
2. **Perfect RFC Alignment**: Directly leverages RFC-0007's `get_instructions()` mechanism in a way that respects single responsibility principle.
3. **Structured Format**: XML format provides clearer element boundaries for LLM parsing.
4. **Future-Proof**: Dynamic instruction functions can evolve to include ML-based skill selection.
5. **Backward Compatible**: Default "off" mode ensures existing configurations continue to work unchanged.

### Why NOT Other Options

- **Option 2 (Hybrid)**: Creates permanent technical debt by maintaining two parallel systems. The complexity outweighs the marginal flexibility benefit.

- **Option 3 (Agent-Level)**: Bypasses the ResourceProvider system, losing benefits like change signals, lifecycle management, and centralized resource handling.

### Accepted Trade-offs

1. **Requires RFC-0007 Knowledge**: Developers need to understand RFC-0007's instruction mechanism. Mitigation: Good documentation and examples.

2. **Async-Only**: Must use async instruction functions. Mitigation: All ResourceProvider methods are already async, consistent with codebase.

3. **XML vs Markdown**: XML is more verbose than markdown, but provides better structure for LLM parsing.

### Conditions

1. Must maintain backward compatibility (default off)
2. Must provide clear migration guide from RFC-0005 approach
3. Must include comprehensive examples
4. Should leave extension points for ML-based selection

---

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AgentsManifest (config.yml)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ skills:                                            │  │
│  │   paths: [...]                                    │  │
│  │   instruction:      # NEW: SkillsInstructionConfig │  │
│  │     mode: metadata                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
  │  │ agents:                                            │  │
  │  │   coder:                                           │  │
  │  │     tools:                                         │  │
  │  │       - type: skills                               │  │
  │  │         injection_mode: full                       │  │
  │  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   AgentPool Initialization                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. SkillsManager discovers skills from paths    │   │
│  │ 2. SkillsRegistry populated with discovered     │   │
│  │ 3. Create SkillsInstructionProvider             │   │
│  │ 4. Add to pool providers                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Agent Initialization (get_agentlet)            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. NativeAgent.get_agentlet() called             │   │
│  │ 2. Collect instructions from all providers:       │   │
│  │    for provider in self.tools.providers:         │   │
│  │        instructions = await provider.get_instructions()│
│  │ 3. SkillsInstructionProvider returns:             │   │
│  │    [_generate_skills_instruction]                 │   │
│  │ 4. Formats skills as XML (structured)            │   │
│  │ 5. Pass to PydanticAgent(instructions=[...])     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Agent Run (Dynamic Evaluation)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ On each agent.run():                              │   │
│  │ 1. PydanticAgent calls instruction functions      │   │
│  │ 2. _generate_skills_instruction(ctx) receives    │   │
│  │    AgentContext with conversation history, etc.  │   │
│  │ 3. Format skills as XML (structured)             │   │
│  │ 4. Return formatted skills section               │   │
│  │ 5. Skills appear in system prompt                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. SkillsInstructionProvider (NEW)

**File**: `src/wolfharness/resource_providers/skills_instruction.py`

Dedicated ResourceProvider for skills injection via RFC-0007's `get_instructions()`.
Keeps concerns separated from SkillsTools.

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from xml.sax.saxutils import escape

from wolfharness.agents.context import AgentContext
from wolfharness.log import get_logger
from wolfharness.resource_providers import ResourceProvider

if TYPE_CHECKING:
    from wolfharness.prompts.instructions import InstructionFunc
    from wolfharness.skills.registry import SkillsRegistry

logger = get_logger(__name__)

InjectionMode = Literal["off", "metadata", "full"]


class SkillsInstructionProvider(ResourceProvider):
    """ResourceProvider that injects skills as dynamic XML-formatted instructions.
    
    This provider implements RFC-0007's get_instructions() to inject skills
    into agent system prompts. It is separate from SkillsTools to maintain
    single responsibility principle.
    """
    
    def __init__(
        self,
        name: str = "skills_instructions",
        skills_registry: SkillsRegistry | None = None,
        injection_mode: InjectionMode = "metadata",
        max_skills: int | None = None,
    ) -> None:
        """Initialize skills instruction provider.
        
        Args:
            name: Provider name
            skills_registry: Registry containing discovered skills
            injection_mode: "metadata" (names/desc) or "full" (complete instructions)
            max_skills: Maximum skills to include (None = all)
        """
        super().__init__(name=name)
        self.registry = skills_registry
        self.injection_mode = injection_mode
        self.max_skills = max_skills
    
    async def get_instructions(self) -> list[InstructionFunc]:
        """Return skill injection instruction functions (RFC-0007)."""
        return [self._generate_skills_instruction]

    async def _generate_skills_instruction(self, ctx: AgentContext) -> str:
        """Generate XML-formatted skills section.

        This instruction function is called on each agent run.
        """
        if self.registry is None:
            return ""

        # 1. Check for overrides in agent context
        injection_mode = self.injection_mode
        max_skills = self.max_skills

        # Traverse providers to find SkillsTools (usually named "skills")
        # and extract overrides if present.
        node = ctx.node
        if (tools := getattr(node, "tools", None)) and (
            providers := getattr(tools, "providers", None)
        ):
            for provider in providers:
                if getattr(provider, "name", None) == "skills":
                    # Check for overrides on the provider instance
                    if (val := getattr(provider, "injection_mode", None)) is not None:
                        injection_mode = val
                    if (val := getattr(provider, "max_skills", None)) is not None:
                        max_skills = val
                    break

        if injection_mode == "off":
            return ""

        # Apply limit if configured
        skill_items = list(self.registry.items())
        if not skill_items:
            return ""

        if max_skills is not None:
            skill_items = skill_items[:max_skills]

        # Build XML
        return await self._format_skills_xml(skill_items, cast(InjectionMode, injection_mode))
    
    async def _format_skills_xml(
        self,
        skill_items: list[tuple[str, Any]],
        mode: InjectionMode,
    ) -> str:
        """Format skills using structured XML format."""
        lines = ["<available-skills>"]

        for name, skill in skill_items:
            try:
                if mode == "metadata":
                    content = self._format_skill_metadata(name, skill)
                elif mode == "full":
                    # Load instructions if available
                    instructions = ""
                    if hasattr(skill, "load_instructions"):
                        instructions = skill.load_instructions()
                    elif hasattr(skill, "instructions"):
                        instructions = skill.instructions or ""

                    content = self._format_skill_full(name, skill, instructions)
                else:
                    continue
                lines.append(content)
            except Exception:
                logger.exception("Failed to format skill for injection", skill=name)
                continue

        lines.append("</available-skills>")
        return "\n".join(lines)
    
    def _format_skill_metadata(self, name: str, skill: Any) -> str:
        """Format skill metadata in XML."""
        desc = escape(str(skill.description)) if hasattr(skill, "description") else ""
        return f'  <skill id="{escape(name)}" name="{escape(name)}" description="{desc}" />'

    def _format_skill_full(self, name: str, skill: Any, instructions: str) -> str:
        """Format full skill content in XML."""
        desc = escape(str(skill.description)) if hasattr(skill, "description") else ""
        path = str(skill.skill_path) if hasattr(skill, "skill_path") else ""

        return f"""  <skill id="{escape(name)}" name="{escape(name)}" description="{desc}">
    <instructions>
      <skill-instruction>
      Base directory for this skill: {path}/
      File references (@path) are relative to this directory.

      {instructions}
      </skill-instruction>

      <user-request>
      $ARGUMENTS
      </user-request>
    </instructions>
  </skill>"""
```

#### 2. SkillsTools (UNCHANGED)

**File**: `src/wolfharness_toolsets/builtin/skills.py`

Existing tools provider remains unchanged. Provides:
- `load_skill` tool
- `list_skills` tool

No modifications needed for RFC-0008.

#### 2. Configuration Models

**File**: `src/wolfharness_config/skills.py`

```python
class SkillsInstructionConfig(BaseModel):
    """Configuration for skills injection via ResourceProvider."""

    mode: Literal["off", "metadata", "full"] = "off"
    max_skills: int | None = None


class SkillsConfig(BaseModel):
    """Extended skills configuration."""

    paths: list[UPath | str] = Field(default_factory=list)
    include_default: bool = Field(default=True)
    instruction: SkillsInstructionConfig | None = Field(
        default=None,
        description="Skills injection configuration. If None, no injection."
    )
```

#### 3. Toolset Configuration

**File**: `src/wolfharness_config/toolsets.py` (extend)

```python
class SkillsToolsetConfig(ToolsetConfig):
    """Configuration for skills toolset."""
    
    type: Literal["skills"] = "skills"
    # Note: injection settings are now handled by the instruction provider
    # but can be overridden here if needed for agent-specific behavior.
    injection_mode: Literal["off", "metadata", "full"] | None = None
    max_skills: int | None = None
```

#### 4. AgentPool Integration

**File**: `src/wolfharness/delegation/pool.py` (extend)

```python
class AgentPool:
    def __init__(self, ...):
        # Existing skills manager
        self.skills = SkillsManager(config.skills)
        
        # NEW: Create SkillsInstructionProvider if configured
        if config.skills and config.skills.instruction:
            instr_config = config.skills.instruction
            if instr_config.mode != "off":
                skills_provider = SkillsInstructionProvider(
                    skills_registry=self.skills.registry,
                    injection_mode=instr_config.mode,
                    max_skills=instr_config.max_skills,
                )
                self.providers.append(skills_provider)
```

### Data Model Changes

**New Files**:
- `src/wolfharness/resource_providers/skills_instruction.py` - Dedicated instruction provider
- `tests/resource_providers/test_skills_instruction.py` - Tests for skills instruction provider

**Modified Files**:
- `src/wolfharness_config/skills.py` - Add `SkillsInstructionConfig`
- `src/wolfharness/delegation/pool.py` - Integrate skills injection provider

### Configuration Examples

#### Example 1: Pool-Wide XML Injection


```yaml
skills:
  paths:
    - ./skills
  instruction:
    mode: metadata  # Default is "off", enable injection with metadata or full
    max_skills: 10

agents:
  coder:
    type: native
    model: openai:gpt-4o
    # Uses pool-wide instruction config (metadata)
```

#### Example 2: Per-Agent Full Injection

```yaml
skills:
  paths:
    - ./skills
  instruction:
    mode: metadata  # Pool-wide default (default is "off")

agents:
  expert:
    type: native
    model: openai:gpt-4o
    tools:
      - type: skills
        injection_mode: full  # Override to full for this agent
        max_skills: 5
```

#### Example 3: Disabled Injection

```yaml
skills:
  paths:
    - ./skills
  # No instruction config = skills not injected into system prompt

agents:
  simple:
    type: native
    model: openai:gpt-4o
    # Skills tools available but not injected
```

#### Example 4: Full XML Output Format

When `mode: full` is used, skills are injected in the following structured XML format:

```xml
<available-skills>
  <skill name="git-workflow">
    <description>Expert in Git workflows and branch management</description>
    <instructions>
      <skill-instruction>
      Base directory for this skill: /path/to/skills/git-workflow/
      File references (@path) in this skill are relative to this directory.

      Git branch creation and merging best practices...
      </skill-instruction>

      <user-request>
      $ARGUMENTS
      </user-request>
    </instructions>
  </skill>
</available-skills>
```

---

## Migration from RFC-0005

### What Changes

| Feature | RFC-0005 (Static) | RFC-0008 (Dynamic) |
|---------|-------------------|-------------------|
| **Injection Type** | Static at agent creation | Dynamic on each agent run |
| **Mechanism** | `SystemPrompts` string manipulation | `ResourceProvider.get_instructions()` |
| **Context Access** | None | `AgentContext` & `RunContext` |
| **Formatting** | Markdown concatenation | Structured XML format |
| **Architecture** | Parallel system | Integrated with RFC-0007 |
| **Responsibility** | Bloated `SystemPrompts` | Dedicated `SkillsInstructionProvider` |

### Migration Guide

#### Before (RFC-0005 Draft Approach)

```python
# SystemPrompts usage
class SystemPrompts:
    def __init__(self, ..., inject_skills: Literal["off", "metadata", "full"] = "off"):
        ...
```

```yaml
# config.yml
agents:
  coder:
    type: native
    skills_injection: full  # Hypothetical field
```

#### After (RFC-0008)

```yaml
# config.yml
skills:
  instruction:
    mode: metadata

agents:
  expert:
    type: native
    tools:
      - type: skills
        injection_mode: full
```

### Breaking Changes

**None** - RFC-0008 is purely additive:
- Default behavior (no injection) remains unchanged.
- Existing toolset configurations for `type: skills` continue to work.
- `skills_injection` field was never implemented in production.

---

## Implementation Plan

### Phase 1: Core Implementation (Day 1)

- **Scope**: Create `SkillsInstructionProvider`
- **Deliverables**:
  - `src/wolfharness/resource_providers/skills_instruction.py`:
    - Implement `get_instructions()` returning dynamic XML generators
    - Implement XML formatting with proper character escaping
  - Unit tests for instruction generation logic

### Phase 2: Configuration (Day 1)

- **Scope**: Add configuration models
- **Deliverables**:
  - Update `wolfharness_config/skills.py` with `SkillsInstructionConfig`
  - Update `wolfharness_config/toolsets.py` to support injection overrides
  - Config validation tests

### Phase 3: AgentPool Integration (Day 2)

- **Scope**: Connect provider to pool and agents
- **Deliverables**:
  - Update `AgentPool` to instantiate `SkillsInstructionProvider` from manifest
  - Ensure `NativeAgent` correctly picks up instructions from the provider
  - Integration tests verifying XML content in system prompts

### Phase 4: Documentation (Day 2)

- **Scope**: Documentation and examples
- **Deliverables**:
  - Update project documentation with new skills injection capabilities
  - Add usage examples to `docs/`

---

## Open Questions

1. **Skill Ordering**
   - Context: Should skills be ordered by relevance or alphabetically?
   - Recommendation: Start with alphabetical for determinism.

2. **XML vs Markdown effectiveness**
   - Context: RFC hypothesizes XML provides clearer boundaries for LLM parsing.
   - Recommendation: Collect feedback/metrics from production use to validate this choice.

3. **Performance Impact**
   - Context: Dynamic re-evaluation adds minimal overhead per run.
   - Recommendation: Measure and ensure overhead remains <5ms.

---

## Decision Record

### Decision: Supersede RFC-0005 with RFC-0008

**Date**: 2026-02-09
**Decision Maker**: Antigravity / Sisyphus

**Decision**: Implement `SkillsInstructionProvider` leveraging RFC-0007 infrastructure.

**Rationale**:
1. **Architectural Purity**: Unified instruction pathway via RFC-0007.
2. **Context Awareness**: Dynamic injection allows for future intelligent skill selection.
3. **Separation of Concerns**: Separate instruction generation from tool provisioning.
4. **Reliability**: XML format provides clear structure for complex prompts.

---

## References

- RFC-0007: Dynamic Instructions for Resource Providers
- `src/wolfharness/resource_providers/base.py`
- `src/wolfharness/prompts/instructions.py`
**Consequences**:
- Positive: Cleaner architecture, better extensibility
- Positive: Aligns with RFC-0007
- Positive: Structured XML format
- Negative: Requires RFC-0007 knowledge
- Negative: RFC-0005 work is abandoned

---

## References

- RFC-0005: Skills Injection into System Prompts (SUPERSEDED)
- RFC-0007: Dynamic Instructions for Resource Providers
- `src/wolfharness/prompts/instructions.py`
- `src/wolfharness/resource_providers/base.py`
- `src/wolfharness/utils/context_wrapping.py`
