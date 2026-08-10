---
rfc_id: RFC-0020
title: MCP Skills Resources Provider Protocol Support
status: IMPLEMENTED
author: Sisyphus
reviewers:
  - Metis (Plan Consultant) - REVIEWED 2025-04-10
  - Momus (Plan Critic) - REVIEWED 2025-04-10
  - Oracle - REVIEWED 2025-04-10
created: 2025-04-11
last_updated: 2025-04-11
decision_date: 2025-04-11
related_rfcs:
  - RFC-0016: Unified Skill-to-Slash Command Architecture
---

# RFC-0020: MCP Skills Resources Provider Protocol Support

**Document Version**: v1.0 (Initial Draft)

**Key Features**:
- **Architecture**: Extends `ResourceProvider` infrastructure for unified skill access
- **Dual MCP Support**: Both prompt-based and resource-based (FastMCP Skills Provider) skills
- **Security**: Path traversal protection with `Path.relative_to()`
- **URI Specification**: Complete encoding rules, validation, and examples for `skill://` scheme
- **Caching**: LRU caching for skill listings with TTL
- **Integration**: Works with existing SkillsManager without parallel systems

## Overview

This RFC specifies the **MCP Skills Resources Provider Protocol** implementation for AgentPool. It enables skills to be exposed as MCP resources using the `skill://` URI scheme while maintaining compatibility with existing skill and slash command interfaces.

**Key Capabilities**:

1. **Consume MCP Skills**: Discover and load skills from external MCP servers
   - Prompt-based skills: Traditional MCP prompts mapped to skills
   - Resource-based skills: FastMCP Skills Provider protocol (`skill://skill-name/SKILL.md`)

2. **Unified Access**: Skills work seamlessly across sources
   - Local filesystem skills: `~/.claude/skills/`
   - MCP-sourced skills: `skill://provider/skill-name`
   
3. **Reference Content**: Access skill-associated files
   - Templates, examples, schemas via `skill://provider/skill/references/file.md`
   
4. **Integration**: Works with RFC-0016 (Slash Commands) for complete skill ecosystem

**Relation to RFC-0016**: This RFC (0020) focuses on the MCP protocol implementation and resource-based skill access. RFC-0016 covers slash command architecture. Together they provide complete skill infrastructure.

**Expected outcome**: Users can type `/skill-name` in OpenCode TUI, Zed (via ACP), or AG-UI clients to activate skills. Skills can bundle reference materials accessible via `skill://` URIs, enabling richer skill ecosystems.

## Background & Context

### Current State

**ResourceProvider Infrastructure** (`src/wolfharness/resource_providers/`):
- `ResourceProvider` base class with `get_skills()`, `get_skill_instructions()` methods
- `skills_changed` signal for change notifications
- `AggregatingResourceProvider` for combining multiple providers
- `MCPResourceProvider` for MCP server integration (needs skill support)
- **Gap**: No local filesystem skill provider; skills not integrated with ResourceProvider

**Skills System** (`src/wolfharness/skills/`):
- Skills are defined in `SKILL.md` files with YAML frontmatter
- `SkillsRegistry` auto-discovers skills from directories
- `SkillsInstructionProvider` injects skills into prompts (not a ResourceProvider)
- **Gap**: Skills not exposed through ResourceProvider interface

**MCP Integration** (`src/wolfharness/mcp_server/`):
- `MCPResourceProvider` exposes MCP tools, prompts, resources
- MCP servers can expose skills in two ways:
  1. **Via Prompts**: Traditional MCP prompts (e.g., `github-copilot/code-review`)
  2. **Via Resources (Skill Protocol)**: Using `skill://` URI scheme with SKILL.md files
     - Main file: `skill://skill-name/SKILL.md`
     - Manifest: `skill://skill-name/_manifest` (JSON file list)
     - References: `skill://skill-name/reference.md`, etc.
- **Gap**: MCP prompts not mapped to skills
- **Gap**: MCP Skills Provider protocol not supported

### Glossary

- **Skill**: A reusable workflow/prompt collection stored in `SKILL.md`
- **ResourceProvider**: Existing abstraction for tool/prompt/skill sources
- **Skill Reference**: Associated content bundled with a skill (templates, examples)
- **Slash Command**: User-triggerable command syntax (e.g., `/test-skill`)
- **skill:// URI**: Unified protocol for accessing skills and references
- **Provider**: ResourceProvider instance (e.g., "local", "github-copilot")
- **MCP Skill Types**:
  - **Prompt-based**: MCP prompts mapped to skills (with argument schemas)
  - **Resource-based**: Skills exposed via `skill://` resources (FastMCP Skills Provider protocol)

## Problem Statement

### Current Pain Points

1. **Poor Discoverability**: Users must know skill names and use `skill` tool explicitly
2. **No Reference Content**: Skills cannot bundle accessible templates/examples
3. **MCP Skill Gap**: MCP prompts have no standardized skill mapping
4. **Fragmented Architecture**: Skills exist outside ResourceProvider ecosystem
5. **Protocol Inconsistency**: Skills behave differently across OpenCode/ACP/AG-UI

### Goals & Non-Goals

### Goals

1. **MCP Skills Provider Protocol Support**: Full compatibility with FastMCP Skills Provider
2. **Dual MCP Skill Types**: Support both prompt-based and resource-based skills
3. **ResourceProvider Integration**: Skills exposed via existing provider infrastructure
4. **Unified Skill Access**: Local and MCP skills accessible via `skill://` URIs
5. **Reference Content**: Skill-associated files accessible via URI paths
6. **Caching**: Efficient caching with TTL for skill listings and content
7. **Security**: Path traversal protection and safe handling of external skills
8. **Backward Compatible**: Works with existing SkillsRegistry and skill commands

### Non-Goals

1. **NO** skill write operations (read-only MCP access)
2. **NO** skill versioning or update mechanisms
3. **NO** skill marketplace or discovery service
4. **NO** changes to SKILL.md structure
5. **NO** slash command implementation (see RFC-0016)

## Evaluation Criteria

| Criterion | Weight | Description | Min Threshold |
|-----------|--------|-------------|---------------|
| Protocol Compatibility | Critical | Works with OpenCode, ACP, AG-UI | All three supported |
| MCP Integration | Critical | MCP prompts work as skills | Full parity |
| Architecture Consistency | Critical | Uses ResourceProvider pattern | No duplicate abstractions |
| URI Scheme Design | High | RFC 3986 compliant | Validated |
| Security | Critical | Path traversal protection | No vulnerabilities |
| Backward Compatibility | Critical | No breaking changes | 100% compatible |
| Performance | High | <50ms command registration | <100ms acceptable |
| Testability | Medium | >80% coverage | Achievable |

## Options Analysis

### Option 1: Extend ResourceProvider (Recommended)

**Description**: Extend existing `ResourceProvider` infrastructure with skill support. Create `LocalResourceProvider` for filesystem skills, extend `MCPResourceProvider` for MCP prompts, use `AggregatingResourceProvider` for unified access.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ResourceProvider Ecosystem                               │
│                                                                              │
│   ┌──────────────────┐         ┌──────────────────┐                         │
│   │ LocalResource    │         │ MCPResource      │                         │
│   │ Provider (NEW)   │         │ Provider (EXTEND)│                         │
│   │ - filesystem     │         │ - MCP prompts    │                         │
│   │ - references/    │         │ - resources      │                         │
│   └────────┬─────────┘         └────────┬─────────┘                         │
│            │                            │                                    │
│            └────────────┬───────────────┘                                    │
│                         ▼                                                    │
│           AggregatingResourceProvider                                        │
│              (combines all providers)                                        │
│                         │                                                    │
│            ┌────────────┼────────────┐                                       │
│            ▼            ▼            ▼                                       │
│     skills_changed  get_skills()  get_skill_instructions()                  │
│                         │                                                    │
│            ┌────────────┼────────────┐                                       │
│            ▼            ▼            ▼                                       │
│   SkillCommandRegistry → Protocol Bridges → Commands                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Advantages**:
- Reuses proven infrastructure (signals, aggregation, lifecycle)
- No duplicate abstractions
- Native AgentPool integration
- Existing change notification system
- Async context manager support

**Disadvantages**:
- Requires refactoring existing SkillsRegistry integration
- Must maintain backward compatibility

**Evaluation**: 9/10 - Best architectural fit

### Option 2: New SkillProvider Abstraction (RFC v3.0)

**Description**: Create new `SkillProvider` ABC parallel to ResourceProvider.

**Advantages**:
- Clean slate design
- Skill-specific interface

**Disadvantages**:
- Duplicates ResourceProvider functionality
- Parallel signal systems
- More complex integration
- Higher maintenance burden

**Evaluation**: 5/10 - Rejected due to duplication

### Option 3: Direct SkillsRegistry Extension

**Description**: Extend `SkillsRegistry` to support MCP sources.

**Advantages**:
- Minimal changes

**Disadvantages**:
- Doesn't leverage ResourceProvider ecosystem
- No automatic aggregation
- Custom change notification needed

**Evaluation**: 4/10 - Doesn't integrate with existing infrastructure

## Recommendation

**Selected**: **Option 1: Extend ResourceProvider**

**Justification**:
1. **Architecture Fit**: Uses existing proven patterns
2. **Code Reuse**: Leverages `AggregatingResourceProvider`, signals, lifecycle
3. **Simpler**: Fewer new abstractions
4. **Maintainable**: Follows established conventions
5. **Extensible**: Easy to add new skill sources

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AgentPool Runtime                                      │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     ResourceProvider Layer                               │   │
│   │                                                                          │   │
│   │  ┌────────────────────┐      ┌────────────────────┐                     │   │
│   │  │ LocalResource      │      │ MCPResource        │                     │   │
│   │  │ Provider           │      │ Provider           │                     │   │
│   │  │ ─────────────────  │      │ ─────────────────  │                     │   │
│   │  │ get_skills()       │      │ get_skills()       │                     │   │
│   │  │   → from registry  │      │   → from prompts   │                     │   │
│   │  │ get_references()   │      │ get_references()   │                     │   │
│   │  │   → references/    │      │   → resources      │                     │   │
│   │  │ skills_changed     │      │ skills_changed     │                     │   │
│   │  │   → file watcher   │      │   → MCP callbacks  │                     │   │
│   │  └────────┬───────────┘      └────────┬───────────┘                     │   │
│   │           │                           │                                  │   │
│   │           └───────────┬───────────────┘                                  │   │
│   │                       ▼                                                  │   │
│   │           AggregatingResourceProvider                                    │   │
│   │              name="skills"                                               │   │
│   │                       │                                                  │   │
│   │              ┌────────┼────────┐                                         │   │
│   │              ▼        ▼        ▼                                         │   │
│   │         get_tools()  get_prompts()  get_skills()                        │   │
│   └──────────────┬──────────────────────────────────────────────────────────┘   │
│                  │                                                               │
│   ┌──────────────┼──────────────────────────────────────────────────────────┐   │
│   │              ▼                                                           │   │
│   │   ┌──────────────────────┐                                               │   │
│   │   │ SkillCommandRegistry │  (watches AggregatingProvider)                │   │
│   │   │ ───────────────────  │                                               │   │
│   │   │ on_skills_changed()  │  ← connects to skills_changed signal          │   │
│   │   │ get_commands()       │                                               │   │
│   │   └──────────┬───────────┘                                               │   │
│   │              │                                                           │   │
│   │   ┌──────────┼───────────┬────────────────────────┐                      │   │
│   │   ▼          ▼           ▼                        │                      │   │
│   │ OpenCode   ACP        AG-UI                      │                      │   │
│   │ Bridge     Bridge     Bridge                     │                      │   │
│   └──────────────────────────────────────────────────┘                      │   │
│                                                                              │   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Exception Hierarchy

```python
# src/wolfharness/skills/exceptions.py

class SkillError(Exception):
    """Base exception for skill-related errors."""
    pass


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found."""
    
    def __init__(self, skill_name: str, available: list[str] | None = None) -> None:
        msg = f"Skill not found: {skill_name!r}"
        if available:
            msg += f". Available: {', '.join(available[:20])}"
        super().__init__(msg)
        self.skill_name = skill_name
        self.available = available


class ReferenceNotFoundError(SkillError):
    """Raised when a skill reference cannot be found."""
    
    def __init__(self, skill_name: str, ref_path: str) -> None:
        super().__init__(f"Reference not found: {ref_path!r} in skill {skill_name!r}")
        self.skill_name = skill_name
        self.ref_path = ref_path


class SecurityError(SkillError):
    """Raised when a security check fails (e.g., path traversal)."""
    pass


class ProviderError(SkillError):
    """Raised when a provider operation fails."""
    pass
```

#### 2. LocalResourceProvider (New)

```python
# src/wolfharness/resource_providers/local.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Self

from upathtools import UPath

from wolfharness.log import get_logger
from wolfharness.resource_providers import ResourceChangeEvent, ResourceProvider
from wolfharness.resource_providers.resource_info import ResourceInfo
from wolfharness.skills.exceptions import ReferenceNotFoundError, SecurityError, SkillNotFoundError
from wolfharness.skills.registry import SkillsRegistry
from wolfharness.skills.skill import Skill

if TYPE_CHECKING:
    from types import TracebackType


logger = get_logger(__name__)


class LocalResourceProvider(ResourceProvider):
    """Resource provider for local filesystem-based skills.
    
    Discovers skills from configured directories and exposes them via the
    standard ResourceProvider interface. Supports skill references from
    the `references/` subdirectory.
    
    Usage:
        provider = LocalResourceProvider(
            name="local",
            skills_dirs=["~/.claude/skills/", ".claude/skills/"]
        )
        async with provider:
            skills = await provider.get_skills()
    """
    
    kind = "local"
    
    def __init__(
        self,
        name: str = "local",
        skills_dirs: list[str | UPath] | None = None,
        owner: str | None = None,
        cache_ttl: float = 60.0,  # 60 seconds default
    ) -> None:
        """Initialize local resource provider.
        
        Args:
            name: Provider name (default: "local")
            skills_dirs: Directories to search for skills
            owner: Optional owner identifier
            cache_ttl: Cache time-to-live in seconds for skill listings
        """
        super().__init__(name=name, owner=owner)
        self.registry = SkillsRegistry(skills_dirs)
        self._watch_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None
        
        # Caching
        self._cache_ttl = cache_ttl
        self._skills_cache: list[Skill] | None = None
        self._cache_timestamp: float = 0.0
    
    async def __aenter__(self) -> Self:
        """Enter async context - discover skills and start watching."""
        # SkillsRegistry.discover_skills() is sync but does I/O
        # Run in thread to avoid blocking event loop
        import anyio
        await anyio.to_thread.run_sync(self.registry.discover_skills)
        
        # Connect SkillsRegistry callbacks to ResourceProvider signals
        self._connect_registry_callbacks()
        
        # Start filesystem watching
        self._start_watching()
        
        logger.info(
            "LocalResourceProvider initialized",
            name=self.name,
            skill_count=len(self.registry),
        )
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context - stop watching."""
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
    
    async def get_skills(self) -> list[Skill]:
        """Get all skills from registry with caching.
        
        Returns:
            List of skills (cached if within TTL)
        """
        import time
        
        now = time.time()
        if (
            self._skills_cache is not None
            and (now - self._cache_timestamp) < self._cache_ttl
        ):
            return self._skills_cache
        
        skills = list(self.registry.values())
        self._skills_cache = skills
        self._cache_timestamp = now
        return skills
    
    async def get_skill(self, name: str) -> Skill | None:
        """Get a specific skill by name."""
        try:
            return self.registry.get(name)
        except Exception:
            return None
    
    async def get_skill_instructions(self, name: str) -> str:
        """Get full instructions for a skill.
        
        Args:
            name: Skill name
            
        Returns:
            Skill instructions from SKILL.md
            
        Raises:
            SkillNotFoundError: If skill not found
        """
        try:
            return self.registry.get_skill_instructions(name)
        except Exception as e:
            available = list(self.registry.keys())
            raise SkillNotFoundError(name, available) from e
    
    async def get_references(self, skill_name: str) -> list[ResourceInfo]:
        """List all references for a skill.
        
        References are files in the skill's `references/` subdirectory.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            List of resource info for reference files
        """
        skill = await self.get_skill(skill_name)
        if not skill:
            return []
        
        refs_dir = skill.skill_path / "references"
        if not refs_dir.exists():
            return []
        
        refs: list[ResourceInfo] = []
        for ref_file in refs_dir.rglob("*"):
            if ref_file.is_file():
                rel_path = ref_file.relative_to(refs_dir)
                refs.append(ResourceInfo(
                    name=f"{skill_name}/{rel_path}",
                    uri=f"skill://{self.name}/{skill_name}/references/{rel_path}",
                    mime_type=self._detect_mime_type(ref_file),
                    description=f"Reference for {skill_name}",
                ))
        return refs
    
    async def read_reference(self, skill_name: str, ref_path: str) -> str:
        """Read reference file content with path traversal protection.
        
        Args:
            skill_name: Skill name
            ref_path: Relative path within references directory
            
        Returns:
            File content as string
            
        Raises:
            SkillNotFoundError: If skill not found
            ReferenceNotFoundError: If reference not found
            SecurityError: If path traversal detected
        """
        skill = await self.get_skill(skill_name)
        if not skill:
            raise SkillNotFoundError(skill_name)
        
        # SECURITY: Validate path components before resolving
        if ".." in Path(ref_path).parts:
            raise SecurityError(
                f"Path traversal detected in reference path: {ref_path!r}"
            )
        
        # Build paths
        refs_dir = (skill.skill_path / "references").resolve()
        target_file = (refs_dir / ref_path).resolve()
        
        # SECURITY: Ensure resolved path is within references directory
        try:
            target_file.relative_to(refs_dir)
        except ValueError:
            raise SecurityError(
                f"Reference path escapes references directory: {ref_path!r}"
            )
        
        if not target_file.exists():
            raise ReferenceNotFoundError(skill_name, ref_path)
        
        return target_file.read_text(encoding="utf-8")
    
    def _detect_mime_type(self, path: Path) -> str:
        """Detect MIME type from file extension."""
        import mimetypes
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"
    
    def _connect_registry_callbacks(self) -> None:
        """Connect SkillsRegistry callbacks to ResourceProvider signals."""
        # SkillsRegistry has on_skill_added/removed callbacks
        # Forward these to the skills_changed signal
        def on_added(name: str, skill: Skill) -> None:
            self._skills_cache = None  # Invalidate cache
            asyncio.create_task(
                self.skills_changed.emit(self.create_change_event("skills"))
            )
        
        def on_removed(name: str) -> None:
            self._skills_cache = None  # Invalidate cache
            asyncio.create_task(
                self.skills_changed.emit(self.create_change_event("skills"))
            )
        
        self.registry.on_skill_added(on_added)
        self.registry.on_skill_removed(on_removed)
    
    def _start_watching(self) -> None:
        """Start filesystem watcher for skill changes."""
        # TODO: Implement with watchdog for real-time updates
        # For now, registry.refresh() can be called manually
        pass
    
    def _invalidate_cache(self) -> None:
        """Invalidate skills cache."""
        self._skills_cache = None
        self._cache_timestamp = 0.0
```

#### 3. Extended MCPResourceProvider

MCP servers can expose skills in two ways:
1. **Via Prompts**: Traditional MCP prompts that map to skills
2. **Via Resources (Skill Protocol)**: Using `skill://` URI scheme with SKILL.md files

```python
# src/wolfharness/resource_providers/mcp_provider.py (extension)

# Add to existing MCPResourceProvider class:

async def get_skills(self) -> list[Skill]:
    """Get skills from all MCP sources.
    
    Combines:
    1. MCP prompts mapped to skills
    2. MCP resources with skill:// URI scheme (FastMCP Skills Provider protocol)
    
    Returns:
        Combined list of skills from all sources
    """
    skills: list[Skill] = []
    
    # Source 1: MCP Prompts as skills
    prompt_skills = await self._get_prompt_skills()
    skills.extend(prompt_skills)
    
    # Source 2: MCP Resources with skill:// scheme
    resource_skills = await self._get_resource_skills()
    skills.extend(resource_skills)
    
    return skills

async def _get_prompt_skills(self) -> list[Skill]:
    """Get MCP prompts mapped to skills."""
    from wolfharness.skills.skill import Skill
    
    prompts = await self.get_prompts()
    skills: list[Skill] = []
    
    for prompt in prompts:
        # Extract argument schema from prompt if available
        arg_schema = None
        if hasattr(prompt, 'arguments') and prompt.arguments:
            arg_schema = [
                {
                    "name": arg.name,
                    "description": getattr(arg, 'description', None),
                    "required": getattr(arg, 'required', False),
                }
                for arg in prompt.arguments
            ]
        
        # Create skill from MCP prompt
        skill = Skill(
            name=prompt.name,
            description=prompt.description or f"MCP prompt from {self.name}",
            skill_path=UPath(f"skill://{self.name}/{prompt.name}"),
            metadata={
                "source": "mcp",
                "server": self.name,
                "provider": self.name,
                "skill_type": "prompt",  # Distinguish from resource-based
                "argument_schema": arg_schema,
                "has_required_args": any(
                    getattr(arg, 'required', False)
                    for arg in (getattr(prompt, 'arguments', []) or [])
                ),
            },
            compatibility=f"mcp:{self.name}",
        )
        skills.append(skill)
    
    return skills

async def _get_resource_skills(self) -> list[Skill]:
    """Get skills from MCP resources using skill:// protocol.
    
    FastMCP Skills Provider protocol exposes skills as resources:
    - skill://skill-name/SKILL.md - Main instruction file
    - skill://skill-name/_manifest - JSON manifest with file list
    - skill://skill-name/reference.md - Supporting files
    
    See: https://gofastmcp.com/servers/providers/skills
    """
    from wolfharness.skills.skill import Skill
    
    resources = await self.get_resources()
    skills: list[Skill] = []
    
    # Find resources with skill:// scheme pointing to SKILL.md
    skill_main_files: dict[str, str] = {}  # skill_name -> resource_uri
    
    for resource in resources:
        uri = resource.uri
        if not uri.startswith("skill://"):
            continue
        
        # Parse skill://skill-name/SKILL.md
        parts = uri[8:].split("/", 2)  # Remove skill://
        if len(parts) >= 2:
            skill_name = parts[0]
            file_path = parts[1] if len(parts) > 1 else ""
            
            if file_path == "SKILL.md" or file_path.endswith("/SKILL.md"):
                skill_main_files[skill_name] = uri
    
    # Create Skill objects for each discovered skill
    for skill_name, main_uri in skill_main_files.items():
        # Try to get manifest for additional metadata
        manifest = await self._get_skill_manifest(skill_name)
        
        # Read main SKILL.md for description
        description = await self._get_skill_description(skill_name, main_uri)
        
        skill = Skill(
            name=skill_name,
            description=description,
            skill_path=UPath(f"skill://{self.name}/{skill_name}"),
            metadata={
                "source": "mcp",
                "server": self.name,
                "provider": self.name,
                "skill_type": "resource",  # Resource-based skill
                "main_uri": main_uri,
                "manifest": manifest,
                "has_references": manifest is not None and len(manifest.get("files", [])) > 1,
            },
            compatibility=f"mcp:{self.name}",
        )
        skills.append(skill)
    
    return skills

async def _get_skill_manifest(self, skill_name: str) -> dict | None:
    """Get skill manifest from MCP resource.
    
    Manifest URI: skill://skill-name/_manifest
    """
    manifest_uri = f"skill://{skill_name}/_manifest"
    
    try:
        contents = await self.read_resource(manifest_uri)
        if contents:
            import json
            return json.loads(contents[0])
    except Exception:
        pass
    
    return None

async def _get_skill_description(self, skill_name: str, main_uri: str) -> str:
    """Extract description from skill's SKILL.md.
    
    Reads the first meaningful line or YAML frontmatter description.
    """
    try:
        contents = await self.read_resource(main_uri)
        if not contents:
            return f"MCP skill from {self.name}"
        
        content = contents[0]
        
        # Check for YAML frontmatter
        if content.startswith("---"):
            import re
            frontmatter_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)
                desc_match = re.search(r"description:\s*(.+)", frontmatter)
                if desc_match:
                    return desc_match.group(1).strip()
        
        # Fallback: first non-empty, non-header line
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("---"):
                return line[:200]  # Limit description length
        
        return f"MCP skill from {self.name}"
    except Exception:
        return f"MCP skill from {self.name}"

async def get_skill_instructions(
    self, 
    name: str, 
    arguments: dict[str, str] | None = None,
) -> str:
    """Get skill instructions from MCP.
    
    Handles both:
    1. Prompt-based skills - renders MCP prompt
    2. Resource-based skills - reads skill:// URI resources
    
    Args:
        name: Skill name
        arguments: Optional arguments (for prompt-based skills)
        
    Returns:
        Skill instructions content
        
    Raises:
        SkillNotFoundError: If skill not found
        ValueError: If required arguments not provided for prompt skills
    """
    # First check if it's a prompt-based skill
    prompts = await self.get_prompts()
    prompt = next((p for p in prompts if p.name == name), None)
    
    if prompt:
        return await self._get_prompt_skill_instructions(prompt, arguments)
    
    # Then check if it's a resource-based skill
    resources = await self.get_resources()
    skill_uri = f"skill://{name}/SKILL.md"
    
    for resource in resources:
        if resource.uri == skill_uri or resource.uri.endswith(f"/{name}/SKILL.md"):
            return await self._get_resource_skill_instructions(name)
    
    raise SkillNotFoundError(name)

async def _get_prompt_skill_instructions(
    self, 
    prompt, 
    arguments: dict[str, str] | None = None,
) -> str:
    """Get instructions from MCP prompt-based skill."""
    # Check if prompt has required arguments
    prompt_args = getattr(prompt, 'arguments', None) or []
    required_args = [
        arg.name for arg in prompt_args 
        if getattr(arg, 'required', False)
    ]
    
    provided_args = arguments or {}
    missing_args = [arg for arg in required_args if arg not in provided_args]
    
    if missing_args:
        return self._format_prompt_skill_template(prompt, missing_args)
    
    # Render with provided arguments
    parts = await self.get_request_parts(prompt.name, arguments=provided_args)
    
    contents: list[str] = []
    for part in parts:
        if hasattr(part, 'content'):
            contents.append(str(part.content))
    
    return "\n\n".join(contents)

async def _get_resource_skill_instructions(self, skill_name: str) -> str:
    """Get instructions from MCP resource-based skill.
    
    Supports both short form (skill://name) and explicit form (skill://name/SKILL.md).
    """
    # Try explicit SKILL.md first
    main_uri = f"skill://{skill_name}/SKILL.md"
    
    try:
        contents = await self.read_resource(main_uri)
        if contents:
            return contents[0]
    except Exception:
        pass
    
    # Fallback: try to find any resource with this skill name prefix
    # that looks like a main skill file
    resources = await self.get_resources()
    for resource in resources:
        uri = resource.uri
        # Match skill://skill-name/ where the last part could be the main file
        if uri.startswith(f"skill://{skill_name}/"):
            parts = uri[len(f"skill://{skill_name}/"):].split("/")
            if len(parts) == 1 and parts[0] not in ["_manifest", ""]:
                # Single file in skill dir, treat as main file
                try:
                    contents = await self.read_resource(uri)
                    if contents:
                        return contents[0]
                except Exception:
                    continue
    
    raise SkillNotFoundError(skill_name)

def _format_prompt_skill_template(self, prompt, missing_args: list[str]) -> str:
    """Format a prompt-based skill template when required arguments are missing."""
    lines = [
        f"# MCP Skill: {prompt.name}",
        "",
        f"{prompt.description or 'No description'}",
        "",
        "## Required Arguments",
    ]
    
    for arg_name in missing_args:
        lines.append(f"- `{arg_name}`: Required argument")
    
    lines.extend([
        "",
        "## Usage",
        f"Use this skill with: skill://{self.name}/{prompt.name}",
        "",
        "Note: This MCP prompt requires arguments to render fully.",
    ])
    
    return "\n".join(lines)

async def get_references(self, skill_name: str) -> list[ResourceInfo]:
    """Get references for a skill.
    
    Handles both:
    1. Prompt-based skills - resources with associated_skill metadata
    2. Resource-based skills - files from skill://skill-name/... URIs
    
    Args:
        skill_name: Name of the skill
        
    Returns:
        List of reference resources
    """
    resources = await self.get_resources()
    associated: list[ResourceInfo] = []
    
    for resource in resources:
        # Method 1: Explicit metadata association (prompt-based skills)
        if resource.metadata.get("associated_skill") == skill_name:
            associated.append(resource)
        
        # Method 2: Resource-based skills (skill:// URI scheme)
        # Match: skill://skill-name/file.ext (but not SKILL.md itself)
        elif resource.uri.startswith(f"skill://{skill_name}/"):
            # Exclude the main SKILL.md file
            if not resource.uri.endswith("/SKILL.md") and "SKILL.md" not in resource.uri.split("/")[-1]:
                associated.append(resource)
    
    return associated

async def read_reference(self, skill_name: str, ref_path: str) -> str:
    """Read reference content for a resource-based skill.
    
    Args:
        skill_name: Skill name
        ref_path: Path to reference file (e.g., "reference.md", "examples/sample.py")
        
    Returns:
        Reference content as string
        
    Raises:
        SkillNotFoundError: If skill not found
        ReferenceNotFoundError: If reference not found
    """
    # SECURITY: Validate reference path
    if ".." in ref_path.split("/"):
        raise SecurityError(f"Path traversal detected: {ref_path!r}")
    
    # Build resource URI
    ref_uri = f"skill://{skill_name}/{ref_path}"
    
    try:
        contents = await self.read_resource(ref_uri)
        if contents:
            return contents[0]
    except Exception as e:
        raise ReferenceNotFoundError(skill_name, ref_path) from e
    
    raise ReferenceNotFoundError(skill_name, ref_path)
```

#### 4. Skill URI Resolver

```python
# src/wolfharness/skills/uri_resolver.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import unquote

from wolfharness.skills.exceptions import SkillNotFoundError, SecurityError

if TYPE_CHECKING:
    from wolfharness.resource_providers import AggregatingResourceProvider


@dataclass(frozen=True)
class ResolvedSkillURI:
    """Parsed skill:// URI components."""
    
    provider: str
    """Provider name (e.g., "local", "github-copilot")."""
    
    skill_name: str
    """Name of the skill."""
    
    reference_path: str | None
    """Optional path to reference (None for main skill)."""
    
    @classmethod
    def parse(cls, uri: str) -> ResolvedSkillURI:
        """Parse a skill:// URI with validation.
        
        Format: skill://{provider}/{skill-name}/{reference-path}
        
        Args:
            uri: URI string to parse
            
        Returns:
            ResolvedSkillURI with parsed components
            
        Raises:
            ValueError: If URI is invalid
            SecurityError: If path traversal detected
        """
        if not uri.startswith("skill://"):
            raise ValueError(f"Not a skill URI: {uri!r}")
        
        # Remove scheme
        path = uri[8:]
        
        # Split into components (max 2 splits for reference path)
        parts = path.split("/", 2)
        
        if len(parts) < 2:
            raise ValueError(f"Invalid skill URI (need provider and name): {uri!r}")
        
        provider = unquote(parts[0])
        skill_name = unquote(parts[1])
        reference_path = unquote(parts[2]) if len(parts) > 2 else None
        
        # Validate provider name
        if not provider or not _is_valid_provider_name(provider):
            raise ValueError(f"Invalid provider name: {provider!r}")
        
        # Validate skill name
        if not skill_name:
            raise ValueError(f"Empty skill name in URI: {uri!r}")
        
        # SECURITY: Check for path traversal in reference path
        if reference_path:
            if ".." in reference_path.split("/"):
                raise SecurityError(f"Path traversal detected in URI: {uri!r}")
            # Check for null bytes
            if "\x00" in reference_path:
                raise SecurityError(f"Null byte in reference path: {uri!r}")
        
        return cls(
            provider=provider,
            skill_name=skill_name,
            reference_path=reference_path,
        )
    
    def __str__(self) -> str:
        """Reconstruct URI string."""
        uri = f"skill://{self.provider}/{self.skill_name}"
        if self.reference_path:
            uri = f"{uri}/{self.reference_path}"
        return uri


def _is_valid_provider_name(name: str) -> bool:
    """Check if provider name is valid.
    
    Rules:
    - Must start with alphanumeric
    - Can contain alphanumeric, hyphen, underscore
    - Max 63 characters
    """
    if not name or len(name) > 63:
        return False
    if not name[0].isalnum():
        return False
    return all(c.isalnum() or c in "-_" for c in name)


class SkillURIResolver:
    """Resolves skill:// URIs using an AggregatingResourceProvider.
    
    Handles skill name resolution with priority ordering for collision detection.
    """
    
    def __init__(
        self, 
        provider: AggregatingResourceProvider,
        provider_priority: list[str] | None = None,
    ) -> None:
        """Initialize resolver with provider.
        
        Args:
            provider: Aggregating provider containing all skill sources
            provider_priority: Ordered list of provider names for collision resolution.
                Providers earlier in the list have higher priority.
                Default: ["local", ...other providers in registration order]
        """
        self._provider = provider
        self._provider_priority = provider_priority or self._default_priority()
    
    def _default_priority(self) -> list[str]:
        """Generate default priority based on provider registration order.
        
        Local provider always has highest priority.
        """
        priority = ["local"]
        for p in self._provider.providers:
            if p.name != "local" and p.name not in priority:
                priority.append(p.name)
        return priority
    
    def _get_priority(self, provider_name: str) -> int:
        """Get priority rank for a provider (lower = higher priority)."""
        try:
            return self._provider_priority.index(provider_name)
        except ValueError:
            return len(self._provider_priority)  # Lowest priority if not in list
    
    async def resolve(self, uri: str) -> ResolvedSkill:
        """Resolve a skill:// URI to skill content.
        
        Supports both short and full URI formats:
        - skill://provider/skill-name (main skill)
        - skill://provider/skill-name/SKILL.md (main skill, explicit)
        - skill://provider/skill-name/references/file.md (reference)
        - skill://provider/skill-name/subdir/file (nested reference)
        
        Args:
            uri: skill:// URI or bare skill name
            
        Returns:
            Resolved skill with content
            
        Raises:
            SkillNotFoundError: If skill or reference not found
            SecurityError: If URI validation fails
        """
        # Handle bare names
        if not uri.startswith("skill://"):
            return await self._resolve_by_name(uri)
        
        # Parse URI
        parsed = ResolvedSkillURI.parse(uri)
        
        # Find provider
        provider = self._find_provider(parsed.provider)
        if not provider:
            raise SkillNotFoundError(
                parsed.skill_name,
                available=await self._list_available_skills()
            )
        
        # Resolve based on URI type
        if parsed.reference_path:
            # Check if this is the main skill file (SKILL.md or explicit short form)
            if parsed.reference_path == "SKILL.md" or not self._is_reference_path(parsed):
                # Treat as main skill request
                instructions = await provider.get_skill_instructions(parsed.skill_name)
                skill = await provider.get_skill(parsed.skill_name)
                
                if not skill:
                    raise SkillNotFoundError(parsed.skill_name)
                
                return ResolvedSkill(
                    name=skill.name,
                    content=instructions,
                    description=skill.description,
                    metadata=skill.metadata,
                    is_reference=False,
                    provider=parsed.provider,
                )
            else:
                # Reference request
                content = await provider.read_reference(
                    parsed.skill_name,
                    parsed.reference_path,
                )
                return ResolvedSkill(
                    name=parsed.skill_name,
                    content=content,
                    is_reference=True,
                    reference_path=parsed.reference_path,
                    provider=parsed.provider,
                )
        else:
            # Short form: skill://provider/skill-name
            # Resolve to main skill
            instructions = await provider.get_skill_instructions(parsed.skill_name)
            skill = await provider.get_skill(parsed.skill_name)
            
            if not skill:
                raise SkillNotFoundError(parsed.skill_name)
            
            return ResolvedSkill(
                name=skill.name,
                content=instructions,
                description=skill.description,
                metadata=skill.metadata,
                is_reference=False,
                provider=parsed.provider,
            )
    
    def _is_reference_path(self, parsed: ResolvedSkillURI) -> bool:
        """Determine if reference_path points to a reference or main skill.
        
        Heuristics:
        - Explicit SKILL.md → main skill (even with subdirs)
        - references/ prefix → reference
        - _manifest → manifest (metadata)
        - Other paths → reference (supporting files)
        
        Args:
            parsed: Parsed URI components
            
        Returns:
            True if reference_path points to a reference file
        """
        if not parsed.reference_path:
            return False
        
        path = parsed.reference_path
        
        # Explicit main skill file
        if path == "SKILL.md" or path.endswith("/SKILL.md"):
            return False
        
        # Manifest file (metadata)
        if path == "_manifest" or path.endswith("/_manifest"):
            return False
        
        # Everything else is a reference
        return True
    
    async def _resolve_by_name(self, name: str) -> ResolvedSkill:
        """Resolve bare skill name across all providers with priority handling.
        
        When multiple providers have skills with the same name, uses provider
        priority order (local > MCP in registration order).
        
        Args:
            name: Skill name to resolve
            
        Returns:
            Resolved skill from highest-priority provider
            
        Raises:
            SkillNotFoundError: If skill not found in any provider
        """
        # Get all skills matching name across providers
        matches: list[tuple[ResolvedSkill, int]] = []  # (skill, priority)
        
        for provider in self._provider.providers:
            try:
                skill = await provider.get_skill(name)
                if skill:
                    instructions = await provider.get_skill_instructions(name)
                    resolved = ResolvedSkill(
                        name=skill.name,
                        content=instructions,
                        description=skill.description,
                        metadata=skill.metadata,
                        provider=provider.name,
                    )
                    priority = self._get_priority(provider.name)
                    matches.append((resolved, priority))
            except Exception as e:
                # Log but continue - do not fail entire resolution for one provider
                logger.debug(
                    "Provider failed to resolve skill",
                    provider=provider.name,
                    skill=name,
                    error=str(e),
                )
                continue
        
        if not matches:
            # Not found in any provider
            available = await self._list_available_skills()
            raise SkillNotFoundError(name, available)
        
        # Sort by priority and return highest (lowest number)
        matches.sort(key=lambda x: x[1])
        
        # Log if there were collisions
        if len(matches) > 1:
            logger.info(
                "Skill name collision resolved by priority",
                skill=name,
                selected_provider=matches[0][0].provider,
                all_providers=[m.provider for m, _ in matches],
            )
        
        return matches[0][0]
    
    def _find_provider(self, name: str):
        """Find provider by name."""
        for provider in self._provider.providers:
            if provider.name == name:
                return provider
        return None
    
    async def _list_available_skills(self) -> list[str]:
        """List all available skill names."""
        names: list[str] = []
        for provider in self._provider.providers:
            try:
                skills = await provider.get_skills()
                names.extend(s.name for s in skills)
            except Exception:
                continue
        return names


@dataclass
class ResolvedSkill:
    """Result of resolving a skill URI."""
    
    name: str
    content: str
    description: str = ""
    metadata: dict | None = None
    provider: str = ""
    is_reference: bool = False
    reference_path: str | None = None
```

#### 5. Updated load_skill Tool

```python
# src/wolfharness_toolsets/builtin/skills.py

from __future__ import annotations

from typing import Literal

from wolfharness.agents.context import AgentContext
from wolfharness.resource_providers import StaticResourceProvider
from wolfharness.skills.exceptions import SkillNotFoundError, SecurityError, ReferenceNotFoundError


SKILL_USAGE_GUIDANCE = """
## Skill URI Format

Skills can be loaded using skill:// URIs or bare names:

### Short Name (Auto-Route)
```python
await load_skill(ctx, "python-expert")
```
Tries all providers in priority order (local first, then MCP).

### Full URI (Explicit Provider)
```python
await load_skill(ctx, "skill://local/python-expert")
await load_skill(ctx, "skill://github-copilot/code-review")
```
Explicitly selects provider. Required when multiple providers have same-named skills.

### MCP Resource-Based Skills (Short Form)
```python
# Short form - automatically resolves to main skill file
await load_skill(ctx, "skill://skills-server/pdf-processing")

# Explicit SKILL.md form (equivalent to short form)
await load_skill(ctx, "skill://skills-server/pdf-processing/SKILL.md")
```
Works with FastMCP Skills Provider protocol. Short form automatically resolves to the main skill file.

### Reference Content
```python
# Local skill references
await load_skill(ctx, "skill://local/python-expert/references/style-guide.md")

# MCP resource-based skill references
await load_skill(ctx, "skill://skills-server/pdf-processing/examples/sample.pdf")
```
Loads bundled reference files from the skill's references directory.

### URI Format
```
skill://{provider}/{skill-name}                    # Short form - main skill
skill://{provider}/{skill-name}/SKILL.md           # Explicit main file
skill://{provider}/{skill-name}/references/file    # Reference files
skill://{provider}/{skill-name}/subdir/file        # Nested references
```

- `provider`: Skill source (e.g., "local", "github-copilot", "skills-server")
- `skill-name`: Name of the skill
- `reference-path`: Optional path to bundled content (omit for main skill)

### Argument Substitution
Skills support bash-style variable substitution:
- `$1`, `$2`, ... - Positional arguments
- `$@` or `$ARGUMENTS` - All arguments
"""


async def load_skill(
    ctx: AgentContext,
    skill_name: str,
    arguments: str | None = None,
) -> str:
    """Load a Claude Code Skill and return its instructions.

    Skills can be loaded by short name (auto-routes across providers) or by
    full skill:// URI for explicit provider selection.
    
    """ + SKILL_USAGE_GUIDANCE + """

    Args:
        ctx: Agent context providing access to pool and skills
        skill_name: Name of the skill to load, or a skill:// URI
        arguments: Optional arguments for variable substitution ($1, $2, $@)

    Returns:
        The skill instructions or reference content
    """
    from wolfharness.log import get_logger
    
    logger = get_logger(__name__)
    
    logger.info(
        "Loading skill",
        skill_name=skill_name,
        arguments=arguments,
        has_pool=ctx.pool is not None,
    )

    if ctx.pool is None:
        return "No agent pool available - skills require pool context"

    # Get resolver from pool
    resolver = ctx.pool.skill_resolver
    
    try:
        # Resolve URI
        resolved = await resolver.resolve(skill_name)
        
        # Process arguments if provided
        if arguments and not resolved.is_reference:
            resolved.content = _substitute_arguments(resolved.content, arguments)
        
        # Format response
        if resolved.is_reference:
            header = f"# Reference: {resolved.reference_path}\n"
            header += f"From skill: {resolved.name} (provider: {resolved.provider})"
            return f"{header}\n\n{resolved.content}"
        
        # Full skill response
        header = f"# {resolved.name}\n\n{resolved.description}"
        if resolved.metadata:
            meta_lines = []
            if resolved.metadata.get("license"):
                meta_lines.append(f"License: {resolved.metadata['license']}")
            if resolved.metadata.get("compatibility"):
                meta_lines.append(f"Compatibility: {resolved.metadata['compatibility']}")
            if meta_lines:
                header += "\n\n" + "\n".join(meta_lines)
        
        return f"{header}\n\n{resolved.content}"
        
    except SkillNotFoundError as e:
        return f"Skill not found. {e}"
    except ReferenceNotFoundError as e:
        return f"Reference not found: {e}"
    except SecurityError as e:
        return f"Security error: {e}"
    except Exception as e:
        logger.exception("Failed to load skill", skill_name=skill_name, error=e)
        return f"Failed to load skill {skill_name!r}: {e}"


def _substitute_arguments(content: str, arguments: str) -> str:
    """Substitute bash-style variables in content.
    
    Args:
        content: Skill instructions with placeholders
        arguments: Arguments string to substitute
        
    Returns:
        Content with substitutions applied
    """
    args_list = arguments.split()
    result = content
    
    # Replace positional args ($1, $2, ...)
    for i, arg in enumerate(args_list, 1):
        result = result.replace(f"${i}", arg)
    
    # Replace special variables
    result = result.replace("$@", arguments)
    result = result.replace("$ARGUMENTS", arguments)
    
    return result


async def list_skills(ctx: AgentContext) -> str:
    """List all available skills from all providers.
    
    Returns:
        Formatted list with provider information
    """
    if ctx.pool is None:
        return "No agent pool available"
    
    resolver = ctx.pool.skill_resolver
    provider = resolver._provider
    
    lines = ["Available skills:", ""]
    
    for prov in provider.providers:
        try:
            skills = await prov.get_skills()
            if skills:
                lines.append(f"## {prov.name} ({len(skills)} skills)")
                for skill in skills:
                    lines.append(f"- **{skill.name}**: {skill.description}")
                    lines.append(f"  URI: `skill://{prov.name}/{skill.name}`")
                lines.append("")
        except Exception as e:
            lines.append(f"## {prov.name} (error: {e})")
            lines.append("")
    
    return "\n".join(lines)


class SkillsTools(StaticResourceProvider):
    """Provider for skills and commands tools."""

    def __init__(
        self,
        name: str = "skills",
        *,
        injection_mode: Literal["off", "metadata", "full"] | None = None,
        max_skills: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.injection_mode = injection_mode
        self.max_skills = max_skills
        self._tools = [
            self.create_tool(load_skill, category="read", read_only=True, idempotent=True),
            self.create_tool(list_skills, category="read", read_only=True, idempotent=True),
        ]
```

### Integration with AgentPool

```python
# src/wolfharness/delegation/pool.py

class AgentPool:
    """Updated AgentPool with unified skill access via ResourceProvider."""
    
    async def _setup_skills_provider(self) -> None:
        """Setup skill resource provider integration.
        
        Integrates with existing SkillsManager to avoid parallel skill systems.
        Creates LocalResourceProvider within SkillsManager, then combines with
        MCP providers via AggregatingResourceProvider.
        """
        from wolfharness.resource_providers.local import LocalResourceProvider
        from wolfharness.resource_providers import AggregatingResourceProvider
        from wolfharness.skills.uri_resolver import SkillURIResolver
        
        # 1. Extend existing SkillsManager with ResourceProvider
        if hasattr(self, 'skills') and self.skills:
            # Create LocalResourceProvider within SkillsManager
            local_provider = LocalResourceProvider(
                name="local",
                skills_dirs=self.skills.config.get_effective_paths() 
                if hasattr(self.skills, 'config') else None,
            )
            await self.exit_stack.enter_async_context(local_provider)
            
            # Store on SkillsManager for unified access
            self.skills._resource_provider = local_provider
        
        # 2. Collect all skill providers
        providers: list[ResourceProvider] = []
        
        # Local skills (if SkillsManager exists)
        if hasattr(self, 'skills') and self.skills:
            providers.append(self.skills.resource_provider)
        
        # MCP servers as skill providers
        if hasattr(self, 'mcp') and self.mcp:
            for mcp_provider in self.mcp.get_providers():
                # Extend existing MCP providers with skill methods
                # They already have get_prompts() which we map to skills
                providers.append(mcp_provider)
        
        # 3. Create aggregating provider with priority order
        # Order matters: local skills have priority over MCP
        self._skill_provider = AggregatingResourceProvider(
            providers=providers,
            name="skills",
        )
        
        # 4. Create URI resolver with collision handling
        self._skill_resolver = SkillURIResolver(
            self._skill_provider,
            provider_priority=["local"] + [
                p.name for p in providers 
                if p.name != "local"
            ],
        )
        
        # 5. Connect to SkillCommandRegistry for slash commands
        self._skill_provider.skills_changed.connect(self._on_skills_changed)
    
    def _on_skills_changed(self, event: ResourceChangeEvent) -> None:
        """Handle skill changes from any provider."""
        # Forward to SkillCommandRegistry for slash command updates
        if hasattr(self, '_skill_command_registry'):
            self._skill_command_registry.handle_resource_change(event)
    
    @property
    def skill_resolver(self) -> SkillURIResolver:
        """Get skill URI resolver."""
        return self._skill_resolver
    
    @property
    def skill_provider(self) -> AggregatingResourceProvider:
        """Get aggregating skill provider."""
        return self._skill_provider
```

### SkillsManager Extension

```python
# src/wolfharness/skills/manager.py (extension)

class SkillsManager:
    """Extended to expose ResourceProvider interface."""
    
    async def __aenter__(self) -> Self:
        """Enter async context."""
        await self.registry.__aenter__()
        
        # Create and enter LocalResourceProvider
        from wolfharness.resource_providers.local import LocalResourceProvider
        
        self._resource_provider = LocalResourceProvider(
            name="local",
            skills_dirs=self.config.get_effective_paths() if self.config else None,
        )
        await self.exit_stack.enter_async_context(self._resource_provider)
        
        return self
    
    @property
    def resource_provider(self) -> LocalResourceProvider:
        """Get the ResourceProvider for local skills."""
        if not hasattr(self, '_resource_provider'):
            raise RuntimeError("SkillsManager not entered as context manager")
        return self._resource_provider
```

## URI Specification (Appendix A)

### Format

```
skill://{provider}/{skill-name}/{reference-path}
```

### Components

| Component | Format | Example |
|-----------|--------|---------|
| `provider` | `[a-z0-9][a-z0-9-_]{0,62}` | `local`, `github-copilot` |
| `skill-name` | Skill-defined | `python-expert` |
| `reference-path` | URL-encoded path | `references/style-guide.md` |

### Validation Rules

1. **Provider Name**:
   - Must start with alphanumeric
   - Can contain alphanumeric, hyphen, underscore
   - Max 63 characters
   - Case-sensitive

2. **Skill Name**:
   - Non-empty string
   - URL-encoded if contains special characters

3. **Reference Path**:
   - Cannot contain `..` (path traversal protection)
   - Cannot contain null bytes
   - Forward slashes only
   - Max 4096 characters

### Examples

```
# Local filesystem skills
skill://local/python-expert
skill://local/python-expert/references/style-guide.md

# MCP prompt-based skills
skill://github-copilot/code-review
skill://my-mcp/custom-prompt

# MCP resource-based skills (FastMCP Skills Provider)
skill://skills-server/pdf-processing/SKILL.md
skill://skills-server/pdf-processing/_manifest
skill://skills-server/pdf-processing/examples/sample.pdf

# With encoding
skill://local/my%20skill/references/file%20with%20spaces.md
```

### MCP Skills via Resources

When an MCP server uses the [FastMCP Skills Provider protocol](https://gofastmcp.com/servers/providers/skills), skills are exposed as resources with the following URI patterns:

| URI Pattern | Purpose | Example |
|-------------|---------|---------|
| `skill://{server}/{skill}` | Short form - resolves to main skill | `skill://skills/pdf` |
| `skill://{server}/{skill}/SKILL.md` | Main instruction file (explicit) | `skill://skills/pdf/SKILL.md` |
| `skill://{server}/{skill}/_manifest` | JSON manifest with file list | `skill://skills/pdf/_manifest` |
| `skill://{server}/{skill}/{ref}` | Reference/supporting files | `skill://skills/pdf/examples/doc.pdf` |

**Short Form Resolution**: The URI `skill://{provider}/{skill-name}` (without path) automatically resolves to the main skill file (SKILL.md for local skills, or the primary skill resource for MCP skills). This provides a consistent shorthand across all skill types.

## Security Considerations

### Path Traversal Protection

1. **Validation Order**:
   - Decode URI components first
   - Check for `..` in path parts
   - Resolve to absolute path
   - Verify path is within allowed directory using `Path.relative_to()`

2. **Symlink Handling**:
   - Resolve symlinks before validation
   - Final path must still be within allowed directory

### Argument Injection

- Arguments are substituted into skill content
- No shell execution occurs (content is instructions, not commands)
- Consider adding escaping for special XML characters if needed

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

1. **Exception classes** (`src/wolfharness/skills/exceptions.py`)
2. **URI resolver** (`src/wolfharness/skills/uri_resolver.py`)
3. **LocalResourceProvider** (`src/wolfharness/resource_providers/local.py`)
4. **Tests** for URI parsing and resolution

### Phase 2: MCP Integration (Week 2)

1. **Extend MCPResourceProvider** with `get_skills()`, `get_references()`
2. **Add MCP skill mapping** logic
3. **Tests** with mock MCP server

### Phase 3: Integration (Week 3)

1. **AgentPool integration** with `AggregatingResourceProvider`
2. **Update load_skill** tool with URI support
3. **SkillCommandRegistry** integration
4. **Protocol bridges** update

### Phase 4: Documentation & Polish (Week 4)

1. **Documentation** update
2. **Performance testing**
3. **Security audit**
4. **Examples** and migration guide

## Review History

Based on comprehensive review by Metis, Momus, and Oracle during initial development:

### Critical Fixes

1. **Async/Sync Lifecycle Fix** (Oracle)
   - Fixed `SkillsRegistry.discover_skills()` sync/async mismatch
   - Now uses `anyio.to_thread.run_sync()` for proper async context
   - Added registry callback connection to ResourceProvider signals

2. **Base Class Method Addition** (Oracle)
   - Added `get_skill()` to `ResourceProvider` base class
   - Added `get_references()` and `read_reference()` to base class
   - Enables polymorphic usage across all providers

3. **SkillsManager Integration** (Oracle)
   - Revised AgentPool integration to use existing SkillsManager
   - Avoids parallel skill systems
   - LocalResourceProvider created within SkillsManager lifecycle

### Quality Improvements

4. **Skill Name Collision Handling** (Metis)
   - Added `provider_priority` configuration to SkillURIResolver
   - Defines resolution order: local > MCP (registration order)
   - Logs collisions with selected provider

5. **Caching Strategy** (Metis)
   - Added LRU-style TTL caching to LocalResourceProvider
   - Default 60-second TTL for skill listings
   - Cache invalidation on skill changes

6. **MCP Prompt Argument Preservation** (Metis)
   - Extended MCP skill mapping to preserve argument schemas
   - Added `argument_schema` and `has_required_args` metadata
   - Added skill template formatting for prompts with required args

### MCP Skills via Resources Support (v4.2)

7. **FastMCP Skills Provider Protocol** (New in v4.2)
   - Full support for MCP skills exposed via resources using `skill://` URI scheme
   - Automatic detection of skill resources (skill://skill-name/SKILL.md)
   - Manifest file support (_manifest) for file discovery
   - Reference file access for skill resources
   - Works with FastMCP's SkillsDirectoryProvider and vendor providers
   - See: https://gofastmcp.com/servers/providers/skills

### Code Quality

7. **Error Handling** (Metis)
   - Added logging for provider failures in resolution
   - Debug logging for skill name collisions
   - Graceful degradation when individual providers fail

8. **Documentation** (Momus)
   - Added integration examples for SkillsManager extension
   - Clarified provider lifecycle management
   - Documented collision resolution strategy

## Decision Record

**Status**: IMPLEMENTED

**Summary**: This RFC specifies the MCP Skills Resources Provider Protocol implementation for AgentPool. It enables consumption of MCP-exposed skills via the `skill://` URI scheme, supporting both prompt-based and resource-based skill types.

**Key Decisions**:
1. ✅ Use ResourceProvider extension pattern for skill access
2. ✅ Support dual MCP skill types: prompts AND resources (FastMCP Skills Provider)
3. ✅ Use `skill://provider/skill-name` URI scheme for unified access
4. ✅ Implement proper path traversal protection with `Path.relative_to()`
5. ✅ Define complete exception hierarchy (SkillError and subclasses)
6. ✅ Add LRU caching with TTL for performance
7. ✅ Integrate with existing SkillsManager to avoid parallel systems
8. ✅ Define provider priority for skill name collision resolution
9. ✅ Maintain backward compatibility with existing skill tools

**Relationship with RFC-0016**:
- RFC-0020 (this): MCP Skills Resources Provider Protocol implementation
- RFC-0016: Slash command architecture for skill invocation
- Together: Complete skill infrastructure (access + commands)

**Implementation Completed**:
- [x] Architecture review passed (Oracle)
- [x] RFC quality review passed (Momus)
- [x] Implementation readiness review passed (Metis)
- [x] Exception hierarchy (`src/wolfharness/skills/exceptions.py`)
- [x] URI resolver (`src/wolfharness/skills/uri_resolver.py`)
- [x] LocalResourceProvider (`src/wolfharness/resource_providers/local.py`)
- [x] MCPResourceProvider skill methods
- [x] AggregatingResourceProvider skill aggregation
- [x] Updated load_skill tool with URI support
- [x] AgentPool integration with skill resolver
- [x] Documentation and examples
- [x] Security audit with path traversal protection

**Documentation**:
- [Skill URI Usage](../../how-to/configuration/skill-uri-usage.md) - Complete usage guide
- [Skill URI Loading Example](../../../examples/skill_uri_loading/) - Loading skills by URI
- [Skills with References Example](../../../examples/skill_with_references/) - Creating skills with references
- [MCP Skills Example](../../../examples/mcp_skills/) - Using MCP-exposed skills
