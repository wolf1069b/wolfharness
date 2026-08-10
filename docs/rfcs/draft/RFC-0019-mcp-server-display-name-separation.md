---
rfc_id: RFC-0019
title: MCP Server Display Name Separation from Client ID
status: ACCEPTED
author: AgentPool Team
created: 2026-04-01
last_updated: 2026-04-02
---

## Overview

Currently, MCP servers in AgentPool are displayed with auto-generated identifiers like `pool_mcp_streamable_http_http://10.147.254.3:8721/mcp` instead of using user-configured friendly names. This RFC proposes separating the internal `client_id` (used for unique identification) from the `display_name` (used for UI presentation), allowing users to configure meaningful names while maintaining system stability.

## Background & Context

### Current Implementation

The MCP server naming follows this chain:

1. **MCPManager initialization** (`src/wolfharness/delegation/pool.py:149`):
   ```python
   self.mcp = MCPManager(name="pool_mcp", servers=servers, owner="pool")
   ```

2. **Provider name construction** (`src/wolfharness/mcp_server/manager.py:137`):
   ```python
   name=f"{self.name}_{config.client_id}"
   ```

3. **Client ID generation** (`src/wolfharness_config/mcp_server.py`):
   - StreamableHTTP: `f"streamable_http_{self.url}"`
   - SSE: `f"sse_{self.url}"`
   - Stdio: `f"{self.command}_{args}"`

4. **Result**: Names like `pool_mcp_streamable_http_http://10.147.254.3:8721/mcp`

### Problem Statement

1. **Poor User Experience**: Auto-generated URLs are hard to read and remember
2. **Configuration Ignored**: The `name` field in config exists but is not used for display
3. **Inconsistent Behavior**: Comment in code acknowledges this limitation: `# Note: client_id is auto-generated from command/url, custom names not supported`

### Glossary

- **client_id**: Unique internal identifier for MCP server connections
- **display_name**: Human-friendly name shown in UI/TUI
- **MCPManager**: Manages lifecycle of MCP server connections
- **MCPResourceProvider**: Wraps an MCP server for tool/resource access

## Goals & Non-Goals

### Goals

- Allow user-defined names to be displayed in OpenCode TUI and other UIs
- Maintain backward compatibility with existing configurations
- Preserve unique identification for internal operations
- Support both configured names and auto-generated fallbacks

### Non-Goals

- Changing the connection/identification mechanism
- Modifying how servers are looked up internally
- Supporting duplicate display names (uniqueness not required for display)
- Renaming existing connected servers dynamically

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Backward Compatibility | High | Must not break existing configurations |
| Implementation Complexity | Medium | Should be a focused, low-risk change |
| User Experience | High | Names should be clear and intuitive |
| Code Maintainability | Medium | Solution should not add significant complexity |
| Test Coverage | High | Must include tests for edge cases |

## Options Analysis

### Option 1: Modify client_id to Return name When Available

**Description**: Change the `client_id` property to return `self.name` if set, otherwise fall back to auto-generated ID.

**Advantages**:
- Simple implementation (single property change per config type)
- Immediate display improvement
- No new abstractions needed

**Disadvantages**:
- Violates single responsibility: `client_id` becomes both identifier and display name
- Risk of breaking internal lookups if names change or conflict
- May cause confusion if two servers have the same display name
- Changes behavior of an existing property

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Backward Compatibility | ⚠️ Medium | Changes existing property semantics |
| Implementation Complexity | ✅ High | Minimal code changes |
| User Experience | ✅ High | Immediate improvement |
| Code Maintainability | ❌ Low | Mixes concerns |

### Option 2: Separate display_name Property (Recommended)

**Description**: Keep `client_id` unchanged for internal use, add a new `display_name` property that returns `name or client_id`.

**Advantages**:
- Clear separation of concerns
- `client_id` remains stable and unique
- `display_name` can change without affecting connections
- Backward compatible: default behavior unchanged
- Easy to reason about: display is presentation-layer concern

**Disadvantages**:
- Requires updates in multiple places (config, manager, routes)
- Slightly more code to maintain

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Backward Compatibility | ✅ High | No changes to existing logic |
| Implementation Complexity | ✅ High | Straightforward changes |
| User Experience | ✅ High | Clean, intuitive names |
| Code Maintainability | ✅ High | Clear separation of concerns |

### Option 3: Store Display Name in MCPResourceProvider

**Description**: Pass both `client_id` and `display_name` to MCPResourceProvider, store display name separately.

**Advantages**:
- Display name available at provider level
- Could support dynamic renaming

**Disadvantages**:
- Requires changes to MCPResourceProvider constructor
- More invasive than necessary
- Over-engineering for current use case

**Evaluation Against Criteria**:
| Criterion | Score | Notes |
|-----------|-------|-------|
| Backward Compatibility | ⚠️ Medium | Constructor changes |
| Implementation Complexity | ❌ Low | More invasive |
| User Experience | ✅ High | Good UX |
| Code Maintainability | ⚠️ Medium | Adds complexity |

## Recommendation

**Adopt Option 2: Separate display_name Property**

### Justification

- **Best Balance**: Highest overall score across criteria
- **Clean Architecture**: Maintains separation between identity and presentation
- **Safe Evolution**: No risk of breaking existing functionality
- **Future-Proof**: Allows future enhancements (dynamic renaming, localized names, etc.)

### Acknowledged Trade-offs

- Slightly more code than Option 1, but significantly more maintainable
- Display names are not guaranteed unique (acceptable for presentation layer)

## Technical Design

### Changes Required

#### 1. Config Layer (`src/wolfharness_config/mcp_server.py`)

Add `display_name` property to base or each config class:

```python
@property
def display_name(self) -> str:
    """Return the display name for UI presentation.
    
    Returns the configured name if available, otherwise falls back
    to the auto-generated client_id.
    """
    return self.name or self.client_id
```

Apply to:
- `StdioMCPServerConfig`
- `SSEMCPServerConfig`
- `StreamableHTTPMCPServerConfig`

#### 2. Manager Layer (`src/wolfharness/mcp_server/manager.py`)

Update provider name construction (line 137):

```python
# Before
name=f"{self.name}_{config.client_id}"

# After
name=f"{self.name}_{config.display_name}"
```

#### 3. API Layer (`src/wolfharness_server/opencode_server/routes/agent_routes.py`)

Update MCP status response (line 178):

```python
# Before
return MCPStatus(name=config.client_id, status="connected")

# After
return MCPStatus(name=config.display_name, status="connected")
```

Internal lookup (line 193) **remains unchanged**:
```python
config = next((s for s in manager.servers if s.client_id == name), None)
```

#### 4. Update Comment (line 149)

Remove or update the comment indicating custom names are not supported.

### Configuration Examples

**With custom name**:
```yaml
mcp_servers:
  - name: "文件系统"
    type: streamable_http
    url: http://localhost:8080/mcp
# Display: pool_mcp_文件系统
```

**Without custom name (fallback)**:
```yaml
mcp_servers:
  - type: streamable_http
    url: http://10.147.254.3:8721/mcp
# Display: pool_mcp_streamable_http_http://10.147.254.3:8721/mcp
```

## Implementation Plan

### Phase 1: Core Changes

1. Add `display_name` property to config classes
2. Update MCPManager to use `display_name` for provider naming
3. Update agent_routes.py to use `display_name` in responses
4. Update/remove outdated comments

### Phase 2: Testing

1. Unit tests for `display_name` property (all three config types)
2. Integration tests for MCP status endpoint
3. Backward compatibility tests (configs without name field)

### Phase 3: Documentation

1. Update configuration documentation
2. Add examples showing custom names
3. Update CHANGELOG

### Rollback Strategy

- Changes are additive only (new property)
- Can revert by changing `display_name` back to `client_id` in usage sites
- No database or persistent state changes

## Open Questions

1. **Should we validate display name uniqueness?**
   - Recommendation: No, display names are presentation-only
   - Internal operations use `client_id` which remains unique

2. **How to handle special characters in display names?**
   - Current: Pass through as-is
   - Consider: URL-encoding or slugification if needed for certain UIs

3. **Should this apply to other protocols (ACP, AG-UI)?**
   - Out of scope for this RFC
   - Can be addressed in follow-up if needed

## Decision Record

**Decision**: ACCEPTED - Option 2 (Separate display_name Property)

**Implementation Summary**:
- Added `display_name` property to `BaseMCPServerConfig` class
- Property returns `self.name.strip() if self.name and self.name.strip() else self.client_id`
- Updated `MCPManager` to use `display_name` for provider naming
- Updated API response to include `display_name` field alongside existing `name` field
- Added comprehensive unit tests (15 tests) and integration tests (7 tests)
- All tests pass, backward compatibility maintained

**Date**: 2026-04-02

---

**Reviewers**: Atlas (Orchestrator)
**Target Completion**: 2026-04-02
