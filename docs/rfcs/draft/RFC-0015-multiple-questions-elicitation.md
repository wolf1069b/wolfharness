---
rfc_id: RFC-0015
title: Multi-Question Elicitation Support for OpenCode Server
status: APPROVED
author: Xeno-Agent Team
reviewers:
created: 2026-03-12
last_updated: 2026-03-12
decision_date: 2026-03-12
related_rfcs:
  - RFC-0010: Multi-Question Tool for User (xeno-agent)
  - RFC-0008: Dynamic Skills Injection
references:
  - /packages/agent-client-protocol/docs/rfds/elicitation.mdx
  - /packages/xeno-agent/docs/rfcs/RFC-0010-multi-question-tool-for-user.md
---

# RFC-0015: Multi-Question Elicitation Support for OpenCode Server

## Overview

This RFC proposes extending wolfharness's OpenCode Server to support multi-question elicitation through the MCP Elicitation protocol. Currently, the `OpenCodeInputProvider` only supports single-question elicitation with enum/array schemas. This limitation prevents tools like `question_for_user` (defined in RFC-0010) from presenting multiple questions in a single interaction.

The proposal involves extending `_handle_question_elicitation` to support `object` type schemas with multiple properties, where each property represents a separate question.

## Table of Contents

- Background & Context
- [Problem Statement](#problem-statement)
- Goals & Non-Goals
- [Technical Design](#technical-design)
- Implementation Plan
- [Open Questions](#open-questions)
- [References](#references)

---

## Background & Context

### Current State

The wolfharness OpenCode Server (`wolfharness_server/opencode_server/`) provides user interaction capabilities through two main input providers:

1. **OpenCodeInputProvider**: Handles elicitation via OpenCode protocol (SSE events)
2. **ACPInputProvider**: Handles elicitation via ACP protocol (permission requests)

The current `OpenCodeInputProvider._handle_question_elicitation()` implementation:

```python
# Current limitation (wolfharness_server/opencode_server/input_provider.py)
async def _handle_question_elicitation(self, params, schema):
    match schema:
        case {"type": "array", "items": {"enum": [...]}}:  # Multi-select single question
            is_multi = True
        case {"enum": [...]}:  # Single-select single question
            is_multi = False
        case _:
            return types.ElicitResult(action="decline")  # ❌ Object schema not supported

    # Creates SINGLE QuestionInfo
    question_info = QuestionInfo(...)
    self.state.pending_questions[question_id] = PendingQuestion(
        questions=[question_info],  # ❌ Hard-coded single question
        ...
    )
```

### Data Model (Already Multi-Question Ready)

Importantly, the OpenCode data models already support multiple questions:

```python
# wolfharness_server/opencode_server/models/question.py
class QuestionRequest:
    questions: list[QuestionInfo]  # ✅ Already supports list

class PendingQuestion:
    questions: list[QuestionInfo]  # ✅ Already supports list
    future: asyncio.Future[list[list[str]]]  # ✅ Answers indexed by question

class QuestionReply:
    answers: list[list[str]]  # ✅ answers[i] for questions[i]
```

### Related Work

RFC-0010 (xeno-agent) defines a `question_for_user` tool that requires this capability:

```xml
<!-- Example questionnaire requiring multi-question support -->
<question header="Model" type="enum">
  <text>Equipment model?</text>
  <suggest>SY215C</suggest>
</question>
<question header="Symptoms" type="multi">
  <text>Select symptoms:</text>
  <suggest>Black smoke</suggest>
</question>
```

This translates to MCP Elicit `requestedSchema`:

```json
{
  "type": "object",
  "properties": {
    "q0": {"type": "string", "enum": ["SY215C"]},
    "q1": {"type": "array", "items": {"enum": ["Black smoke"]}}
  }
}
```

---

## Problem Statement

### The Problem

When a tool sends an elicitation request with an object schema containing multiple properties (representing multiple questions), the current implementation rejects it with `action="decline"`.

### Evidence

1. **Code analysis**: `_handle_question_elicitation` only matches `enum` and `array` schemas
2. **Type mismatch**: Single-question assumption in current implementation vs. multi-question data model
3. **User impact**: Tools cannot collect multiple related answers in one interaction

### Impact of Inaction

- **User Experience**: Multiple round-trips for related questions
- **Tool Limitations**: RFC-0010 `question_for_user` tool cannot function as designed
- **Protocol Compliance**: Partial MCP Elicitation implementation

---

## Goals & Non-Goals

### Goals (In Scope)

1. Extend `_handle_question_elicitation` to support object schemas with multiple properties
2. Map each object property to a `QuestionInfo` in the `PendingQuestion.questions` list
3. Support all existing question types (enum, multi-select, input) within the object schema
4. Maintain backward compatibility with existing single-question enum/array schemas
5. Preserve existing SSE event format and client protocol

### Non-Goals (Out of Scope)

1. ACPInputProvider changes (ACP uses PermissionOption buttons, unsuitable for multi-question forms)
2. Nested object schemas (properties within properties)
3. Schema validation beyond MCP restricted subset
4. Conditional questions (show/hide based on previous answers)
5. Support for protocols other than OpenCode and ACP

### Success Criteria

- [x] Object schema with 2+ properties creates corresponding number of questions
- [x] Each property type (string/enum/array) is correctly rendered
- [x] Answers maintain correct index mapping (answers[i] ↔ questions[i])
- [x] Existing single-enum questions continue to work unchanged
- [x] RFC-0010 `question_for_user` tool functions correctly

---

## Implementation Notes

### Implementation Location

**Primary File**: `src/wolfharness_server/opencode_server/input_provider.py`

The implementation extends `OpenCodeInputProvider._handle_question_elicitation()` to support `object` type schemas with multiple properties.

### Design Decisions Applied

#### Single-Property Object Handling (Option A)

Object schemas with only 1 property use the existing single-question flow. The multi-question handler only triggers when `len(props) >= 2`. This minimizes disruption to existing behavior while maintaining clean separation of concerns.

```python
# From input_provider.py
match schema:
    # ... existing single-question handlers
    case {"type": "object", "properties": dict() as props} if len(props) >= 2:
        return await self._handle_multi_question(params, props)
```

#### Unsupported Property Types (Option C)

Unsupported property types are converted to text input (free text behavior). This provides maximum flexibility - users can always provide an answer even when the schema doesn't match expected patterns.

```python
# Fallback behavior for unsupported types
case _:
    # Free text input - no predefined options
    is_multi = False
    options = []
```

#### Max Questions Limit

A soft limit of 10 questions is enforced with warning log for UX considerations:

```python
if len(properties) > 10:
    logger.warning(f"Large question set: {len(properties)} questions")
```

#### Property Key Preservation

Original property keys from the schema are preserved in the answer object. The implementation does NOT convert to `q{i}` format - keys maintain their semantic meaning from the source schema.

```python
# Answers mapped with original keys preserved
content = {key: answers[i] for i, key in enumerate(properties.keys())}
```

### Deviations from Original RFC

**None.** The implementation follows the RFC specification exactly with the design decisions documented above (Option A for single-property objects, Option C for unsupported types).

---

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Question Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tool (question_for_user)                                   │
│     │                                                       │
│     ▼ MCP ElicitRequestFormParams                           │
│  ┌─────────────────────┐                                    │
│  │ requestedSchema: {  │                                    │
│  │   "type": "object", │                                    │
│  │   "properties": {   │                                    │
│  │     "q0": {...},   │  ──┐                              │
│  │     "q1": {...}    │  ──┼──► Multiple properties       │
│  │   }                 │  ──┘                              │
│  │ }                   │                                    │
│  └─────────────────────┘                                    │
│     │                                                       │
│     ▼                                                       │
│  OpenCodeInputProvider                                      │
│     │                                                       │
│     ▼ _handle_question_elicitation()                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Detect object schema                             │   │
│  │ 2. For each property:                               │   │
│  │    - Parse type (enum/array/string)                 │   │
│  │    - Create QuestionInfo                            │   │
│  │    - Extract title/description                      │   │
│  │ 3. Build QuestionInfo[] list                        │   │
│  └─────────────────────────────────────────────────────┘   │
│     │                                                       │
│     ▼                                                       │
│  PendingQuestion                                            │
│     questions: [QuestionInfo_0, QuestionInfo_1, ...]        │
│     future: asyncio.Future                                  │
│     │                                                       │
│     ▼ SSE Event                                             │
│  QuestionAskedEvent                                         │
│     questions: [...]  ◄─── Already supported!               │
│     │                                                       │
│     ▼ Client UI                                             │
│  OpenCode TUI renders multiple question cards               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 1. Schema Parsing Extension

```python
# wolfharness_server/opencode_server/input_provider.py

async def _handle_question_elicitation(
    self,
    params: types.ElicitRequestFormParams,
    schema: dict[str, Any],
) -> types.ElicitResult | types.ErrorData:
    """Handle elicitation via OpenCode question system.

    Extended to support:
    - Single enum schemas (existing)
    - Single array schemas (existing)
    - Object schemas with multiple properties (NEW)
    """
    match schema:
        # Existing: Single enum question
        case {"type": "array", "items": {"enum": [...]}}:
            return await self._handle_single_enum(params, schema, is_multi=True)
        case {"enum": [...]}:
            return await self._handle_single_enum(params, schema, is_multi=False)

        # NEW: Object schema with multiple questions
        case {"type": "object", "properties": dict() as props} if len(props) > 1:
            return await self._handle_multi_question(params, props)

        case _:
            return types.ElicitResult(action="decline")
```

#### 2. Multi-Question Handler

```python
async def _handle_multi_question(
    self,
    params: types.ElicitRequestFormParams,
    properties: dict[str, dict[str, Any]],
) -> types.ElicitResult | types.ErrorData:
    """Handle object schema with multiple properties as multiple questions."""
    from wolfharness_server.opencode_server.models import QuestionInfo, QuestionOption
    from wolfharness_server.opencode_server.models.events import QuestionAskedEvent

    question_id = self._generate_permission_id()
    questions: list[QuestionInfo] = []

    for key, prop_schema in properties.items():
        question = self._property_to_question(key, prop_schema)
        questions.append(question)

    # Create future for multi-question response
    future: asyncio.Future[list[list[str]]] = asyncio.get_event_loop().create_future()

    self.state.pending_questions[question_id] = PendingQuestion(
        session_id=self.session_id,
        questions=questions,  # ✅ List of QuestionInfo
        future=future,
    )

    # Broadcast multi-question event
    event = QuestionAskedEvent.create(
        request_id=question_id,
        session_id=self.session_id,
        questions=questions,  # ✅ Already supports list
    )
    await self.state.broadcast_event(event)

    # Wait for answers
    try:
        answers = await future  # list[list[str]]
        # Map answers to object property format
        content = {f"q{i}": ans for i, ans in enumerate(answers)}
        return types.ElicitResult(action="accept", content=content)
    except asyncio.CancelledError:
        return types.ElicitResult(action="cancel")
    finally:
        self.state.pending_questions.pop(question_id, None)
```

#### 3. Property to Question Conversion

```python
def _property_to_question(
    self,
    key: str,
    prop_schema: dict[str, Any],
) -> QuestionInfo:
    """Convert a JSON Schema property to QuestionInfo."""
    from wolfharness_server.opencode_server.models import QuestionInfo, QuestionOption

    title = prop_schema.get("title", key)
    description = prop_schema.get("description", "")

    # Determine question type and options
    prop_type = prop_schema.get("type")
    is_multi = False
    options: list[QuestionOption] = []

    match prop_schema:
        case {"type": "array", "items": {"enum": enum_values}}:
            is_multi = True
            options = [QuestionOption(label=str(v), description="") for v in enum_values]
        case {"enum": enum_values}:
            is_multi = False
            options = [QuestionOption(label=str(v), description="") for v in enum_values]
        case {"type": "string"}:
            # Free text input - no predefined options
            is_multi = False
            options = []
        case {"oneOf": one_of} if isinstance(one_of, list):
            # Use oneOf for titled enum values
            is_multi = False
            options = [
                QuestionOption(
                    label=opt.get("const", ""),
                    description=opt.get("title", "")
                )
                for opt in one_of
            ]

    return QuestionInfo(
        question=description or title,
        header=title[:12],  # Truncate per OpenCode spec
        options=options,
        multiple=is_multi or None,
    )
```

#### 4. Client Response Handling

The existing `resolve_question` method already supports `list[list[str]]`:

```python
# Existing method - no changes needed
def resolve_question(self, question_id: str, answers: list[list[str]]) -> bool:
    """Resolve a pending question request.

    Args:
        question_id: The question request ID
        answers: User's answers (array of arrays per OpenCode format)
               answers[i] corresponds to questions[i]

    Returns:
        True if the question was found and resolved, False otherwise
    """
    pending = self.state.pending_questions.get(question_id)
    if pending is None:
        return False

    future = pending.future
    if future.done():
        return False

    future.set_result(answers)  # ✅ Already supports multi-question
    return True
```

---

## Implementation Plan

### Phase 1: Core Extension

**Scope**: Extend `_handle_question_elicitation` with object schema support

**Tasks**:
1. Add object schema detection in `_handle_question_elicitation`
2. Implement `_handle_multi_question` method
3. Implement `_property_to_question` conversion method
4. Add unit tests for multi-question scenarios

**Files Modified**:
- `wolfharness_server/opencode_server/input_provider.py`

### Phase 2: Testing & Validation

**Scope**: Comprehensive testing against RFC-0010 requirements

**Tasks**:
1. Test with RFC-0010 `question_for_user` tool
2. Verify single-question backward compatibility
3. Test edge cases (empty answers, cancellations)
4. Performance testing (many questions)

### Phase 3: Documentation

**Scope**: Update relevant documentation

**Tasks**:
1. Update OpenCode Server documentation
2. Add examples to developer guide
3. Mark RFC-0015 as APPROVED

---

## Open Questions

1. **Question Ordering**
   - Should we preserve property order from schema?
   - Current Python dicts maintain insertion order (3.7+)

2. **Property Naming**
   - Schema uses `q0`, `q1` keys from xeno-agent
   - Should we support custom property names?

3. **Maximum Questions**
   - Should we limit number of questions for UX?
   - Proposal: Soft limit of 10, warning logged above

4. **Nested Objects**
   - Out of scope for now
   - Could be added in future RFC if needed

---

## References

### Related RFCs

- RFC-0010: Multi-Question Tool for User Interaction

### Code References

- `/packages/wolfharness/src/wolfharness_server/opencode_server/input_provider.py`
- `/packages/wolfharness/src/wolfharness_server/opencode_server/models/question.py`
- `/packages/wolfharness/src/wolfharness_server/opencode_server/state.py`

### Protocol References

- [MCP Elicitation Specification](https://modelcontextprotocol.io/specification/draft/client/elicitation)
- [ACP Elicitation RFD](/packages/agent-client-protocol/docs/rfds/elicitation.mdx)
