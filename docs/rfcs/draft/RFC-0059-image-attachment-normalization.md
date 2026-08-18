---
rfc_id: RFC-0059
title: "Image Attachment Normalization: Resize and Re-encode Oversized Images Before Provider Requests"
status: DRAFT
author: pinjun.mo
reviewers:
  - name: yuchen.liu
    status: pending
created: 2026-08-17
last_updated: 2026-08-17
decision_date:
related_rfcs: []
related_specs: []
---

# RFC-0059: Image Attachment Normalization

## Table of Contents

- [Overview](#overview)
- [Background & Context](#background--context)
- [Problem Statement](#problem-statement)
- [Goals & Non-Goals](#goals--non-goals)
- [Evaluation Criteria](#evaluation-criteria)
- [Options Analysis](#options-analysis)
- [Recommendation](#recommendation)
- [Technical Design](#technical-design)
- [Security Considerations](#security-considerations)
- [Implementation Plan](#implementation-plan)
- [Open Questions](#open-questions)
- [Decision Record](#decision-record)
- [References](#references)

---

## Overview

AgentPool already provides **centralized image normalization** for the tool-read path: `resize_image_if_needed()` in `src/wolfharness_toolsets/fsspec_toolset/image_utils.py` (default 2000 px, 4.5 MB, Pillow) is wired into both the `read` tool and the fsspec toolset. However, **protocol user-upload paths — Python API `run_agent(image_url=...)`, ACP attachments, and OpenCode server `FilePartInput` — bypass it entirely**: they are converted to pydantic-ai `ImageUrl`/`BinaryContent` at `to_user_content()` with no size or byte-budget checks. Oversized field photos (commonly 4000×3000 px, 10–20 MB) can exceed provider input limits, causing context overflow errors and inflated token costs.

This RFC proposes **reusing the existing centralized normalization** by inserting it at the `FilePart → pydantic-ai content` conversion point (alongside `to_user_content`), so all protocol entry points inherit the same constraint capability already used by the tool-read path.

**Expected outcome**: every protocol entry point (Python API / ACP / OpenCode server) constrains image attachments within the same configurable limits as the tool path; oversized user input fails with a clear error; tool-result images keep their current omission behavior; resizer unavailability degrades to passing the original through (a property to be preserved; note that the existing tool path hard-depends on Pillow, so unavailability is not currently a live code path there).

The design's failure semantics are modeled on opencode's reference implementation (`packages/opencode/src/image/image.ts`), but the normalization engine itself already exists in AgentPool and does not need to be recreated.

---

## Background & Context

### Current State

Image capability in the AgentPool stack flows through three layers, none of which normalize images:

1. **Python API entry**: `src/wolfharness/functional/run.py:21-50` — `run_agent(prompt, image_url=...)` maps the URL directly to pydantic-ai `ImageUrl(url=image_url)`. (Note: the agentpool CLI `run` command has no `--image` flag; this path is exposed only as a Python API.)
2. **Conversion point**: `src/wolfharness/utils/pydantic_ai_helpers.py:87-133` — `to_user_content()` maps `image/*` → `ImageUrl`, and `data:` URIs → `BinaryContent.from_data_uri()`. Pure MIME-to-content-type mapping; no size/byte logic.
3. **OpenCode server input**: `src/wolfharness_server/opencode_server/routes/message_routes.py:320` — `FilePartInput` branch calls `add_file_part(mime, url, ...)` with no image-specific handling.
4. **ACP server input**: `src/wolfharness_server/acp_server/converters.py:146-148,178-181` — `ImageContentBlock` / `BlobResourceContents` with `image/*` MIME decode base64 into `BinaryImage` with no size/byte checks.

The `FilePart` model (`src/wolfharness_server/opencode_server/models/parts.py:188`, fields `mime`/`filename`/`url`/`source`) mirrors opencode's `SessionV1.FilePart` structurally, but carries no normalization logic.

### Reference Implementation: opencode

opencode implements centralized image normalization in `packages/opencode/src/image/image.ts`:

- **Limits**: `max_width`/`max_height` 2000×2000 px; `max_base64_bytes` 5 MB (base64 payload); `auto_resize: true` by default.
- **Algorithm**: photon-wasm iterative 0.75× downscale loop (max 32 steps); for each candidate size, tries PNG then JPEG at quality ladder `[80, 85, 70, 55, 40]`; returns the first candidate under `max_base64_bytes`.
- **Failure semantics**:
  - User input over-limit and not resizable → `SizeError`, prompt fails with a clear error.
  - Tool-result over-limit → silently omitted, output annotated `[N images omitted: could not be resized below the image size limit.]`.
  - Resizer unavailable (`ResizerUnavailableError`) → original image passed through unchanged.
- **Config**: `attachment.image.{auto_resize, max_width, max_height, max_base64_bytes}` (`packages/core/src/v1/config/attachment.ts`).
- **Model-modality gating**: `unsupportedParts()` replaces images with error text for models lacking image modality.

### Related Work

- `tests/test_multimodal_storage.py`, `tests/test_modality_filter.py` — existing modality handling is at the storage/filter layer, not attachment normalization.

### Glossary

| Term | Definition |
|------|------------|
| FilePart | OpenCode protocol file-content part: `mime`/`filename`/`url` (data URI or `file://` path) |
| Normalization | Resize + re-encode applied to images exceeding configured limits |
| base64 payload | Byte length of the base64 string in `data:<mime>;base64,...` (measured by opencode in UTF-8 bytes) |
| pydantic-ai content | Pydantic-AI multimodal content abstraction: `ImageUrl` / `BinaryContent` / `DocumentUrl` |

---

## Problem Statement

### The Problem

AgentPool servers started via `serve-opencode` / `serve-acp` / `run_agent(image_url=...)` perform **no image preprocessing**:

1. No dimension or byte-budget checks — images are forwarded at original size.
2. No resize or re-encode — no `Image.normalize` equivalent.
3. No over-limit failure semantics — neither rejection nor omission; the downstream provider is the only backstop.

Images travel as base64 data URIs into `BinaryContent`/`ImageUrl` and are sent to the model unchanged. The only backstop is post-hoc error reporting from pydantic-ai/SDK/provider.

### Evidence

- `message_routes.py:320` `FilePartInput` branch does only `add_file_part`, no image branch.
- `pydantic_ai_helpers.py` performs MIME→content-type mapping only; no size/byte logic.
- The codebase contains modality tests (`test_multimodal_storage`, `test_modality_filter`) at the storage/filter layers, none covering attachment normalization.

### Impact of Inaction

- **Risk**: field photos (4000×3000 px, 10–20 MB originals common from mobile devices) base64-encode to ~15–27 MB, exceeding typical single-input limits of many models, triggering:
  - context window overflow (4xx / input token limits),
  - server-side memory spikes when decoding large images,
  - opaque errors surfaced by SDK/provider rather than a clear user-visible message.
- **Cost**: request token usage scales with pixel count; images larger than the model's effective resolution waste inference budget.
- **Opportunity**: no scenario-specific trade-off between high-fidelity diagnostic detail and cost control.

---

## Goals & Non-Goals

### Goals (In Scope)

1. Centralized normalization at the `FilePart → pydantic-ai content` conversion point, with default limits 2000×2000 px / 5 MB base64.
2. Configuration support: `attachment.image.{auto_resize, max_width, max_height, max_base64_bytes}`, overridable via AgentPool config schema.
3. Distinct over-limit failure semantics for user input vs. tool-result attachments.
4. All protocol entry points (Python API / ACP / OpenCode server) share the same normalization capability.

### Non-Goals (Out of Scope)

1. Model-modality capability negotiation/gating (opencode's `unsupportedParts` equivalent) — handled by pydantic-ai SDK; not in scope.
2. GIF animation frame preservation — re-encoding outputs static PNG/JPEG only.
3. Image provenance / session-persistent URI management.
4. EXIF orientation correction / metadata stripping (possible follow-up).
5. Audio / Video / PDF attachment normalization — `image/*` only.

### Success Criteria

- [x] Oversized images are resized/re-encoded within configured limits before reaching the model.
- [x] `attachment.image` config independently overrides defaults.
- [x] Over-limit user attachment (with `auto_resize: false`) yields a clear user-visible error.
- [ ] Over-limit tool-result attachment is omitted and annotated.
- [x] Resizer unavailable → original image passed through; session not interrupted.
- [ ] p99 added latency for normalization of a typical-size image < 500 ms.

---

## Evaluation Criteria

| Criterion | Weight | Description | Minimum Threshold |
|-----------|--------|-------------|-------------------|
| Coverage | High | All protocol entry points unified | Python API/ACP/OpenCode unified |
| Configurability | High | Limits overridable via config | At least width/height/bytes configurable |
| Failure-semantic clarity | High | Over-limit behavior controllable and explainable | User input errors; tool result omits |
| Resource overhead | Medium | CPU/memory/latency of normalization | Typical-size p99 < 500 ms |
| Implementation complexity | Medium | Cost to implement/maintain | No cross-package architecture reshuffle |
| Dependency risk | Medium | Reliability of new image-library dependency | Prefer pure-Python / mature WASM |

---

## Options Analysis

### Option 1: AgentPool Core Layer (normalize at conversion point)

**Description**: Add an `ImageNormalizer` service in the agentpool core. Hook it ahead of `pydantic_ai_helpers.to_user_content` / OpenCode `converters`, so image attachments are normalized before conversion to pydantic-ai content across all protocol entries. Config provided by AgentPool config schema (mirroring opencode's `attachment.image`).

**Advantages**:
- Covers all protocol entry points with one implementation.
- Architecturally aligned with opencode (normalize at the conversion point, not inside each tool).
- Transparent to consuming packages (e.g. xeno-agent) — they gain the capability without changes.

**Disadvantages**:
- Introduces an image-processing dependency (Pillow / sharp / photon-wasm) into the agentpool core dependency set.
- AgentPool maintainers must accept this as a general capability (not xeno-specific).
- Config schema changes affect all agentpool consumers.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Coverage | 5/5 | Three entry points unified |
| Configurability | 5/5 | Native config-schema support |
| Failure-semantic clarity | 4/5 | Unified semantics; must distinguish tool-result context |
| Resource overhead | 4/5 | Centralized implementation enables result caching/reuse |
| Implementation complexity | 3/5 | Core schema change required |
| Dependency risk | 3/5 | Image library enters core deps |

**Effort Estimate**: High (core change + tests + config schema).

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Maintainers reject scope | Medium | High | Cite opencode's precedent; prove generality for all agents |
| Image library install issues in pure-Python envs | Low | Medium | Prefer Pillow (pure wheel) or lazy-load |
| Impact on other agentpool consumers | Medium | Medium | Enabled by default, but `attachment.image.auto_resize: false` escape hatch |

---

### Option 2: Consumer Package Layer (xeno-agent-specific)

**Description**: Implement `ImageNormalizer` inside a consuming package (xeno-agent), invoked at that package's own entry points (e.g. `equipment_expert` tool chain, custom resource provider). Config via that package's YAML.

**Advantages**:
- No core agentpool changes; no core dependency introduced.
- Consumer teams control limits/algorithm and can tune for diagnostic scenarios (e.g. JPEG quality preserving detail fidelity).
- Small, isolated footprint.

**Disadvantages**:
- Only covers the consuming package's scenarios; other AgentPool consumers get no benefit.
- Requires explicit invocation at each consumer tool entry — omissible for future tools.
- Tool-result attachment path (handled centrally in opencode's processor) is not aligned; consumer must handle it separately.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Coverage | 3/5 | Consumer scope only |
| Configurability | 5/5 | Consumer-owned YAML |
| Failure-semantic clarity | 4/5 | Consumer-controllable |
| Resource overhead | 4/5 | On-demand invocation |
| Implementation complexity | 4/5 | Confined to one package |
| Dependency risk | 4/5 | Dependency confined to consumer package |

**Effort Estimate**: Low (single-package addition + single entry-point wiring).

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tool-chain entry-point omissions | Medium | Medium | Wire at unified entry (resource provider), not per-tool |
| Duplicated implementation vs. future core need | High | Medium | Keep abstraction; migrate if agentpool generalizes |

---

### Option 3: Client-Side Preprocessing (considered and set aside)

**Description**: Resize images at the caller (IDE, `opencode attach` client, client CLI) before upload.

**Advantages**:
- No AgentPool / consumer changes.

**Disadvantages**:
- Cannot constrain tool-result images (`read` / fsspec toolset / MCP-returned).
- Depends on unpredictable client behavior.
- Conflicts with server-side unified failure semantics.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Coverage | 2/5 | User-upload path only |
| Configurability | 1/5 | Client-driven, server-uncontrollable |
| Failure-semantic clarity | 1/5 | Server cannot guarantee |
| Resource overhead | 3/5 | Client CPU |
| Implementation complexity | 3/5 | Simple but ineffective |
| Dependency risk | 2/5 | Uncontrolled client environment |

**Risk Assessment**:

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cannot cover tool results | Certain | High | Essentially rules out this option |

---

### Options Comparison Summary

| Criterion | Option 1 (Core) | Option 2 (Consumer) | Option 3 (Client) |
|-----------|------------------|----------------------|-------------------|
| Coverage | 5/5 | 3/5 | 2/5 |
| Configurability | 5/5 | 5/5 | 1/5 |
| Failure-semantic clarity | 4/5 | 4/5 | 1/5 |
| Resource overhead | 4/5 | 4/5 | 3/5 |
| Implementation complexity | 3/5 | 4/5 | 3/5 |
| Dependency risk | 3/5 | 4/5 | 2/5 |
| **Total** | **24/30** | **24/30** | **12/30** |

---

## Recommendation

### Recommended Option

**[Option 1: AgentPool Core Layer]** — with Option 2 as an acceptable transitional path.

### Justification

- Architecturally, opencode normalizes at the data-to-model conversion point (not inside tools). AgentPool introducing the same capability at `to_user_content` is the **broadest-coverage, least-omissible** placement, and consistent with the reference implementation.
- Options 1 and 2 score identically on the summed criteria; the decisive factor is **ownership**: if the capability is general (this RFC argues it is — every agent faces large-image input), it belongs in the core; if the consumer team wants to validate value first with lower startup cost, Option 2 provides a transitional path with an abstraction that can migrate later.
- Option 3 cannot constrain the tool-result path and is retained only for comparison.

### Accepted Trade-offs

1. **Image library enters core deps**: Option 1 introduces Pillow (recommended) or equivalent; accepted in exchange for globally consistent constraint capability.
2. **GIF animation staticized**: re-encode outputs static frames only; no impact on static diagnostic field photos.
3. **Config schema change**: `attachment.image` requires AgentPool config schema update; backward compatible (defaults apply when absent).

### Conditions

- `auto_resize: true` by default; `attachment.image.auto_resize: false` escape hatch.
- Normalization failure (resizer unavailable) must pass the original through; never block the session.
- Defaults aligned with opencode (2000 / 2000 / 5242880) for consistent cross-project expectations.

---

## Technical Design (Preliminary)

> To be finalized after approval. Draft for review.

### Architecture Overview

```
Image sources                        Normalization pipeline                          Model
┌────────────────┐      ┌────────────────────────────────────────────────┐
│ Python API      │      │ ImageNormalizer (agentpool core)                │
│  run_agent()    │      │ NEW: inserted before to_user_content()          │
│ ACP attachment  │────▶│ ┌──────────────┐   ┌──────────────────────┐     │────▶ ImageUrl /
│ OpenCode FilePart│     │ │ size check   │──▶│ resize + re-encode  │     │      BinaryContent
│                  │      │ │ (px + bytes) │   │ PNG/JPEG candidates │     │      (pydantic-ai)
│ Tool results    │      │ └──────────────┘   └──────────────────────┘     │
│ (read/fsspec)   │      │  ALREADY normalized by resize_image_if_needed() │
│  MCP            │      └────────────────────────────────────────────────┘
└────────────────┘
```

### Key Components

#### ImageNormalizer

- Responsibility: accept `FilePart` (`image/*` only), return a constrained `FilePart`.
- Input: data URI or `file://` path; output: data URI.
- Algorithm: mirror opencode `image.ts` — dimension/byte check → iterative 0.75× downscale → PNG/JPEG quality-ladder candidate selection.

#### Config (`attachment.image`)

```
attachment:
  image:
    auto_resize: true      # default
    max_width: 2000
    max_height: 2000
    max_base64_bytes: 5242880
```

### Data Flow

1. User or tool-result produces `FilePart(image/*)`.
2. Normalization: byte check → if over-limit, resize + re-encode → new FilePart.
3. Persist (original or normalized — see Open Questions).
4. Convert to pydantic-ai content (`ImageUrl` / `BinaryContent`).

### API Design

```
ImageNormalizer.normalize(input: FilePart) -> FilePart
  raises: InvalidDataUrlError | DecodeError | SizeError

Config: attachment.image.<key> (injected into agentpool config schema)
```

---

## Security Considerations

### Threat Analysis

| Threat | Impact | Likelihood | Mitigation |
|--------|--------|------------|------------|
| Malicious oversized image → memory spike | Medium | Medium | Pre-check base64 length; reject input > N× limit before decode |
| Decoder vulnerability (decompression bomb) | Medium | Low | Maintained library + validate dimensions post-decode |
| Client-constructed malformed data URI | Low | Low | Unified `InvalidDataUrlError` handling |

### Security Measures

- [ ] Pre-check base64 length before decode (reject > N× limit directly)
- [ ] Accept `data:` base64 URIs only (no remote URL fetching; avoids SSRF)

### Compliance

No regulatory requirements identified.

---

## Implementation Plan

### Phase 1: AgentPool Core Normalization (if Option 1 adopted)

- **Scope**: Add `ImageNormalizer` + `attachment.image` config + wiring ahead of `to_user_content`.
- **Deliverables**: normalization service, config schema, unit tests, compatibility validation against `test_multimodal_storage`.
- **Dependencies**: Pillow introduction (or photon-wasm matching opencode).

### Phase 2: Consumer Validation (if Option 2 as transition)

- **Scope**: Wire normalization into a consumer resource provider; end-to-end validation with real field photos.
- **Deliverables**: consumer-side wiring + one end-to-end test (real photo; verify >5 MB / >2000² is compressed).
- **Dependencies**: Phase 1 deliverables or consumer-local implementation.

### Milestones

| Milestone | Description | Target | Status |
|-----------|-------------|--------|--------|
| M1 | ImageNormalizer + config schema | TBD | Implemented |
| M2 | Entry-point wiring + unit tests | TBD | Implemented |
| M3 | End-to-end validation + docs | TBD | In Progress |

### Rollback Strategy

- `attachment.image.auto_resize: false` disables resizing (byte check retained).
- Normalization failure always passes the original through — a natural escape path.

---

## Open Questions

1. **[Original vs. Normalized Persistence]**
   - Context: should session history store the original or the normalized image? opencode persists the normalized result.
   - Owner: AgentPool maintainers
   - Status: Open

2. **[Image Library Selection]**
   - Context: Pillow (pure wheel, mature ecosystem) vs. photon-wasm (matches opencode but requires WASM runtime) vs. sharp (native bindings, heavier install).
   - Owner: AgentPool maintainers
   - Status: Open

3. **[Normalization Trigger Point]**
   - Context: normalize at conversion-to-pydantic-ai-content time vs. at session-write time. First saves storage; second preserves more fidelity.
   - Owner: AgentPool maintainers
   - Status: Open

4. **[GIF / Animated Image Policy]**
   - Context: is static-frame re-encoding acceptable? Does the diagnostic scenario require animation preservation?
   - Owner: Consumer teams (e.g. xeno-agent)
   - Status: Open

5. **[Compatibility]**
   - Context: do other AgentPool consumers accept default normalization? Is a feature flag needed?
   - Owner: AgentPool maintainers
   - Status: Open

---

## Decision Record

> To be completed after review concludes.

### Decision

**Status**: Pending

**Date**: —

**Approvers**: —

### Decision Summary

—

### Key Discussion Points

1. —

### Conditions of Approval

—

---

## References

### Related Documents

- opencode reference implementation: `packages/opencode/src/image/image.ts`, `provider/transform.ts`, `session/message-v2.ts`
- opencode config documentation: `packages/opencode/packages/web/src/content/docs/config.mdx:425-451`
- AgentPool current image path: files cited in §2.1

### Appendix

- Normalization flow details follow opencode `image.ts` as the baseline template; error layering (`SizeError` / `ResizerUnavailableError` / `InvalidDataUrlError`) adapted from the same source.