---
rfc_id: RFC-0061
title: "Information-Preserving Degradation for ModalityFilter: From Bare Placeholders to Retrievable References"
status: DRAFT
author: pinjun.mo
reviewers: []
created: 2026-08-18
last_updated: 2026-08-19
decision_date:
related_rfcs:
  - RFC-0059 (Image Attachment Normalization: Resize and Re-encode Oversized Images Before Provider Requests)
related_specs: []
---

# RFC-0061: Information-Preserving Degradation for ModalityFilter

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

AgentPool's `ModalityFilterCapability` degrades multimodal content that the active model does not support. Its default `describe` strategy replaces an unsupported image with a **bare MIME placeholder** — e.g. `[image/png]` — before the provider request is sent. This RFC argues that the placeholder destroys the **retrievability** of the content: the text-only model receives a token with no filename, no location, and no way to hand the image to a vision-capable tool or subagent, so the original user intent (analyze *this* image) is silently lost.

Unlike RFC-0059 (which constrains image *size* for vision-capable providers), this RFC is about what happens when the provider **cannot read the image at all**. It proposes replacing the bare placeholder with an **information-preserving degradation** that keeps retrievable metadata on the image, and discusses whether AgentPool should persist unsupported media so that other agents — vision-capable subagents, file tools, or MCP resources — can actually access it.

The design question posed here is deliberately left open for discussion. The RFC surveys the candidate strategies and presents the trade-offs, mirroring how the opencode community converged on "don't destroy the image reference in the first place" without yet landing an implementation.

---

## Background & Context

### Current State

`ModalityFilterCapability` (in `src/wolfharness/capabilities/modality_filter.py`) is an opt-in capability that is **not auto-injected**. It is enabled by declaring `type: modality_filter` in a manifest's capabilities and configuring a per-category strategy (`describe` / `drop` / `pass`):

- `describe` (default): replace unsupported content with a text placeholder.
- `drop`: remove the content entirely.
- `pass`: forward the content unchanged.

The degradation is applied in two places:

1. **`before_model_request`** — scans `ModelRequest` / `ModelResponse` messages and rewrites unsupported content in `UserPromptPart` and `ToolReturnPart` via `dataclasses.replace()`.
2. **`wrap_tool_execute`** — filters multimodal content in tool results.

The placeholder comes from `describe_multimodal_content()` in `src/wolfharness/capabilities/modality_utils.py`. As this RFC was being authored, the original bare placeholder was:

```python
case BinaryImage(media_type=media) | BinaryContent(media_type=media):
    return f"[{media}]"        # e.g. "[image/png]"
```

This function has 12 call sites, most of which are genuinely for **logging / display / persistence** (`helpers.py:_summarize_content_block` documents itself as "for logging/display"). The problem addressed by this RFC is that **the same placeholder is reused as the degradation payload delivered to the model**, where a bare MIME token is neither visible content nor a usable reference.

### Reference Implementation: opencode

The opencode ecosystem has explored the exact same problem and produced a documented consensus. Findings from the `anomalyco/opencode` codebase (HEAD `040b856`):

**Current behavior — error text, not a bare placeholder.** `unsupportedParts()` in `packages/opencode/src/provider/transform.ts:410-442` rewrites the unsupported part to:

```
ERROR: Cannot read "photo.png" (this model does not support image input). Inform the user.
```

**The bare-placeholder form was explicitly rejected.** PR [#29279](https://github.com/anomalyco/opencode/pull/29279) attempted to replace the error text with `[Attached image: "photo.png" (image/png)]`. Reviewers rejected it on hallucination risk:

> "Calling something 'Attached' when the model has no direct access to its content is misleading and invites the LLM to invent details."

Its counter-proposal (still unmerged) carried three signals — a factual claim, an anti-hallucination guardrail, and an escape hatch:

```
[User provided image: "photo.png" (image/png).
 Direct processing unsupported — available via tools and filesystem.]
```

**The bare placeholder is a known bug complaint.** The `stripMedia` path in `packages/opencode/src/session/message-v2.ts:213-218` produces `[Attached image/png: file]`. Issue [#42758](https://github.com/anomalyco/opencode/issues/42758) (author environment: `deepseek-v4-flash`, a text-only model — the same downstream scenario motivating this RFC) reports that this leaves the agent with **no way to access the actual image content**, since the placeholder carries no reference to how the image can be retrieved.

**Community consensus: don't destroy the reference.** Issue [#29216](https://github.com/anomalyco/opencode/issues/29216) articulates the design philosophy that two further open issues (`#36006`, `#40495`) and three open PRs (`#32680`, `#26164`, `#21633` — persist unsupported images to temp files for vision MCP tools) all converge on:

> "don't destroy the image reference in the first place" — the model doesn't need native vision; it needs an undestroyed reference (file path / MIME / filename) so it can dispatch to an MCP tool.

**Automatic model switching is explicitly not planned.** Issues `#32601` (NOT_PLANNED) and `#31936` (CLOSED) confirm opencode will not auto-switch to a vision model when an image arrives.

In short: the opencode community identifies bare-placeholder degradation as a defect, agrees the fix should preserve retrievability, and has not shipped it. AgentPool has an opportunity to design this properly rather than inheriting the same defect.

---

## Problem Statement

When `ModalityFilterCapability` degrades an image to `[image/png]` inside the provider request:

1. **The content is unrecoverable by the model.** The placeholder is a MIME-only token. It carries no filename, no path, no identifier, and no hint of how the image could be opened.
2. **Delegation is impossible.** AgentPool's native agents can delegate to subagents and call tools. A vision-capable subagent or a file-reading tool would be perfectly able to consume a *reference*, but a bare placeholder gives the delegating model nothing to pass along.
3. **The degradation is misleading.** It reads like a description of content that is actually absent, which — exactly as opencode's reviewer warned — invites the model to invent details about an image it has never seen.
4. **The failure is silent.** The user attached an image with an intent; the session proceeds as if that intent were preserved. There is no surfaced signal that the image was not understood.

The same defect affects `describe` degradation in every modality category (image/audio/video/document) and every content type (`BinaryContent`, `BinaryImage`, `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`, `UploadedFile`).

---

## Goals & Non-Goals

### Goals

- Preserve **retrievable metadata** (filename, MIME, storage location where applicable) in the degradation output so the model and downstream agents can act on it.
- Support the realistic working pattern: **text-only primary model + vision-capable subagent or tool** consuming the degraded content via AgentPool's existing delegation / tool infrastructure.
- Make the degradation **explicit and honest**: signal "this content exists but was not directly read" rather than pretending it was.
- Keep the change contained to the `modality_filter` capability layer; no protocol server rewrites.

  > **Status (2026-08-19): superseded.** Landing `reference` + mime integrity required minimal
  > converter-layer changes so binary content actually reaches the model intact. Specifically:
  > - `src/wolfharness/mcp_server/conversions.py`: pass `BlobResourceContents.mimeType` through
  >   to `BinaryContent` instead of hardcoding `application/octet-stream`.
  > - `src/wolfharness_server/acp_server/converters.py`: drop the `_DOCUMENT_FORMATS` whitelist
  >   gate in `resource_to_content` so arbitrary-mime embedded blobs surface as `BinaryContent`
  >   (with octet-stream fallback), deferring modality decisions to `ModalityFilterCapability`.
  >   This is consistent with the filter treating `"unknown"` mime as pass-through.
  >
  > The principle still holds: capability-layer degradation is the decision point; converters
  > only stop destroying the data before it gets there.

### Non-Goals

- Auto-switch models when unsupported content arrives (opencode's NOT_PLANNED position; aligned with AgentPool's explicit model-pinning design).
- Preserve the *bytes* of unsupported media when no consumer exists — persistence should be scoped to retrievability, not archival.
- Change what vision-capable models receive. When the model supports a modality, content passes through unchanged and RFC-0059 normalization applies.
- Resolve the storage backend question in this RFC (see [Open Questions](#open-questions)); this RFC defines the degradation surface, storage is the implementation detail.

---

## Evaluation Criteria

| Criterion | Question it answers |
|-----------|---------------------|
| Retrievability | Can a downstream agent (subagent/tool) actually obtain the content from the degradation output? |
| Honesty | Does the output truthfully represent "content exists, not directly read" vs. pretending content was parsed? |
| Hallucination resistance | Does the wording discourage the model from inventing details about unseen content? |
| Boundary safety | Can the output be safely surfaced in logs, prompts, and (if persisted) an HTTP server without leaking or breaking? |
| Backward compatibility | Does the change keep working for existing manifests that rely on `describe` today? |
| Implementation complexity | How much of the capability layer / storage layer must change? |

---

## Options Analysis

### Option 1: Status Quo — Bare MIME Placeholder

**Description**: Keep `describe` producing `[image/png]` (and `[image: <url>]` for URL types).

**Advantages**:
- Zero change; current tests pass.
- Byte-lean prompt footprint.

**Disadvantages**:
- Unrecoverable content, no delegation path, silent failure, hallucination-prone (see [Problem Statement](#problem-statement)).
- Directly contradicted by opencode PR #29279 review and issue #42758 — this is the documented defect.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Retrievability | 1/5 | Nothing to retrieve |
| Honesty | 1/5 | Reads like content, content absent |
| Hallucination resistance | 1/5 | Invites invention |
| Boundary safety | 4/5 | No new attack surface |
| Backward compatibility | 5/5 | Unchanged |
| Implementation complexity | 5/5 | None |

**Effort Estimate**: None.

---

### Option 2: Informational Metadata Placeholder (no persistence)

**Description**: Keep the degradation as a *text* replacement, but enrich the placeholder with retrievable metadata and explicit signals. Modeled on opencode PR #29279's counter-proposal:

```
[User supplied image: "photo.png" (image/png).
 Direct model processing is unsupported by the active model.
 The file is NOT inlined into this context — a vision-capable subagent or file tool may open the original.]
```

For URL types: `[image: https://...]` already carries a reference and could stay as-is. For `BinaryContent`/`BinaryImage` (which have no filename), the placeholder would degrade to `[User supplied image (image/png), source not preserved]` — honest about what is *not* available.

**Advantages**:
- Cheap: confined to `describe_multimodal_content()` / the `describe` branch in `modality_filter.py`.
- Honest and hallucination-resistant wording; aligns with opencode's accepted direction.
- No storage, no security surface.

**Disadvantages**:
- **Still not retrievable in the binary case**: `BinaryContent`/`BinaryImage` have no on-disk origin, so the metadata placeholder still can't hand anything to a tool.
- Requires a vision-capable *tool* that accepts a filename — AgentPool today has `read` (fsspec) which can open local paths, so the pattern is realistic only for content that originated on disk.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Retrievability | 2/5 | Filename present, bytes not persisted for binary input |
| Honesty | 5/5 | Explicit "unsupported / not inlined" signal |
| Hallucination resistance | 5/5 | Active discouragement wording |
| Boundary safety | 4/5 | Metadata only; filename is the only new surface |
| Backward compatibility | 4/5 | New string shape is a drop-in for downstream parsers; tests updated |
| Implementation complexity | 4/5 | Single function + tests |

**Effort Estimate**: Low (one function + unit tests + one behavior test).

---

### Option 3: Session-Scoped Persistence + Retrievable Reference

**Description**: When `describe` (or a new `reference` strategy) degrades binary multimodal content, persist the bytes into the session storage under a deterministic key, and produce a reference the model can hand to tools:

```
[User supplied image: "photo.png" (image/png).
 Direct processing unsupported — the file is persisted for this session.
 A vision-capable subagent can access it via <storage://session-id/… or a host-relative path>]
```

The persisted reference is designed to be consumable by AgentPool's existing delegation model: the primary agent can spawn a vision-capable subagent and pass the reference in the prompt, and the subagent's file/`read` tools open it. This is exactly the pattern opencode PRs #32680 / #21633 / #26164 tried to ship, restated in AgentPool's session-storage terms.

**Advantages**:
- **Truly retrievable** — the delegation loop closes: text model → reference → vision subagent.
- Solves the binary-content case (no on-disk origin) by creating one.
- Consistent with RFC-0059's storage-aware direction and with AgentPool's existing `StorageManager` / session persistence.

**Disadvantages**:
- **Storage backend coupling**: requires deciding where bytes live (in-memory per session? SQLite? filesystem?), TTL/cleanup, and multi-process visibility. This is a real design fork this RFC does not close.
- **When does persistence run, if ever?** Degradation happens in `before_model_request`, which is synchronous and hot-path. Writing bytes there couples the capability to storage I/O.
- **Boundary risk**: exposing a reference means exposing bytes; must scope to the session and consider authz on any HTTP-served reference.
- If no vision-capable consumer exists in the manifest, persistence is wasted I/O.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Retrievability | 5/5 | Bytes persisted, subagent can consume |
| Honesty | 5/5 | Explicit signal |
| Hallucination resistance | 5/5 | Active discouragement wording |
| Boundary safety | 3/5 | Storage-cleanness + potential HTTP exposure |
| Backward compatibility | 4/5 | New value-add; existing `describe` behavior superseded by configurable strategy |
| Implementation complexity | 2/5 | Capability + storage + lifecycle |

**Effort Estimate**: High (new strategy or storage hook + lifecycle + tests + storage decision).

---

### Option 4: Hybrid — Metadata First, Persistence Optional

**Description**: Ship Option 2 (metadata placeholder) as the default behavior now, and add Option 3's persistence behind a *new explicit strategy* (e.g. `reference`) that manifests opt into when they actually have vision-capable consumers. A manifest that wants the delegation loop declares:

```yaml
capabilities:
  - type: modality_filter
    image_strategy: reference    # persist + emit retrievable reference
```

**Advantages**:
- Immediate defect fix (metadata honesty) with zero storage commitment.
- Persistence only engaged when a consumer exists — no wasted I/O, no new security surface for manifests that don't need it.
- Preserves `describe` semantics for existing manifests (backward compatible), adds `reference` for opt-in.
- Matches opencode's trajectory exactly: metadata placeholder as the agreed direction; persistence as the still-open PR.

**Disadvantages**:
- Two strategies to maintain.
- `reference` still has all of Option 3's open storage questions.

**Evaluation Against Criteria**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Retrievability | 4/5 | Metadata now; full retrievability when `reference` used |
| Honesty | 5/5 | Explicit signal in both modes |
| Hallucination resistance | 5/5 | Same wording |
| Boundary safety | 4/5 | Persistence opt-in → less default surface |
| Backward compatibility | 5/5 | `describe` unchanged, new strategy added |
| Implementation complexity | 3/5 | Two pieces, but second is opt-in |

**Effort Estimate**: Medium (default metadata change + opt-in reference strategy + storage decision for reference).

---

### Options Comparison Summary

| Criterion | 1: Bare | 2: Metadata | 3: Persist+Ref | 4: Hybrid |
|-----------|---------|--------------|----------------|-----------|
| Retrievability | 1/5 | 2/5 | 5/5 | 4/5 |
| Honesty | 1/5 | 5/5 | 5/5 | 5/5 |
| Hallucination resistance | 1/5 | 5/5 | 5/5 | 5/5 |
| Boundary safety | 4/5 | 4/5 | 3/5 | 4/5 |
| Backward compatibility | 5/5 | 4/5 | 4/5 | 5/5 |
| Implementation complexity | 5/5 | 4/5 | 2/5 | 3/5 |
| **Total** | **17/30** | **24/30** | **24/30** | **26/30** |

---

## Recommendation

### Recommended Option

**[Option 4: Hybrid — Metadata First, Persistence Optional]** — with Option 2 as the minimum acceptable landing scope.

### Justification

- Option 1 is the documented defect (opencode PR #29279 rejection, issue #42758) and does not meet the goals.
- Options 2 and 3 score equally overall, but Option 3 commits to a storage design this RFC deliberately leaves open. Shipping storage-backed degradation speculatively, for manifests that may have no vision-capable consumer, is over-engineering.
- Option 4 captures the immediate, cheap, broadly-shared win (honest, retrievable-by-filename metadata — the opencode-consensus direction) and defers the costly, uncertain part (persistence) behind an explicit opt-in strategy that only engages when the user actually wants the delegaton loop.
- It is backward compatible: existing `describe` manifests keep their current behavior (upgraded to the informative string shape); the new `reference` strategy is opt-in.

### Accepted Trade-offs

1. **Filename-preserving degradation still needs a consumer.** For binary input with no on-disk origin, Option 4's metadata form (like Option 2) cannot hand a tool anything; full retrievability requires `reference`, which is out of scope for the initial landing unless reviewers prefer Option 3.
2. **`reference` inherits storage questions.** Persistence backend, TTL, cleanup, and multi-process visibility remain open (see [Open Questions](#open-questions)); they bubble to whichever option includes persistence.

### Conditions

- The metadata placeholder wording must include the anti-hallucination guardrail "direct model processing unsupported" (not just a filename).
- The `reference` strategy, if added, must be documented as requiring a vision-capable subagent or file tool in the manifest.
- Any persistence must be scoped to the session and cleaned up with the session (consistent with AgentPool's session-scoped storage).

---

## Technical Design (Preliminary)

> To be finalized after approval. Draft for review.

### Architecture Overview

```
Unsupported image input
        │
        ▼
ModalityFilterCapability._filter_single_content()
        │  strategy = describe (default)
        ▼
describe_multimodal_content()  ──NEW──▶  information-preserving placeholder
        │                                   (filename · mime · "unsupported" signal)
        │
        ├── default: text replacement in request
        │
        └── strategy = reference (opt-in, if approved)
                    ▼
            session-scoped persist  ──▶  retrievable reference in prompt
                                                 │
                                                 ▼
                              text-only model delegates to
                              vision-capable subagent / file tool
```

### Key Components

#### `describe_multimodal_content()` (information-preserving variant)

- Signature unchanged: `(content: MultiModalContent) -> str`.
- Output string now depends on which metadata is available:
  - `ImageUrl` / `AudioUrl` / `VideoUrl` / `DocumentUrl` (URL types): keep `[image: <url>]` — the URL *is* a retrievable reference.
  - `BinaryImage` / `BinaryContent`: emit `[User supplied <media> — direct model processing is unsupported by the active model (not inlined); a vision-capable subagent or file tool may open the original if it is on disk]`.
  - `UploadedFile`: emit `[User supplied uploaded file (file_id: <id>)]` — the id is a reference if the upload store is queryable.

#### New strategy: `reference` (opt-in, pending discussion)

- Persist degraded binary bytes to session storage under a deterministic key.
- Emit the reference into the replacement text with the same honesty guardrails.
- Never runs unless the manifest declares `image_strategy: reference`.

### Data Flow

1. Manifest declares `modality_filter` capability (existing) — defaults unchanged.
2. Degradation path in `before_model_request` now produces honest metadata text (Option 2 behavior, default).
3. If `strategy: reference`, bytes are persisted first, then the reference is emitted.
4. The model may delegate the reference to a vision-capable subagent (existing delegation infra) which opens it via file tools (existing `read` / fsspec).

### API Design

```
# Existing (unchanged)
describe_multimodal_content(content) -> str          # now information-preserving

# New (if Option 3/4's reference strategy is approved)
enum ModalityStrategy += "reference"
ModalityFilterCapability.reference_strategy(...)      # persists + emits reference
```

---

## Security Considerations

- **Placeholder text is user-influenced**: filenames are provided by the caller/OS. They are already interpolated today (in logs); emitting them into the model prompt requires care with control characters and prompt-injection-ish filenames. Recommend rendering filenames via a safe repr/escape before interpolation.
- **Persistence is a new write path**: any `reference`/Option-3 persistence must scope bytes to the session, bound total size (reuse RFC-0059's byte limits), and guarantee cleanup on session teardown. Persisted references must never be world-readable by default.
- **No new secrets surface**: unsupported media can contain sensitive pixels (screenshots). Emitting "persisted at <path>" only makes sense inside an already-trusted runtime; do not echo full absolute paths into prompts unless the deployment is trusted.

---

## Implementation Plan

### Phase 1 — Honest metadata placeholder (Option 2 scope, in-core)

1. Rewrite `describe_multimodal_content()` to produce the information-preserving strings above.
2. Update unit tests in `tests/test_modality_utils.py` and the modality-filter behavior tests (`tests/test_modality_filter.py`, `tests/test_agent_factory_modality.py`).
3. Keep URL-type placeholders unchanged (already references).
4. Add a changelog entry under `changelog/unreleased/`.

### Phase 2 — Opt-in `reference` strategy (only if reviewers approve the storage direction)

1. Storage decision first (see [Open Questions](#open-questions)).
2. Add `reference` to `ModalityStrategy`; wire persistence in the degradation path.
3. VCR/E2E test: text-only primary + vision-capable subagent consuming the persisted reference.

### Phase 3 — Docs

- Update `docs/rfcs/draft/` status, `docs/reference` for the capability, and the configuration reference (`docs/how-to/` modality-filter page if present).

---

## Open Questions

1. **Is the bare placeholder genuinely a defect, or intended lossy behavior?** The RFC assumes the former; reviewers may decide the loss is acceptable for byte-budget reasons.
2. **Does AgentPool want a storage-backed `reference` strategy at all?** opencode's community wants it but hasn't shipped it; AgentPool can differentiate. (Opencode status: PRs #32680 / #21633 / #26164 open, unmerged.)
3. **Where do persisted bytes live?** In-memory per-run, session SQLite, filesystem scratch, or a host-served reference? Affects multi-process and restart visibility.
4. **Should degradation emit a surfaced signal to the caller** (e.g. an event) so the user knows their image was not read — beyond the honest placeholder?
5. **Filename safety**: should the replacement text escape/redact filenames before interpolation?

---

## Decision Record

- **2026-08-18**: RFC opened as DRAFT by pinjun.mo. Problem identified while debugging a real deployment (`glm52`/kimi-k2, a text-only model, failing on pasted images via the opencode server). Three opencode-codebase investigations (HEAD `040b856`) support the problem framing. No decision made yet.
- **2026-08-19**: Option 4 (hybrid) landed as the working direction. Implementation commits on this branch:
  - `d06514464` — information-preserving `describe` (honest metadata placeholder) + opt-in `reference` strategy (session-scoped scratch persistence under `tempfile`).
  - `115ca7784` — converter-layer mime integrity: `BlobResourceContents.mimeType` passthrough in MCP conversions, removal of the `_DOCUMENT_FORMATS` whitelist in ACP `resource_to_content` (see Goals & Non-Goals status note). Both changes were necessary so binary content reaches `ModalityFilterCapability` intact; the filter remains the single decision point for unsupported modalities.
  - Remaining open items carried forward unchanged: storage backend/TTL for `reference` (see [Open Questions](#open-questions)), surfaced degradation signal to the caller.

---

## References

- `src/wolfharness/capabilities/modality_filter.py` — current degradation implementation
- `src/wolfharness/capabilities/modality_utils.py` — `describe_multimodal_content()` (placeholder source)
- `src/wolfharness/agents/native_agent/helpers.py` — `_summarize_content_block` (logging/display call site; documents the dual-use tension)
- RFC-0059: Image Attachment Normalization (Resize and Re-encode Oversized Images)
- opencode `anomalyco/opencode` HEAD `040b856`:
  - `packages/opencode/src/provider/transform.ts:410-442` — `unsupportedParts()` (error-text degradation)
  - `packages/opencode/src/session/message-v2.ts:213-218` — `stripMedia` bare-placeholder path
  - PR [#29279](https://github.com/anomalyco/opencode/pull/29279) — bare placeholder rejected (hallucination risk)
  - Issue [#42758](https://github.com/anomalyco/opencode/issues/42758) — "agent has no way to access the actual image content"
  - Issue [#29216](https://github.com/anomalyco/opencode/issues/29216) — "don't destroy the image reference"
  - Issues [#32601](https://github.com/anomalyco/opencode/issues/32601) / [#31936](https://github.com/anomalyco/opencode/issues/31936) — auto-switch NOT_PLANNED
  - PRs [#32680](https://github.com/anomalyco/opencode/pull/32680) / [#21633](https://github.com/anomalyco/opencode/pull/21633) / [#26164](https://github.com/anomalyco/opencode/pull/26164) — persist-to-temp-file attempts (open, unmerged)