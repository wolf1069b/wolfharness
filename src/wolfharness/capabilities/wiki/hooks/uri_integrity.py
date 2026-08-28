"""URI integrity check hook.

Validates that wiki entity URIs in entity content follow the canonical
readable format ``<root_uri>/<Concept>/<Class>/<Object>`` (where
``<root_uri>`` is the configured wiki root, e.g.
``viking://resources/<namespace>``), or the legacy 24-hex hash form
``<root_uri>/<Concept>/<hash>``.  A readable URI is a file path, so it
cannot be a "phantom hash"; only empty / placeholder / object-less tails
are flagged.
"""

from __future__ import annotations

from wolfharness.capabilities.wiki.quality import (
    extract_malformed_wiki_uris,
    extract_source_uris,
    extract_wiki_uris,
    is_wiki_uri,
)

from .base import BaseHook, HookResult


class URIIntegrityHook(BaseHook):
    """Check that wiki URIs in entity content are well-formed.

    Validates:
    - Entity heading URI matches readable or legacy-hash format.
    - Body references to ``<root_uri>/...`` are well-formed (no empty,
      object-less, or placeholder tails).
    """

    @property
    def name(self) -> str:
        return "uri_integrity"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        lines = content.splitlines()

        heading_uri = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("# ---"):
                heading_uris = sorted(extract_source_uris(stripped))
                if heading_uris and is_wiki_uri(heading_uris[0]):
                    heading_uri = heading_uris[0]
                break

        if heading_uri and heading_uri not in extract_wiki_uris(heading_uri):
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=(
                    f"Entity heading URI '{heading_uri}' does not match "
                    f"expected format <root_uri>/<Concept>/<Class>/<Object>."
                ),
                severity="warning",
            )

        bad_uris = extract_malformed_wiki_uris(content)
        if bad_uris:
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=(
                    f"Found {len(bad_uris)} malformed wiki URI(s) with empty/"
                    f"object-less/placeholder tails: "
                    f"{', '.join(bad_uris[:3])}"
                    f"{'...' if len(bad_uris) > 3 else ''}"
                ),
                severity="warning",
            )

        return HookResult(
            hook_name=self.name,
            passed=True,
            message="All wiki URIs are well-formed.",
        )
