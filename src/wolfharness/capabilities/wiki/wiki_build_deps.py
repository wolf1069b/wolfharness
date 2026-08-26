"""Shared dependency contract for WikiBuildTools mixins.

The ``WikiBuildTools`` implementation behind the ``build_tools`` facade is split across several
mixins (:mod:`text_parsers`, :mod:`opa`, :mod:`drafts`, ...) purely for
file organization.  Because the mixins call methods on one another through
``self`` at runtime, pyright needs the cross-mixin surface declared once.

``WikiBuildDeps`` captures that contract: the shared instance fields plus
every method one mixin calls on a sibling.  Each mixin subclasses
``WikiBuildDeps`` so pyright can resolve ``self.<sibling_method>()`` calls
within it.  At runtime the real implementations come from composition into
``WikiBuildTools``.

Methods are declared with an explicit ``self`` parameter as the callable
contract, matching how mixins invoke them (``self.method(...)``).  The
concrete implementations may be instance methods, ``@staticmethod``, or
``@classmethod``; pyright treats a ``self``-style declaration as the common
callable shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from pathlib import Path

    from wolfharness.capabilities.build.build_logger import WikiBuildLogger
    from wolfharness.capabilities.storage.backend import FSBackend
    from wolfharness.capabilities.storage.storage import WikiStore
    from wolfharness.capabilities.wiki.quality import (
        BuildProfile,
        SourceReadResult,
        WikiAuditReport,
    )


class WikiBuildDeps(Protocol):
    """Contract a WikiBuildTools mixin relies on from its siblings."""

    store: WikiStore
    _log: WikiBuildLogger | None
    _audit_cache: Any
    _case_root: Path | None
    _faultannotated_root: Path | None
    _raw_fs: FSBackend
    _bom_fs: FSBackend | None

    # -- service skeleton (fields + early helpers) -------------------------
    def _invalidate_audit_cache(self) -> None: ...
    def _reject_phantom_body_refs(
        self,
        content: str,
        *,
        extra_known: set[str] | None = None,
    ) -> None: ...

    # -- TextParsersMixin / remaining pieces --------------------------------
    def _apply_operations(self, content: str, operations: list[dict]) -> str: ...
    def read_resource(self, uri: str, line_numbers: bool = False) -> str | None: ...
    def read_raw_source(self, uri: str) -> SourceReadResult: ...
    def find_wiki(
        self,
        query: str,
        *,
        target_uri: str = "",
        limit: int = 10,
        deep: bool = False,
    ) -> list[dict]: ...
    def _sync_component_narrative_links(
        self,
        component_uri: str,
        *,
        fault_links: list[tuple[str, str]] | None = None,
        procedure_links: list[tuple[str, str]] | None = None,
    ) -> None: ...
    def write_symptom_profile(
        self,
        symptom_uri: str,
        profile_id: str,
        content: str,
        *,
        expected_sha256: str = "",
    ) -> str: ...
    def merge_entity(
        self,
        concept: str,
        class_name: str,
        object_name: str,
        content: str,
        *,
        expected_sha256: str = "",
    ) -> str: ...
    def list_symptom_profiles(self, symptom_uri: str) -> list[dict[str, str]]: ...
    def trace_diagnostic_path(
        self,
        device_uri: str = "",
        symptom_uri: str = "",
        limit: int = 100,
    ) -> dict[str, object]: ...
    def diff_symptom_profile(
        self,
        symptom_uri: str,
        profile_id: str,
        candidate_content: str,
    ) -> dict[str, object]: ...
    def patch_symptom_profile(
        self,
        symptom_uri: str,
        profile_id: str,
        operations: list[dict],
        *,
        expected_sha256: str = "",
    ) -> str: ...

    # -- AuditMixin / OPAMixin cross-contract -----------------------------
    def audit_wiki(
        self,
        *,
        concept: str = "",
        code: str = "",
        offset: int = 0,
        limit: int = 100,
        profile: BuildProfile = "manual",
        entity_uris: list[str] | None = None,
        force_refresh: bool = False,
    ) -> WikiAuditReport: ...
    def get_opas(
        self,
        *,
        target_uri: str = "",
        status: str = "",
        category: str = "",
        reason_code: str = "",
        scope: str = "",
        source_chapter: str = "",
        limit: int = 50,
    ) -> list[dict]: ...
    def _is_explicit_tracked_record(self, record: dict) -> bool: ...
