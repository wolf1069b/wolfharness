"""Wiki build quality hooks — post-tool-call validation for entity writes."""

from __future__ import annotations

from .base import BaseHook, HookResult
from .diagnostic_closure import DiagnosticClosureHook
from .frontmatter_schema import FrontmatterSchemaHook
from .source_ref import SourceReferenceHook
from .uri_integrity import URIIntegrityHook

__all__ = [
    "BaseHook",
    "DiagnosticClosureHook",
    "FrontmatterSchemaHook",
    "HookResult",
    "SourceReferenceHook",
    "URIIntegrityHook",
]
