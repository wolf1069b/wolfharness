"""Source reference format hook.

Validates that source reference tags (``[Menu]``, ``[Case]``) in entity
body text are accompanied by a ``viking://`` or ``file://`` URI. Bare tags
without URIs should be stripped during post-processing (step 16), per
design_717.md.

Example of valid reference:
    ``[Menu][viking://resources/<namespace>/.../chapters/.../chapter.md]``

Example of bare reference (flagged):
    ``[Menu]``
    ``[Case] some text``
"""

from __future__ import annotations

import re

from .base import BaseHook, HookResult


# Bare source tags: [Menu] or [Case] NOT followed by a [scheme://...] URI
# Matches [Menu] or [Case] that are not immediately followed by a recognised
# source URI scheme.
_BARE_TAG_RE = re.compile(r"\[(Menu|Case)\](?!\[[a-z][a-z0-9+.\-]*://)")

# Valid source tags: [Menu][viking://...] / [Case][file://...]
_VALID_TAG_RE = re.compile(r"\[(Menu|Case)\]\[[a-z][a-z0-9+.\-]*://[^\]]+\]")


class SourceReferenceHook(BaseHook):
    """Check that source reference tags have accompanying URIs.

    Per design_717.md post-processing step 16:
    "裸引用标签清洗（剥离无 URI 的 [Menu]/[Case]）"

    This hook flags bare ``[Menu]`` and ``[Case]`` tags that lack a
    following ``[viking://...]`` or ``[file://...]`` source URI link, so
    they can be cleaned up.
    """

    @property
    def name(self) -> str:
        return "source_ref"

    def check(
        self,
        content: str,
        concept: str = "",
        class_name: str = "",
        object_name: str = "",
    ) -> HookResult:
        # Extract frontmatter to skip it (sources field in frontmatter
        # uses a different format — list of URIs, not [Menu][...] tags)
        lines = content.splitlines()
        body_start = 0
        if lines and lines[0].strip() == "---":
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    body_start = i + 1
                    break

        body = "\n".join(lines[body_start:])

        bare_matches = list(_BARE_TAG_RE.finditer(body))
        valid_count = len(_VALID_TAG_RE.findall(body))

        if bare_matches:
            bare_tags = [m.group(0) for m in bare_matches]
            return HookResult(
                hook_name=self.name,
                passed=False,
                message=(
                    f"Found {len(bare_tags)} bare source tag(s) without URI: "
                    f"{', '.join(bare_tags[:5])}"
                    f"{'...' if len(bare_tags) > 5 else ''}. "
                    f"These will be stripped in post-processing. "
                    f"({valid_count} valid tagged reference(s) found.)"
                ),
                severity="warning",
            )

        return HookResult(
            hook_name=self.name,
            passed=True,
            message=(
                f"All source reference tags have URIs. ({valid_count} valid reference(s) found.)"
            ),
        )
