"""Language-agnostic chapter importance scoring via character n-grams.

Ranks a manual's chapters by how much distinctive technical content they
carry, using only character bigrams learned from the manual itself — no
per-language vocabulary and no entity keyword lists.  A chapter that repeats
the manual's technical vocabulary (part names, procedures, fault steps,
measured values) scores high; boilerplate (prefaces, safety notes, bare
schematics) shares few distinctive bigrams and scores near zero.

Two passes over the manual:

1. :func:`build_fingerprint` — TF-IDF-lite over all chapters produces the
   small set of bigrams that best characterize the manual's technical
   content ("the fingerprint").  Ubiquitous grams self-exclude via inverse
   document frequency, so the vocabulary adapts to whatever language the
   manual is written in.
2. :func:`score_chapter_record` — a chapter's score is the share of
   fingerprint bigrams it contains, mapped to a 0-100 scale so scores stay
   comparable across manuals of different size/language.

Pure statistics (Counter/math/re), no model calls, milliseconds across a
whole manual, and fully unit-testable with synthetic corpora.  The only
assumptions are the byte-level ``_KEEP_RE`` character whitelist and the HTML
strip — plain text without markup works unchanged.
"""

from __future__ import annotations

from collections import Counter
import math
import re

from wolfharness.capabilities.wiki.section_constants import ADMIN_SECTION_KEYWORDS


_HTML_STRIP_RE = re.compile(r"<[^>]+>")
_KEEP_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff]")  # letters + digits only
_LETTER_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]")


def _clean(content: str) -> str:
    """Normalize a chapter to a letter/digit character stream.

    HTML/OCR scaffolding, entities and spaces are dropped (CJK and Latin both
    reduce to continuous streams), so character bigrams are language-agnostic.
    """
    return _KEEP_RE.sub("", _HTML_STRIP_RE.sub("", content).lower())


def _bigrams(text: str) -> Counter[str]:
    if len(text) < 2:
        return Counter()
    return Counter(text[i : i + 2] for i in range(len(text) - 1))


def build_fingerprint(
    chapters: list[str],
    *,
    size: int = 1000,
    min_df: int = 1,
) -> frozenset[str]:
    """Select the bigrams that best characterize a manual's technical content.

    Weights each bigram by corpus TF scaled by inverse document frequency
    (TF-IDF-lite over chapters as documents); ubiquitous grams (e.g. ``的了``
    or ``th``) self-exclude and distinctive terms surface regardless of
    language.  Pure-digit or pure-noise grams are dropped.

    Args:
        chapters: Raw chapter texts (HTML is stripped internally).
        size: Target fingerprint cardinality (top-N by weight).
        min_df: Drop grams seen in fewer chapters than this.  Default ``1``
            keeps rare-but-technical bigrams in the running for small
            corpora; raise it only to suppress OCR junk in very noisy corps.

    Returns:
        Immutable set of selected bigrams.
    """
    grams_by_doc: list[Counter[str]] = [_bigrams(_clean(c)) for c in chapters]
    doc_freq: Counter[str] = Counter()
    for grams in grams_by_doc:
        doc_freq.update(grams.keys())

    n_docs = len(grams_by_doc)
    idf = {
        gram: math.log(n_docs / (1 + count))
        for gram, count in doc_freq.items()
        if count >= min_df and _LETTER_RE.search(gram)
    }
    weight: dict[str, float] = {}
    for grams in grams_by_doc:
        total = sum(grams.values()) or 1
        for gram, count in grams.items():
            inverse = idf.get(gram)
            if inverse is not None:
                weight[gram] = weight.get(gram, 0.0) + (count / total) * inverse
    return frozenset(
        gram for gram, _ in sorted(weight.items(), key=lambda kv: kv[1], reverse=True)[:size]
    )


def classify_score(score: float, *, skip_below: float = 10.0, read_above: float = 25.0) -> str:
    """Map a 0-100 importance score to a build action.

    Args:
        score: Importance score from :func:`score_chapter_record`.
        skip_below: Below this → low-value → register no_entity.
        read_above: At/above this → high-value → priority read.

    Returns:
        ``"skip"``, ``"read"``, or ``"priority"``.
    """
    if score < skip_below:
        return "skip"
    if score >= read_above:
        return "priority"
    return "read"


def score_chapter_record(
    content: str,
    fingerprint: frozenset[str],
    *,
    skip_below: float = 10.0,
    read_above: float = 25.0,
) -> dict[str, object]:
    """Score one chapter against a corpus fingerprint.

    Score = 100 × (distinct fingerprint bigrams present) ÷ fingerprint
    size — the share of the manual's technical vocabulary this chapter
    carries, on a 0-100 scale comparable across manuals of any size and
    language.  Chapters that fail to substantively reuse the manual's
    vocabulary (prefaces/safety/bare schematics) score near zero and
    classify as ``"skip"`` so the model never reads them.

    Args:
        content: Raw chapter markdown text.
        fingerprint: Fingerprint from :func:`build_fingerprint`.
        skip_below: Threshold for ``"skip"`` classification.
        read_above: Threshold for ``"priority"`` classification.

    Returns:
        ``{"score": float, "action": str, "signal_breakdown": {name: count}}``.
    """
    grams = _bigrams(_clean(content))
    hits = 0
    for gram in grams:
        if gram in fingerprint:
            hits += 1
    score = round(hits * 100 / max(len(fingerprint), 1), 2)
    return {
        "score": score,
        "action": classify_score(score, skip_below=skip_below, read_above=read_above),
        "signal_breakdown": {"fingerprint_hits": hits},
    }


def should_auto_register_no_entity_from_toc(*, rootsection: str, section: str, title: str) -> bool:
    """True when a chapter is administrative boilerplate by its TOC position."""
    probe = f"{rootsection} {section} {title}".lower()
    return any(keyword in probe for keyword in ADMIN_SECTION_KEYWORDS)


def should_auto_register_no_entity(
    content: str,
    score_record: dict[str, object] | None,
    *,
    directory_administrative: bool,
) -> bool:
    """True when a chapter is administrative or scores below the skip threshold."""
    if directory_administrative:
        return True
    if not isinstance(score_record, dict):
        return False
    return str(score_record.get("action", "")) == "skip"
