"""Test that merge_queue_into_iterator raises ImportError.

Regression test for structured concurrency migration:
- merge_queue_into_iterator should no longer be importable
- Trying to import it should raise ImportError
- Other stream utilities should still work
"""

from __future__ import annotations

import pytest

from wolfharness.utils.streams import FileChange, FileOpsTracker


def test_merge_queue_into_iterator_raises_import_error() -> None:
    """Verify merge_queue_into_iterator import raises ImportError.

    Regression test for structured concurrency migration:
    - merge_queue_into_iterator was removed in Phase 5
    - Attempting to import should raise ImportError
    - Other stream utilities should still work
    """
    with pytest.raises(ImportError):
        from wolfharness.utils.streams import merge_queue_into_iterator  # noqa: F401

    # Verify other utilities still work
    assert FileChange is not None
    assert FileOpsTracker is not None
