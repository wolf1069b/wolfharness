"""Span helpers for safe instrumentation in async generators.

When ``with logfire.span(...)`` is used inside an async generator that
contains ``yield``, OpenTelemetry's context management can raise
``ValueError: Token was created in a different Context`` during
``GeneratorExit``. This happens because the span's ``__exit__`` calls
``context.detach(token)`` which tries to ``ContextVar.reset(token)``,
but the token was created in a different ``contextvars.Context`` than
the one active when the generator is closed.

Logfire's ``LogfireSpan.__exit__`` is decorated with
``@handle_internal_errors`` which catches this ``ValueError`` and
suppresses it — BUT this also prevents ``_end()`` from being called,
since the exception fires in ``_detach()`` before ``_end()`` is reached.
An unended span is never exported, causing "Missing Span" in SigNoz.

This module fixes the issue by calling ``_detach()`` and ``_end()``
separately, ensuring the span is always ended even if context detach
fails. It also suppresses the OTel log warning for cleanliness.

Implementation note: ``safe_span`` is a CLASS (not ``@contextmanager``)
because ``@contextmanager``'s ``__exit__`` explicitly skips
``throw(GeneratorExit)`` for safety — meaning the ``finally`` block
would never run when ``GeneratorExit`` propagates through an async
generator that uses ``with safe_span(...)``.
"""

from __future__ import annotations

import logging
from typing import Any

import logfire

from wolfharness.log import get_logger


_log = get_logger(__name__)


class _DetachContextFilter(logging.Filter):
    """Suppress 'Failed to detach context' warnings from OpenTelemetry.

    These warnings occur when async generators are closed via
    GeneratorExit and the OTel context token was created in a different
    contextvars.Context. The error is non-fatal — spans are still
    recorded correctly.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to detach context" not in record.getMessage()


# Install the filter on the opentelemetry.context logger.
_otel_context_logger = logging.getLogger("opentelemetry.context")
_otel_context_logger.addFilter(_DetachContextFilter())


class safe_span:  # noqa: N801
    """Create a logfire span safe for use in async generators.

    Use this instead of ``with logfire.span(...)`` inside async
    generators that contain ``yield`` statements.

    Unlike ``@contextmanager``-based implementations, this class-based
    version ensures ``__exit__`` always runs — even when
    ``GeneratorExit`` propagates through the enclosing async generator.
    This is critical because ``@contextmanager``'s ``__exit__``
    explicitly skips ``throw(GeneratorExit)``, which would leave the
    span unended and unexported.

    Args:
        name: Span name (e.g., ``"turn.native"``).
        **attributes: Span attributes (e.g., ``session_id=...``).
    """

    def __init__(self, name: str, **attributes: Any) -> None:
        self._span = logfire.span(name, **attributes)

    def __enter__(self) -> logfire.LogfireSpan:
        self._span.__enter__()
        return self._span

    def __exit__(self, *exc_info: object) -> None:
        # Detach context first. This may fail with ValueError when the
        # async generator is closed via GeneratorExit in a different
        # contextvars.Context than where the span was entered.
        try:
            self._span._detach()  # type: ignore[no-untyped-call]
        except Exception:  # noqa: BLE001
            _log.debug("safe_span _detach() failed for %s", self._span.name, exc_info=True)
        # End the span. This MUST always run, even if detach failed.
        # An unended span is never exported by the OTel exporter,
        # causing "Missing Span" in backends like SigNoz.
        try:
            self._span._end()
        except Exception:  # noqa: BLE001
            _log.warning("safe_span _end() failed for %s", self._span.name, exc_info=True)
