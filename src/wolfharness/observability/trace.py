"""Trace context helpers for logging trace IDs at entry points.

Provides a utility to extract the current W3C trace ID from the active
OpenTelemetry span, making it easy to correlate logs with traces in
SigNoz/Jaeger/Honeycomb.
"""

from __future__ import annotations

from opentelemetry import trace


def get_trace_id() -> str | None:
    """Return the current trace ID as a 32-char hex string, or None.

    Use at entry points to log the trace ID for easy search:

    >>> logger.info("Turn started", trace_id=get_trace_id(), turn_id=...)
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx is not None and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return None
