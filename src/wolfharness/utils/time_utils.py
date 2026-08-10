"""Date and time utilities."""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Literal


TimeZoneMode = Literal["utc", "local"]


def get_now(tz_mode: TimeZoneMode = "utc") -> datetime:
    """Get current datetime in UTC or local timezone."""
    now = datetime.now(UTC)
    return now.astimezone() if tz_mode == "local" else now


def now_ms() -> int:
    """Return current wall-clock time in milliseconds as integer.

    Uses ``time.time_ns()`` (integer nanoseconds) with floor division to avoid
    the float truncation bug in ``int(time.time() * 1000)`` which can produce
    non-monotonic results at microsecond boundaries (off by 1ms).

    !!! warning "Not monotonic"
        ``time.time_ns()`` reads the system real-time clock (CLOCK_REALTIME),
        which can be adjusted by NTP. Small backward jumps (1-100ms) are
        common on macOS/Linux. Callers that require monotonic ordering
        (e.g., ID generation in ``identifiers.py``) must handle backward
        clock jumps themselves.
    """
    return time.time_ns() // 1_000_000


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime (UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def parse_iso_timestamp(value: str, *, fallback: datetime | None = None) -> datetime:
    """Parse an ISO 8601 timestamp string, always returning a timezone-aware datetime.

    Handles 'Z' suffix and assumes UTC for timezone-naive strings. Falls back
    to the provided fallback or current UTC time on parse failure.

    Args:
        value: ISO timestamp string (may use 'Z' instead of '+00:00', may be naive)
        fallback: Datetime to return on parse failure (defaults to current UTC time)

    Returns:
        Parsed timezone-aware datetime in UTC, or fallback on failure.
    """
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return fallback if fallback is not None else get_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
