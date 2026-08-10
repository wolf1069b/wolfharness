"""Configuration for SessionPool tunable parameters.

Extracted from hardcoded values in SessionPool and SessionController
as part of session-debt-cleanup Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionPoolConfig:
    """Configuration for SessionPool tunable parameters.

    Attributes:
        message_cache_maxsize: Maximum number of sessions in the
            ``_message_cache`` LRU. When exceeded, least-recently-used
            entries for **inactive** sessions are evicted. Active
            sessions are never evicted. Default: 1000.
        session_ttl_seconds: Time-to-live for idle sessions in seconds.
            Sessions with no active run whose ``last_active_at`` is
            older than this are eligible for cleanup. Default: 3600 (1h).
        cleanup_interval_seconds: Interval for the session TTL cleanup
            loop in seconds. Default: 1800 (30 min, i.e. TTL / 2).
        deferred_cleanup_interval_seconds: Interval for the deferred
            call expiry cleanup loop in seconds. Default: 60.
    """

    message_cache_maxsize: int = 1000
    session_ttl_seconds: float = 3600.0
    cleanup_interval_seconds: float = 1800.0
    deferred_cleanup_interval_seconds: float = 60.0
