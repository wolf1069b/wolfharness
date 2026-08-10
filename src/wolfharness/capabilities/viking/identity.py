"""Viking identity resolution — ``VikingIdentity`` and API key decoding.

This module provides the pure-function API key decoder and the
``VikingIdentity`` frozen dataclass used by ``VikingCapability`` to
construct user-scoped URIs dynamically.

Resolution chain (implemented on ``VikingCapability._resolve_identity``):
1. Explicit config fields (``self.user`` + ``self.account``)
2. New-format API key decode (``_try_decode_api_key``)
3. ``/health`` endpoint query
4. Fallback to ``"default"``
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VikingIdentity:
    """Resolved Viking identity for URI construction.

    Attributes:
        account_id: Viking account identifier.
        user_id: Viking user identifier — used in ``viking://user/{user_id}/`` URIs.
        role: Viking role (e.g. ``"user"``, ``"admin"``).
    """

    account_id: str
    user_id: str
    role: str = "user"


def _try_decode_api_key(api_key: str) -> tuple[str, str] | None:
    """Attempt to decode ``account_id`` and ``user_id`` from a new-format API key.

    Viking new-format API keys follow the pattern
    ``base64(account).base64(user).base64(signature)``.
    The first two dot-separated parts are decoded as standard base64
    to extract ``account_id`` and ``user_id``.

    Args:
        api_key: The raw API key string.

    Returns:
        A ``(account_id, user_id)`` tuple if decoding succeeds,
        or ``None`` if the key is malformed or decoding fails.
    """
    if not api_key or "." not in api_key:
        return None

    parts = api_key.split(".")
    # Need at least account, user, and signature parts.
    if len(parts) < 3:  # noqa: PLR2004
        return None

    try:
        account_id = base64.b64decode(parts[0] + "==").decode("utf-8")
        user_id = base64.b64decode(parts[1] + "==").decode("utf-8")
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None

    if not account_id or not user_id:
        return None

    return (account_id, user_id)
