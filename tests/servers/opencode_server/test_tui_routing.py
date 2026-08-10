"""Tests for TUI event routing filter and /global/routing-check endpoint.

Covers all 3 rules of the OpenCode TUI routing filter:
1. Sync events always dropped
2. Global directory always passes (except sync)
3. Directory must match exactly (string comparison, no normalization)

Plus edge cases and the HTTP endpoint integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from wolfharness_server.opencode_server.models import GlobalEvent
from wolfharness_server.opencode_server.routes.routing import (
    RoutingCheckResponse,
    tui_event_filter,
)


pytestmark = pytest.mark.integration


if TYPE_CHECKING:
    from httpx import AsyncClient


# =============================================================================
# Helper
# =============================================================================


def _make_event(
    directory: str = "/project",
    payload_type: str | None = None,
) -> GlobalEvent:
    """Create a GlobalEvent for testing."""
    payload: dict[str, Any] = {}
    if payload_type is not None:
        payload["type"] = payload_type
    return GlobalEvent(directory=directory, payload=payload)


# =============================================================================
# Rule 1: sync events always dropped
# =============================================================================


def test_sync_event_dropped() -> None:
    """Sync events are always dropped regardless of other conditions."""
    event = _make_event(directory="global", payload_type="sync")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "sync_dropped"


def test_sync_event_dropped_even_with_matching_directory() -> None:
    """Sync event with matching directory is still dropped."""
    event = _make_event(directory="/project", payload_type="sync")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "sync_dropped"


def test_non_sync_payload_type_not_dropped() -> None:
    """Non-sync payload types are not affected by rule 1."""
    event = _make_event(directory="/project", payload_type="session.status")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is True
    assert reason == "directory_match"


def test_no_payload_type_not_dropped() -> None:
    """Event without a payload type is not affected by rule 1."""
    event = _make_event(directory="/project")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is True
    assert reason == "directory_match"


# =============================================================================
# Rule 2: global directory always passes (except sync)
# =============================================================================


def test_global_directory_passes() -> None:
    """Event with directory='global' always passes (non-sync)."""
    event = _make_event(directory="global")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is True
    assert reason == "global_directory"


def test_global_directory_passes_regardless_of_project_directory() -> None:
    """Global directory passes even when project_directory differs."""
    event = _make_event(directory="global")
    passed, reason = tui_event_filter(event, "/completely/different")
    assert passed is True
    assert reason == "global_directory"


def test_global_directory_with_sync_still_dropped() -> None:
    """Sync event with global directory is dropped (rule 1 takes precedence)."""
    event = _make_event(directory="global", payload_type="sync")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "sync_dropped"


# =============================================================================
# Rule 3: directory must match exactly
# =============================================================================


def test_directory_match_passes() -> None:
    """Exact directory match → passes."""
    event = _make_event(directory="/project")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is True
    assert reason == "directory_match"


def test_directory_mismatch_fails() -> None:
    """Directory mismatch → fails."""
    event = _make_event(directory="/other")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "directory_mismatch"


def test_directory_trailing_slash_not_normalized() -> None:
    """Trailing slash vs no trailing slash → NOT equal (no normalization)."""
    event = _make_event(directory="/project/")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "directory_mismatch"


def test_directory_no_trailing_vs_trailing_not_normalized() -> None:
    """No trailing slash vs trailing slash → NOT equal (no normalization)."""
    event = _make_event(directory="/project")
    passed, reason = tui_event_filter(event, "/project/")
    assert passed is False
    assert reason == "directory_mismatch"


def test_directory_case_sensitive() -> None:
    """Directory comparison is case-sensitive."""
    event = _make_event(directory="/Project")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "directory_mismatch"


def test_directory_empty_string_mismatch() -> None:
    """Empty string directory doesn't match a real path."""
    event = _make_event(directory="")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "directory_mismatch"


def test_directory_match_with_spaces() -> None:
    """Directory with spaces matches exactly."""
    event = _make_event(directory="/path with spaces/project")
    passed, reason = tui_event_filter(event, "/path with spaces/project")
    assert passed is True
    assert reason == "directory_match"


# =============================================================================
# Rule priority / interaction edge cases
# =============================================================================


def test_rule_priority_sync_over_global() -> None:
    """Rule 1 (sync) takes precedence over rule 2 (global directory)."""
    event = _make_event(directory="global", payload_type="sync")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "sync_dropped"


def test_full_path_cjk_directory() -> None:
    """CJK characters in directory match exactly."""
    event = _make_event(directory="/项目/代码")
    passed, reason = tui_event_filter(event, "/项目/代码")
    assert passed is True
    assert reason == "directory_match"


# =============================================================================
# RoutingCheckResponse model tests
# =============================================================================


def test_routing_check_response_serialization() -> None:
    """RoutingCheckResponse serializes correctly with camelCase aliases."""
    response = RoutingCheckResponse(would_pass=True, reason="directory_match")
    dumped = response.model_dump(by_alias=True, exclude_none=True)
    assert dumped["wouldPass"] is True
    assert dumped["reason"] == "directory_match"


def test_routing_check_response_all_reasons() -> None:
    """All valid reason values produce valid RoutingCheckResponse."""
    valid_reasons = [
        "sync_dropped",
        "global_directory",
        "directory_match",
        "directory_mismatch",
    ]
    for reason in valid_reasons:
        response = RoutingCheckResponse(would_pass=True, reason=reason)
        assert response.reason == reason


# =============================================================================
# /global/routing-check HTTP endpoint tests
# =============================================================================


@pytest.mark.anyio
async def test_routing_check_directory_match(async_client: AsyncClient) -> None:
    """Directory match returns would_pass=True, reason=directory_match."""
    response = await async_client.get(
        "/global/routing-check",
        params={"directory": "/my/project", "project_directory": "/my/project"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["wouldPass"] is True
    assert data["reason"] == "directory_match"


@pytest.mark.anyio
async def test_routing_check_directory_mismatch(async_client: AsyncClient) -> None:
    """Directory mismatch returns would_pass=False, reason=directory_mismatch."""
    response = await async_client.get(
        "/global/routing-check",
        params={"directory": "/different/path"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["wouldPass"] is False
    assert data["reason"] == "directory_mismatch"


@pytest.mark.anyio
async def test_routing_check_global_directory(async_client: AsyncClient) -> None:
    """Global directory returns would_pass=True, reason=global_directory."""
    response = await async_client.get(
        "/global/routing-check",
        params={"directory": "global"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["wouldPass"] is True
    assert data["reason"] == "global_directory"


@pytest.mark.anyio
async def test_routing_check_sync_dropped(async_client: AsyncClient) -> None:
    """Sync event with global directory is dropped by rule 1.

    Note: the endpoint constructs a GlobalEvent with an empty payload {},
    so we can't directly test sync_dropped via the endpoint (no payload.type
    parameter is exposed). This test verifies the general behavior.
    """
    # The endpoint doesn't support payload_type param, so we test the
    # pure function directly for sync_dropped and verify the endpoint
    # works for the other cases
    event = _make_event(directory="global", payload_type="sync")
    passed, reason = tui_event_filter(event, "/project")
    assert passed is False
    assert reason == "sync_dropped"


@pytest.mark.anyio
async def test_routing_check_custom_project_directory(async_client: AsyncClient) -> None:
    """Custom project_directory parameter overrides state.working_dir."""
    response = await async_client.get(
        "/global/routing-check",
        params={
            "directory": "/custom/project",
            "project_directory": "/custom/project",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["wouldPass"] is True
    assert data["reason"] == "directory_match"


@pytest.mark.anyio
async def test_routing_check_default_project_directory(async_client: AsyncClient) -> None:
    """Without project_directory param, uses server's working_dir.

    Since the server's working_dir is a temp directory, we use a known
    different directory to verify it fails (directory_mismatch), which
    confirms the default is being used rather than matching any value.
    """
    response = await async_client.get(
        "/global/routing-check",
        params={"directory": "/definitely/not/the/working/dir"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["wouldPass"] is False
    assert data["reason"] == "directory_mismatch"
