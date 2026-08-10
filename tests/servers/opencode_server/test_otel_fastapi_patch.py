"""Tests for the OpenTelemetry FastAPI route-details patch.

FastAPI >= 0.136 wraps ``include_router`` sub-routers in ``_IncludedRouter``
nodes that carry no ``path`` attribute. OTel's ``_get_route_details`` reads
``route.path`` in its ``Match.PARTIAL`` branch without guarding, crashing on
dynamic routes (e.g. ``/session/{id}/message``) and turning valid requests into
500s. ``_safe_get_route_details`` guards that branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from starlette.routing import Match, Route

from wolfharness_server.opencode_server.otel_fastapi_patch import (
    _safe_get_route_details,
    patch_otel_fastapi_route_details,
)


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from fastapi import FastAPI


def _app_with_included_router() -> FastAPI:
    """Build a FastAPI app with a dynamic route inside an included router."""
    from fastapi import APIRouter, FastAPI

    sub = APIRouter()

    @sub.get("/session/{session_id}/message")
    async def session_message(session_id: str) -> dict[str, str]:
        return {"ok": session_id}

    app = FastAPI()
    app.include_router(sub)
    return app


def _scope_for(app: FastAPI, path: str) -> dict[str, Any]:
    """Build a minimal ASGI scope for an HTTP request to ``path``."""
    return {
        "type": "http",
        "app": app,
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "scheme": "http",
    }


def test_included_router_entries_have_no_path() -> None:
    """Sanity: FastAPI >= 0.136 wraps include_router routes in _IncludedRouter."""
    app = _app_with_included_router()

    pathless = [r for r in app.routes if not hasattr(r, "path")]
    assert pathless, "expected at least one _IncludedRouter entry without .path"
    assert all(type(r).__name__ == "_IncludedRouter" for r in pathless), (
        "pathless routes should be _IncludedRouter wrappers"
    )


def test_original_otel_route_details_crashes_on_partial_match() -> None:
    """The unpatched OTel helper crashes on a partial-match pathless route.

    Guards the fix: this proves the bug exists and the patch is necessary.

    A partial match against an ``_IncludedRouter`` wrapper happens when the
    request method differs from the route's (e.g. POST against a GET route):
    the path-prefix matches (``Match.PARTIAL``) but the wrapper has no ``path``
    attribute, so OTel's unguarded ``route.path`` access raises.
    """
    from fastapi import APIRouter, FastAPI

    verify_app = FastAPI()
    sub = APIRouter()

    @sub.get("/session/{session_id}/message")
    async def sm(session_id: str) -> dict[str, str]:
        return {"ok": session_id}

    verify_app.include_router(sub)

    # POST against a GET-only route inside an included router → PARTIAL match.
    partial_routes = []
    for route in verify_app.routes:
        if isinstance(route, Route):
            continue  # skip explicit Route objects (they carry .path)
        match, _ = route.matches({
            "path": "/session/ses_x/message",
            "path_params": {},
            "root_path": "",
            "method": "POST",
            "type": "http",
        })
        if match == Match.PARTIAL:
            partial_routes.append(route)

    assert partial_routes, "expected a partial match against _IncludedRouter"
    for route in partial_routes:
        with pytest.raises(AttributeError):
            object.__getattribute__(route, "path")

    # The guarded helper must resolve the same POST without crashing.
    resolved_route = _safe_get_route_details(_scope_for(verify_app, "/session/ses_x/message"))
    assert resolved_route is not None


def test_safe_route_details_handles_pathless_included_router() -> None:
    """The patched helper falls back to scope path, never crashing."""
    from fastapi import APIRouter, FastAPI

    app = FastAPI()
    sub = APIRouter()

    @sub.post("/session/{session_id}/message")
    async def sm(session_id: str) -> dict[str, str]:
        return {"ok": session_id}

    app.include_router(sub)

    resolved_route = _safe_get_route_details(_scope_for(app, "/session/ses_x/message"))
    assert resolved_route is not None


def test_patch_is_idempotent_and_effective() -> None:
    """patch_otel_fastapi_route_details rewires the OTel module global."""
    import opentelemetry.instrumentation.fastapi as otel_fastapi

    patch_otel_fastapi_route_details()
    patch_otel_fastapi_route_details()

    assert otel_fastapi._get_route_details is _safe_get_route_details
