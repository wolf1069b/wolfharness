"""Workaround for OpenTelemetry FastAPI instrumentation with FastAPI >= 0.136.

FastAPI >= 0.136 wraps sub-routers added via ``include_router`` in an
``_IncludedRouter`` wrapper. These wrappers are matchable base routes that
intentionally carry no ``path`` attribute.

``opentelemetry.instrumentation.fastapi._get_route_details`` walks
``app.routes`` and reads ``route.path``. Its ``Match.FULL`` branch already
guards the missing-``path`` case with try/except, but the ``Match.PARTIAL``
branch does not — so any request that only *partially* matches a
lazily-included route (for example a dynamic path like ``/session/{id}/message``
inside an included router) raises ``AttributeError`` and turns into a 500.

This module replaces the module-level ``_get_route_details`` helper in
``opentelemetry.instrumentation.fastapi`` with a guarded copy.
``_get_default_span_details`` resolves it as a module global on every call, so
the reassignment takes effect without re-instrumenting the app.

Why a runtime patch instead of upgrading the dependency?
--------------------------------------------------------
The upstream OpenTelemetry fix landed in ``opentelemetry-instrumentation-
fastapi>=0.64b0``, but it requires ``opentelemetry-semantic-conventions``
pinned to the exact same ``0.6xb0`` version, i.e. ``==0.64b0``. The pinned
``mistralai`` (a transitive dep of ``pydantic-ai-slim[mistral]``) declares
``opentelemetry-semantic-conventions<0.61`` and cannot coexist with that
upgrade, so the dependency graph is unsatisfiable unless we override
``mistralai``'s metadata.

We deliberately avoid that override (see below) and keep this patch instead:

- ``[[tool.uv.dependency-metadata]]`` overrides DO NOT propagate to downstream
  projects that depend on ``wolfharness`` (e.g. ``xeno-agent``). Every consumer
  would have to repeat the same override in its own ``pyproject.toml``, which
  turns a single dependency bump into a coordination burden across repositories.
- The patch is fully local to this package: it changes no dependency versions,
  so all downstream projects keep working without edits.
- ``mistralai`` only imports stable attribute constants from
  ``opentelemetry.semconv.attributes``; the semantic-conventions contents it
  relies on are stable across the 0.60-0.65 betas.
- ``opentelemetry-instrumentation-fastapi`` is a pre-release (``0.6xb0``); the
  patch is immune to beta-version churn and works on any version in range.

Revisit the upstream upgrade (removing this module) if either condition changes:
(1) ``mistralai`` relaxes its ``opentelemetry-semantic-conventions`` pin, or
(2) ``dependency-metadata`` overrides start propagating transitively.
"""

from __future__ import annotations

from typing import Any

from starlette.routing import Match, Route


def _safe_get_route_details(scope: dict[str, Any]) -> str | None:
    """Route-details resolver matching OTel's with a guarded PARTIAL branch.

    Identical to the OTel implementation except the ``Match.PARTIAL`` branch
    falls back to ``scope["path"]`` when the matched route has no ``path``
    attribute (FastAPI >= 0.136 ``_IncludedRouter`` wrappers).
    """
    app = scope["app"]
    route: str | None = None

    for starlette_route in app.routes:
        match, _ = (
            Route.matches(starlette_route, scope)
            if isinstance(starlette_route, Route)
            else starlette_route.matches(scope)
        )
        if match == Match.FULL:
            try:
                route = starlette_route.path
            except AttributeError:
                # routes added via host routing won't have a path attribute
                route = scope.get("path")
            break
        if match == Match.PARTIAL:
            try:
                route = starlette_route.path
            except AttributeError:
                # FastAPI >= 0.136 ``_IncludedRouter`` wrappers carry no path.
                route = scope.get("path")
    return route


def patch_otel_fastapi_route_details() -> None:
    """Replace OTel's ``_get_route_details`` with the guarded version.

    Idempotent: re-applying ends with the OTel module pointing at our guarded
    function either way. Safe to call before ``logfire.instrument_fastapi``.
    """
    import opentelemetry.instrumentation.fastapi as otel_fastapi

    otel_fastapi.__dict__["_get_route_details"] = _safe_get_route_details


__all__ = ["patch_otel_fastapi_route_details"]
