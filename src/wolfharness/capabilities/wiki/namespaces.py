"""Canonical OpenViking resource namespace resolution.

Single source of truth for the ``viking://resources/<namespace>`` roots used
across the harness and the wiki build store. Historically each module
hard-coded a deployment-specific namespace. Runtime namespaces are now
required environment inputs, so changing a library or target never requires
a code edit and a missing deployment value fails before any write begins.

The runtime wiki store consumes these through the same environment variables,
so the resolvers here are consistent with how ``create_wiki_store`` picks its root:
``VIKING_NAMESPACE`` -> ``viking://resources/<ns>``.

Environment variables:

- ``VIKING_NAMESPACE``: wiki build target resource namespace.
- ``VIKING_RAW_NAMESPACE``: raw manual library resource namespace.
"""

from __future__ import annotations

import os


_RESOURCES_PREFIX = "viking://resources/"


class NamespaceConfigurationError(RuntimeError):
    """Raised when a required Viking namespace was not configured."""


def _required_namespace(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip().strip("/")
    if not value:
        raise NamespaceConfigurationError(
            f"{env_name} is required when WIKI_STORAGE_BACKEND=viking",
        )
    return value


def wiki_namespace() -> str:
    """Return the wiki build target resource namespace.

    Resolved from the required ``VIKING_NAMESPACE`` deployment setting.
    """
    return _required_namespace("VIKING_NAMESPACE")


def raw_namespace() -> str:
    """Return the raw manual library resource namespace.

    Resolved from the required ``VIKING_RAW_NAMESPACE`` deployment setting.
    """
    return _required_namespace("VIKING_RAW_NAMESPACE")


def resources_root(namespace: str) -> str:
    """Return the ``viking://resources/<namespace>`` root for a namespace."""
    normalized = namespace.strip().strip("/")
    if not normalized:
        raise NamespaceConfigurationError("Viking resource namespace must not be empty")
    return f"{_RESOURCES_PREFIX}{normalized}"


def wiki_resources_root() -> str:
    """Return the wiki build target root URI (``viking://resources/<ns>``).

    Mirrors ``WikiStore.root_uri`` for the configured namespace, so module
    defaults stay consistent with the runtime store root even before a
    store is created.
    """
    return resources_root(wiki_namespace())


def raw_resources_root() -> str:
    """Return the raw manual library root URI (``viking://resources/<ns>``)."""
    return resources_root(raw_namespace())
