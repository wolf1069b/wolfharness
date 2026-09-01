"""L4 subprocess E2E test: ACP server startup failures must surface on stderr.

Reproduces the user scenario where a config passes early validation but fails
at agent construction (e.g. ``model: nosuchprovider:nonexistent``): the
process must exit with code 1 and print the exception plus the log file path
to stderr instead of exiting silently.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.e2e.conftest import SKIP_WINDOWS


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_WINDOWS, reason="stdio subprocess issues"),
]


BAD_CONFIG = """\
agents:
  bad_agent:
    type: native
    model: nosuchprovider:nonexistent
    system_prompt: "x"
"""


def test_acp_startup_failure_prints_to_stderr(tmp_path) -> None:
    """Config that fails agent construction surfaces the error on stderr."""
    config_path = tmp_path / "bad_config.yml"
    config_path.write_text(BAD_CONFIG)

    env = os.environ.copy()
    env["OBSERVABILITY_ENABLED"] = "false"
    env["LOGFIRE_DISABLE"] = "true"
    env["HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "wolfharness_cli", "serve-acp", str(config_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )

    assert result.returncode == 1
    assert "failed to start" in result.stderr
    assert "Unknown provider" in result.stderr
    assert "log file" in result.stderr
    assert result.stdout == ""
