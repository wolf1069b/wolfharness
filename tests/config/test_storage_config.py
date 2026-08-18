"""Tests for portable WolfHarness storage configuration."""

from pathlib import Path

import pytest

from wolfharness_config.storage import get_database_path


pytestmark = pytest.mark.unit


def test_database_path_uses_configured_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WOLFHARNESS_DATABASE_URL", raising=False)
    monkeypatch.setenv("WOLFHARNESS_DATA_DIR", str(tmp_path / "history"))

    assert get_database_path() == f"sqlite:///{tmp_path / 'history' / 'history.db'}"


def test_database_url_takes_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOLFHARNESS_DATA_DIR", str(tmp_path / "history"))
    monkeypatch.setenv("WOLFHARNESS_DATABASE_URL", "sqlite:////private/tmp/wolfharness-smoke.db")

    assert get_database_path() == "sqlite:////private/tmp/wolfharness-smoke.db"
