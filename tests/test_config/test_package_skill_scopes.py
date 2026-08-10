from __future__ import annotations

import pytest
import yaml

from wolfharness_config import resolution


pytestmark = pytest.mark.unit


def test_include_package_records_package_skill_scopes(tmp_path, monkeypatch):
    config_path = tmp_path / "wolfharness.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "include_packages": ["rebuttal_agent.config:agents.yaml"],
                "skills": {"paths": [str(tmp_path / "host-skills")]},
                "agents": {"engineer": {"type": "native", "model": "test"}},
            },
        ),
        encoding="utf-8",
    )

    def fake_load_package_yaml(ref: str) -> dict:
        assert ref == "rebuttal_agent.config:agents.yaml"
        return {
            "skills": {"paths": [str(tmp_path / "rebuttal-skills")]},
            "agents": {"rebuttal_agent": {"type": "native", "model": "test"}},
            "teams": {"fta_content_review_team": {"mode": "parallel", "members": []}},
        }

    monkeypatch.setattr(resolution, "_load_package_yaml", fake_load_package_yaml)

    resolved = resolution.resolve_config(
        explicit_path=config_path,
        include_global=False,
        include_project=False,
    )

    scopes = resolved.data["_skill_scopes"]
    assert scopes["nodes"] == {
        "rebuttal_agent": "rebuttal_agent",
        "fta_content_review_team": "rebuttal_agent",
    }
    assert {"scope": "host", "path": str(tmp_path / "host-skills")} in scopes["paths"]
    assert {
        "scope": "rebuttal_agent",
        "path": str(tmp_path / "rebuttal-skills"),
    } in scopes["paths"]
