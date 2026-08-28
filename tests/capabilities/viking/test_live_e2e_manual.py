"""Manual live end-to-end: real Viking server, real capability tools.

Run explicitly (requires live server):
    uv run pytest tests/capabilities/viking/test_live_e2e_manual.py -q
"""

import json
import pathlib
import tempfile
import uuid

import pytest

from wolfharness.capabilities.viking import VikingCapability
from wolfharness.capabilities.viking.tools import build_tools


def _get_tool(tools: list, name: str):
    for t in tools:
        if getattr(t, "__name__", "") == name:
            return t
    raise AssertionError(f"tool {name} not found")


class _Ctx:
    deps = None


@pytest.mark.asyncio
@pytest.mark.real_mcp
async def test_live_upload_tree_then_link_relations(allow_model_requests: None) -> None:
    conf = json.loads(pathlib.Path.home().joinpath(".openviking/ovcli.conf").read_text())
    ns = f"e2e_live_{uuid.uuid4().hex[:6]}"
    root = pathlib.Path(tempfile.mkdtemp(prefix="ov_live_"))
    (root / "Fault").mkdir()
    (root / "Timeout.md").write_text("# 超时故障", encoding="utf-8")
    (root / "Fault" / "NoStart.md").write_text("# 无法启动", encoding="utf-8")
    (root / "Fault" / "BlackSmoke.md").write_text("# 冒黑烟", encoding="utf-8")
    rel_file = root / "backlinks_index.json"
    rel_file.write_text(
        json.dumps({
            f"viking://resources/{ns}/Fault/NoStart.md": [
                f"viking://resources/{ns}/Fault/BlackSmoke.md"
            ],
            f"viking://resources/{ns}/Timeout.md": [f"viking://resources/{ns}/Fault/NoStart.md"],
        }),
        encoding="utf-8",
    )

    cap = VikingCapability(
        mode="all",
        url=conf["url"],
        api_key=conf["api_key"],
        enable_link=True,
        timeout=180.0,
    )
    async with cap:
        tools = build_tools(cap)
        upload = _get_tool(tools, "viking_upload_tree")
        up_res = await upload(_Ctx(), path=str(root), to=f"viking://resources/{ns}")
        print("\nLIVE upload FULL:", up_res.return_value)
        assert not up_res.return_value.lower().startswith("viking_upload_tree error")

        links = _get_tool(tools, "viking_link_relations")
        lk_res = await links(
            _Ctx(), relations_file=str(rel_file), namespace_base=f"viking://resources/{ns}"
        )
        print("LIVE link_relations FULL:", lk_res.return_value)
        assert not lk_res.return_value.lower().startswith("viking_link_relations error")

        client = await cap._ensure_client()
        r1 = await client.relations(f"viking://resources/{ns}/Fault/NoStart/NoStart.md")
        r2 = await client.relations(f"viking://resources/{ns}/Timeout/Timeout.md")
        r3 = await client.relations(f"viking://resources/{ns}/Fault/BlackSmoke/BlackSmoke.md")
        print("LIVE relations NoStart:", json.dumps(r1, ensure_ascii=False))
        print("LIVE relations Timeout:", json.dumps(r2, ensure_ascii=False))
        print("LIVE relations BlackSmoke:", json.dumps(r3, ensure_ascii=False))
        assert r1, f"missing edges r1={r1}"
        assert r2, f"missing edges r2={r2}"
        assert r3, f"missing edges r3={r3}"
        assert "linked=3, failed=0" in lk_res.return_value, f"link failures: {lk_res.return_value}"

        await client.rm(f"viking://resources/{ns}", recursive=True)
    print("LIVE E2E PASSED")
