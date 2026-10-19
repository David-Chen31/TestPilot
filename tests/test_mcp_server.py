"""端到端：启动 MCP server 子进程，经 stdio 协议调用工具。"""

import asyncio
from pathlib import Path

from testpilot.sandbox import Workspace
from testpilot.tools.client import call, connect
from testpilot.tools.server import TOOL_NAMES


def test_tools_over_mcp(sample_repo: Path, tmp_path: Path):
    async def scenario():
        with Workspace(sample_repo, base_dir=tmp_path) as ws:
            async with connect(ws.path) as session:
                listed = await session.list_tools()
                assert sorted(t.name for t in listed.tools) == sorted(TOOL_NAMES)
                # 每个工具都有给 LLM 看的描述和参数 schema
                assert all(t.description and t.inputSchema for t in listed.tools)

                symbols = await call(session, "list_symbols", {"path": "sample/calc.py"})
                assert [s["name"] for s in symbols["symbols"]] == ["add", "clamp"]

                bad = await call(session, "write_test_file", {"path": "sample/calc.py", "content": "x"})
                assert "tests_testpilot" in bad["error"]

                await call(
                    session,
                    "write_test_file",
                    {"path": "tests_testpilot/test_a.py", "content": "def test_a():\n    assert True\n"},
                )
                result = await call(session, "run_pytest")
                assert result["all_passed"] and result["passed"] == 1

    asyncio.run(scenario())
