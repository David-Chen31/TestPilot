"""通过 MCP（stdio）调用 TestPilot 工具的小客户端，用于手动验证和集成测试。

单独调用一个工具（在已有目录上，会写入文件，请对副本使用）::

    python -m testpilot.tools.client --root <目录> read_file '{"path": "sample/calc.py"}'

演示全部工具（先把仓库复制到临时工作副本）::

    python -m testpilot.tools.client --demo tests/fixtures/sample_repo --target sample/calc.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def connect(root: str | Path, python_paths: Sequence[str] = (".",)) -> AsyncIterator[ClientSession]:
    args = ["-m", "testpilot.tools.server", "--root", str(root)]
    for p in python_paths:
        args += ["--python-path", p]
    params = StdioServerParameters(command=sys.executable, args=args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call(session: ClientSession, name: str, arguments: dict[str, Any] | None = None) -> dict:
    """调用工具并返回结构化结果；工具抛异常时返回 {"error": ...}。"""
    result = await session.call_tool(name, arguments or {})
    text = "".join(getattr(c, "text", "") for c in result.content)
    if result.isError:
        return {"error": text}
    if result.structuredContent is not None:
        # 返回类型标注为 dict 时，FastMCP 会包一层 {"result": ...}
        return result.structuredContent.get("result", result.structuredContent)
    return json.loads(text)


DEMO_TEST = '''from sample.calc import add, clamp


def test_add():
    assert add(1, 2) == 3


def test_clamp_runs():
    # 覆盖了 clamp 的一条路径，但断言很弱：抓不到大多数 bug
    assert clamp(5, 0, 10) is not None
'''


async def demo(repo: Path, target: str) -> None:
    from testpilot.sandbox.workspace import Workspace

    with Workspace(repo) as ws:
        async with connect(ws.path) as session:
            tools = await session.list_tools()
            _show("list_tools", [t.name for t in tools.tools])
            _show("read_file", await call(session, "read_file", {"path": target, "end_line": 8}))
            _show("list_symbols", await call(session, "list_symbols", {"path": target}))
            _show("search_code", await call(session, "search_code", {"query": "def clamp"}))
            _show(
                "write_test_file",
                await call(session, "write_test_file", {"path": "tests_testpilot/test_calc.py", "content": DEMO_TEST}),
            )
            _show("run_pytest", await call(session, "run_pytest"))
            _show("coverage_report", await call(session, "coverage_report", {"target": target}))
            _show("run_mutation", await call(session, "run_mutation", {"target": target}))


def _show(title: str, value: Any) -> None:
    print(f"===== {title}")
    print(json.dumps(value, ensure_ascii=False, indent=2))


async def _single(root: Path, tool: str, arguments: dict) -> None:
    async with connect(root) as session:
        _show(tool, await call(session, tool, arguments))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", metavar="REPO", help="复制 REPO 到临时副本并依次调用全部工具")
    parser.add_argument("--target", default="sample/calc.py", help="--demo 时的目标源文件")
    parser.add_argument("--root", help="单个工具调用时 server 绑定的目录")
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON 格式的工具参数")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 管道默认 GBK，中文会乱码

    if args.demo:
        asyncio.run(demo(Path(args.demo), args.target))
    elif args.root and args.tool:
        asyncio.run(_single(Path(args.root), args.tool, json.loads(args.arguments)))
    else:
        parser.error("需要 --demo REPO，或 --root DIR TOOL [JSON]")


if __name__ == "__main__":
    main()
