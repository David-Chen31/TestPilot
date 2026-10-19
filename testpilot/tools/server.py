"""TestPilot 的 MCP server：把 RepoTools 的方法注册成 MCP 工具，走 stdio 传输。

每个任务启动一个 server 进程，绑定到该任务的工作副本::

    python -m testpilot.tools.server --root /tmp/testpilot-xxx/repo

一个进程只服务一个副本，任务之间天然隔离，也不需要在每次工具调用里传仓库路径。
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from testpilot.tools.repo_tools import RepoTools

TOOL_NAMES = [
    "read_file",
    "list_symbols",
    "search_code",
    "write_test_file",
    "run_pytest",
    "check_validity",
    "coverage_report",
    "run_mutation",
]


def build_server(tools: RepoTools) -> FastMCP:
    server = FastMCP("testpilot", log_level="WARNING")
    for name in TOOL_NAMES:
        server.add_tool(getattr(tools, name), name=name)
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TestPilot MCP server (stdio)")
    parser.add_argument("--root", required=True, help="工作副本根目录")
    parser.add_argument("--python-path", action="append", default=None, help="相对 root 的导入路径，可多次指定")
    parser.add_argument("--pytest-timeout", type=float, default=60.0)
    parser.add_argument("--mutation-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    tools = RepoTools(
        args.root,
        python_paths=args.python_path or ["."],
        pytest_timeout=args.pytest_timeout,
        mutation_timeout=args.mutation_timeout,
    )
    build_server(tools).run(transport="stdio")


if __name__ == "__main__":
    main()
