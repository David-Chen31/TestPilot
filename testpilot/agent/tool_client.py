"""Agent 调用工具的统一接口。

正式运行时工具来自 MCP server（经 langchain-mcp-adapters 加载成 LangChain 工具）；
单元测试用 LocalToolClient 直接调用 RepoTools，省掉进程启动，逻辑完全相同。
每次调用都被记录下来，节点把它们写进轨迹。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Protocol, Sequence

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp.client.stdio import stdio_client

from testpilot.tools.repo_tools import RepoTools

_MCP_ERROR_PREFIX = "Error executing tool"


class ToolError(RuntimeError):
    pass


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    duration: float
    ok: bool
    error: str = ""

    def to_dict(self) -> dict:
        # 参数里可能有整个测试文件，轨迹里只留摘要
        args = {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v) for k, v in self.args.items()}
        return {"name": self.name, "args": args, "duration": round(self.duration, 2), "ok": self.ok, "error": self.error}


class ToolClient(Protocol):
    calls: list[ToolCallRecord]

    async def call(self, name: str, args: dict | None = None) -> dict: ...


@dataclass
class _RecordingClient:
    calls: list[ToolCallRecord] = field(default_factory=list)

    async def call(self, name: str, args: dict | None = None) -> dict:
        args = args or {}
        start = time.monotonic()
        try:
            result = await self._call(name, args)
        except Exception as e:
            self.calls.append(ToolCallRecord(name, args, time.monotonic() - start, False, str(e)[:300]))
            raise
        self.calls.append(ToolCallRecord(name, args, time.monotonic() - start, True))
        return result

    async def _call(self, name: str, args: dict) -> dict:
        raise NotImplementedError


class MCPToolClient(_RecordingClient):
    def __init__(self, tools: Sequence[BaseTool]):
        super().__init__()
        self._tools = {t.name: t for t in tools}

    async def _call(self, name: str, args: dict) -> dict:
        out = await self._tools[name].ainvoke(args)
        # adapters 返回文本块列表；工具抛异常时不会向上抛，而是返回 "Error executing tool ..." 文本
        text = "".join(b.get("text", "") for b in out) if isinstance(out, list) else str(out)
        if text.startswith(_MCP_ERROR_PREFIX):
            raise ToolError(text)
        return json.loads(text)


class LocalToolClient(_RecordingClient):
    def __init__(self, repo_tools: RepoTools):
        super().__init__()
        self._repo = repo_tools

    async def _call(self, name: str, args: dict) -> dict:
        try:
            # 工具里有阻塞的子进程调用，放到线程里，不阻塞事件循环
            return await asyncio.to_thread(getattr(self._repo, name), **args)
        except (ValueError, FileNotFoundError) as e:
            raise ToolError(f"{_MCP_ERROR_PREFIX} {name}: {e}") from e


@asynccontextmanager
async def mcp_tool_client(root: str | Path, python_paths: Sequence[str] = (".",)) -> AsyncIterator[MCPToolClient]:
    """启动绑定到 ``root`` 的 MCP server 子进程，返回可调用全部工具的客户端。"""
    args = ["-m", "testpilot.tools.server", "--root", str(root)]
    for p in python_paths:
        args += ["--python-path", p]
    # stdio_client 默认只传 PATH、HOME 等少数变量；另外只转发 TESTPILOT_ 开头的设置（不含任何密钥）
    env = {**get_default_environment(), **{k: v for k, v in os.environ.items() if k.startswith("TESTPILOT_")}}
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPToolClient(await load_mcp_tools(session))
