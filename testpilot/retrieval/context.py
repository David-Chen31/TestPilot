"""为目标模块召回写测试所需的上下文（只用 AST 依赖分析，向量检索在 M6 加入）。

召回三类内容：
1. 目标模块源码（带行号：之后变异反馈按行号指出存活变异体，LLM 需要能对上）；
2. 目标模块从本仓库 import 的符号的定义（它调用的东西）；
3. 公开符号在仓库其他地方的调用处（帮助理解用法）。
全部通过工具（MCP）获取，和 Agent 其他步骤一样留下工具调用记录。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Sequence

from testpilot.agent.tool_client import ToolClient, ToolError
from testpilot.tools.repo_tools import GENERATED_TESTS_DIR

MAX_DEPENDENCY_CHARS = 12_000
MAX_USAGES = 20
MAX_USAGES_PER_SYMBOL = 2
_NUMBERED = re.compile(r"^\s*\d+ \| ?(.*)$")


@dataclass
class Context:
    source: str  # 原始源码
    numbered_source: str  # 带行号
    public_symbols: list[str]
    dependencies: list[dict] = field(default_factory=list)  # {file, name, start_line, end_line, code}
    usages: list[dict] = field(default_factory=list)  # {file, line, text}

    def render(self) -> str:
        parts = []
        if self.dependencies:
            parts.append("Definitions imported by the module from elsewhere in the repository:")
            for d in self.dependencies:
                parts.append(f"# {d['file']} lines {d['start_line']}-{d['end_line']}\n```python\n{d['code']}\n```")
        if self.usages:
            parts.append("How the module's public API is used elsewhere in the repository:")
            parts.append("\n".join(f"- {u['file']}:{u['line']}: `{u['text']}`" for u in self.usages))
        return "\n\n".join(parts)


async def read_source(tools: ToolClient, path: str) -> tuple[str, str]:
    """分页读完整个文件（read_file 每次最多 400 行），返回 (原文, 带行号文本)。"""
    numbered: list[str] = []
    start = 1
    while True:
        r = await tools.call("read_file", {"path": path, "start_line": start})
        if r["content"]:
            numbered += r["content"].split("\n")
        if r["end_line"] >= r["total_lines"]:
            break
        start = r["end_line"] + 1
    plain = [m.group(1) if (m := _NUMBERED.match(line)) else line for line in numbered]
    return "\n".join(plain) + "\n", "\n".join(numbered)


async def build_context(
    tools: ToolClient,
    target: str,
    module: str,
    python_paths: Sequence[str] = (".",),
    public: list[str] | None = None,
) -> Context:
    """``public``：要覆盖的公开符号（Agent 的 plan 节点已经算好时直接传入）。"""
    source, numbered = await read_source(tools, target)
    if public is None:
        symbols = (await tools.call("list_symbols", {"path": target}))["symbols"]
        public = [s["qualname"] for s in symbols if s["kind"] != "method" and not s["name"].startswith("_")]
    ctx = Context(source=source, numbered_source=numbered, public_symbols=public)
    ctx.dependencies = await _imported_definitions(tools, source, target, module, python_paths)
    ctx.usages = await _usages(tools, public, target)
    return ctx


async def _imported_definitions(
    tools: ToolClient, source: str, target: str, module: str, python_paths: Sequence[str]
) -> list[dict]:
    deps: list[dict] = []
    budget = MAX_DEPENDENCY_CHARS
    is_package = PurePosixPath(target).name == "__init__.py"
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ImportFrom):
            continue
        imported = _resolve_module(node, module, is_package)
        file = await _module_file(tools, imported, python_paths) if imported else None
        if file is None or file == target:
            continue  # 第三方库 / 标准库 / 自身
        symbols = {s["qualname"]: s for s in (await tools.call("list_symbols", {"path": file}))["symbols"]}
        for alias in node.names:
            sym = symbols.get(alias.name)
            if sym is None:
                continue  # 常量、子模块等不是函数 / 类的名字
            code, _ = await _read_lines(tools, file, sym["start_line"], sym["end_line"])
            if len(code) > budget:
                continue
            budget -= len(code)
            deps.append({"file": file, "name": alias.name, "start_line": sym["start_line"], "end_line": sym["end_line"], "code": code})
    return deps


def _resolve_module(node: ast.ImportFrom, module: str, is_package: bool) -> str | None:
    if node.level == 0:
        return node.module
    package = module.split(".") if is_package else module.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
    return ".".join([*base, *([node.module] if node.module else [])]) or None


async def _module_file(tools: ToolClient, module: str, python_paths: Sequence[str]) -> str | None:
    rel = module.replace(".", "/")
    for root in python_paths:
        prefix = "" if root == "." else f"{root.rstrip('/')}/"
        for candidate in (f"{prefix}{rel}.py", f"{prefix}{rel}/__init__.py"):
            try:
                await tools.call("read_file", {"path": candidate, "start_line": 1, "end_line": 1})
                return candidate
            except ToolError:
                continue
    return None


async def _read_lines(tools: ToolClient, file: str, start: int, end: int) -> tuple[str, str]:
    r = await tools.call("read_file", {"path": file, "start_line": start, "end_line": end})
    lines = r["content"].split("\n")
    return "\n".join(m.group(1) if (m := _NUMBERED.match(x)) else x for x in lines), r["content"]


async def _usages(tools: ToolClient, public: list[str], target: str) -> list[dict]:
    usages: list[dict] = []
    for name in public:
        r = await tools.call("search_code", {"query": f"{name}(", "max_results": 50})
        found = [
            m
            for m in r["matches"]
            if m["file"] != target
            and not m["file"].startswith(GENERATED_TESTS_DIR + "/")
            and not m["text"].lstrip().startswith(("def ", "class ", "#"))
        ]
        usages += found[:MAX_USAGES_PER_SYMBOL]
        if len(usages) >= MAX_USAGES:
            break
    return usages[:MAX_USAGES]
