"""在工作副本目录里运行：读取 mutmut 结果，输出 JSON 到 stdout。

单独成进程的原因：mutmut 的内部 API 依赖当前目录（读 ./pyproject.toml 和 ./mutants/），
不能在服务进程里直接 import。依赖 mutmut 3.8 的内部函数，所以 mutmut 版本是锁死的。

mutmut 给出的 diff 行号是相对函数开头的，这里换算成原文件的绝对行号。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

_SEP = "ǁ"  # mutmut 对类方法的命名分隔符：xǁClassǁmethod
_HUNK = re.compile(r"^@@ -(\d+)")


def function_starts(source: str) -> dict[str, tuple[int, int]]:
    """mutmut 改写后的函数名 -> (起始行, 结束行)，起始行含装饰器。"""
    spans: dict[str, tuple[int, int]] = {}

    def span(node: ast.AST) -> tuple[int, int]:
        decos = [d.lineno for d in getattr(node, "decorator_list", [])]
        return min([node.lineno, *decos]), node.end_lineno or node.lineno

    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans[f"x_{node.name}"] = span(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans[f"x{_SEP}{node.name}{_SEP}{item.name}"] = span(item)
    return spans


def locate(diff: str, func_span: tuple[int, int] | None, source_lines: list[str]) -> tuple[int | None, str, str]:
    """从函数内相对 diff 求出 (绝对行号, 原代码, 变异后代码)。"""
    removed: list[str] = []
    added: list[str] = []
    first_removed_rel: int | None = None
    rel = 0
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            rel = int(m.group(1))
            continue
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
            if first_removed_rel is None:
                first_removed_rel = rel
            rel += 1
        elif line.startswith("+"):
            added.append(line[1:])
        else:
            rel += 1
    original, mutated = "\n".join(removed), "\n".join(added)
    if func_span is None or first_removed_rel is None:
        return None, original, mutated

    start, end = func_span
    guess = start + first_removed_rel - 1
    first = removed[0].strip()
    if 1 <= guess <= len(source_lines) and source_lines[guess - 1].strip() == first:
        return guess, original, mutated
    # 函数头被 mutmut 改写过时偏移可能不准，退回到在函数范围内按内容查找
    for n in range(start, min(end, len(source_lines)) + 1):
        if source_lines[n - 1].strip() == first:
            return n, original, mutated
    return None, original, mutated


def main(target: str) -> None:
    from mutmut.__main__ import SourceFileMutationData, get_diff_for_mutant, status_by_exit_code

    data = SourceFileMutationData(path=Path(target))
    data.load()
    source = Path(target).read_text(encoding="utf-8")
    source_lines = source.splitlines()
    spans = function_starts(source)

    out = []
    for key, exit_code in data.exit_code_by_key.items():
        diff = get_diff_for_mutant(key, path=Path(target))
        mangled = key.rsplit(".", 1)[-1].split("__mutmut_")[0]
        line, original, mutated = locate(diff, spans.get(mangled), source_lines)
        out.append(
            {
                "name": key,
                "status": status_by_exit_code[exit_code],
                "file": target,
                "line": line,
                "original": original,
                "mutated": mutated,
                "diff": diff,
            }
        )
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1])
