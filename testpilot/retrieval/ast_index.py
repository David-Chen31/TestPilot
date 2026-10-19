"""按函数 / 类切分 Python 源码，记录位置和调用关系。

只用标准库 ``ast``，不执行被测代码。调用关系是**名字级**的近似：记录函数体里
出现的被调用名（``foo()`` 记 ``foo``，``self.bar()`` / ``mod.bar()`` 记 ``bar``），
不做类型推断。召回时按名字匹配，宁可多召回一点。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal

SymbolKind = Literal["function", "method", "class"]


@dataclass
class Symbol:
    name: str
    qualname: str  # 类方法为 "Cls.method"
    kind: SymbolKind
    file: str  # 相对仓库根目录，正斜杠
    start_line: int  # 含装饰器
    end_line: int
    signature: str
    calls: list[str] = field(default_factory=list)  # 去重后的被调用名，保持出现顺序

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualname": self.qualname,
            "kind": self.kind,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "calls": self.calls,
        }


def extract_symbols(source: str, file: str) -> list[Symbol]:
    """提取模块顶层函数、类，以及类里的方法（嵌套函数不单独切分）。"""
    tree = ast.parse(source)
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_function_symbol(node, file, class_name=None))
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                Symbol(
                    name=node.name,
                    qualname=node.name,
                    kind="class",
                    file=file,
                    start_line=_start_line(node),
                    end_line=node.end_lineno or node.lineno,
                    signature=_class_signature(node),
                    calls=_called_names(node),
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(_function_symbol(item, file, class_name=node.name))
    return symbols


def _function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef, file: str, class_name: str | None
) -> Symbol:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return Symbol(
        name=node.name,
        qualname=f"{class_name}.{node.name}" if class_name else node.name,
        kind="method" if class_name else "function",
        file=file,
        start_line=_start_line(node),
        end_line=node.end_lineno or node.lineno,
        signature=f"{prefix} {node.name}({ast.unparse(node.args)}){returns}",
        calls=_called_names(node),
    )


def _start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(b) for b in node.bases] + [ast.unparse(k) for k in node.keywords]
    return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"


def _called_names(node: ast.AST) -> list[str]:
    seen: dict[str, None] = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                seen.setdefault(func.id, None)
            elif isinstance(func, ast.Attribute):
                seen.setdefault(func.attr, None)
    return list(seen)
