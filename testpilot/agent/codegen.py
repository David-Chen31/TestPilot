"""与“生成测试代码”相关的纯函数：模块名推导、测试文件路径、从回复中提取代码。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from testpilot.tools.repo_tools import GENERATED_TESTS_DIR

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def module_name(target: str, python_paths: Sequence[str] = (".",)) -> str:
    """toolz/dicttoolz.py -> toolz.dicttoolz；src 布局时去掉 python_path 前缀。"""
    path = Path(target).with_suffix("")
    for root in python_paths:
        root_parts = Path(root).parts if root != "." else ()
        if path.parts[: len(root_parts)] == root_parts:
            parts = path.parts[len(root_parts) :]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
    raise ValueError(f"{target} 不在任何 python_path 下: {python_paths}")


def generated_test_path(target: str) -> str:
    return f"{GENERATED_TESTS_DIR}/test_{Path(target).stem}.py"


def extract_code(text: str) -> str:
    """取回复中最长的 python 代码块。没有代码块时返回空串（调用方把它当作一次失败的生成）。"""
    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        return ""
    return max(blocks, key=len).strip() + "\n"
