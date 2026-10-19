"""基线 B0：把目标模块的源码一次性交给 LLM，让它写出整个测试文件，不运行、不修改。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from testpilot import prompts
from testpilot.llm import Usage, invoke
from testpilot.tools.repo_tools import GENERATED_TESTS_DIR

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


@dataclass
class GenerationResult:
    test_files: list[str]
    usage: Usage
    trace: list[dict] = field(default_factory=list)  # 结构化轨迹：每一步的输入、输出、耗时


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
    """取回复中最长的 python 代码块；没有代码块时把整个回复当作代码。"""
    blocks = _CODE_BLOCK.findall(text)
    return (max(blocks, key=len) if blocks else text).strip() + "\n"


def build_messages(target: str, source: str, module: str) -> list[BaseMessage]:
    user = prompts.B0_USER.format(
        module=module,
        target=target,
        requirements=prompts.TEST_REQUIREMENTS.format(module=module),
        source=source,
    )
    return [SystemMessage(prompts.SYSTEM), HumanMessage(user)]


def generate(
    model: BaseChatModel, workdir: Path, target: str, python_paths: Sequence[str] = (".",)
) -> GenerationResult:
    module = module_name(target, python_paths)
    source = (workdir / target).read_text(encoding="utf-8")
    messages = build_messages(target, source, module)
    call = invoke(model, messages)
    code = extract_code(call.content)

    test_file = generated_test_path(target)
    out = workdir / test_file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8")

    usage = Usage()
    usage.add(call)
    trace = [
        {
            "node": "generate",
            "messages": [{"role": m.type, "content": m.content} for m in messages],
            "response": call.content,
            "finish_reason": call.finish_reason,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "reasoning_tokens": call.reasoning_tokens,
            "latency": round(call.latency, 2),
        }
    ]
    return GenerationResult(test_files=[test_file], usage=usage, trace=trace)
