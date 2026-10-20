"""独立采样：同一个目标独立生成 N 份测试（每份都是一次完整的 Agent 运行）。

用途：
- 等成本对照：B2 的 token 花费是 B0 的数倍，B2 更好可能只是因为“花得多”。
  把同样多的 token 花在 N 次独立的 B0 生成上，再按不用变异测试的信号挑选 / 合并，
  才能把“变异反馈”本身的作用和“多花 token”分开；
- 测试时扩展（M9）：保存的 N 份样本可以离线换不同的选择器（随机 / 覆盖率 / 变异得分 / LLM 评委），
  不需要重新调用 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.language_models import BaseChatModel

from testpilot.agent.graph import B0, AgentConfig, run_agent
from testpilot.agent.tool_client import ToolClient, mcp_tool_client
from testpilot.llm import Usage


@dataclass
class SampleSet:
    codes: list[str] = field(default_factory=list)  # 每份样本最终的测试文件内容
    infos: list[dict] = field(default_factory=list)  # 每份样本的停止原因等
    tokens: list[int] = field(default_factory=list)  # 每份样本的 token 数（输入 + 输出）
    usages: list[Usage] = field(default_factory=list)  # 每份样本各自的用量
    traces: list[list[dict]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)  # 全部样本合计


async def sample_tests(
    model: BaseChatModel,
    workdir: str | Path,
    target: str,
    python_paths: Sequence[str],
    n: int,
    config: AgentConfig = B0,
    tools: ToolClient | None = None,
) -> SampleSet:
    """顺序运行 ``n`` 次 Agent。每次都写同一个测试文件，跑完立即读出保存，下一次覆盖它。"""
    if tools is None:
        async with mcp_tool_client(workdir, python_paths) as mcp_tools:
            return await sample_tests(model, workdir, target, python_paths, n, config, mcp_tools)

    out = SampleSet()
    for _ in range(n):
        gen = await run_agent(model, workdir, target, python_paths, config, tools)
        out.codes.append((Path(workdir) / gen.test_files[0]).read_text(encoding="utf-8"))
        out.infos.append(gen.info)
        out.tokens.append(gen.usage.total_tokens)
        out.usages.append(gen.usage)
        out.traces.append(gen.trace)
        out.usage.merge(gen.usage)
    return out
