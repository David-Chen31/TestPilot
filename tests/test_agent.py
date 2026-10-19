"""Agent 状态机：用按脚本回复的假 LLM 驱动，在样例仓库上真实运行 pytest / coverage。"""

import asyncio
from dataclasses import replace
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from testpilot import prompts
from testpilot.agent.graph import B0, B1, run_agent
from testpilot.agent.tool_client import LocalToolClient
from testpilot.sandbox import Workspace
from testpilot.tools.repo_tools import RepoTools

TARGET = "sample/calc.py"
TEST_FILE = "tests_testpilot/test_calc.py"

GOOD = """from sample.calc import add, clamp

def test_add():
    assert add(1, 2) == 3

def test_clamp():
    assert clamp(-1, 0, 10) == 0
"""
ONE_BAD = GOOD + """
def test_wrong():
    assert add(2, 2) == 5
"""
ALL_BAD = """from sample.calc import add

def test_wrong():
    assert add(2, 2) == 5
"""


def _reply(code: str | None, tokens: int = 0) -> AIMessage:
    content = f"```python\n{code}```" if code is not None else "Sorry, here is an explanation without code."
    usage = {"prompt_tokens": tokens, "completion_tokens": tokens, "prompt_cache_hit_tokens": 0}
    return AIMessage(content=content, response_metadata={"token_usage": usage, "finish_reason": "stop"})


def _run(sample_repo: Path, tmp_path: Path, replies: list[AIMessage], config):
    model = GenericFakeChatModel(messages=iter(replies))
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        tools = LocalToolClient(RepoTools(ws.path))
        result = asyncio.run(run_agent(model, ws.path, TARGET, (".",), config, tools=tools))
        final_code = ws.read_text(TEST_FILE)
    return result, final_code


def _nodes(result) -> list[str]:
    return [e["node"] for e in result.trace]


def test_b0_single_shot_keeps_failures(sample_repo: Path, tmp_path: Path):
    result, code = _run(sample_repo, tmp_path, [_reply(ONE_BAD)], B0)
    assert result.info["iterations"] == 1
    assert result.info["stop_reason"] == "单次生成（无反馈）"
    assert code == ONE_BAD
    assert _nodes(result) == ["plan", "retrieve_context", "write_tests", "run_tests", "analyze", "decide", "finalize"]


def test_b1_fixes_failures_with_pytest_feedback(sample_repo: Path, tmp_path: Path):
    result, code = _run(sample_repo, tmp_path, [_reply(ONE_BAD), _reply(GOOD)], B1)
    assert result.info["iterations"] == 2
    assert result.info["stop_reason"].startswith("全部测试通过")
    assert code == GOOD
    # 第二轮的提示词里带着第一轮的失败信息
    second = [e for e in result.trace if e["node"] == "write_tests"][1]
    assert "test_wrong" in second["llm"]["task"] and "assert 4 == 5" in second["llm"]["task"]
    # 每一轮的工具调用都记录在轨迹里
    run_events = [e for e in result.trace if e["node"] == "run_tests"]
    assert all(e["tool_calls"][0]["name"] == "run_pytest" for e in run_events)


def test_stops_at_max_iterations(sample_repo: Path, tmp_path: Path):
    # 每轮都比上一轮多通过一个测试（一直在提升），只能靠最大迭代数停下
    replies = [_reply(ALL_BAD), _reply(GOOD.split("def test_clamp")[0] + ALL_BAD.split("\n", 1)[1]), _reply(ONE_BAD)]
    result, _ = _run(sample_repo, tmp_path, replies, replace(B1, max_iterations=3, patience=10))
    assert result.info["iterations"] == 3
    assert result.info["stop_reason"] == "达到最大迭代数 3"


def test_stops_without_improvement_and_keeps_best_round(sample_repo: Path, tmp_path: Path):
    # 第 1 轮最好（2 个通过），之后两轮都更差：连续 2 轮无提升后停止，最终采用第 1 轮
    replies = [_reply(ONE_BAD), _reply(ALL_BAD), _reply(ALL_BAD), _reply(GOOD)]
    result, code = _run(sample_repo, tmp_path, replies, replace(B1, patience=2))
    assert result.info["iterations"] == 3
    assert result.info["stop_reason"] == "连续 2 轮没有提升"
    assert result.info["best_iteration"] == 1
    assert code == ONE_BAD


def test_stops_when_token_budget_exhausted(sample_repo: Path, tmp_path: Path):
    replies = [_reply(ALL_BAD, tokens=600), _reply(ALL_BAD, tokens=600)]
    result, _ = _run(sample_repo, tmp_path, replies, replace(B1, token_budget=1000))
    assert result.info["iterations"] == 1
    assert result.info["stop_reason"].startswith("token 预算用完")
    assert result.usage.total_tokens == 1200


def test_reply_without_code_gets_feedback(sample_repo: Path, tmp_path: Path):
    result, code = _run(sample_repo, tmp_path, [_reply(None), _reply(GOOD)], B1)
    second = [e for e in result.trace if e["node"] == "write_tests"][1]
    assert prompts.FEEDBACK_NO_CODE in second["llm"]["task"]
    assert code == GOOD


def test_no_code_at_all_leaves_empty_test_file(sample_repo: Path, tmp_path: Path):
    result, code = _run(sample_repo, tmp_path, [_reply(None)], B0)
    assert code.startswith("# no tests generated")


def test_prompt_prefix_shared_across_iterations(sample_repo: Path, tmp_path: Path):
    result, _ = _run(sample_repo, tmp_path, [_reply(ONE_BAD), _reply(GOOD)], B1)
    [ctx] = [e for e in result.trace if e["node"] == "retrieve_context"]
    prefix = ctx["prompt_prefix"]
    assert "    8 | def clamp(x: int, low: int, high: int) -> int:" in prefix  # 带行号的源码
    assert "`add`, `clamp`" in prefix
    assert "from sample.calc import" in prefix


def test_end_to_end_over_mcp(sample_repo: Path, tmp_path: Path):
    """不注入工具：真的启动 MCP server，经 langchain-mcp-adapters 调用。"""
    model = GenericFakeChatModel(messages=iter([_reply(ONE_BAD), _reply(GOOD)]))
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        result = asyncio.run(run_agent(model, ws.path, TARGET, (".",), B1))
        assert ws.read_text(TEST_FILE) == GOOD
    assert result.info["stop_reason"].startswith("全部测试通过")
    analyze = [e for e in result.trace if e["node"] == "analyze"][-1]
    assert analyze["coverage"]["line_rate"] == 0.6  # add + clamp 的 x < low 分支：10 条语句执行了 6 条
