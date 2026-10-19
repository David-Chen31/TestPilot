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


# ---------------- B2：变异反馈 ----------------

import os  # noqa: E402

import pytest  # noqa: E402

from testpilot.agent.graph import B2, TestPilotAgent  # noqa: E402
from tests.test_repo_tools import STRONG_TEST, WEAK_TEST  # noqa: E402


def _agent(config=B2) -> TestPilotAgent:
    return TestPilotAgent(model=None, tools=LocalToolClient(RepoTools(".")), config=config)


def _survivor(line, mutated, status="survived"):
    return {"line": line, "original": "x", "mutated": mutated, "status": status}


def test_mutation_feedback_prioritizes_unseen_and_unexecuted():
    agent = _agent()
    mutation = {
        "error": "", "killed": 5, "survived": 3, "score": 5 / 8,
        "survivors": [_survivor(3, "a"), _survivor(9, "b", "no tests"), _survivor(1, "c")],
    }  # fmt: skip
    validity = {"invalid": {}}
    # line 1 已经展示过（多半是等价变异体），排到最后；没有测试执行到的 line 9 排最前
    text, shown = agent._mutation_feedback("m.py", {"passed": 4}, validity, mutation, ["1|c"])
    order = [text.index(f"line {n}:") for n in (9, 3, 1)]
    assert order == sorted(order)
    assert "(no test executes this line)" in text
    assert "5 of 8 mutants were killed" in text
    assert shown == ["1|c", "9|b", "3|a"]


def test_mutation_feedback_reports_flaky_tests():
    agent = _agent()
    validity = {"invalid": {"t.py::test_x": "flaky, run 2: failed: boom"}}
    text, _ = agent._mutation_feedback("m.py", {"passed": 4}, validity, None, [])
    assert "did not pass consistently over 3 runs" in text and "t.py::test_x" in text


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ({"error": "", "survived": 0}, "全部测试有效，且没有存活的变异体"),
        ({"error": "mutmut 超时"}, "变异测试出错"),
    ],
)
def test_decide_stop_reasons_for_b2(mutation, expected):
    state = {"feedback": "", "mutation": mutation, "iteration": 1, "tokens": 0, "stale_rounds": 0, "run": {"all_passed": True}}
    out = asyncio.run(_agent().decide(state))
    assert out["next"] == "finalize" and out["stop_reason"].startswith(expected)


@pytest.mark.skipif(os.name == "nt", reason="mutmut 不支持原生 Windows")
def test_b2_feeds_back_survivors_and_stops_on_equivalent_mutants(sample_repo: Path, tmp_path: Path):
    # 第 1 轮弱测试；第 2 轮强测试（只剩 2 个等价变异体）；之后两轮没有提升 → 按 patience 停止
    replies = [_reply(WEAK_TEST), _reply(STRONG_TEST), _reply(STRONG_TEST), _reply(STRONG_TEST)]
    result, code = _run(sample_repo, tmp_path, replies, B2)
    writes = [e for e in result.trace if e["node"] == "write_tests"]
    assert "line 9: `if low > high:` → `if low >= high:`" in writes[1]["llm"]["task"]
    analyses = [e for e in result.trace if e["node"] == "analyze"]
    assert [round(a["mutation"]["score"], 3) for a in analyses] == [round(1 / 7, 3), round(5 / 7, 3)] + [round(5 / 7, 3)] * 2
    assert analyses[0]["validity"]["valid"] == 2  # 变异测试只用有效测试
    assert result.info["stop_reason"] == "连续 2 轮没有提升"
    assert result.info["best_iteration"] == 2
    assert code == STRONG_TEST.lstrip()  # extract_code 去掉了首尾空行
