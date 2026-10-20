"""独立采样与按覆盖率挑选（等成本对照）。"""

import asyncio
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from testpilot.agent.graph import B0
from testpilot.agent.sampling import sample_tests
from testpilot.agent.tool_client import LocalToolClient
from testpilot.eval.selection import screen_sample, select_best
from testpilot.sandbox import Workspace
from testpilot.tools.repo_tools import RepoTools
from tests.test_agent import ALL_BAD, FULL, GOOD, ONE_BAD, TARGET, _reply


def test_sample_tests_runs_independent_generations(sample_repo: Path, tmp_path: Path):
    model = GenericFakeChatModel(messages=iter([_reply(GOOD, 10), _reply(ONE_BAD, 20), _reply(FULL, 30)]))
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        out = asyncio.run(sample_tests(model, ws.path, TARGET, (".",), 3, B0, LocalToolClient(RepoTools(ws.path))))
    assert out.codes == [GOOD, ONE_BAD, FULL]
    assert out.tokens == [20, 40, 60] and out.usage.total_tokens == 120 and out.usage.calls == 3
    # 每份样本都是一次完整的 B0 运行，提示词里没有其他样本的信息
    tasks = [e["llm"]["task"] for trace in out.traces for e in trace if e["node"] == "write_tests"]
    assert len(tasks) == 3 and len(set(tasks)) == 1


def test_select_best_by_valid_coverage(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        files = []
        for k, code in enumerate([ALL_BAD, ONE_BAD, FULL, FULL], 1):
            ws.write_text(f"tests_testpilot/test_calc_s{k}.py", code)
            files.append(f"tests_testpilot/test_calc_s{k}.py")
        rows = [screen_sample(ws.path, TARGET, f, (".",)) for f in files]
    assert [r["tests_valid"] for r in rows] == [0, 2, 2, 2]
    assert rows[0]["line_rate"] == 0.0  # 没有有效测试：不计覆盖率
    assert rows[1]["line_rate"] == 0.6  # 失败的 test_wrong 不贡献覆盖率
    assert rows[2]["line_rate"] == 1.0
    assert select_best(rows) == 2  # 并列时取最早的一份
