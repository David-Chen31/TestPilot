import os
from pathlib import Path

import pytest

from testpilot.sandbox import Workspace
from testpilot.sandbox._mutmut_export import function_starts, locate
from testpilot.sandbox.mutation import MutationReport, Mutant, run_mutation
from tests.test_repo_tools import STRONG_TEST, WEAK_TEST

linux_only = pytest.mark.skipif(os.name == "nt", reason="mutmut 不支持原生 Windows")

# ---------- 纯函数：行号换算、得分口径（各平台都跑） ----------

SRC = '''class A:
    @property
    def p(self):
        return 1

    def m(self, x):
        y = x + 1
        return y


def f(a, b):
    return a + b
'''


def test_function_starts_mangled_names():
    spans = function_starts(SRC)
    assert spans["xǁAǁp"] == (2, 4)
    assert spans["xǁAǁm"] == (6, 8)
    assert spans["x_f"] == (11, 12)


def test_locate_relative_diff_to_absolute_line():
    diff = (
        "--- a.py\n+++ a.py\n@@ -1,3 +1,3 @@\n"
        " def xǁAǁm__mutmut_orig(self, x):\n"
        "-    y = x + 1\n"
        "+    y = x - 1\n"
        "     return y"
    )
    line, original, mutated = locate(diff, (6, 8), SRC.splitlines())
    assert line == 7
    assert original.strip() == "y = x + 1"
    assert mutated.strip() == "y = x - 1"


def test_locate_falls_back_to_content_search():
    # 装饰器让相对偏移不准时，在函数范围内按内容找
    diff = "@@ -1,2 +1,2 @@\n def xǁAǁp__mutmut_orig(self):\n-    return 1\n+    return 2"
    line, _, _ = locate(diff, (2, 4), SRC.splitlines())
    assert line == 4


def _report(*statuses: str) -> MutationReport:
    return MutationReport(
        target="t.py",
        mutants=[Mutant(f"m{i}", s, "t.py", 1, "", "", "") for i, s in enumerate(statuses)],
    )


def test_score_definition():
    rep = _report("killed", "killed", "survived", "no tests", "timeout", "suspicious")
    # 分母 = killed + survived + no tests；timeout / suspicious 排除，也不算杀死
    assert (rep.killed, rep.survived, rep.excluded) == (2, 2, 2)
    assert rep.score == 0.5
    assert [m.status for m in rep.survivors()] == ["survived", "no tests"]


def test_score_none_when_no_scorable_mutants():
    assert _report("timeout").score is None


@pytest.mark.skipif(os.name != "nt", reason="仅 Windows")
def test_windows_reports_unsupported(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        rep = run_mutation(ws.path, "sample/calc.py", ["tests_testpilot"])
    assert "Windows" in rep.error


# ---------- 真实运行 mutmut（只在 Linux / Docker） ----------


def _run(sample_repo: Path, tmp_path: Path, test_code: str, target: str = "sample/calc.py") -> MutationReport:
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_testpilot/test_gen.py", test_code)
        return run_mutation(ws.path, target, ["tests_testpilot"], timeout=300)


@linux_only
def test_weak_tests_leave_survivors(sample_repo: Path, tmp_path: Path):
    rep = _run(sample_repo, tmp_path, WEAK_TEST)
    assert rep.error == ""
    assert (rep.killed, rep.survived) == (1, 6)
    by_line = {(m.line, m.mutated.strip()) for m in rep.survivors()}
    assert (9, "if low >= high:") in by_line
    assert (11, "if x <= low:") in by_line
    assert all(m.file == "sample/calc.py" for m in rep.mutants)


@linux_only
def test_strong_tests_kill_more(sample_repo: Path, tmp_path: Path):
    rep = _run(sample_repo, tmp_path, STRONG_TEST)
    assert rep.error == ""
    assert (rep.killed, rep.survived) == (5, 2)
    # 剩下两个是等价变异体：x == low 时 `return low` 与 `return x` 结果相同，
    # 任何测试都杀不死。它们让变异得分有一个低于 100% 的上限。
    assert {(m.line, m.mutated.strip()) for m in rep.survivors()} == {
        (11, "if x <= low:"),
        (13, "if x >= high:"),
    }


@linux_only
def test_only_target_file_is_mutated(sample_repo: Path, tmp_path: Path):
    rep = _run(sample_repo, tmp_path, WEAK_TEST)
    assert {m.name.rsplit(".", 1)[0] for m in rep.mutants} == {"sample.calc"}


@linux_only
def test_method_line_numbers(sample_repo: Path, tmp_path: Path):
    code = "from sample.shapes import Rect\n\ndef test_scale():\n    assert Rect(2, 3).scale(2) is not None\n"
    rep = _run(sample_repo, tmp_path, code, target="sample/shapes.py")
    assert rep.error == ""
    source = (sample_repo / "sample" / "shapes.py").read_text(encoding="utf-8").splitlines()
    assert rep.mutants
    for m in rep.mutants:
        assert m.line is not None, m.name
        assert source[m.line - 1].strip() == m.original.splitlines()[0].strip()


@linux_only
def test_repeated_runs_in_same_workspace(sample_repo: Path, tmp_path: Path):
    """Agent 会在同一个副本里反复做变异测试：第二次必须用新测试重新跑，而不是报配置冲突或复用旧结果。"""
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        (ws.path / "pyproject.toml").write_text('[project]\nname = "sample"\n', encoding="utf-8")
        ws.write_text("tests_testpilot/test_gen.py", WEAK_TEST)
        first = run_mutation(ws.path, "sample/calc.py", ["tests_testpilot"], timeout=300)
        ws.write_text("tests_testpilot/test_gen.py", STRONG_TEST)
        second = run_mutation(ws.path, "sample/calc.py", ["tests_testpilot"], timeout=300)
        pyproject = ws.read_text("pyproject.toml")
    assert (first.error, second.error) == ("", "")
    assert (first.killed, second.killed) == (1, 5)
    assert pyproject.count("[tool.mutmut]") == 1
    assert pyproject.startswith('[project]\nname = "sample"\n')


@linux_only
def test_rejects_repo_with_own_mutmut_config(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        (ws.path / "pyproject.toml").write_text("[tool.mutmut]\nsource_paths = ['x']\n", encoding="utf-8")
        ws.write_text("tests_testpilot/test_gen.py", WEAK_TEST)
        rep = run_mutation(ws.path, "sample/calc.py", ["tests_testpilot"])
    assert rep.error


@linux_only
def test_tests_failing_on_original_are_rejected(sample_repo: Path, tmp_path: Path):
    rep = _run(sample_repo, tmp_path, "from sample.calc import add\n\ndef test_bad():\n    assert add(1, 2) == 4\n")
    assert rep.error
    assert rep.mutants == []
