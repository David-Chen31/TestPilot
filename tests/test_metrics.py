import os
from pathlib import Path

import pytest

from testpilot.eval.metrics import check_validity, defined_test_functions, evaluate
from testpilot.sandbox import Workspace

F = "tests_testpilot/test_gen.py"


def test_defined_test_functions():
    code = """
def test_a(): pass
def helper(): pass
async def test_b(): pass
class TestX:
    def test_c(self): pass
    def setup_method(self): pass
class Helper:
    def test_d(self): pass
"""
    assert defined_test_functions(code, F) == [f"{F}::test_a", f"{F}::test_b", f"{F}::TestX::test_c"]
    assert defined_test_functions("def (:", F) == []


def test_defined_test_functions_counts_unparsable_file():
    # 一处语法错误让整个文件无法解析：测试仍按写出来的数量计数
    # test_b 就是 M4 中 B0 真实出现过的错误：def test_merge_with_single_iterable':
    code = """def test_a():
    pass

def test_b':
    pass

class TestX:
    def test_c(self):
        pass
"""
    assert defined_test_functions(code, F) == [f"{F}::test_a@L1", f"{F}::test_b@L4", f"{F}::test_c@L8"]


def test_unparsable_file_all_invalid_with_reason(sample_repo: Path, tmp_path: Path):
    v = _validity(sample_repo, tmp_path, "def test_a():\n    pass\n\ndef test_b':\n    pass\n")
    assert len(v.total) == 2 and v.valid == []
    assert all("SyntaxError" in reason for reason in v.invalid.values())


def _validity(sample_repo: Path, tmp_path: Path, code: str, **kw):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text(F, code)
        return check_validity(ws.path, [F], **kw)


def test_validity_pass_fail_and_param(sample_repo: Path, tmp_path: Path):
    code = """
import pytest
from sample.calc import add

def test_ok():
    assert add(1, 1) == 2

@pytest.mark.parametrize("a,b,c", [(1, 2, 3), (2, 2, 5)])
def test_param_one_bad(a, b, c):
    assert add(a, b) == c

class TestCls:
    def test_method(self):
        assert add(0, 0) == 0
"""
    v = _validity(sample_repo, tmp_path, code)
    assert sorted(v.valid) == [f"{F}::TestCls::test_method", f"{F}::test_ok"]
    # 参数化用例只要有一个失败，整个测试函数无效
    assert f"{F}::test_param_one_bad" in v.invalid
    assert len(v.runs) == 3


def test_validity_detects_flaky(sample_repo: Path, tmp_path: Path):
    # 第一次运行通过、之后失败：用工作副本里的计数文件模拟
    code = """
from pathlib import Path

def test_flaky():
    p = Path(__file__).with_name("counter.txt")
    n = int(p.read_text()) if p.exists() else 0
    p.write_text(str(n + 1))
    assert n == 0

def test_stable():
    assert True
"""
    v = _validity(sample_repo, tmp_path, code)
    assert v.valid == [f"{F}::test_stable"]
    assert v.flaky == [f"{F}::test_flaky"]
    assert v.invalid[f"{F}::test_flaky"].startswith("flaky, run 2")


def test_validity_import_error_invalidates_all(sample_repo: Path, tmp_path: Path):
    code = "from sample.calc import nope\n\ndef test_a():\n    pass\n\ndef test_b():\n    pass\n"
    v = _validity(sample_repo, tmp_path, code)
    assert v.valid == []
    assert len(v.total) == 2 and len(v.invalid) == 2
    assert "ImportError" in v.invalid[f"{F}::test_a"]  # 原因里带上真正的报错
    assert len(v.runs) == 1  # 第一轮就全部无效，不再重复运行


@pytest.mark.skipif(
    os.name == "nt", reason="Windows 上 pytest-timeout 只能用 thread 方式，超时会结束整个 pytest 进程"
)
def test_validity_hanging_test_only_fails_itself(sample_repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("testpilot.eval.metrics.PER_TEST_TIMEOUT", 1.0)
    code = "def test_hang():\n    while True:\n        pass\n\ndef test_ok():\n    assert True\n"
    v = _validity(sample_repo, tmp_path, code, n_runs=1)
    assert v.valid == [f"{F}::test_ok"]
    assert "Timeout" in v.invalid[f"{F}::test_hang"]


def test_coverage_counts_only_valid_tests(sample_repo: Path, tmp_path: Path):
    # 失败的测试执行到了 clamp 的所有分支，但它无效，覆盖率不能算它的
    code = """
from sample.calc import add, clamp

def test_add():
    assert add(1, 2) == 3

def test_clamp_wrong():
    clamp(-1, 0, 10); clamp(11, 0, 10); clamp(5, 0, 10)
    assert False
"""
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text(F, code)
        ev = evaluate(ws.path, "sample/calc.py", [F], with_mutation=False)
    assert ev.validity.valid == [f"{F}::test_add"]
    assert ev.line_rate == pytest.approx(3 / 10)  # 只有 import 时执行的 def 行和 add 的函数体


def test_no_valid_tests_scores_zero(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text(F, "def test_bad():\n    assert False\n")
        ev = evaluate(ws.path, "sample/calc.py", [F], with_mutation=True)
    assert (ev.line_rate, ev.branch_rate) == (0.0, 0.0)
    assert ev.mutation["score"] == 0.0
    assert ev.summary()["tests_valid"] == 0
