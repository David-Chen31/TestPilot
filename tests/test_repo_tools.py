from pathlib import Path

import pytest

from testpilot.sandbox import Workspace
from testpilot.tools.repo_tools import GENERATED_TESTS_DIR, MAX_READ_LINES, RepoTools

WEAK_TEST = """
from sample.calc import add, clamp

def test_add():
    assert add(1, 2) == 3

def test_clamp_runs():
    assert clamp(5, 0, 10) is not None
"""

STRONG_TEST = """
import pytest
from sample.calc import add, clamp

def test_add():
    assert add(1, 2) == 3

def test_clamp_inside():
    assert clamp(5, 0, 10) == 5

def test_clamp_bounds():
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10

def test_clamp_equal_bounds():
    assert clamp(3, 3, 3) == 3

def test_clamp_invalid():
    with pytest.raises(ValueError, match="^low must not exceed high$"):
        clamp(1, 5, 0)
"""


@pytest.fixture
def tools(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        yield RepoTools(ws.path, pytest_timeout=60)


# ---------- read_file ----------


def test_read_file_with_line_numbers(tools: RepoTools):
    res = tools.read_file("sample/calc.py", start_line=4, end_line=5)
    assert res["content"].splitlines() == ["    4 | def add(a: int, b: int) -> int:", "    5 |     return a + b"]
    assert res["total_lines"] == 15


def test_read_file_caps_lines(tools: RepoTools):
    (tools.root / "big.py").write_text("x = 1\n" * (MAX_READ_LINES + 50))
    res = tools.read_file("big.py")
    assert res["end_line"] == MAX_READ_LINES


@pytest.mark.parametrize("bad", ["../outside.py", "/etc/passwd"])
def test_read_file_rejects_escape(tools: RepoTools, bad: str):
    with pytest.raises(ValueError):
        tools.read_file(bad)


def test_read_file_missing(tools: RepoTools):
    with pytest.raises(FileNotFoundError):
        tools.read_file("nope.py")


# ---------- list_symbols / search_code ----------


def test_list_symbols_file(tools: RepoTools):
    names = [s["qualname"] for s in tools.list_symbols("sample/calc.py")["symbols"]]
    assert names == ["add", "clamp"]


def test_list_symbols_directory_skips_tool_dirs(tools: RepoTools):
    (tools.root / "mutants").mkdir()
    (tools.root / "mutants" / "junk.py").write_text("def junk(): pass\n")
    files = {s["file"] for s in tools.list_symbols(".")["symbols"]}
    assert files == {"sample/calc.py", "sample/shapes.py"}


def test_list_symbols_reports_syntax_errors(tools: RepoTools):
    (tools.root / "broken.py").write_text("def (:\n")
    res = tools.list_symbols(".")
    assert any(e.startswith("broken.py") for e in res["errors"])


def test_search_code_literal(tools: RepoTools):
    res = tools.search_code("clamp(")
    hits = {(m["file"], m["line"]) for m in res["matches"]}
    assert ("sample/calc.py", 8) in hits
    assert ("sample/shapes.py", 17) in hits


def test_search_code_regex_and_invalid(tools: RepoTools):
    assert tools.search_code(r"def \w+\(self", regex=True)["matches"]
    assert "invalid regex" in tools.search_code("(", regex=True)["error"]


def test_search_code_truncates(tools: RepoTools):
    res = tools.search_code("self", max_results=2)
    assert len(res["matches"]) == 2
    assert res["truncated"]


# ---------- write_test_file ----------


def test_write_test_file(tools: RepoTools):
    tools.write_test_file(f"{GENERATED_TESTS_DIR}/test_x.py", "def test_x(): pass\n")
    assert (tools.root / GENERATED_TESTS_DIR / "test_x.py").exists()


@pytest.mark.parametrize(
    "bad",
    [
        "sample/calc.py",  # 不能改源码
        "tests/test_x.py",  # 必须在生成目录下
        f"{GENERATED_TESTS_DIR}/helper.py",  # 必须是 test_*.py
        f"{GENERATED_TESTS_DIR}/../sample/test_x.py",  # 越界
    ],
)
def test_write_test_file_rejects(tools: RepoTools, bad: str):
    with pytest.raises(ValueError):
        tools.write_test_file(bad, "x = 1\n")


# ---------- run_pytest / coverage ----------


def test_run_pytest_reports_only_problems(tools: RepoTools):
    tools.write_test_file(
        f"{GENERATED_TESTS_DIR}/test_mixed.py",
        "def test_ok():\n    assert True\n\ndef test_bad():\n    assert 1 == 2\n",
    )
    res = tools.run_pytest()
    assert (res["passed"], res["failed"], res["all_passed"]) == (1, 1, False)
    [problem] = res["problems"]
    assert problem["test"].endswith("::test_bad")
    assert "assert 1 == 2" in problem["message"]


def test_run_pytest_collection_error_includes_output(tools: RepoTools):
    tools.write_test_file(f"{GENERATED_TESTS_DIR}/test_broken.py", "def test_x(:\n")
    res = tools.run_pytest()
    assert not res["all_passed"]
    assert res["errors"] == 1


def test_coverage_weak_vs_strong(tools: RepoTools):
    tools.write_test_file(f"{GENERATED_TESTS_DIR}/test_calc.py", WEAK_TEST)
    weak = tools.coverage_report("sample/calc.py")
    assert weak["tests_all_passed"]
    assert weak["line_rate"] == pytest.approx(0.7)
    assert weak["branch_rate"] == pytest.approx(0.5)
    assert weak["missing_lines"] == [10, 12, 14]

    tools.write_test_file(f"{GENERATED_TESTS_DIR}/test_calc.py", STRONG_TEST)
    strong = tools.coverage_report("sample/calc.py")
    assert strong["line_rate"] == 1.0
    assert strong["branch_rate"] == 1.0
    assert strong["missing_lines"] == []


def test_coverage_target_not_imported(tools: RepoTools):
    tools.write_test_file(f"{GENERATED_TESTS_DIR}/test_other.py", "def test_x():\n    assert True\n")
    res = tools.coverage_report("sample/calc.py")
    assert res["line_rate"] is None
    assert res["error"]
