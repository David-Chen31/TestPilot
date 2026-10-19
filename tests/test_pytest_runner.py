import time
from pathlib import Path

from testpilot.sandbox import Workspace, run_pytest

PASSING = """
from sample.calc import add, clamp

def test_add():
    assert add(1, 2) == 3

def test_clamp_low():
    assert clamp(-5, 0, 10) == 0
"""

FAILING = """
from sample.calc import add

def test_add_wrong():
    assert add(1, 2) == 4
"""

IMPORT_ERROR = """
from sample.calc import does_not_exist

def test_x():
    pass
"""

HANGING = """
def test_forever():
    while True:
        pass
"""


def test_passing_tests(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_calc.py", PASSING)
        res = run_pytest(ws.path, ["tests_gen/test_calc.py"], timeout=60)
    assert res.all_passed
    assert res.count("passed") == 2
    assert {c.name.split("::")[1] for c in res.cases} == {"test_add", "test_clamp_low"}


def test_failing_test_reports_message(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_calc.py", FAILING)
        res = run_pytest(ws.path, ["tests_gen/test_calc.py"], timeout=60)
    assert not res.all_passed
    assert res.exec.returncode == 1
    [case] = res.cases
    assert case.outcome == "failed"
    assert "assert 3 == 4" in case.message


def test_import_error_is_collection_error(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_calc.py", IMPORT_ERROR)
        res = run_pytest(ws.path, ["tests_gen/test_calc.py"], timeout=60)
    assert not res.all_passed
    assert res.count("error") == 1
    assert "does_not_exist" in res.cases[0].message


def test_hanging_test_is_killed(sample_repo: Path, tmp_path: Path):
    start = time.monotonic()
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_hang.py", HANGING)
        res = run_pytest(ws.path, ["tests_gen/test_hang.py"], timeout=3)
    assert res.timed_out
    assert not res.all_passed
    assert res.cases == []  # 被杀时 pytest 来不及写报告
    assert time.monotonic() - start < 20


def test_no_tests_collected_is_not_passing(sample_repo: Path, tmp_path: Path):
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_empty.py", "X = 1\n")
        res = run_pytest(ws.path, ["tests_gen/test_empty.py"], timeout=60)
    assert res.exec.returncode == 5
    assert not res.all_passed


def test_secrets_not_visible_to_tests(sample_repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("REDIS_URL", "redis://secret-host:6379/0")
    probe = """
import os

def test_no_secrets():
    assert "LLM_API_KEY" not in os.environ
    assert "REDIS_URL" not in os.environ
"""
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_env.py", probe)
        res = run_pytest(ws.path, ["tests_gen/test_env.py"], timeout=60)
    assert res.all_passed


def test_source_repo_untouched(sample_repo: Path, tmp_path: Path):
    before = sorted(p.relative_to(sample_repo) for p in sample_repo.rglob("*"))
    with Workspace(sample_repo, base_dir=tmp_path) as ws:
        ws.write_text("tests_gen/test_calc.py", PASSING)
        run_pytest(ws.path, ["tests_gen/test_calc.py"], timeout=60)
    after = sorted(p.relative_to(sample_repo) for p in sample_repo.rglob("*"))
    assert before == after
