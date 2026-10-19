"""测试有效性与评价指标。B0 / B1 / B2 和 Agent 内部都用这一份实现，保证口径一致。

有效性规则（docs/design.md）：
- 有效测试 = 在**未修改的原代码**上连续 ``N_VALIDITY_RUNS`` 次通过的测试函数。
  参数化测试的所有用例都通过，这个函数才算通过。
- 覆盖率和变异得分**只在有效测试上计算**。在原代码上就失败的测试会让所有变异体
  都“失败”，把变异得分虚高——这是 SWE-bench 复现测试审计里看到的问题。
- 计数单位是测试函数（AST 中定义的 ``test*`` 函数 / ``Test*`` 类中的 ``test*`` 方法），
  这样连导入都失败、一个用例都收集不到的文件也能正确计数（全部无效）。
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from testpilot.sandbox.coverage_runner import measure_coverage
from testpilot.sandbox.mutation import run_mutation
from testpilot.sandbox.pytest_runner import DEFAULT_PER_TEST_TIMEOUT, run_pytest

N_VALIDITY_RUNS = 3
PER_TEST_TIMEOUT = DEFAULT_PER_TEST_TIMEOUT
RUN_TIMEOUT = 300.0  # 一次 pytest 运行的硬上限（秒）
# 不要求名字后面紧跟 "("：语法错误往往就出在函数头上
_TEST_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(test\w*)", re.MULTILINE)


def defined_test_functions(code: str, file: str) -> list[str]:
    """测试文件中定义的测试函数 id（与 pytest nodeid 去掉参数化后缀的形式一致）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 文件解析失败时测试一个都跑不了，但仍要按写出来的数量计数（全部无效），
        # 否则“生成测试数”会被算成 0，有效率变成 0/0
        return [f"{file}::{m.group(1)}@L{code.count(chr(10), 0, m.start()) + 1}" for m in _TEST_DEF.finditer(code)]
    ids: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            ids.append(f"{file}::{node.name}")
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                    ids.append(f"{file}::{node.name}::{item.name}")
    return ids


@dataclass
class Validity:
    total: list[str] = field(default_factory=list)  # 所有测试函数 id
    valid: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)  # id -> 原因
    flaky: list[str] = field(default_factory=list)  # 先通过、后失败的
    runs: list[dict] = field(default_factory=list)  # 每轮运行摘要

    def to_dict(self) -> dict:
        return {
            "total": len(self.total),
            "valid": len(self.valid),
            "valid_ids": self.valid,
            "invalid": self.invalid,
            "flaky": self.flaky,
            "runs": self.runs,
        }


def check_validity(
    workdir: str | Path,
    test_files: Sequence[str],
    python_paths: Sequence[str] = (".",),
    n_runs: int = N_VALIDITY_RUNS,
) -> Validity:
    """在原代码上把测试跑 ``n_runs`` 次，返回哪些测试函数有效。"""
    workdir = Path(workdir)
    v = Validity()
    for f in test_files:
        v.total += defined_test_functions((workdir / f).read_text(encoding="utf-8"), f)

    passed_before: set[str] = set()
    for k in range(1, n_runs + 1):
        candidates = [t for t in v.total if t not in v.invalid]
        if not candidates:
            break
        res = run_pytest(
            workdir, test_files, timeout=RUN_TIMEOUT, python_paths=python_paths, per_test_timeout=PER_TEST_TIMEOUT
        )
        v.runs.append(
            {
                "run": k,
                "returncode": res.exec.returncode,
                "timed_out": res.timed_out,
                "duration": round(res.exec.duration, 2),
                "passed": res.count("passed"),
                "failed": res.count("failed"),
                "errors": res.count("error"),
                "skipped": res.count("skipped"),
            }
        )
        outcomes: dict[str, list] = {}
        collection_errors: list[str] = []
        for c in res.cases:
            if c.nodeid.split("::", 1)[0] not in test_files:
                # 收集阶段的错误（例如 import 失败）在 JUnit 里是一个没有文件路径的伪用例
                collection_errors.append(_error_line(c.message))
                continue
            outcomes.setdefault(c.function_id, []).append(c)
            if c.function_id not in v.total:  # 例如继承来的测试方法
                v.total.append(c.function_id)
        no_result = "; ".join(collection_errors) or _first_line(res.exec.stdout or res.exec.stderr) or "无结果"
        for t in [t for t in v.total if t not in v.invalid]:
            cases = outcomes.get(t)
            if not cases:
                reason = f"run {k}: 未收集到（{'超时' if res.timed_out else no_result}）"
            else:
                bad = next((c for c in cases if c.outcome != "passed"), None)
                if bad is None:
                    passed_before.add(t)
                    continue
                reason = f"run {k}: {bad.outcome}: {_first_line(bad.message)}"
            if t in passed_before:
                v.flaky.append(t)
                reason = "flaky, " + reason
            v.invalid[t] = reason
    v.valid = [t for t in v.total if t not in v.invalid]
    return v


@dataclass
class Evaluation:
    target: str
    validity: Validity
    line_rate: float = 0.0
    branch_rate: float = 0.0
    coverage_error: str = ""
    mutation: dict | None = None
    durations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "validity": self.validity.to_dict(),
            "line_rate": self.line_rate,
            "branch_rate": self.branch_rate,
            "coverage_error": self.coverage_error,
            "mutation": self.mutation,
            "durations": self.durations,
        }

    def summary(self) -> dict:
        m = self.mutation or {}
        return {
            "tests_total": len(self.validity.total),
            "tests_valid": len(self.validity.valid),
            "line_rate": self.line_rate,
            "branch_rate": self.branch_rate,
            "mutation_score": m.get("score"),
            "mutants_total": m.get("total"),
            # 变异测试出错时必须显式可见，不能只留下一个 None 得分
            "mutation_error": (m.get("error") or "")[-300:],
        }


def evaluate(
    workdir: str | Path,
    target: str,
    test_files: Sequence[str],
    python_paths: Sequence[str] = (".",),
    with_mutation: bool = True,
    mutation_timeout: float = 1800.0,
) -> Evaluation:
    t0 = time.monotonic()
    validity = check_validity(workdir, test_files, python_paths)
    ev = Evaluation(target=target, validity=validity)
    ev.durations["validity"] = round(time.monotonic() - t0, 2)
    if not validity.valid:
        # 没有有效测试：什么都没覆盖，也杀不死任何变异体
        ev.coverage_error = "没有有效测试"
        if with_mutation:
            ev.mutation = {"score": 0.0, "total": None, "error": "没有有效测试"}
        return ev

    t0 = time.monotonic()
    cov = measure_coverage(workdir, target, validity.valid, timeout=RUN_TIMEOUT, python_paths=python_paths)
    ev.line_rate = cov.line_rate or 0.0  # None 表示目标文件一行都没执行到
    ev.branch_rate = cov.branch_rate or 0.0
    ev.coverage_error = cov.error
    ev.durations["coverage"] = round(time.monotonic() - t0, 2)

    if with_mutation:
        t0 = time.monotonic()
        rep = run_mutation(workdir, target, validity.valid, timeout=mutation_timeout, python_paths=python_paths)
        ev.mutation = {
            "score": rep.score,
            "total": len(rep.mutants),
            "killed": rep.killed,
            "survived": rep.survived,
            "excluded": rep.excluded,
            "status_counts": rep.status_counts(),
            "survivors": [
                {"line": m.line, "original": m.original.strip(), "mutated": m.mutated.strip(), "status": m.status}
                for m in rep.survivors()
            ],
            "error": rep.error,
        }
        ev.durations["mutation"] = round(time.monotonic() - t0, 2)
    return ev


def _error_line(text: str) -> str:
    """traceback 里最有信息量的一行：最后一个 ``E   ...`` 行（异常类型和消息）。"""
    errors = [line.strip() for line in text.splitlines() if line.startswith("E ")]
    return (errors[-1][1:].strip() if errors else _first_line(text))[:300]


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""
