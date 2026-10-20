"""从多份独立样本中挑选测试（等成本对照；M9 的测试时扩展会在这里加更多选择器）。

这里的选择器**不用变异测试**：等成本对照要回答的是“不用变异信号、只多花 token，能不能追上 B2”。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from testpilot.eval.metrics import RUN_TIMEOUT, check_validity
from testpilot.sandbox.coverage_runner import measure_coverage


def screen_sample(workdir: str | Path, target: str, test_file: str, python_paths: Sequence[str]) -> dict:
    """单份样本的廉价指标：有效测试数（原代码上 3 次通过）和有效测试的覆盖率。"""
    v = check_validity(workdir, [test_file], python_paths)
    row = {"file": test_file, "tests_total": len(v.total), "tests_valid": len(v.valid), "line_rate": 0.0, "branch_rate": 0.0}
    if v.valid:
        cov = measure_coverage(workdir, target, v.valid, timeout=RUN_TIMEOUT, python_paths=python_paths)
        row["line_rate"] = cov.line_rate or 0.0
        row["branch_rate"] = cov.branch_rate or 0.0
        row["coverage_error"] = cov.error
    return row


def coverage_key(row: dict) -> tuple:
    """按有效测试的覆盖率选：行覆盖 > 分支覆盖 > 有效测试数。"""
    return (row["tests_valid"] > 0, row["line_rate"], row["branch_rate"], row["tests_valid"])


def select_best(rows: Sequence[dict], key=coverage_key) -> int:
    """返回得分最高的样本下标；并列时取最早的一份（与采样顺序无关的确定性规则）。"""
    return max(range(len(rows)), key=lambda i: (key(rows[i]), -i))
