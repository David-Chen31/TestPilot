"""在 coverage.py 下运行 pytest，返回目标文件的行覆盖率和分支覆盖率。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from testpilot.sandbox.executor import run
from testpilot.sandbox.pytest_runner import PytestRun, run_pytest


@dataclass
class CoverageResult:
    pytest: PytestRun
    target: str
    num_statements: int = 0
    covered_lines: int = 0
    num_branches: int = 0
    covered_branches: int = 0
    missing_lines: list[int] = field(default_factory=list)
    missing_branches: list[tuple[int, int]] = field(default_factory=list)  # (from, to)，to < 0 表示退出函数
    error: str = ""

    @property
    def line_rate(self) -> float | None:
        return self.covered_lines / self.num_statements if self.num_statements else None

    @property
    def branch_rate(self) -> float | None:
        return self.covered_branches / self.num_branches if self.num_branches else None


def measure_coverage(
    workdir: str | Path,
    target: str,
    tests: Sequence[str],
    timeout: float = 120.0,
    python_paths: Sequence[str] = (".",),
) -> CoverageResult:
    """运行 ``tests``，统计 ``target``（相对 workdir 的源文件路径）的覆盖率。

    注意：覆盖率本身不区分测试是否通过。调用方应只对有效测试（原代码上通过）调用，
    否则一个执行到一半就失败的测试也会贡献覆盖率。
    """
    workdir = Path(workdir).resolve()
    data_file = workdir / ".testpilot" / ".coverage"
    data_file.unlink(missing_ok=True)

    py_run = run_pytest(
        workdir,
        tests,
        timeout=timeout,
        python_paths=python_paths,
        coverage_data=data_file,
        coverage_include=[target],
    )
    result = CoverageResult(pytest=py_run, target=target)
    if not data_file.exists():
        result.error = "没有生成覆盖率数据（pytest 超时或未能启动）"
        return result

    report = workdir / ".testpilot" / "coverage.json"
    export = run(
        [sys.executable, "-m", "coverage", "json", f"--data-file={data_file}", "-o", str(report)],
        cwd=workdir,
        timeout=60,
        max_output_chars=2000,
    )
    # 退出码 1 + "No data to report" 表示目标文件一行都没执行到（例如测试根本没 import 它）
    if not report.exists():
        result.error = (export.stdout + export.stderr).strip() or "coverage json 失败"
        return result

    files = json.loads(report.read_text(encoding="utf-8"))["files"]
    target_path = (workdir / target).resolve()
    for name, data in files.items():
        if (workdir / name).resolve() == target_path:
            s = data["summary"]
            result.num_statements = s["num_statements"]
            result.covered_lines = s["covered_lines"]
            result.num_branches = s.get("num_branches", 0)
            result.covered_branches = s.get("covered_branches", 0)
            result.missing_lines = data["missing_lines"]
            result.missing_branches = [tuple(b) for b in data.get("missing_branches", [])]
            break
    else:
        result.error = f"覆盖率报告中没有 {target}"
    return result
