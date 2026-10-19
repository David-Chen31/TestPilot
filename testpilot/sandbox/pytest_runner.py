"""在工作副本里运行 pytest，并把结果解析成按测试用例的结构化数据。

用 JUnit XML 而不是解析终端输出：终端格式随 pytest 版本和插件变化，
XML 是稳定的机器接口。后面“有效测试 = 原代码上连续 3 次通过”就基于逐用例结果。
"""

from __future__ import annotations

import os
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from testpilot.sandbox.executor import ExecResult, run, truncate

Outcome = Literal["passed", "failed", "error", "skipped"]

# pytest 退出码：0 全部通过；1 有失败；2 被中断；3 内部错误；4 用法错误；5 没收集到测试
EXIT_NO_TESTS = 5

DEFAULT_MAX_OUTPUT_CHARS = 4000
_MAX_CASE_MESSAGE_CHARS = 1500


@dataclass(frozen=True)
class CaseResult:
    name: str  # "<classname>::<name>"，classname 形如 tests.test_calc.TestAdd
    outcome: Outcome
    message: str = ""  # 失败 / 错误 / 跳过原因（已截断）


@dataclass
class PytestRun:
    exec: ExecResult
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def timed_out(self) -> bool:
        return self.exec.timed_out

    def count(self, outcome: Outcome) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def all_passed(self) -> bool:
        """有用例、全部通过、没有超时。没收集到测试不算通过。"""
        return bool(self.cases) and self.exec.ok and all(c.outcome == "passed" for c in self.cases)


def run_pytest(
    workdir: str | Path,
    targets: Sequence[str] = (),
    timeout: float = 60.0,
    extra_args: Sequence[str] = (),
    python_paths: Sequence[str] = (".",),
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> PytestRun:
    """在 ``workdir`` 中运行 ``pytest targets``。

    ``python_paths`` 相对 ``workdir``，加入 PYTHONPATH，让测试能 import 被测包
    （src 布局的仓库传 ``("src",)``）。
    """
    workdir = Path(workdir).resolve()
    report_dir = workdir / ".testpilot"
    report_dir.mkdir(exist_ok=True)
    report = report_dir / f"junit-{uuid.uuid4().hex}.xml"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",  # 不在副本里写 .pytest_cache
        "-o",
        "addopts=",  # 忽略被测仓库配置里的 addopts（例如 --cov），避免依赖缺失导致跑不起来
        "-o",
        "junit_family=xunit2",
        f"--junitxml={report}",
        *extra_args,
    ]
    env = sandbox_env()
    env["PYTHONPATH"] = os.pathsep.join(str((workdir / p).resolve()) for p in python_paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = run(cmd, cwd=workdir, timeout=timeout, env=env, max_output_chars=max_output_chars)
    cases = _parse_junit(report) if report.exists() else []
    report.unlink(missing_ok=True)
    return PytestRun(exec=result, cases=cases)


_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_SECRET_NAMES = {"LLM_BASE_URL", "REDIS_URL"}


def sandbox_env() -> dict[str, str]:
    """当前进程的环境变量，去掉密钥类变量。被测代码和生成的测试都是不可信代码，
    不能让它们读到 LLM API key 等凭据。"""
    return {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in _SECRET_NAMES and not any(m in k.upper() for m in _SECRET_MARKERS)
    }


def _parse_junit(path: Path) -> list[CaseResult]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    cases: list[CaseResult] = []
    for tc in root.iter("testcase"):
        name = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        outcome: Outcome = "passed"
        message = ""
        for child in tc:
            if child.tag in ("failure", "error", "skipped"):
                outcome = "failed" if child.tag == "failure" else child.tag  # type: ignore[assignment]
                detail = child.text or ""
                message = f"{child.get('message', '')}\n{detail}".strip()
                break
        cases.append(CaseResult(name, outcome, truncate(message, _MAX_CASE_MESSAGE_CHARS)))
    return cases
