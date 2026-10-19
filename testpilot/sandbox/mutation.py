"""用 mutmut 对目标文件做变异测试，返回每个变异体的状态和位置。

mutmut 依赖 fork，不支持原生 Windows，只在 Linux / Docker 中运行。

变异得分口径（见 docs/design.md）::

    score = killed / (total - timeout - 无法编译)

- killed：至少一个测试在变异体上失败。
- 存活（计入分母、不计入分子）：``survived``，以及 ``no tests``（没有测试执行到这个变异体）。
- 排除出分母：``timeout``；mutmut 用 libcst 生成变异体并保证语法合法，所以“无法编译”恒为 0。
  其余 mutmut 的非常规状态（suspicious / segfault / skipped / not checked 等）也排除并单独计数，
  不把它们算作“杀死”，避免虚高。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from testpilot.sandbox.executor import ExecResult, run, truncate
from testpilot.sandbox.pytest_runner import sandbox_env

_BEGIN = "# >>> testpilot: mutmut config (auto-generated) >>>"

KILLED = {"killed"}
SURVIVED = {"survived", "no tests"}
# 其余状态全部排除出分母


@dataclass
class Mutant:
    name: str  # mutmut 的变异体 key，例如 sample.calc.x_clamp__mutmut_3
    status: str
    file: str
    line: int | None  # 原文件中的绝对行号
    original: str  # 被替换的原代码行
    mutated: str  # 替换后的代码行
    diff: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MutationReport:
    target: str
    exec: ExecResult | None = None
    mutants: list[Mutant] = field(default_factory=list)
    error: str = ""

    def count(self, statuses: set[str]) -> int:
        return sum(1 for m in self.mutants if m.status in statuses)

    @property
    def killed(self) -> int:
        return self.count(KILLED)

    @property
    def survived(self) -> int:
        return self.count(SURVIVED)

    @property
    def excluded(self) -> int:
        return len(self.mutants) - self.killed - self.survived

    @property
    def score(self) -> float | None:
        denom = self.killed + self.survived
        return self.killed / denom if denom else None

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.mutants:
            counts[m.status] = counts.get(m.status, 0) + 1
        return counts

    def survivors(self) -> list[Mutant]:
        return [m for m in self.mutants if m.status in SURVIVED]


def run_mutation(
    workdir: str | Path,
    target: str,
    tests: Sequence[str],
    timeout: float = 600.0,
    python_paths: Sequence[str] = (".",),
    max_children: int | None = None,
) -> MutationReport:
    """对 ``target``（相对 workdir 的 .py 文件）做变异，用 ``tests`` 去杀。

    调用前提：``tests`` 在原代码上全部通过（有效性规则，由调用方保证）。
    mutmut 自己也会先跑一遍原代码上的测试，不通过就拒绝继续，这里把它报成 error。
    """
    workdir = Path(workdir).resolve()
    report = MutationReport(target=target)
    if os.name == "nt":
        report.error = "mutmut 不支持原生 Windows，请在 Docker / WSL2 中运行变异测试"
        return report
    if not target.endswith(".py") or not (workdir / target).is_file():
        report.error = f"目标文件不存在或不是 .py 文件: {target}"
        return report
    if not tests:
        report.error = "没有指定测试"
        return report

    try:
        _write_config(workdir, target, tests, python_paths)
    except ValueError as e:
        report.error = str(e)
        return report
    shutil.rmtree(workdir / "mutants", ignore_errors=True)  # 不复用上一轮的缓存结果

    env = sandbox_env()
    env["PYTHONPATH"] = os.pathsep.join(str((workdir / p).resolve()) for p in python_paths)
    cmd = [sys.executable, "-m", "mutmut", "run"]
    if max_children:
        cmd += ["--max-children", str(max_children)]
    report.exec = run(cmd, cwd=workdir, timeout=timeout, env=env, max_output_chars=3000)
    if report.exec.timed_out:
        report.error = f"mutmut 超时（{timeout:.0f}s）"
        return report
    if report.exec.returncode != 0:
        report.error = "mutmut 运行失败（常见原因：测试在原代码上没有全部通过）\n" + _tail(report.exec)
        return report

    export = run(
        [sys.executable, "-m", "testpilot.sandbox._mutmut_export", target],
        cwd=workdir,
        timeout=120,
        env=env,
    )
    if export.returncode != 0:
        report.error = "读取 mutmut 结果失败\n" + truncate(export.stderr, 2000)
        return report
    report.mutants = [Mutant(**m) for m in json.loads(export.stdout)]
    return report


def _write_config(workdir: Path, target: str, tests: Sequence[str], python_paths: Sequence[str]) -> None:
    """把 mutmut 配置追加到工作副本的 pyproject.toml（副本可以随便改）。"""
    pyproject = workdir / "pyproject.toml"
    existing = pyproject.read_text(encoding="utf-8") if pyproject.exists() else ""
    existing = existing.split(_BEGIN, 1)[0]  # 去掉上一次运行写入的配置块（同一副本会多次做变异测试）
    if "[tool.mutmut]" in existing:
        raise ValueError("被测仓库自带 [tool.mutmut] 配置，暂不支持")

    # source_paths 必须是可导入的顶层包（或顶层模块），mutmut 会把它整体复制到 mutants/ 下；
    # only_mutate 把变异范围限制在目标文件
    rel = Path(target)
    root = next((Path(p) for p in python_paths if _is_under(rel, Path(p))), Path("."))
    top = (root / rel.relative_to(root).parts[0]).as_posix()
    config = {
        "source_paths": [top],
        "only_mutate": [rel.as_posix()],
        "pytest_add_cli_args": ["-o", "addopts="],
        "pytest_add_cli_args_test_selection": list(tests),
        # mutmut 只把 tests/ 等默认目录复制进 mutants/，生成的测试要显式声明
        "also_copy": list(tests),
        "use_git_change_detection": False,
    }
    # 配置块固定追加在文件末尾，TOML 表到文件结束为止，不会吞掉原有内容
    lines = [_BEGIN, "[tool.mutmut]"] + [f"{k} = {json.dumps(v)}" for k, v in config.items()]
    pyproject.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _is_under(path: Path, root: Path) -> bool:
    return root == Path(".") or path.parts[: len(root.parts)] == root.parts


def _tail(res: ExecResult) -> str:
    return truncate((res.stdout + "\n" + res.stderr).strip(), 2000)
