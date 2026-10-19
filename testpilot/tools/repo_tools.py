"""MCP 工具的实现。绑定到一个工作副本根目录，所有路径都经过越界检查。

逻辑放在普通类里，MCP server（server.py）只做注册，这样工具可以脱离 MCP 直接单测。
方法的 docstring 会成为给 LLM 看的工具描述，所以用英文写、写清参数含义。
所有返回值都限制了长度，避免一次工具调用塞满上下文。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from testpilot.eval.metrics import check_validity
from testpilot.retrieval.ast_index import extract_symbols
from testpilot.sandbox.coverage_runner import measure_coverage
from testpilot.sandbox.mutation import run_mutation as _run_mutation
from testpilot.sandbox.pytest_runner import DEFAULT_PER_TEST_TIMEOUT
from testpilot.sandbox.pytest_runner import run_pytest as _run_pytest

# 生成的测试只能写在这个目录下，不允许改源码（否则“原代码上通过”的有效性规则就没有意义了）
GENERATED_TESTS_DIR = "tests_testpilot"
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", ".tox", ".nox", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".testpilot", "mutants",
}  # fmt: skip

MAX_READ_LINES = 400
MAX_SEARCH_RESULTS = 50
MAX_CASES = 30
MAX_SURVIVORS = 20


class RepoTools:
    def __init__(
        self,
        root: str | Path,
        python_paths: Sequence[str] = (".",),
        pytest_timeout: float = 60.0,
        mutation_timeout: float = 600.0,
    ):
        self.root = Path(root).resolve()
        self.python_paths = tuple(python_paths)
        self.pytest_timeout = pytest_timeout
        self.mutation_timeout = mutation_timeout

    # ---------- 读代码 ----------

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        """Read a text file in the repository, with line numbers.

        Args:
            path: file path relative to the repository root.
            start_line: first line to return (1-based).
            end_line: last line to return (inclusive). At most 400 lines are returned per call.
        """
        file = self._resolve(path)
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line or len(lines), start + MAX_READ_LINES - 1)
        body = "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(start, end + 1))
        return {"path": path, "start_line": start, "end_line": end, "total_lines": len(lines), "content": body}

    def list_symbols(self, path: str = ".") -> dict:
        """List functions, classes and methods with their line ranges, signatures and the names they call.

        Args:
            path: a .py file or a directory (searched recursively) relative to the repository root.
        """
        target = self._resolve(path)
        files = [target] if target.is_file() else self._python_files(target)
        symbols, errors = [], []
        for f in files:
            rel = f.relative_to(self.root).as_posix()
            try:
                symbols += [s.to_dict() for s in extract_symbols(f.read_text(encoding="utf-8"), rel)]
            except (SyntaxError, UnicodeDecodeError) as e:
                errors.append(f"{rel}: {e}")
        return {"path": path, "symbols": symbols, "errors": errors}

    def search_code(self, query: str, regex: bool = False, max_results: int = 30) -> dict:
        """Search Python files for a string (or a regular expression if regex=True), case-sensitive.

        Returns matching lines with file and line number. Useful for finding where a function is used.
        """
        try:
            pattern = re.compile(query if regex else re.escape(query))
        except re.error as e:
            return {"query": query, "matches": [], "truncated": False, "error": f"invalid regex: {e}"}
        limit = max(1, min(max_results, MAX_SEARCH_RESULTS))
        matches: list[dict] = []
        for f in self._python_files(self.root):
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append({"file": f.relative_to(self.root).as_posix(), "line": n, "text": line.strip()[:200]})
                    if len(matches) > limit:
                        return {"query": query, "matches": matches[:limit], "truncated": True}
        return {"query": query, "matches": matches, "truncated": False}

    # ---------- 写测试 ----------

    def write_test_file(self, path: str, content: str) -> dict:
        """Create or overwrite a generated test file.

        Args:
            path: must be inside the tests_testpilot/ directory and named test_*.py,
                e.g. "tests_testpilot/test_calc.py". Source files cannot be modified.
            content: full Python source of the test file.
        """
        file = self._resolve(path, must_exist=False)
        # 检查解析后的真实路径，而不是字符串前缀：tests_testpilot/../sample/test_x.py 前缀合法但会写进源码目录
        allowed = (self.root / GENERATED_TESTS_DIR).resolve()
        if not (file.is_relative_to(allowed) and file.name.startswith("test_") and file.suffix == ".py"):
            raise ValueError(f"tests must be written to {GENERATED_TESTS_DIR}/test_*.py, got: {path}")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        return {"path": file.relative_to(self.root).as_posix(), "bytes": len(content.encode("utf-8"))}

    # ---------- 运行与评价 ----------

    def run_pytest(self, tests: list[str] | None = None) -> dict:
        """Run pytest on the given test files (default: all generated tests) and report per-test results.

        Failure messages and output are truncated (head and tail kept).
        """
        res = _run_pytest(
            self.root,
            self._tests(tests),
            timeout=self.pytest_timeout,
            python_paths=self.python_paths,
            per_test_timeout=DEFAULT_PER_TEST_TIMEOUT,
        )
        not_passed = [c for c in res.cases if c.outcome != "passed"]
        return {
            "all_passed": res.all_passed,
            "passed": res.count("passed"),
            "failed": res.count("failed"),
            "errors": res.count("error"),
            "skipped": res.count("skipped"),
            "timed_out": res.timed_out,
            "returncode": res.exec.returncode,
            # 只列出没通过的用例，通过的只给数量，节省上下文
            "problems": [{"test": c.name, "outcome": c.outcome, "message": c.message} for c in not_passed[:MAX_CASES]],
            "output": res.exec.stdout if not res.cases else "",  # 没有用例结果时（收集失败 / 超时）才需要原始输出
        }

    def check_validity(self, tests: list[str] | None = None) -> dict:
        """Run the tests 3 times on the unmodified code and report which test functions are valid.

        A test function is valid only if it passed in all 3 runs (all parametrized cases included).
        Only valid tests should be used for coverage and mutation testing.
        """
        v = check_validity(self.root, self._tests(tests), python_paths=self.python_paths)
        return {
            "total": len(v.total),
            "valid": len(v.valid),
            "valid_ids": v.valid,
            "invalid": dict(list(v.invalid.items())[:MAX_CASES]),
            "flaky": v.flaky,
        }

    def coverage_report(self, target: str, tests: list[str] | None = None) -> dict:
        """Measure line and branch coverage of one source file by the given tests (default: all generated tests).

        Args:
            target: source file path relative to the repository root, e.g. "pkg/module.py".
        """
        self._resolve(target)
        cov = measure_coverage(
            self.root, target, self._tests(tests), timeout=self.pytest_timeout * 2, python_paths=self.python_paths
        )
        return {
            "target": target,
            "tests_all_passed": cov.pytest.all_passed,
            "line_rate": cov.line_rate,
            "branch_rate": cov.branch_rate,
            "statements": cov.num_statements,
            "branches": cov.num_branches,
            "missing_lines": cov.missing_lines,
            "missing_branches": [list(b) for b in cov.missing_branches],
            "error": cov.error,
        }

    def run_mutation(self, target: str, tests: list[str] | None = None, max_survivors: int = MAX_SURVIVORS) -> dict:
        """Run mutation testing on one source file and report the mutation score and surviving mutants.

        Each surviving mutant shows the line number, the original code and the mutated code:
        a test that fails on the mutated code but passes on the original would kill it.
        The tests must already pass on the original code.

        Args:
            target: source file path relative to the repository root.
            tests: test files or pytest node ids to run against the mutants (default: all generated tests).
            max_survivors: maximum number of surviving mutants to list (at most 200).
        """
        self._resolve(target)
        limit = max(1, min(max_survivors, 200))
        rep = _run_mutation(
            self.root, target, self._tests(tests), timeout=self.mutation_timeout, python_paths=self.python_paths
        )
        survivors = rep.survivors()
        return {
            "target": target,
            "score": rep.score,
            "total": len(rep.mutants),
            "killed": rep.killed,
            "survived": rep.survived,
            "excluded": rep.excluded,
            "status_counts": rep.status_counts(),
            "survivors": [
                {"line": m.line, "original": m.original.strip(), "mutated": m.mutated.strip(), "status": m.status}
                for m in survivors[:limit]
            ],
            "survivors_truncated": len(survivors) > limit,
            "error": rep.error,
        }

    # ---------- 内部 ----------

    def _resolve(self, rel: str, must_exist: bool = True) -> Path:
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"path escapes the repository: {rel}")
        if must_exist and not target.exists():
            raise FileNotFoundError(f"no such file or directory: {rel}")
        return target

    def _tests(self, tests: list[str] | None) -> list[str]:
        tests = tests or [GENERATED_TESTS_DIR]
        for t in tests:
            self._resolve(t.split("::", 1)[0])  # 可以是 nodeid，检查文件部分
        return tests

    def _python_files(self, directory: Path) -> list[Path]:
        return sorted(
            p
            for p in directory.rglob("*.py")
            if not any(part in SKIP_DIRS for part in p.relative_to(self.root).parts)
        )
