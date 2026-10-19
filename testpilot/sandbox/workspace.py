"""每个任务的独立工作副本：复制仓库到临时目录，所有写操作都只发生在副本里。"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

# 不复制的内容：版本库、缓存、虚拟环境。被测代码的 import 不依赖它们，
# 复制它们只会拖慢速度，或者让旧的 .pyc / 覆盖率数据干扰结果。
_IGNORE = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    ".coverage",
    ".coverage.*",
    "node_modules",
)


class Workspace:
    """用作上下文管理器::

        with Workspace(repo) as ws:
            ws.write_text("tests/test_x.py", code)
            run_pytest(ws.path, ["tests/test_x.py"], timeout=60)

    退出时删除整个临时目录（``keep=True`` 时保留，便于调试）。
    """

    def __init__(self, source: str | Path, base_dir: str | Path | None = None, keep: bool = False):
        self.source = Path(source).resolve()
        if not self.source.is_dir():
            raise NotADirectoryError(self.source)
        self._base_dir = base_dir
        self._keep = keep
        self._tmp: Path | None = None
        self.path: Path = Path()

    def __enter__(self) -> Workspace:
        self._tmp = Path(tempfile.mkdtemp(prefix="testpilot-", dir=self._base_dir))
        self.path = self._tmp / "repo"
        # symlinks=True：符号链接按链接本身复制，不跟随到仓库外部
        shutil.copytree(self.source, self.path, ignore=_IGNORE, symlinks=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._tmp is not None and not self._keep:
            _rmtree(self._tmp)
            self._tmp = None

    def resolve(self, rel_path: str | Path) -> Path:
        """把相对路径解析到副本内；越界（``../``、绝对路径、链接逃逸）则报错。

        这个检查之后会挡在 LLM 可调用的读写工具前面。
        """
        target = (self.path / rel_path).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError(f"路径越出工作副本: {rel_path}")
        return target

    def read_text(self, rel_path: str | Path) -> str:
        return self.resolve(rel_path).read_text(encoding="utf-8")

    def write_text(self, rel_path: str | Path, content: str) -> Path:
        target = self.resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target


def _rmtree(path: Path) -> None:
    # Windows 上只读文件（如 git 对象）直接删除会失败，先去掉只读位再重试
    def _on_error(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:
        shutil.rmtree(path, onerror=_on_error)
